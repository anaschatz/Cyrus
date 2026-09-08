from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_voice.action_approval import ActionApprovalJournal
from jarvis_voice.cloud_approval import CloudApprovalJournal
from jarvis_voice.delivery_journal import DeliveryDedupJournal
from jarvis_voice.scheduler_journal import SchedulerClientJournal
from jarvis_voice.private_state import (
    VoicePrivateStateError,
    audit_voice_journal_state,
    harden_voice_journal_state,
    prepare_voice_journal_state,
    sqlite_journal_family,
)


@contextmanager
def _umask(value: int):
    previous = os.umask(value)
    try:
        yield
    finally:
        os.umask(previous)


@unittest.skipUnless(os.name == "posix", "POSIX owner/mode contract")
class VoicePrivateStateTests(unittest.TestCase):
    def test_default_audit_includes_scheduler_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scheduler = root / "scheduler.sqlite3"
            scheduler.write_bytes(b"fixture")
            scheduler.chmod(0o644)
            with (
                patch("jarvis_voice.delivery_journal.default_delivery_journal_path", return_value=root / "delivery.db"),
                patch("jarvis_voice.cloud_approval.default_cloud_approval_journal_path", return_value=root / "cloud.db"),
                patch("jarvis_voice.action_approval.default_action_approval_journal_path", return_value=root / "actions.db"),
                patch("jarvis_voice.scheduler_journal.default_scheduler_journal_path", return_value=scheduler),
            ):
                self.assertFalse(audit_voice_journal_state(trusted_boundary=root).ready)
                self.assertTrue(harden_voice_journal_state(trusted_boundary=root).ready)
            self.assertEqual(stat.S_IMODE(scheduler.stat().st_mode), 0o600)

    def test_hardener_rejects_fifo_substitution_without_blocking(self) -> None:
        script = '''import os, sys
from pathlib import Path
from unittest.mock import patch
from jarvis_voice.private_state import _chmod_owned_no_follow
p=Path(sys.argv[1]); p.write_bytes(b"fixture"); p.chmod(0o644)
real_open=os.open
def swapped(name, flags, *args, **kwargs):
    if Path(name)==p:
        p.unlink(); os.mkfifo(p, 0o644)
    return real_open(name, flags, *args, **kwargs)
with patch.object(os, "open", side_effect=swapped):
    assert _chmod_owned_no_follow(p, directory=False, mode=0o600) is False
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-c", script, str(Path(directory) / "journal")],
                                    capture_output=True, timeout=2)
            self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_audit_is_passive_and_hardener_migrates_every_existing_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "application" / "state"
            state.mkdir(parents=True)
            state.chmod(0o755)
            journals = tuple(state / f"journal-{index}.sqlite3" for index in range(3))
            existing = []
            for journal in journals:
                journal.write_bytes(b"")
                journal.chmod(0o644)
                sidecar = Path(f"{journal}-wal")
                sidecar.write_bytes(b"")
                sidecar.chmod(0o640)
                existing.extend((journal, sidecar))

            audit = audit_voice_journal_state(
                journals,
                trusted_boundary=root,
            )
            self.assertFalse(audit.ready)
            self.assertEqual(stat.S_IMODE(state.stat().st_mode), 0o755)
            self.assertTrue(
                all(stat.S_IMODE(path.lstat().st_mode) != 0o600 for path in existing)
            )

            hardened = harden_voice_journal_state(
                journals,
                trusted_boundary=root,
            )
            self.assertTrue(hardened.ready)
            self.assertEqual(hardened.hardened_directory_count, 1)
            self.assertEqual(hardened.hardened_file_count, 6)
            self.assertEqual(stat.S_IMODE(state.stat().st_mode), 0o700)
            for path in existing:
                self.assertEqual(stat.S_IMODE(path.lstat().st_mode), 0o600)
                self.assertEqual(path.lstat().st_uid, os.getuid())
            encoded = json.dumps(hardened.to_dict(), sort_keys=True)
            self.assertNotIn(str(root), encoded)
            self.assertNotIn("journal-0", encoded)

    def test_missing_tree_is_safe_to_audit_and_created_private_by_hardener(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _umask(0):
            root = Path(directory)
            journal = root / "application" / "state" / "journal.sqlite3"

            audit = audit_voice_journal_state(
                (journal,),
                trusted_boundary=root,
            )
            self.assertTrue(audit.ready)
            self.assertFalse(journal.exists())

            hardened = harden_voice_journal_state(
                (journal,),
                trusted_boundary=root,
            )
            self.assertTrue(hardened.ready)
            self.assertEqual(
                stat.S_IMODE(journal.parent.lstat().st_mode),
                0o700,
            )
            self.assertEqual(stat.S_IMODE(journal.lstat().st_mode), 0o600)

    def test_symlinked_ancestor_below_trusted_boundary_is_never_followed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            state = real / "state"
            state.mkdir(parents=True)
            state.chmod(0o700)
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            journal = alias / "state" / "journal.sqlite3"

            report = harden_voice_journal_state(
                (journal,),
                trusted_boundary=root,
            )
            with patch(
                "jarvis_voice.delivery_journal.sqlite3.connect"
            ) as connect:
                with self.assertRaises(VoicePrivateStateError):
                    DeliveryDedupJournal(
                        journal,
                        trusted_boundary=root,
                    )

            self.assertFalse(report.ready)
            self.assertEqual(report.violation_counts, {"ancestor_symlink": 1})
            connect.assert_not_called()
            self.assertFalse((state / "journal.sqlite3").exists())
            self.assertTrue(alias.is_symlink())

    def test_final_state_directory_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real-state"
            real.mkdir(mode=0o700)
            linked = root / "state"
            linked.symlink_to(real, target_is_directory=True)

            report = audit_voice_journal_state(
                (linked / "journal.sqlite3",),
                trusted_boundary=root,
            )

            self.assertFalse(report.ready)
            self.assertEqual(report.violation_counts, {"directory_symlink": 1})

    def test_database_and_sidecar_symlinks_fail_before_sqlite_connect(self) -> None:
        constructors = (
            (
                "jarvis_voice.delivery_journal.sqlite3.connect",
                DeliveryDedupJournal,
            ),
            (
                "jarvis_voice.cloud_approval.sqlite3.connect",
                CloudApprovalJournal,
            ),
            (
                "jarvis_voice.action_approval.sqlite3.connect",
                ActionApprovalJournal,
            ),
            (
                "jarvis_voice.scheduler_journal.sqlite3.connect",
                partial(SchedulerClientJournal, cipher=None),
            ),
        )
        for suffix in ("", "-wal", "-shm", "-journal"):
            for connect_name, constructor in constructors:
                with self.subTest(suffix=suffix or "main", constructor=constructor):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        state = root / "state"
                        state.mkdir(mode=0o700)
                        journal = state / "journal.sqlite3"
                        outside = root / "outside"
                        outside.write_bytes(b"outside")
                        outside.chmod(0o644)
                        if suffix:
                            journal.write_bytes(b"")
                            journal.chmod(0o600)
                        Path(f"{journal}{suffix}").symlink_to(outside)

                        with patch(connect_name) as connect:
                            with self.assertRaises(VoicePrivateStateError) as raised:
                                constructor(
                                    journal,
                                    trusted_boundary=root,
                                )

                        connect.assert_not_called()
                        self.assertEqual(outside.read_bytes(), b"outside")
                        self.assertEqual(stat.S_IMODE(outside.stat().st_mode), 0o644)
                        error = json.dumps(raised.exception.to_log_record())
                        self.assertNotIn(str(root), error)
                        self.assertNotIn("outside", error)

    def test_special_database_entry_is_rejected_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir(mode=0o700)
            journal = state / "journal.sqlite3"
            journal.mkdir(mode=0o700)

            report = harden_voice_journal_state(
                (journal,),
                trusted_boundary=root,
            )

            self.assertFalse(report.ready)
            self.assertEqual(report.violation_counts, {"file_type": 1})
            self.assertTrue(journal.is_dir())

    def test_wrong_owner_boundary_is_not_hardened(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir(mode=0o755)
            journal = state / "journal.sqlite3"
            journal.write_bytes(b"")
            journal.chmod(0o644)
            with patch(
                "jarvis_voice.private_state.os.getuid",
                return_value=os.getuid() + 1,
            ):
                report = harden_voice_journal_state(
                    (journal,),
                    trusted_boundary=root,
                )

            self.assertFalse(report.ready)
            self.assertEqual(report.violation_counts, {"ancestor_owner": 1})
            self.assertEqual(stat.S_IMODE(state.stat().st_mode), 0o755)
            self.assertEqual(stat.S_IMODE(journal.stat().st_mode), 0o644)

    def test_real_journals_create_exact_modes_even_under_open_umask(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _umask(0):
            root = Path(directory)
            state = root / "state"
            paths = tuple(state / f"journal-{index}.sqlite3" for index in range(4))
            journals = (
                DeliveryDedupJournal(paths[0], trusted_boundary=root),
                CloudApprovalJournal(paths[1], trusted_boundary=root),
                ActionApprovalJournal(paths[2], trusted_boundary=root),
                SchedulerClientJournal(paths[3], cipher=None, trusted_boundary=root),
            )
            try:
                self.assertEqual(stat.S_IMODE(state.lstat().st_mode), 0o700)
                for path in paths:
                    for candidate in sqlite_journal_family(path):
                        if candidate.exists():
                            self.assertEqual(
                                stat.S_IMODE(candidate.lstat().st_mode),
                                0o600,
                            )
                            self.assertEqual(candidate.lstat().st_uid, os.getuid())
                self.assertTrue(
                    audit_voice_journal_state(
                        paths,
                        trusted_boundary=root,
                    ).ready
                )
            finally:
                for journal in journals:
                    journal.close()

    def test_prepare_rejects_untrusted_path_outside_explicit_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary = root / "trusted"
            boundary.mkdir(mode=0o700)
            journal = root / "outside" / "journal.sqlite3"

            with self.assertRaises(VoicePrivateStateError) as raised:
                prepare_voice_journal_state(
                    (journal,),
                    trusted_boundary=boundary,
                )

            self.assertEqual(
                raised.exception.code,
                "voice.private_state_boundary_invalid",
            )
            self.assertNotIn(str(root), str(raised.exception))


if __name__ == "__main__":
    unittest.main()
