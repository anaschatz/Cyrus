from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator.config import (
    _MAX_CONFIGURATION_BYTES,
    OrchestratorConfig,
)
from jarvis_orchestrator.windows_private_file import harden_windows_private_path

CONTROL_TOKEN = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"
INFERENCE_TOKEN = "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8"


class ConfigSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        if os.name == "nt":  # pragma: no cover - native Windows CI
            harden_windows_private_path(self.root)
        self.path = self.root / "orchestrator.json"

    def document(self) -> dict[str, object]:
        return {
            "database_path": str(self.root / "state.sqlite3"),
            "control_bearer_token": CONTROL_TOKEN,
            "inference_bearer_token": INFERENCE_TOKEN,
        }

    def write(self, raw: bytes, *, mode: int = 0o600) -> Path:
        self.path.write_bytes(raw)
        self.path.chmod(mode)
        if os.name == "nt":  # pragma: no cover - native Windows CI
            harden_windows_private_path(self.path)
        return self.path

    def write_document(self, value: dict[str, object] | None = None) -> Path:
        raw = json.dumps(
            self.document() if value is None else value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return self.write(raw + b"\n")

    def test_owner_only_regular_strict_config_loads(self) -> None:
        config = OrchestratorConfig.load(self.write_document())

        self.assertEqual(config.database_path, self.root / "state.sqlite3")
        self.assertEqual(config.control_bearer_token, CONTROL_TOKEN)
        self.assertEqual(config.inference_bearer_token, INFERENCE_TOKEN)
        self.assertNotIn(CONTROL_TOKEN, repr(config))
        self.assertNotIn(INFERENCE_TOKEN, repr(config))

    def test_response_language_is_closed_and_research_fails_closed(self) -> None:
        for value in ("el", "auto", "EN", ""):
            with self.subTest(value=value):
                document = self.document()
                document["response_language"] = value
                with self.assertRaisesRegex(ValueError, "response_language"):
                    OrchestratorConfig.load(self.write_document(document))

        document = self.document()
        document.update({"response_language": "en", "research_enabled": True})
        with self.assertRaisesRegex(ValueError, "do not yet support research"):
            OrchestratorConfig.from_test_mapping(document)

    def test_file_size_is_bounded_before_parsing(self) -> None:
        self.write(b" " * (_MAX_CONFIGURATION_BYTES + 1))

        with self.assertRaisesRegex(ValueError, "maximum size"):
            OrchestratorConfig.load(self.path)

    @unittest.skipUnless(os.name == "posix", "POSIX symlink contract")
    def test_final_symlink_is_rejected(self) -> None:
        target = self.write_document()
        link = self.root / "orchestrator-link.json"
        link.symlink_to(target)

        with self.assertRaisesRegex(ValueError, "private regular file"):
            OrchestratorConfig.load(link)

    def test_non_regular_path_is_rejected(self) -> None:
        directory = self.root / "config-directory"
        directory.mkdir(mode=0o700)

        with self.assertRaisesRegex(ValueError, "private regular file"):
            OrchestratorConfig.load(directory)

    @unittest.skipUnless(os.name == "posix", "POSIX owner contract")
    def test_non_owner_file_is_rejected(self) -> None:
        self.write_document()
        owner = self.path.stat().st_uid

        with (
            patch("jarvis_orchestrator.config.os.geteuid", return_value=owner + 1),
            self.assertRaisesRegex(ValueError, "private regular file"),
        ):
            OrchestratorConfig.load(self.path)

    @unittest.skipUnless(os.name == "posix", "POSIX mode contract")
    def test_group_world_or_executable_modes_are_rejected(self) -> None:
        for mode in (0o640, 0o604, 0o700):
            with self.subTest(mode=oct(mode)):
                self.write_document()
                self.path.chmod(mode)
                with self.assertRaisesRegex(ValueError, "private regular file"):
                    OrchestratorConfig.load(self.path)

    @unittest.skipUnless(os.name == "nt", "native Windows reparse contract")
    def test_windows_reparse_point_is_rejected(self) -> None:
        target = self.write_document()
        link = self.root / "orchestrator-link.json"
        try:
            link.symlink_to(target)
        except OSError as exc:  # pragma: no cover - host policy dependent
            self.skipTest(f"Windows symlink creation unavailable: {type(exc).__name__}")

        with self.assertRaisesRegex(ValueError, "private regular file"):
            OrchestratorConfig.load(link)

    @unittest.skipUnless(os.name == "nt", "native Windows DACL contract")
    def test_windows_foreign_allow_ace_is_rejected(self) -> None:
        import ntsecuritycon  # type: ignore[import-not-found]
        import win32security  # type: ignore[import-not-found]

        self.write_document()
        security = win32security.GetNamedSecurityInfo(
            str(self.path),
            win32security.SE_FILE_OBJECT,
            win32security.DACL_SECURITY_INFORMATION,
        )
        dacl = security.GetSecurityDescriptorDacl()
        world_sid = win32security.CreateWellKnownSid(win32security.WinWorldSid, None)
        dacl.AddAccessAllowedAce(
            win32security.ACL_REVISION,
            ntsecuritycon.FILE_GENERIC_READ,
            world_sid,
        )
        win32security.SetNamedSecurityInfo(
            str(self.path),
            win32security.SE_FILE_OBJECT,
            win32security.DACL_SECURITY_INFORMATION
            | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
            None,
            None,
            dacl,
            None,
        )

        with self.assertRaisesRegex(ValueError, "private regular file"):
            OrchestratorConfig.load(self.path)

    def test_utf8_bom_and_invalid_utf8_are_rejected(self) -> None:
        valid = json.dumps(self.document(), separators=(",", ":")).encode("utf-8")
        cases = (
            (b"\xef\xbb\xbf" + valid, "BOM"),
            (valid[:-1] + b"\xff}", "strict UTF-8"),
        )
        for raw, expected in cases:
            with self.subTest(expected=expected):
                self.write(raw)
                with self.assertRaisesRegex(ValueError, expected):
                    OrchestratorConfig.load(self.path)

    def test_duplicate_keys_are_rejected_without_echoing_values(self) -> None:
        other_token = "P8byCX4ezFV6gwHU9ijKR3loMS5npQT7rsVW0uxYZ1_A2cdE"
        raw = (
            b'{"database_path":"/tmp/state.sqlite3",'
            + json.dumps("control_bearer_token").encode("ascii")
            + b":"
            + json.dumps(CONTROL_TOKEN).encode("ascii")
            + b","
            + json.dumps("control_bearer_token").encode("ascii")
            + b":"
            + json.dumps(other_token).encode("ascii")
            + b","
            + json.dumps("inference_bearer_token").encode("ascii")
            + b":"
            + json.dumps(INFERENCE_TOKEN).encode("ascii")
            + b"}"
        )
        self.write(raw)

        with self.assertRaisesRegex(ValueError, "duplicate object key") as raised:
            OrchestratorConfig.load(self.path)
        rendered = str(raised.exception)
        self.assertNotIn(CONTROL_TOKEN, rendered)
        self.assertNotIn(other_token, rendered)

    def test_unknown_key_error_does_not_echo_the_key(self) -> None:
        document = self.document()
        document["TOP_SECRET_UNKNOWN_KEY"] = True

        with self.assertRaisesRegex(
            ValueError, "^unknown configuration field$"
        ) as raised:
            OrchestratorConfig.from_mapping(document)
        self.assertNotIn("TOP_SECRET_UNKNOWN_KEY", str(raised.exception))

    def test_non_finite_numbers_are_rejected(self) -> None:
        for constant in (b"NaN", b"Infinity", b"-Infinity", b"1e10000"):
            with self.subTest(constant=constant):
                document = json.dumps(self.document(), separators=(",", ":"))
                raw = (
                    document[:-1].encode("utf-8")
                    + b',"model_timeout_seconds":'
                    + constant
                    + b"}"
                )
                self.write(raw)
                with self.assertRaisesRegex(ValueError, "non-finite number"):
                    OrchestratorConfig.load(self.path)

    def test_unexpected_trailing_json_is_rejected(self) -> None:
        valid = json.dumps(self.document(), separators=(",", ":")).encode("utf-8")
        self.write(valid + b"\n{}")

        with self.assertRaisesRegex(ValueError, "unexpected trailing data"):
            OrchestratorConfig.load(self.path)

    def test_same_size_mutation_during_read_is_rejected(self) -> None:
        raw = json.dumps(self.document(), separators=(",", ":")).encode("utf-8")
        self.write(raw + b" " * (130 * 1024 - len(raw)))
        original_read = os.read
        original_mtime = self.path.stat().st_mtime_ns
        mutated = False

        def read_then_mutate(descriptor: int, amount: int) -> bytes:
            nonlocal mutated
            chunk = original_read(descriptor, amount)
            if chunk and not mutated:
                mutated = True
                with self.path.open("r+b") as stream:
                    stream.seek(-1, os.SEEK_END)
                    stream.write(b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.utime(
                    self.path,
                    ns=(self.path.stat().st_atime_ns, original_mtime + 1_000_000),
                )
            return chunk

        with (
            patch("jarvis_orchestrator.config.os.read", side_effect=read_then_mutate),
            self.assertRaisesRegex(ValueError, "changed while reading"),
        ):
            OrchestratorConfig.load(self.path)
        self.assertTrue(mutated)

    def test_excessive_json_nesting_fails_closed(self) -> None:
        self.write(b"[" * 2_000 + b"0" + b"]" * 2_000)

        with self.assertRaisesRegex(ValueError, "nesting is invalid"):
            OrchestratorConfig.load(self.path)

    def test_noncanonical_or_undersized_bearer_tokens_are_rejected(self) -> None:
        invalid_tokens = (
            "short",
            CONTROL_TOKEN + "=",
            CONTROL_TOKEN[:-1] + "9",
            base64.b64encode(b"\xfb" * 32).decode("ascii").rstrip("="),
            "A9bc DEfghJKLmNPQrSTUvWXYZ0123456789_-A9bc",
            "αλφα" * 16,
        )
        for token in invalid_tokens:
            with self.subTest(token_length=len(token)):
                document = self.document()
                document["control_bearer_token"] = token
                self.write_document(document)
                with self.assertRaisesRegex(
                    ValueError, "canonical bounded bearer token"
                ) as raised:
                    OrchestratorConfig.load(self.path)
                self.assertNotIn(token, str(raised.exception))

    def test_capacity_format_does_not_claim_value_randomness(self) -> None:
        structurally_valid = base64.urlsafe_b64encode(b"\0" * 32).rstrip(b"=").decode()
        document = self.document()
        document["control_bearer_token"] = structurally_valid
        self.write_document(document)

        config = OrchestratorConfig.load(self.path)

        self.assertEqual(config.control_bearer_token, structurally_valid)

    def test_inference_bearer_token_format_and_capacity_are_checked(self) -> None:
        document = self.document()
        document["inference_bearer_token"] = "weak-inference-token"
        self.write_document(document)

        with self.assertRaisesRegex(ValueError, "inference_bearer_token"):
            OrchestratorConfig.load(self.path)

    def test_inline_bearer_token_has_a_hard_maximum(self) -> None:
        oversized = base64.urlsafe_b64encode(b"x" * 385).rstrip(b"=").decode()
        document = self.document()
        document["control_bearer_token"] = oversized
        self.write_document(document)

        with self.assertRaisesRegex(
            ValueError, "canonical bounded bearer token"
        ) as raised:
            OrchestratorConfig.load(self.path)
        self.assertNotIn(oversized, str(raised.exception))

    def test_nonempty_research_token_format_and_capacity_are_checked(self) -> None:
        document = self.document()
        document["research_bearer_token"] = "research-secret"
        self.write_document(document)

        with self.assertRaisesRegex(ValueError, "research_bearer_token"):
            OrchestratorConfig.load(self.path)

    def test_programmatic_fixture_construction_remains_explicitly_separate(
        self,
    ) -> None:
        fixture = {
            "database_path": str(self.root / "state.sqlite3"),
            "control_bearer_token": "fixture-control",
            "inference_bearer_token": "fixture-inference",
        }

        with self.assertRaisesRegex(ValueError, "canonical bounded bearer token"):
            OrchestratorConfig.from_mapping(fixture)
        config = OrchestratorConfig.from_test_mapping(fixture)

        self.assertEqual(config.control_bearer_token, "fixture-control")
        with self.assertRaisesRegex(ValueError, "canonical bounded bearer token"):
            self.write_document(fixture)
            OrchestratorConfig.load(self.path)


if __name__ == "__main__":
    unittest.main()
