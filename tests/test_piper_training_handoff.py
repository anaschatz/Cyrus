from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.macos.compact_greek_tts_pilot import ArtifactSpec
from jarvis_voice.piper_training_handoff import (
    BUNDLE_SCHEMA,
    HANDOFF_SCHEMA,
    build_piper_f1_handoff,
    verify_piper_f1_handoff,
    write_piper_f1_handoff,
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class PiperTrainingHandoffTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, tuple[ArtifactSpec, ...]]:
        data = root / "f1-training-data"
        audio = data / "audio"
        audio.mkdir(parents=True)
        clips = {"Greek-F1-A.wav": b"wav-a", "Greek-F1-B.wav": b"wav-b"}
        for name, value in clips.items():
            (audio / name).write_bytes(value)
        metadata = b"Greek-F1-A.wav|alpha\nGreek-F1-B.wav|beta\n"
        (data / "metadata.csv").write_bytes(metadata)
        bundle = {
            "schema": BUNDLE_SCHEMA,
            "dataset_revision": "dataset-revision",
            "recipe_tree": "recipe-tree",
            "speaker": "F1",
            "rows": 2,
            "rejected_f2_rows": 0,
            "source_sample_rate": 44100,
            "piper_sample_rate": 22050,
            "metadata_sha256": digest(metadata),
            "audio_sha256": {name: digest(value) for name, value in clips.items()},
            "network_used": False,
        }
        (data / "bundle-manifest.json").write_text(json.dumps(bundle), encoding="utf-8")

        artifact_root = root / "artifacts"
        checkpoint = artifact_root / "checkpoint" / "el/model"
        checkpoint.mkdir(parents=True)
        checkpoint_specs = []
        for name, value in (("MODEL_CARD", b"card"), ("config.json", b"config"), ("model.ckpt", b"model")):
            (checkpoint / name).write_bytes(value)
            checkpoint_specs.append(ArtifactSpec(f"el/model/{name}", len(value), digest(value), "https://example.invalid"))
        return data, artifact_root, tuple(checkpoint_specs)

    def patches(self, specs: tuple[ArtifactSpec, ...]):
        return mock.patch.multiple(
            "jarvis_voice.piper_training_handoff",
            STOMA_SPEAKER_CLIPS=2,
            STOMA_REVISION="dataset-revision",
            STOMA_GITHUB_TREE="recipe-tree",
            PIPER_CHECKPOINT_DIRECTORY="el/model",
            PIPER_CHECKPOINT_ARTIFACTS=specs,
        )

    def test_build_write_and_verify_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, artifacts, specs = self.fixture(root)
            with self.patches(specs):
                manifest = build_piper_f1_handoff(data, artifacts)
                output = root / "handoff.json"
                write_piper_f1_handoff(output, manifest)
                transfer = root / "transfer"
                transfer.mkdir()
                (transfer / "f1-training-data").symlink_to(data, target_is_directory=True)
                (transfer / "checkpoint").symlink_to(artifacts / "checkpoint", target_is_directory=True)
                # Symlinked transfer inputs are intentionally rejected.
                with self.assertRaisesRegex(BackendError, "path_invalid"):
                    verify_piper_f1_handoff(transfer, output)

                for link in transfer.iterdir():
                    link.unlink()
                shutil.copytree(data, transfer / "f1-training-data")
                shutil.copytree(artifacts / "checkpoint", transfer / "checkpoint")
                report = verify_piper_f1_handoff(transfer, output)
                self.assertTrue(report["ready"])
                self.assertEqual(report["files"], 7)

    def test_changed_audio_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, artifacts, specs = self.fixture(root)
            (data / "audio" / "Greek-F1-A.wav").write_bytes(b"changed")
            with self.patches(specs), self.assertRaisesRegex(BackendError, "audio_hash_mismatch"):
                build_piper_f1_handoff(data, artifacts)

    def test_extra_audio_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, artifacts, specs = self.fixture(root)
            (data / "audio" / "Greek-F1-extra.wav").write_bytes(b"extra")
            with self.patches(specs), self.assertRaisesRegex(BackendError, "audio_set_mismatch"):
                build_piper_f1_handoff(data, artifacts)

    def test_verify_rejects_self_consistent_transfer_with_wrong_pins_or_bundle(self) -> None:
        for relative, error in (
            ("checkpoint/el/model/model.ckpt", "checkpoint_hash_mismatch"),
            ("f1-training-data/metadata.csv", "metadata_hash_mismatch"),
            ("f1-training-data/audio/Greek-F1-A.wav", "audio_hash_mismatch"),
            ("f1-training-data/bundle-manifest.json", "bundle_incompatible"),
        ):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data, artifacts, specs = self.fixture(root)
                with self.patches(specs):
                    manifest = build_piper_f1_handoff(data, artifacts)
                    shutil.move(artifacts / "checkpoint", root / "checkpoint")
                    changed = root / relative
                    if relative.endswith("bundle-manifest.json"):
                        bundle = json.loads(changed.read_text())
                        bundle["dataset_revision"] = "wrong-revision"
                        changed.write_text(json.dumps(bundle))
                    else:
                        changed.write_bytes(b"altered")
                    item = next(item for item in manifest["files"] if item["path"] == relative)
                    item.update(size=changed.stat().st_size, sha256=digest(changed.read_bytes()))
                    manifest["totals"]["bytes"] = sum(item["size"] for item in manifest["files"])
                    output = root / "handoff.json"
                    write_piper_f1_handoff(output, manifest)
                    with self.assertRaisesRegex(BackendError, error):
                        verify_piper_f1_handoff(root, output)

    def test_write_never_overwrites_a_racing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "handoff.json"
            original_link = os.link

            def racing_link(source: Path, destination: Path) -> None:
                Path(destination).write_bytes(b"racing-writer")
                original_link(source, destination)

            with mock.patch(
                "jarvis_voice.piper_training_handoff.os.link",
                side_effect=racing_link,
            ), self.assertRaises(OSError):
                write_piper_f1_handoff(output, {"schema": HANDOFF_SCHEMA})
            self.assertEqual(output.read_bytes(), b"racing-writer")
            self.assertEqual(list(output.parent.glob(".piper-f1-handoff-*")), [])


if __name__ == "__main__":
    unittest.main()
