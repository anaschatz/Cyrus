from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.artifact_audit import discover_artifacts, run_artifact_audit
from jarvis_voice.cli import main
from jarvis_voice.errors import BackendError
from jarvis_voice.manifest import ModelFileSpec, ModelManifest


class ArtifactAuditTests(unittest.TestCase):
    def populate(self, root: Path) -> None:
        ct2 = root / "ct2"
        ct2.mkdir()
        for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
            (ct2 / name).write_bytes(("ct2-" + name).encode())
        (ct2 / "extra.txt").write_text("allowed")
        (root / "ggml-large-v3-turbo-q5_0.bin").write_bytes(b"ggml-model")
        (root / "base.pt").write_bytes(b"checkpoint")
        (root / "silero_vad.onnx").write_bytes(b"vad")
        server = root / "whisper-server"
        server.write_bytes(b"sidecar")
        server.chmod(0o700)
        # These are deliberately not approved patterns.
        (root / "model.bin").write_bytes(b"not-a-standalone-candidate")
        (root / "unrelated.gguf").write_bytes(b"llm")
        (root / "weights.pt").write_bytes(b"unrelated")
        (root / "network.onnx").write_bytes(b"unrelated")

    def test_exact_classification_false_positive_rejection_and_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            models = base / "models"
            evidence = base / "evidence"
            models.mkdir(); evidence.mkdir()
            self.populate(models)
            output = evidence / "summary.json"
            private = evidence / "private.json"
            summary = run_artifact_audit(
                [models],
                output=output,
                private_manifest=private,
                now=datetime(2026, 8, 10, tzinfo=UTC),
            )
            private_raw = json.loads(private.read_text())
            redacted = output.read_text()
            private_mode = stat.S_IMODE(private.stat().st_mode)
        self.assertTrue(summary["completed"])
        self.assertEqual(summary["counts"], {
            "faster_whisper_ct2": 1,
            "whisper_cpp_model": 1,
            "openai_whisper_checkpoint": 1,
            "silero_vad_onnx": 1,
            "whisper_cpp_server": 1,
        })
        self.assertEqual(summary["next_action"], "artifact.cross_platform_candidates_found")
        self.assertFalse(summary["model_content_read"])
        self.assertNotIn(str(models), redacted)
        self.assertNotIn("ggml-large", redacted)
        self.assertNotIn("sha256", redacted)
        self.assertTrue(private_raw["path_disclosure"])
        self.assertEqual(len(private_raw["candidates"]), 5)
        self.assertEqual(private_mode, 0o600)

    def test_hashing_is_explicit_and_candidate_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            models = base / "models"
            evidence = base / "evidence"
            models.mkdir(); evidence.mkdir()
            self.populate(models)
            specs = []
            for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
                data = (models / "ct2" / name).read_bytes()
                specs.append(ModelFileSpec(name, len(data), hashlib.sha256(data).hexdigest()))
            manifest = ModelManifest("test/model", "test-revision", "faster-whisper/ctranslate2", tuple(specs))
            with mock.patch("jarvis_voice.artifact_audit.BASE_MODEL_MANIFEST", manifest):
                summary = run_artifact_audit(
                    [models],
                    output=evidence / "summary.json",
                    private_manifest=evidence / "private.json",
                    hash_candidates=True,
                )
            private = json.loads((evidence / "private.json").read_text())
        self.assertTrue(summary["model_content_read"])
        self.assertEqual(summary["scan"]["hashed_artifacts"], 8)
        self.assertEqual(summary["pinned_manifest_matches"], 1)
        artifact_records = [artifact for candidate in private["candidates"] for artifact in candidate["artifacts"]]
        self.assertTrue(all(record["hash_status"] == "complete" and len(record["sha256"]) == 64 for record in artifact_records))
        disclosed_names = {Path(record["path"]).name for record in artifact_records}
        self.assertNotIn("unrelated.gguf", disclosed_names)
        self.assertNotIn("weights.pt", disclosed_names)

    def test_hugging_face_file_symlinks_are_classified_without_following_directory_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            blobs = root / "blobs"
            snapshot = root / "snapshot"
            blobs.mkdir(); snapshot.mkdir()
            for index, name in enumerate(("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")):
                target = blobs / f"blob-{index}"
                target.write_bytes(name.encode())
                (snapshot / name).symlink_to(target)
            outside = root / "outside"
            outside.mkdir()
            (root / "linked-directory").symlink_to(outside, target_is_directory=True)
            _, candidates, state = discover_artifacts([root])
        self.assertEqual([(item.kind, item.symlink) for item in candidates], [("faster_whisper_ct2", True)])
        self.assertEqual(state.symlink_directories_skipped, 1)

    def test_limits_make_partial_result_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            models = base / "models"
            evidence = base / "evidence"
            models.mkdir(); evidence.mkdir()
            (models / "ggml-a.bin").write_bytes(b"a")
            (models / "ggml-b.bin").write_bytes(b"b")
            summary = run_artifact_audit(
                [models],
                output=evidence / "summary.json",
                private_manifest=evidence / "private.json",
                max_candidates=1,
            )
        self.assertFalse(summary["completed"])
        self.assertTrue(summary["scan"]["truncated"])
        self.assertEqual(summary["scan"]["candidates"], 1)

    def test_roots_outputs_and_limits_are_guarded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(BackendError, "artifact.root_not_absolute"):
                discover_artifacts(["relative"])
            with self.assertRaisesRegex(BackendError, "artifact.root_too_broad"):
                discover_artifacts([Path("/")])
            with self.assertRaisesRegex(BackendError, "artifact.invalid_limit"):
                discover_artifacts([root], max_depth=33)
            same = root / "same.json"
            with self.assertRaisesRegex(BackendError, "artifact.output_collision"):
                run_artifact_audit([root], output=same, private_manifest=same)
            with self.assertRaisesRegex(BackendError, "artifact.output_not_absolute"):
                run_artifact_audit([root], output="summary.json", private_manifest=root / "private.json")

    def test_cli_runs_without_stdout_paths_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            models = base / "models"
            evidence = base / "evidence"
            models.mkdir(); evidence.mkdir()
            (models / "base.pt").write_bytes(b"checkpoint")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([
                    "artifact-audit",
                    "--root", str(models),
                    "--output", str(evidence / "summary.json"),
                    "--private-manifest", str(evidence / "private.json"),
                ])
            summary = json.loads((evidence / "summary.json").read_text())
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(summary["next_action"], "artifact.conversion_would_be_required")
        self.assertFalse(summary["network_used"])


if __name__ == "__main__":
    unittest.main()
