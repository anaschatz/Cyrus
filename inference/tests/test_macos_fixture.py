from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jarvis_inference.macos_fixture import (
    FixtureFailure,
    QualificationClient,
    _memory_drift,
    _read_identity,
    _write_evidence,
)


class MacosFixtureTests(unittest.TestCase):
    def test_loopback_url_is_exact(self) -> None:
        client = QualificationClient("http://127.0.0.1:1234/v1", "secret", "jarvis-local")
        self.assertEqual(client.port, 1234)
        for value in (
            "https://127.0.0.1:1234/v1",
            "http://localhost:1234/v1",
            "http://127.0.0.1:1234/other",
        ):
            with self.subTest(value=value), self.assertRaises(FixtureFailure):
                QualificationClient(value, "secret", "jarvis-local")

    def test_memory_drift_is_bounded_and_deterministic(self) -> None:
        self.assertEqual(_memory_drift([1_000_000] * 200), (0.0, 0.0))
        final, projected = _memory_drift([1_000_000 + index * 1_000 for index in range(200)])
        self.assertGreater(final, 0.05)
        self.assertGreater(projected, 0.05)
        with self.assertRaises(FixtureFailure):
            _memory_drift([1] * 199)

    def test_identity_comes_only_from_current_in_progress_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "alias": "jarvis-local",
                "qualification_report": "qualification-report.json",
            }))
            report = root / "qualification-report.json"
            report.write_text(json.dumps({
                "status": "in_progress",
                "challenge": "c" * 64,
                "identity": {"manifest_digest": "d" * 64},
            }))
            alias, identity = _read_identity(manifest, "c" * 64)
            self.assertEqual(alias, "jarvis-local")
            self.assertEqual(identity["manifest_digest"], "d" * 64)
            with self.assertRaises(FixtureFailure):
                _read_identity(manifest, "e" * 64)

    def test_evidence_write_is_fresh_and_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence.json"
            _write_evidence(output, {"ready": True})
            self.assertEqual(json.loads(output.read_text()), {"ready": True})
            with self.assertRaises(FixtureFailure):
                _write_evidence(output, {"ready": False})


if __name__ == "__main__":
    unittest.main()
