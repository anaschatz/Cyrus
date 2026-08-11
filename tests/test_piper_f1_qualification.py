from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.macos.compact_greek_tts_pilot import (
    MAX_EXPORTED_MODEL_BYTES,
    MAX_RUNTIME_RSS_BYTES,
    PIPER_TARGET_SAMPLE_RATE,
)
from jarvis_voice.macos.piper_f1_qualification import (
    CONFIG_NAME,
    FIXTURES,
    MODEL_NAME,
    PcmChunk,
    inspect_piper_f1_candidate,
    piper_f1_qualification_preflight,
    run_piper_f1_qualification,
)


def _config() -> dict[str, object]:
    return {
        "dataset": "stoma_f1",
        "audio": {"sample_rate": PIPER_TARGET_SAMPLE_RATE, "quality": "medium"},
        "espeak": {"voice": "el"},
        "language": {"code": "el"},
        "phoneme_type": "espeak",
        "phoneme_id_map": {"_": [0], "^": [1], " ": [3]},
        "num_speakers": 1,
        "num_symbols": 256,
    }


def _candidate(root: Path) -> None:
    (root / MODEL_NAME).write_bytes(b"fake-onnx")
    (root / CONFIG_NAME).write_text(json.dumps(_config()), encoding="utf-8")


class _FakeSession:
    def __init__(self, model: Path, config: Path) -> None:
        self.model = model
        self.config = config
        self.aborted = False
        self.closed = False
        self.calls: list[str] = []

    @property
    def peak_rss_bytes(self) -> int:
        return MAX_RUNTIME_RSS_BYTES // 2

    def stream(self, text: str):
        self.calls.append(text)
        yield PcmChunk(b"\x00\x00" * PIPER_TARGET_SAMPLE_RATE, PIPER_TARGET_SAMPLE_RATE)

    def abort(self) -> float:
        self.aborted = True
        return 2.0

    def close(self) -> None:
        self.closed = True


class _FakeSink:
    def __init__(self) -> None:
        self.opened = False
        self.aborted = False

    def open(self, chunk: PcmChunk) -> None:
        self.opened = chunk.sample_rate == PIPER_TARGET_SAMPLE_RATE

    def write(self, pcm16: bytes) -> None:
        if not self.opened or not pcm16:
            raise AssertionError("sink was not opened")

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        pass


class PiperF1QualificationTests(unittest.TestCase):
    def test_missing_candidate_is_a_read_only_not_ready_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before = list(root.iterdir())
            report = inspect_piper_f1_candidate(root)
            preflight = piper_f1_qualification_preflight(root, runtime_probe=lambda: False)
            after = list(root.iterdir())
        self.assertFalse(report["ready"])
        self.assertEqual(report["code"], "tts.piper_f1.model_missing")
        self.assertFalse(preflight["installation_performed"])
        self.assertFalse(preflight["network_used"])
        self.assertFalse(preflight["production_backend_changed"])
        self.assertEqual(before, after)

    def test_candidate_validation_pins_names_hashes_and_greek_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _candidate(root)
            report = inspect_piper_f1_candidate(root)
        self.assertTrue(report["ready"])
        self.assertEqual(report["model"]["name"], MODEL_NAME)
        self.assertEqual(len(report["model"]["sha256"]), 64)
        self.assertEqual(len(report["config"]["sha256"]), 64)

    def test_candidate_rejects_incompatible_config_and_oversize_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _candidate(root)
            incompatible = _config()
            incompatible["language"] = {"code": "en"}
            (root / CONFIG_NAME).write_text(json.dumps(incompatible), encoding="utf-8")
            self.assertEqual(
                inspect_piper_f1_candidate(root)["code"],
                "tts.piper_f1.config_incompatible",
            )
            (root / CONFIG_NAME).write_text(json.dumps(_config()), encoding="utf-8")
            with (root / MODEL_NAME).open("r+b") as stream:
                stream.truncate(MAX_EXPORTED_MODEL_BYTES + 1)
            self.assertEqual(
                inspect_piper_f1_candidate(root)["code"],
                "tts.piper_f1.model_size_invalid",
            )

    def test_fake_runtime_exercises_all_gates_without_piper_install(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "candidate"
            candidate.mkdir()
            _candidate(candidate)
            output = root / "output"
            sessions: list[_FakeSession] = []

            def factory(model: Path, config: Path) -> _FakeSession:
                session = _FakeSession(model, config)
                sessions.append(session)
                return session

            report = run_piper_f1_qualification(
                candidate,
                output,
                session_factory=factory,
                playback_sink_factory=_FakeSink,
            )

            wavs = sorted(output.glob("*.wav"))
        self.assertEqual(len(wavs), len(FIXTURES))
        self.assertTrue(report["automated_pass"])
        self.assertFalse(report["ready_for_promotion"])
        self.assertEqual(report["code"], "tts.piper_f1.human_approval_required")
        self.assertTrue(report["metrics"]["cancellation"]["measured"])
        self.assertTrue(sessions[0].aborted)
        self.assertTrue(sessions[0].closed)
        self.assertFalse(report["installation_performed"])
        self.assertFalse(report["production_backend_changed"])

    def test_no_playback_measurement_cannot_pass_cancellation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "candidate"
            candidate.mkdir()
            _candidate(candidate)
            report = run_piper_f1_qualification(
                candidate,
                root / "output",
                session_factory=_FakeSession,
            )
        self.assertFalse(report["automated_pass"])
        self.assertFalse(report["gates"]["cancellation"])


if __name__ == "__main__":
    unittest.main()
