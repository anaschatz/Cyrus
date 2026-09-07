from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401
from jarvis_voice.errors import BackendError
from jarvis_voice.manifest import ModelFileSpec, ModelManifest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GreekTtsTrainingScriptTests(unittest.TestCase):
    def test_scoring_scripts_verify_the_model_before_loading(self):
        for name in ("score_greektts15_candidates", "score_greektts15_pronunciation", "score_greektts15_quality_candidate"):
            for valid in (False, True):
                with self.subTest(script=name, valid=valid), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp).resolve()
                    model_dir = root / "model"
                    model_dir.mkdir()
                    specs = []
                    for filename in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
                        content = filename.encode()
                        (model_dir / filename).write_bytes(content)
                        specs.append(ModelFileSpec(filename, len(content), hashlib.sha256(content).hexdigest()))
                    manifest = ModelManifest("fixture/whisper", "fixture-revision", "faster-whisper/ctranslate2", tuple(specs))
                    if not valid:
                        (model_dir / "model.bin").write_bytes(b"x" * len(b"model.bin"))
                    directories = {}
                    for label in ("baseline", "candidate", "second", "dataset"):
                        directory = root / label
                        directory.mkdir()
                        (directory / "01-fixture.wav").write_bytes(b"synthetic audio")
                        directories[label] = str(directory)
                    audio = root / "dataset" / "audio"
                    audio.mkdir()
                    (audio / "native.wav").write_bytes(b"synthetic audio")
                    output = root / "report.json"
                    args = [name, "--model-dir", str(model_dir), "--baseline", directories["baseline"], "--report", str(output)]
                    if name.endswith("candidates"):
                        args += ["--lr-5e-7", directories["candidate"], "--lr-1e-6", directories["second"]]
                    elif name.endswith("pronunciation"):
                        args += ["--dataset-root", directories["dataset"], "--blend", directories["candidate"]]
                    else:
                        args += ["--candidate", directories["candidate"], "--candidate-label", "attention-only"]
                    module = load_script(name)
                    model = mock.Mock()
                    model.transcribe.side_effect = lambda *a, **k: (iter([SimpleNamespace(text="γειά", avg_logprob=-0.1)]), SimpleNamespace(language_probability=0.9))
                    factory = mock.Mock(return_value=model)
                    fixture_name = "PRONUNCIATION_REFERENCES" if name.endswith("pronunciation") else "QUALITY_FIXTURES"
                    fixtures = (SimpleNamespace(fixture_id="fixture", filename="native.wav", text="γειά"),) if name.endswith("pronunciation") else (("fixture", "γειά"),)
                    with mock.patch.object(module, "BASE_MODEL_MANIFEST", manifest), mock.patch.object(module, fixture_name, fixtures), mock.patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=factory)}), mock.patch.object(sys, "argv", args), mock.patch.dict(os.environ), redirect_stdout(io.StringIO()):
                        if valid:
                            self.assertEqual(module.main(), 0)
                            report = json.loads(output.read_text())
                            self.assertEqual(report["asr"]["model"], manifest.model_id)
                            self.assertFalse(report["promotion_allowed"])
                            self.assertTrue(report["human_listening_required"])
                            self.assertTrue(factory.call_args.kwargs["local_files_only"])
                        else:
                            with self.assertRaisesRegex(SystemExit, "model.hash_mismatch"):
                                module.main()
                            factory.assert_not_called()
                            self.assertFalse(output.exists())

    def test_cache_and_training_forward_arguments_and_report_backend_errors(self):
        for name in ("cache_greektts15_snac", "train_greektts15_pilot"):
            for failed in (False, True):
                with self.subTest(script=name, failed=failed), tempfile.TemporaryDirectory() as temp, ExitStack() as patches:
                    module = load_script(name)
                    output = Path(temp) / "report.json"
                    args = [name, "--plan", "/fixture/plan", "--cache-root", "/fixture/cache", "--report", str(output)]
                    if name.startswith("cache"):
                        args += ["--dataset-root", "/fixture/dataset", "--snac-root", "/fixture/snac", "--limit", "3"]
                        encoder = object()
                        factory = patches.enter_context(mock.patch.object(module, "TorchSnacEncoder", return_value=encoder))
                        operation_name = "cache_snac_samples"
                        positional = ("/fixture/plan", "/fixture/dataset", "/fixture/cache", encoder)
                        keywords = {"limit": 3}
                    else:
                        args += ["--model-root", "/fixture/model", "--gradient-report", "/fixture/gradient", "--output-dir", "/fixture/output", "--learning-rate", "2.5e-7", "--target-profile", "mlp-only", "--sample-profile", "jarvis-lexical"]
                        operation_name = "run_training_pilot"
                        positional = ("/fixture/model", "/fixture/plan", "/fixture/cache", "/fixture/gradient", "/fixture/output")
                        keywords = dict(learning_rate=2.5e-7, steps=25, target_profile="mlp-only", eos_loss_weight=1,
                                        sample_profile="jarvis-lexical", source_adapter_directory=None, general_stage_index=0)
                    operation = patches.enter_context(mock.patch.object(module, operation_name, return_value={"ready": True},
                                                                         side_effect=BackendError("tts.fixture_failed", "fixture", subsystem="tts") if failed else None))
                    patches.enter_context(mock.patch.object(sys, "argv", args))
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(module.main(), 2 if failed else 0)
                    expected = {"ready": False, "code": "tts.fixture_failed"} if failed else {"ready": True}
                    self.assertEqual(json.loads(output.read_text()), expected)
                    operation.assert_called_once_with(*positional, **keywords)
                    if name.startswith("cache"):
                        factory.assert_called_once_with("/fixture/snac")


if __name__ == "__main__":
    unittest.main()
