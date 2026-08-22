from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_orchestrator.domain import canonical_json
from jarvis_orchestrator.translation import (
    TRANSLATION_CONVERSION_POLICY_SHA256,
    TRANSLATION_MODEL_REVISION,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPOSITORY_ROOT / "orchestrator" / "scripts" / "build_translation_pack.py"
BUILD_LOCK = (
    REPOSITORY_ROOT
    / "orchestrator"
    / "translation"
    / "translation-build-lock.json"
)
BUILD = runpy.run_path(str(BUILD_SCRIPT))


class TranslationBuildTests(unittest.TestCase):
    def test_tracked_build_lock_is_canonical_and_bound_to_runtime(self) -> None:
        document = BUILD["load_build_lock"](BUILD_LOCK)

        self.assertEqual(document["revision"], TRANSLATION_MODEL_REVISION)
        self.assertEqual(
            hashlib.sha256(canonical_json(document).encode()).hexdigest(),
            TRANSLATION_CONVERSION_POLICY_SHA256,
        )
        self.assertEqual(
            [item["path"] for item in document["runtime_files"]],
            [
                "config.json",
                "model.bin",
                "shared_vocabulary.json",
                "source.spm",
                "target.spm",
            ],
        )
        self.assertEqual(
            sum(item["size"] for item in document["runtime_files"]),
            243_579_143,
        )

    def test_build_lock_rejects_duplicate_keys_and_policy_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            raw = BUILD_LOCK.read_bytes().replace(
                b'"version": "jarvis.translation-build-lock/1"',
                b'"version": "bad", "version": "jarvis.translation-build-lock/1"',
            )
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                BUILD["load_build_lock"](path)

            document = json.loads(BUILD_LOCK.read_text())
            document["conversion"]["force"] = True
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conversion policy"):
                BUILD["load_build_lock"](path)

    def test_closed_artifact_verifier_rejects_extra_or_changed_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.bin"
            artifact.write_bytes(b"model")
            expected = (
                {
                    "path": "model.bin",
                    "size": 5,
                    "sha256": hashlib.sha256(b"model").hexdigest(),
                },
            )
            BUILD["verify_closed_artifacts"](root, expected)

            (root / "extra").mkdir()
            with self.assertRaisesRegex(ValueError, "not closed"):
                BUILD["verify_closed_artifacts"](root, expected)
            (root / "extra").rmdir()
            artifact.write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "digest"):
                BUILD["verify_closed_artifacts"](root, expected)

    def test_builder_calls_exact_offline_conversion_api(self) -> None:
        captured: dict[str, object] = {}

        class Converter:
            def __init__(self, model: str, **kwargs: object) -> None:
                captured["model"] = model
                captured["constructor"] = kwargs

            def convert(self, output: str, **kwargs: object) -> None:
                captured["output"] = output
                captured["convert"] = kwargs
                Path(output).mkdir()

        converters = types.ModuleType("ctranslate2.converters")
        converters.TransformersConverter = Converter
        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.converters = converters

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            output = root / "model"
            manifest = root / "manifest.json"
            with (
                patch.dict(
                    sys.modules,
                    {
                        "ctranslate2": ctranslate2,
                        "ctranslate2.converters": converters,
                    },
                ),
                patch.object(sys, "addaudithook"),
                patch.dict(os.environ, {}, clear=False),
            ):
                with patch.dict(
                    BUILD["build_pack"].__globals__,
                    {
                        "verify_builder": lambda document: None,
                        "verify_closed_artifacts": lambda selected, artifacts: None,
                        "load_translation_manifest": lambda *args: None,
                    },
                ):
                    result = BUILD["build_pack"](
                        source_root=source,
                        output_root=output,
                        build_lock_path=BUILD_LOCK,
                        manifest_output=manifest,
                    )

        self.assertEqual(captured["model"], str(source))
        self.assertEqual(
            captured["constructor"],
            {
                "activation_scales": None,
                "copy_files": ["source.spm", "target.spm"],
                "load_as_float16": True,
                "revision": None,
                "low_cpu_mem_usage": False,
                "trust_remote_code": False,
            },
        )
        self.assertEqual(
            captured["convert"],
            {"vmap": None, "quantization": "int8", "force": False},
        )
        self.assertEqual(
            result["build_lock_sha256"],
            TRANSLATION_CONVERSION_POLICY_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
