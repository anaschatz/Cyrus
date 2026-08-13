from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.macos.greektts15_pilot import (
    BASE_ARTIFACTS,
    BASE_8BIT_ARTIFACTS,
    CONVERSION_RESERVE_BYTES,
    EXPECTED_MLX_KEYS,
    MINIMUM_ACQUISITION_FREE_BYTES,
    MODEL_DOWNLOAD_BYTES,
    MODEL_LAYERS,
    PEFT_ARTIFACTS,
    PRONUNCIATION_REFERENCES,
    QUALITY_FIXTURES,
    QUALITY_MAX_MEAN_REALTIME_FACTOR,
    QUALITY_MAX_MLX_MEMORY_BYTES,
    QUALITY_MAX_RSS_BYTES,
    SNAC_ARTIFACTS,
    START_OF_SPEECH,
    END_OF_SPEECH,
    AUDIO_TOKEN_OFFSET,
    CODEBOOK_SIZE,
    _deny_network_during_pilot,
    expected_converted_weight_keys,
    extract_snac_codes,
    format_pilot_prompt,
    greektts15_pilot_preflight,
    remap_peft_weight_key,
    run_greektts15_pilot,
    run_greektts15_pronunciation_qualification,
    run_greektts15_quality_qualification,
    validate_greektts_model_root,
    _native_adapter_config,
)


class GreekTts15PilotTests(unittest.TestCase):
    def test_native_training_adapter_config_remains_rank_64(self) -> None:
        config = _native_adapter_config()
        self.assertEqual(config["lora_parameters"]["rank"], 64)
        self.assertEqual(config["lora_parameters"]["scale"], 1.0)
        self.assertEqual(len(config["lora_parameters"]["keys"]), 7)

    def test_candidate_temperature_choices_are_bounded(self) -> None:
        source = Path("src/jarvis_voice/macos/greektts15_pilot.py").read_text(encoding="utf-8")
        self.assertIn('temperature not in {0.0, 0.2, 0.4, 0.6}', source)
    def test_network_guard_fails_closed_and_restores_socket(self) -> None:
        original_socket = socket.socket

        @_deny_network_during_pilot
        def guarded() -> dict[str, object]:
            with socket.socket() as stream:
                with self.assertRaises(BackendError) as raised:
                    stream.connect(("127.0.0.1", 9))
            self.assertEqual(raised.exception.code, "tts.greektts15.network_forbidden")
            return {"ready": True}

        self.assertTrue(guarded()["ready"])
        self.assertIs(socket.socket, original_socket)

    def test_preflight_is_read_only_and_blocks_low_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = greektts15_pilot_preflight(
                temp,
                Path(temp) / "models",
                platform_name="darwin",
                machine="arm64",
                python_version=(3, 11, 14),
                pointer_bits=64,
                disk_usage=lambda path: SimpleNamespace(free=4 * 1024**3),
                dependency_probe=lambda name, version: False,
            )
        self.assertFalse(report["ready"])
        self.assertFalse(report["acquisition_ready"])
        self.assertEqual(report["code"], "tts.greektts15.disk_space_insufficient")
        self.assertEqual(report["disk"]["acquisition_required_mib"], MINIMUM_ACQUISITION_FREE_BYTES // 1024**2)
        self.assertFalse(report["network_used"])
        self.assertFalse(report["installation_performed"])

    def test_exact_dependency_set_uses_download_conversion_and_run_reserve(self) -> None:
        required = MODEL_DOWNLOAD_BYTES + CONVERSION_RESERVE_BYTES + 2 * 1024**3
        with tempfile.TemporaryDirectory() as temp:
            report = greektts15_pilot_preflight(
                temp,
                Path(temp) / "models",
                platform_name="darwin",
                machine="arm64",
                python_version=(3, 11, 14),
                pointer_bits=64,
                disk_usage=lambda path: SimpleNamespace(free=required),
                dependency_probe=lambda name, version: True,
            )
        self.assertTrue(report["acquisition_ready"])
        self.assertEqual(report["disk"]["acquisition_required_mib"], required // 1024**2)

    def test_host_gate_requires_native_arm64_python_311(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = greektts15_pilot_preflight(
                temp,
                Path(temp) / "models",
                platform_name="darwin",
                machine="arm64",
                python_version=(3, 14, 0),
                pointer_bits=64,
                disk_usage=lambda path: SimpleNamespace(free=20 * 1024**3),
                dependency_probe=lambda name, version: True,
            )
        self.assertEqual(report["code"], "tts.greektts15.host_unsupported")
        self.assertFalse(report["acquisition_ready"])

    def test_model_validation_accepts_exact_pinned_artifacts_and_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            groups = (
                (root / "base", BASE_ARTIFACTS),
                (root / "peft", PEFT_ARTIFACTS),
                (root / "snac", SNAC_ARTIFACTS),
            )
            hashes: dict[str, str] = {}
            for directory, artifacts in groups:
                for artifact in artifacts:
                    path = directory / artifact.relative_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("wb") as stream:
                        stream.truncate(artifact.size)
                    hashes[str(path)] = artifact.sha256
            (root / "base" / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "llama",
                        "num_hidden_layers": 28,
                        "hidden_size": 3072,
                        "intermediate_size": 8192,
                        "vocab_size": 156940,
                        "quantization": {"group_size": 64, "bits": 4},
                    }
                ).ljust(BASE_ARTIFACTS[0].size)
            )
            (root / "peft" / "checkpoint-264000" / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "peft_type": "LORA",
                        "r": 64,
                        "lora_alpha": 64,
                        "lora_dropout": 0,
                        "use_dora": False,
                        "use_rslora": False,
                        "target_modules": [key.split(".")[-1] for key in EXPECTED_MLX_KEYS],
                    }
                ).ljust(PEFT_ARTIFACTS[0].size)
            )
            (root / "snac" / "config.json").write_text(
                json.dumps({"sampling_rate": 24000, "codebook_size": 4096, "vq_strides": [4, 2, 1]}).ljust(
                    SNAC_ARTIFACTS[0].size
                )
            )
            with mock.patch(
                "jarvis_voice.macos.greektts15_pilot._sha256",
                side_effect=lambda path: hashes[str(path)],
            ):
                result = validate_greektts_model_root(root)
        self.assertTrue(result.ready)

    def test_peft_key_mapping_is_exact(self) -> None:
        key = "base_model.model.model.layers.27.mlp.down_proj.lora_B.weight"
        self.assertEqual(remap_peft_weight_key(key), "model.layers.27.mlp.down_proj.lora_b")
        self.assertIsNone(remap_peft_weight_key(key.replace("layers.27", "layers.28")))
        self.assertIsNone(remap_peft_weight_key(key.replace("down_proj", "lm_head")))
        self.assertEqual(len(expected_converted_weight_keys()), MODEL_LAYERS * len(EXPECTED_MLX_KEYS) * 2)

    def test_8bit_variant_requires_8bit_artifacts_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            groups = (
                (root / "base-8bit", BASE_8BIT_ARTIFACTS),
                (root / "peft", PEFT_ARTIFACTS),
                (root / "snac", SNAC_ARTIFACTS),
            )
            hashes: dict[str, str] = {}
            for directory, artifacts in groups:
                for artifact in artifacts:
                    path = directory / artifact.relative_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("wb") as stream:
                        stream.truncate(artifact.size)
                    hashes[str(path)] = artifact.sha256
            (root / "base-8bit" / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "llama",
                        "num_hidden_layers": 28,
                        "hidden_size": 3072,
                        "intermediate_size": 8192,
                        "vocab_size": 156940,
                        "quantization": {"group_size": 64, "bits": 8},
                    }
                ).ljust(BASE_8BIT_ARTIFACTS[0].size)
            )
            (root / "peft" / "checkpoint-264000" / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "peft_type": "LORA",
                        "r": 64,
                        "lora_alpha": 64,
                        "lora_dropout": 0,
                        "use_dora": False,
                        "use_rslora": False,
                        "target_modules": [key.split(".")[-1] for key in EXPECTED_MLX_KEYS],
                    }
                ).ljust(PEFT_ARTIFACTS[0].size)
            )
            (root / "snac" / "config.json").write_text(
                json.dumps({"sampling_rate": 24000, "codebook_size": 4096, "vq_strides": [4, 2, 1]}).ljust(
                    SNAC_ARTIFACTS[0].size
                )
            )
            with mock.patch(
                "jarvis_voice.macos.greektts15_pilot._sha256",
                side_effect=lambda path: hashes[str(path)],
            ):
                result = validate_greektts_model_root(root, base_variant="8bit")
        self.assertTrue(result.ready)

    def test_snac_code_redistribution_matches_official_layout(self) -> None:
        values = [
            1,
            CODEBOOK_SIZE + 2,
            2 * CODEBOOK_SIZE + 3,
            3 * CODEBOOK_SIZE + 4,
            4 * CODEBOOK_SIZE + 5,
            5 * CODEBOOK_SIZE + 6,
            6 * CODEBOOK_SIZE + 7,
        ]
        generated = [START_OF_SPEECH, *(AUDIO_TOKEN_OFFSET + value for value in values), END_OF_SPEECH]
        self.assertEqual(extract_snac_codes(generated), ([1], [2, 5], [3, 4, 6, 7]))

    def test_snac_code_extraction_rejects_out_of_range_values(self) -> None:
        generated = [START_OF_SPEECH, *([AUDIO_TOKEN_OFFSET - 1] * 7), END_OF_SPEECH]
        with self.assertRaises(BackendError) as raised:
            extract_snac_codes(generated)
        self.assertEqual(raised.exception.code, "tts.greektts15.audio_code_invalid")

    def test_experimental_speaker_prompt_is_strict_and_explicit(self) -> None:
        self.assertEqual(
            format_pilot_prompt("leo"),
            "leo: Ο μπαμπάς πήγε στην αντάρα για να βρει τα αγκάθια.",
        )
        self.assertEqual(format_pilot_prompt(), "Ο μπαμπάς πήγε στην αντάρα για να βρει τα αγκάθια.")
        with self.assertRaises(BackendError) as raised:
            format_pilot_prompt("unknown")
        self.assertEqual(raised.exception.code, "tts.greektts15.speaker_invalid")

    def test_quality_fixture_set_and_resource_gates_are_bounded(self) -> None:
        self.assertEqual(len(QUALITY_FIXTURES), 4)
        self.assertEqual(len({fixture_id for fixture_id, _ in QUALITY_FIXTURES}), 4)
        self.assertTrue(all(text.strip() for _, text in QUALITY_FIXTURES))
        self.assertEqual(QUALITY_MAX_RSS_BYTES, 4 * 1024**3)
        self.assertEqual(QUALITY_MAX_MLX_MEMORY_BYTES, 3 * 1024**3)
        self.assertEqual(QUALITY_MAX_MEAN_REALTIME_FACTOR, 1.0)

    def test_pronunciation_references_are_pinned_and_phonetically_bounded(self) -> None:
        self.assertEqual(len(PRONUNCIATION_REFERENCES), 4)
        self.assertEqual(len({item.fixture_id for item in PRONUNCIATION_REFERENCES}), 4)
        self.assertTrue(all(item.filename.startswith("Greek-F1-Harvard-") for item in PRONUNCIATION_REFERENCES))
        self.assertTrue(all(len(item.audio_sha256) == 64 for item in PRONUNCIATION_REFERENCES))

    def test_pronunciation_qualification_pairs_native_and_synthetic_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            dataset = Path(temp) / "dataset"
            audio_root = dataset / "audio"
            audio_root.mkdir(parents=True)
            hashes = {}
            for item in PRONUNCIATION_REFERENCES:
                path = audio_root / item.filename
                path.write_bytes(b"reference")
                hashes[str(path)] = item.audio_sha256
            suite = {"ready": True, "samples": []}
            with (
                mock.patch("jarvis_voice.macos.greektts15_pilot._sha256", side_effect=lambda path: hashes[str(path)]),
                mock.patch("jarvis_voice.macos.greektts15_pilot._run_greektts15_suite", return_value=suite) as run,
            ):
                report = run_greektts15_pronunciation_qualification("/models", dataset, Path(temp) / "out")
        self.assertEqual(report["code"], "tts.greektts15.pronunciation_qualification_completed")
        self.assertTrue(report["pronunciation"]["same_text_pairing"])
        self.assertFalse(report["pronunciation"]["production_promotion_allowed"])
        fixtures = run.call_args.kwargs["fixtures"]
        self.assertEqual(fixtures, tuple((item.fixture_id, item.text) for item in PRONUNCIATION_REFERENCES))

    def test_legacy_pilot_flattens_single_suite_sample(self) -> None:
        suite = {
            "ready": True,
            "peak_rss_bytes": 1,
            "peak_mlx_memory_bytes": 1,
            "samples": [
                {
                    "fixture_id": "greek_consonants",
                    "file": "01-greek_consonants.wav",
                    "realtime_factor": 0.5,
                }
            ],
        }
        with mock.patch(
            "jarvis_voice.macos.greektts15_pilot._run_greektts15_suite",
            return_value=suite,
        ):
            report = run_greektts15_pilot("/models", "/output")
        self.assertEqual(report["code"], "tts.greektts15.pilot_completed")
        self.assertEqual(report["fixture_id"], "greek_consonants")
        self.assertEqual(report["sample"]["file"], "01-greek_consonants.wav")
        self.assertNotIn("samples", report)
        self.assertNotIn("file", report)

    def test_quality_qualification_never_promotes_without_streaming(self) -> None:
        suite = {
            "ready": True,
            "peak_rss_bytes": QUALITY_MAX_RSS_BYTES,
            "peak_mlx_memory_bytes": QUALITY_MAX_MLX_MEMORY_BYTES,
            "samples": [
                {"fixture_id": fixture_id, "realtime_factor": 0.5}
                for fixture_id, _ in QUALITY_FIXTURES
            ],
        }
        with mock.patch(
            "jarvis_voice.macos.greektts15_pilot._run_greektts15_suite",
            return_value=suite,
        ):
            report = run_greektts15_quality_qualification("/models", "/output")
        qualification = report["qualification"]
        self.assertFalse(qualification["automated_gates_passed"])
        self.assertFalse(qualification["production_promotion_allowed"])
        self.assertFalse(qualification["automated_gates"]["incremental_audio_streaming"])


if __name__ == "__main__":
    unittest.main()
