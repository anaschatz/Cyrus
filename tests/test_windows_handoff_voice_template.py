from __future__ import annotations

import json
import unittest
from pathlib import Path

from jarvis_voice.config import load_voice_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPOSITORY_ROOT / "config.windows-rtx5080.template.json"


class WindowsHandoffVoiceTemplateTests(unittest.TestCase):
    def test_template_is_parseable_and_frozen_to_safe_initial_profile(self) -> None:
        config = load_voice_config(TEMPLATE)

        self.assertEqual(config.ptt_key, "F9")
        self.assertEqual(config.orchestrator_url, "ws://127.0.0.1:8765")
        self.assertEqual(config.asr.model_profile, "large-v3-turbo")
        self.assertEqual(config.asr.execution_profile, "cuda")
        self.assertEqual(config.asr.cuda.device_index, 0)
        self.assertEqual(config.asr.cuda.compute_type, "int8_float16")
        self.assertEqual(config.asr.cuda.residency, "turn_scoped")
        self.assertEqual(config.asr.cuda.minimum_free_vram_mb, 2048)
        self.assertEqual(config.asr.cuda.reserve_for_other_workloads_mb, 2048)
        self.assertTrue(config.asr.cuda.fallback_to_cpu)
        self.assertEqual(config.windows_tts_backend, "sapi5")
        self.assertFalse(config.scheduler.enabled)

    def test_template_cannot_claim_materialized_readiness(self) -> None:
        raw = json.loads(TEMPLATE.read_text(encoding="utf-8"))

        self.assertIsNone(raw["whisper_model_directory"])
        self.assertIsNone(raw["voice_auth_key_file"])
        self.assertIsNone(raw["vad"]["artifact"]["path"])
        self.assertEqual(raw["sapi_voices"], {"el-GR": "", "en-US": ""})
        self.assertFalse(raw["scheduler"]["enabled"])
        serialized = TEMPLATE.read_text(encoding="utf-8")
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("C:\\Users\\", serialized)
        self.assertNotIn("sk-or-", serialized)


if __name__ == "__main__":
    unittest.main()
