from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.macos.greektts15_adapter_analysis import parse_lora_key


class GreekTts15AdapterAnalysisTests(unittest.TestCase):
    def test_lora_key_parser_preserves_targeting_dimensions(self) -> None:
        self.assertEqual(
            parse_lora_key("model.layers.27.self_attn.q_proj.lora_b"),
            (27, "self_attn", "q_proj", "b"),
        )
        self.assertEqual(
            parse_lora_key("model.layers.0.mlp.down_proj.lora_a"),
            (0, "mlp", "down_proj", "a"),
        )

    def test_lora_key_parser_rejects_non_adapter_weights(self) -> None:
        with self.assertRaises(BackendError) as raised:
            parse_lora_key("model.layers.0.self_attn.q_proj.weight")
        self.assertEqual(raised.exception.code, "tts.greektts15.adapter_key_invalid")


if __name__ == "__main__":
    unittest.main()
