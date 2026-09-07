from __future__ import annotations

import json
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.macos.greektts15_adapter_analysis import analyze_adapter_delta, parse_lora_key


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

    @unittest.skipUnless(find_spec("numpy"), "optional numerical dependency")
    def test_analysis_rejects_invalid_tensor_maps_and_nonfinite_energy(self):
        import numpy as np

        keys = [
            f"model.layers.{layer}.{section}.{module}.lora_{matrix}"
            for layer in range(28)
            for section, modules in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")), ("mlp", ("gate_proj", "up_proj", "down_proj")))
            for module in modules
            for matrix in ("a", "b")
        ]
        baseline = {key: np.ones(1, dtype=np.float32) for key in keys}
        candidate = {key: np.full(1, 2.0, dtype=np.float32) for key in keys}
        mx = ModuleType("mlx.core")
        mx.sum = np.sum
        mlx = ModuleType("mlx")
        mlx.core = mx
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / name for name in ("baseline.safetensors", "candidate.safetensors")]
            for path in paths:
                path.write_bytes(b"synthetic")
            with patch.dict(sys.modules, {"mlx": mlx, "mlx.core": mx}), np.errstate(invalid="ignore", over="ignore"):
                mx.load = Mock(side_effect=[baseline, candidate])
                report = analyze_adapter_delta(*paths)
                self.assertTrue(report["ready"])
                self.assertEqual(report["tensor_groups"], 392)
                self.assertAlmostEqual(report["total_delta_l2"], 392 ** 0.5)
                json.dumps(report, allow_nan=False)
                for malformed in (None, [[1]]):
                    with self.subTest(candidate=malformed):
                        mx.load = Mock(side_effect=[baseline, malformed])
                        with self.assertRaisesRegex(BackendError, "adapter_analysis_mismatch"):
                            analyze_adapter_delta(*paths)
                for source, value in ((False, float("nan")), (True, float("nan")), (False, float("inf")), (False, 1e30)):
                    with self.subTest(source=source, value=value):
                        invalid = dict(baseline if source else candidate)
                        invalid[keys[0]] = np.full(1, value, dtype=np.float32)
                        mx.load = Mock(side_effect=[invalid if source else baseline, candidate if source else invalid])
                        with self.assertRaisesRegex(BackendError, "adapter_analysis_nonfinite"):
                            analyze_adapter_delta(*paths)


if __name__ == "__main__":
    unittest.main()
