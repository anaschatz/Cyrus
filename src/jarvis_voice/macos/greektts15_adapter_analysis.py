"""Analyze GreekTTS LoRA update concentration before targeted training."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

from jarvis_voice.errors import BackendError


_LORA_KEY = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.(?P<section>self_attn|mlp)\."
    r"(?P<module>[a-z_]+)\.lora_(?P<matrix>a|b)$"
)


def parse_lora_key(key: str) -> tuple[int, str, str, str]:
    match = _LORA_KEY.fullmatch(key)
    if match is None:
        raise BackendError(
            "tts.greektts15.adapter_key_invalid",
            "GreekTTS LoRA tensor key is invalid",
            subsystem="tts",
        )
    return (
        int(match.group("layer")),
        match.group("section"),
        match.group("module"),
        match.group("matrix"),
    )


def analyze_adapter_delta(
    baseline_file: str | Path,
    candidate_file: str | Path,
) -> dict[str, object]:
    """Measure where a candidate update moved the immutable rank-64 adapter."""

    baseline_path = Path(baseline_file).expanduser()
    candidate_path = Path(candidate_file).expanduser()
    if any(
        not path.is_absolute() or path.is_symlink() or not path.is_file()
        for path in (baseline_path, candidate_path)
    ):
        raise BackendError(
            "tts.greektts15.adapter_analysis_input_invalid",
            "GreekTTS adapter analysis input is invalid",
            subsystem="tts",
        )
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise BackendError(
            "tts.greektts15.dependencies_missing",
            "MLX is required for GreekTTS adapter analysis",
            subsystem="tts",
        ) from exc
    baseline = mx.load(str(baseline_path))
    candidate = mx.load(str(candidate_path))
    if not isinstance(baseline, dict) or not isinstance(candidate, dict) or set(baseline) != set(candidate) or len(baseline) != 392:
        raise BackendError(
            "tts.greektts15.adapter_analysis_mismatch",
            "GreekTTS adapter analysis tensors do not match",
            subsystem="tts",
        )
    tensors: list[dict[str, object]] = []
    squared_by_layer: defaultdict[int, float] = defaultdict(float)
    squared_by_module: defaultdict[str, float] = defaultdict(float)
    squared_by_section: defaultdict[str, float] = defaultdict(float)
    squared_by_matrix: defaultdict[str, float] = defaultdict(float)
    total_squared = 0.0
    for key in sorted(baseline):
        layer, section, module, matrix = parse_lora_key(key)
        source, target = baseline[key], candidate[key]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise BackendError(
                "tts.greektts15.adapter_analysis_mismatch",
                "GreekTTS adapter analysis tensor shapes do not match",
                subsystem="tts",
            )
        delta = target - source
        delta_squared = float(mx.sum(delta * delta).item())
        source_squared = float(mx.sum(source * source).item())
        if not math.isfinite(delta_squared) or not math.isfinite(source_squared):
            raise BackendError(
                "tts.greektts15.adapter_analysis_nonfinite",
                "GreekTTS adapter analysis energies must be finite",
                subsystem="tts",
            )
        total_squared += delta_squared
        squared_by_layer[layer] += delta_squared
        squared_by_module[module] += delta_squared
        squared_by_section[section] += delta_squared
        squared_by_matrix[matrix] += delta_squared
        tensors.append(
            {
                "key": key,
                "layer": layer,
                "section": section,
                "module": module,
                "matrix": matrix,
                "parameters": int(source.size),
                "delta_l2": round(delta_squared**0.5, 8),
                "relative_delta_percent": round(100.0 * (delta_squared / max(source_squared, 1e-30)) ** 0.5, 6),
            }
        )
    if total_squared <= 0:
        raise BackendError(
            "tts.greektts15.adapter_analysis_empty",
            "GreekTTS adapter candidate contains no update",
            subsystem="tts",
        )

    def contribution(items: dict[object, float], label: str) -> list[dict[str, object]]:
        return sorted(
            (
                {label: key, "delta_energy_percent": round(100.0 * value / total_squared, 6)}
                for key, value in items.items()
            ),
            key=lambda item: float(item["delta_energy_percent"]),
            reverse=True,
        )

    tensors.sort(key=lambda item: float(item["delta_l2"]), reverse=True)
    return {
        "schema": "jarvis.voice.greektts15-adapter-delta-analysis/1",
        "ready": True,
        "tensor_groups": len(tensors),
        "total_delta_l2": round(total_squared**0.5, 8),
        "by_layer": contribution(squared_by_layer, "layer"),
        "by_module": contribution(squared_by_module, "module"),
        "by_section": contribution(squared_by_section, "section"),
        "by_matrix": contribution(squared_by_matrix, "matrix"),
        "largest_tensors": tensors[:28],
        "diagnostic_only": True,
        "targeted_training_approved": False,
    }
