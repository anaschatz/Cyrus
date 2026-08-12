"""Fail-closed ONNX export for a locally trained Piper MPS checkpoint."""

from __future__ import annotations

import importlib
import json
import os
import re
from pathlib import Path
from typing import Callable, Sequence

from .errors import BackendError


UnsafeGlobalsProbe = Callable[[str], Sequence[str]]
MINIMUM_CHECKPOINT_BYTES = 100 * 1024 * 1024
MAXIMUM_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
EXPECTED_OUTPUT_NAME = "el_GR-jarvis-f1-medium.onnx"
_CHECKPOINT_NAME = re.compile(r"^epoch=\d+-val_mel=\d+\.\d+\.ckpt$")
_LIGHTNING_VERSION = re.compile(r"^version_\d+$")
LANGUAGE_METADATA = {
    "code": "el",
    "family": "el",
    "region": "GR",
    "name_native": "Ελληνικά",
    "name_english": "Greek",
    "country_english": "Greece",
}


def _error(code: str, message: str) -> BackendError:
    return BackendError(code, message, subsystem="tts")


def validate_piper_mps_export(
    checkpoint_path: str | Path,
    output_path: str | Path,
    unsafe_globals_probe: UnsafeGlobalsProbe,
    *,
    minimum_size: int = MINIMUM_CHECKPOINT_BYTES,
    maximum_size: int = MAXIMUM_CHECKPOINT_BYTES,
) -> tuple[Path, Path]:
    """Validate local export paths and require an empty unsafe-global set."""

    checkpoint = Path(checkpoint_path).expanduser()
    output = Path(output_path).expanduser()
    checkpoint_layout_ready = (
        _CHECKPOINT_NAME.fullmatch(checkpoint.name) is not None
        and checkpoint.parent.name == "checkpoints"
        and _LIGHTNING_VERSION.fullmatch(checkpoint.parent.parent.name) is not None
        and checkpoint.parent.parent.parent.name == "lightning_logs"
    )
    if (
        not checkpoint.is_absolute()
        or not checkpoint.is_file()
        or checkpoint.is_symlink()
        or not checkpoint_layout_ready
    ):
        raise _error(
            "tts.f1_mps.export_checkpoint_invalid",
            "Piper export checkpoint is invalid",
        )
    try:
        size = checkpoint.stat().st_size
    except OSError as exc:
        raise _error(
            "tts.f1_mps.export_checkpoint_unreadable",
            "Piper export checkpoint could not be read",
        ) from exc
    if size < minimum_size or size > maximum_size:
        raise _error(
            "tts.f1_mps.export_checkpoint_size_invalid",
            "Piper export checkpoint size is outside the bounded profile",
        )
    try:
        observed = tuple(unsafe_globals_probe(str(checkpoint)))
    except (OSError, RuntimeError, ValueError) as exc:
        raise _error(
            "tts.f1_mps.export_checkpoint_scan_failed",
            "Piper export checkpoint safe-global scan failed",
        ) from exc
    if observed:
        raise _error(
            "tts.f1_mps.export_checkpoint_globals_invalid",
            "Piper export checkpoint contains unsafe globals",
        )

    if (
        not output.is_absolute()
        or output.name != EXPECTED_OUTPUT_NAME
        or not output.parent.is_dir()
        or output.parent.is_symlink()
        or output.exists()
        or output.with_name(f"{output.name}.json").is_symlink()
        or not output.with_name(f"{output.name}.json").is_file()
    ):
        raise _error(
            "tts.f1_mps.export_output_invalid",
            "Piper export output path or companion config is invalid",
        )
    return checkpoint, output


def finalize_piper_mps_config(config_path: str | Path) -> None:
    """Add canonical Greek discovery metadata to Piper's generated config."""

    config = Path(config_path).expanduser()
    if not config.is_absolute() or not config.is_file() or config.is_symlink():
        raise _error(
            "tts.f1_mps.export_config_invalid",
            "Piper export config is invalid",
        )
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _error(
            "tts.f1_mps.export_config_unreadable",
            "Piper export config could not be read",
        ) from exc
    audio = payload.get("audio") if isinstance(payload, dict) else None
    espeak = payload.get("espeak") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or not isinstance(audio, dict)
        or audio.get("sample_rate") != 22_050
        or not isinstance(espeak, dict)
        or espeak.get("voice") != "el"
        or payload.get("phoneme_type") != "espeak"
        or payload.get("num_speakers") != 1
        or payload.get("num_symbols") != 256
        or not isinstance(payload.get("phoneme_id_map"), dict)
    ):
        raise _error(
            "tts.f1_mps.export_config_incompatible",
            "Piper export config is incompatible with the Greek F1 profile",
        )
    observed_language = payload.get("language")
    if observed_language not in (None, LANGUAGE_METADATA):
        raise _error(
            "tts.f1_mps.export_language_conflict",
            "Piper export config contains conflicting language metadata",
        )
    if observed_language == LANGUAGE_METADATA:
        return
    payload["language"] = LANGUAGE_METADATA
    partial = config.with_name(f".{config.name}.part")
    if partial.exists() or partial.is_symlink():
        raise _error(
            "tts.f1_mps.export_config_partial_exists",
            "A previous Piper config partial already exists",
        )
    try:
        partial.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.chmod(partial, config.stat().st_mode & 0o777)
        os.replace(partial, config)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise _error(
            "tts.f1_mps.export_config_write_failed",
            "Piper export config could not be finalized",
        ) from exc


def run_piper_mps_export(checkpoint_path: str | Path, output_path: str | Path) -> None:
    """Export through PyTorch's installed legacy ONNX path, then validate it."""

    torch = importlib.import_module("torch")
    checkpoint, output = validate_piper_mps_export(
        checkpoint_path,
        output_path,
        torch.serialization.get_unsafe_globals_in_checkpoint,
    )
    vits_module = importlib.import_module("piper.train.vits.lightning")
    onnx = importlib.import_module("onnx")
    model_class = getattr(vits_module, "VitsModel")
    original_torch_load = torch.load

    def weights_only_load(source: object, *args: object, **kwargs: object) -> object:
        try:
            candidate = Path(os.fspath(source)).expanduser().absolute()
        except TypeError:
            candidate = None
        if candidate == checkpoint.absolute():
            kwargs["weights_only"] = True
        return original_torch_load(source, *args, **kwargs)

    partial = output.with_name(f".{output.name}.part")
    if partial.exists() or partial.is_symlink():
        raise _error(
            "tts.f1_mps.export_partial_exists",
            "A previous Piper export partial already exists",
        )
    torch.load = weights_only_load
    try:
        model = model_class.load_from_checkpoint(checkpoint, map_location="cpu")
    finally:
        torch.load = original_torch_load
    model_g = model.model_g
    model_g.eval()
    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    def infer_forward(text, text_lengths, scales, sid=None):
        audio = model_g.infer(
            text,
            text_lengths,
            noise_scale=scales[0],
            length_scale=scales[1],
            noise_scale_w=scales[2],
            sid=sid,
        )[0].unsqueeze(1)
        return audio

    model_g.forward = infer_forward
    sequence = torch.randint(
        low=0,
        high=model_g.n_vocab,
        size=(1, 50),
        dtype=torch.long,
    )
    lengths = torch.LongTensor([sequence.size(1)])
    speaker = torch.LongTensor([0]) if model_g.n_speakers > 1 else None
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    try:
        torch.onnx.export(
            model=model_g,
            args=(sequence, lengths, scales, speaker),
            f=partial,
            verbose=False,
            opset_version=15,
            input_names=["input", "input_lengths", "scales", "sid"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "phonemes"},
                "input_lengths": {0: "batch_size"},
                "output": {0: "batch_size", 2: "time"},
            },
            dynamo=False,
        )
        onnx.checker.check_model(onnx.load(partial))
        finalize_piper_mps_config(output.with_name(f"{output.name}.json"))
        os.replace(partial, output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
