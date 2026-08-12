"""Strict safe-global bridge for the pinned Piper warm-start checkpoint."""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import sys
from pathlib import Path, PosixPath
from typing import Callable, Sequence

from .errors import BackendError
from .macos.compact_greek_tts_pilot import PIPER_CHECKPOINT_ARTIFACTS


UnsafeGlobalsProbe = Callable[[str], Sequence[str]]
EXPECTED_UNSAFE_GLOBALS = frozenset({"pathlib.PosixPath"})
WARM_START_SPEC = PIPER_CHECKPOINT_ARTIFACTS[-1]
WARM_START_ARGUMENT = "--model.warmstart_ckpt"
RESUME_ARGUMENT = "--ckpt_path"
MINIMUM_RESUME_CHECKPOINT_BYTES = 100 * 1024 * 1024
MAXIMUM_RESUME_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
_LIGHTNING_VERSION = re.compile(r"^version_\d+$")


def _error(code: str, message: str) -> BackendError:
    return BackendError(code, message, subsystem="tts")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_piper_mps_checkpoint(
    piper_argv: Sequence[str],
    unsafe_globals_probe: UnsafeGlobalsProbe,
    *,
    expected_size: int = WARM_START_SPEC.size,
    expected_sha256: str = WARM_START_SPEC.sha256,
) -> Path:
    """Validate the exact checkpoint and its complete weights-only global set."""

    indices = [
        index for index, value in enumerate(piper_argv) if value == WARM_START_ARGUMENT
    ]
    if len(indices) != 1 or indices[0] + 1 >= len(piper_argv):
        raise _error(
            "tts.f1_mps.checkpoint_argument_invalid",
            "Exactly one Piper checkpoint path is required",
        )
    checkpoint = Path(piper_argv[indices[0] + 1]).expanduser()
    if (
        not checkpoint.is_absolute()
        or not checkpoint.is_file()
        or checkpoint.is_symlink()
    ):
        raise _error(
            "tts.f1_mps.checkpoint_invalid",
            "Piper warm-start checkpoint is invalid",
        )
    try:
        size = checkpoint.stat().st_size
        checksum = _sha256(checkpoint)
    except OSError as exc:
        raise _error(
            "tts.f1_mps.checkpoint_unreadable",
            "Piper warm-start checkpoint could not be read",
        ) from exc
    if size != expected_size or checksum != expected_sha256:
        raise _error(
            "tts.f1_mps.checkpoint_hash_mismatch",
            "Piper warm-start checkpoint does not match the pinned artifact",
        )
    try:
        observed = frozenset(unsafe_globals_probe(str(checkpoint)))
    except (OSError, RuntimeError, ValueError) as exc:
        raise _error(
            "tts.f1_mps.checkpoint_scan_failed",
            "Piper warm-start safe-global scan failed",
        ) from exc
    if observed != EXPECTED_UNSAFE_GLOBALS:
        raise _error(
            "tts.f1_mps.checkpoint_globals_mismatch",
            "Piper warm-start checkpoint contains an unexpected global set",
        )
    return checkpoint


def validate_piper_mps_resume_checkpoint(
    piper_argv: Sequence[str],
    unsafe_globals_probe: UnsafeGlobalsProbe,
    *,
    minimum_size: int = MINIMUM_RESUME_CHECKPOINT_BYTES,
    maximum_size: int = MAXIMUM_RESUME_CHECKPOINT_BYTES,
) -> Path:
    """Validate a local Lightning `last.ckpt` before weights-only resume."""

    resume_indices = [
        index for index, value in enumerate(piper_argv) if value == RESUME_ARGUMENT
    ]
    warm_value = _single_value(piper_argv, WARM_START_ARGUMENT)
    if (
        warm_value not in (None, "null")
        or len(resume_indices) != 1
        or resume_indices[0] + 1 >= len(piper_argv)
    ):
        raise _error(
            "tts.f1_mps.resume_argument_invalid",
            "Resume requires exactly one checkpoint and no warm-start",
        )
    checkpoint = Path(piper_argv[resume_indices[0] + 1]).expanduser()
    checkpoint_parts_ready = (
        checkpoint.name == "last.ckpt"
        and checkpoint.parent.name == "checkpoints"
        and _LIGHTNING_VERSION.fullmatch(checkpoint.parent.parent.name) is not None
        and checkpoint.parent.parent.parent.name == "lightning_logs"
    )
    if (
        not checkpoint.is_absolute()
        or not checkpoint.is_file()
        or checkpoint.is_symlink()
        or not checkpoint_parts_ready
    ):
        raise _error(
            "tts.f1_mps.resume_checkpoint_invalid",
            "Piper resume checkpoint is invalid",
        )
    try:
        size = checkpoint.stat().st_size
    except OSError as exc:
        raise _error(
            "tts.f1_mps.resume_checkpoint_unreadable",
            "Piper resume checkpoint could not be read",
        ) from exc
    if size < minimum_size or size > maximum_size:
        raise _error(
            "tts.f1_mps.resume_checkpoint_size_invalid",
            "Piper resume checkpoint size is outside the bounded profile",
        )
    try:
        observed = frozenset(unsafe_globals_probe(str(checkpoint)))
    except (OSError, RuntimeError, ValueError) as exc:
        raise _error(
            "tts.f1_mps.resume_checkpoint_scan_failed",
            "Piper resume safe-global scan failed",
        ) from exc
    if observed:
        raise _error(
            "tts.f1_mps.resume_checkpoint_globals_invalid",
            "Piper resume checkpoint contains unsafe globals",
        )
    return checkpoint


def _single_value(piper_argv: Sequence[str], argument: str) -> str | None:
    indices = [index for index, value in enumerate(piper_argv) if value == argument]
    if not indices:
        return None
    if len(indices) != 1 or indices[0] + 1 >= len(piper_argv):
        raise _error(
            "tts.f1_mps.training_argument_invalid",
            "Piper training argument is missing or duplicated",
        )
    return piper_argv[indices[0] + 1]


def _prepare_piper_main(
    piper_argv: Sequence[str],
) -> tuple[object, tuple[object, ...], object | None, object | None]:
    training_module = importlib.import_module("piper.train.__main__")
    callbacks = tuple(getattr(training_module, "_DEFAULT_CALLBACKS", ()))
    if _single_value(piper_argv, "--model.mos_metric") == "none":
        monitors = tuple(getattr(callback, "monitor", None) for callback in callbacks)
        if monitors.count("val_mos") != 1 or monitors.count("val_mel") != 1:
            raise _error(
                "tts.f1_mps.checkpoint_callbacks_invalid",
                "Pinned Piper checkpoint callback profile changed",
            )
        training_module._DEFAULT_CALLBACKS = [
            callback
            for callback in callbacks
            if getattr(callback, "monitor", None) != "val_mos"
        ]

    model_class = None
    original_on_fit_start = None
    if _single_value(piper_argv, RESUME_ARGUMENT) is not None:
        model_class = getattr(training_module, "VitsModel", None)
        original_on_fit_start = getattr(model_class, "on_fit_start", None)
        if model_class is None or not callable(original_on_fit_start):
            training_module._DEFAULT_CALLBACKS = list(callbacks)
            raise _error(
                "tts.f1_mps.resume_hook_invalid",
                "Pinned Piper resume hook changed",
            )

        def resume_on_fit_start(instance: object) -> object:
            instance._warmstart_ckpt = None
            instance._vocoder_warmstart_ckpt = None
            return original_on_fit_start(instance)

        model_class.on_fit_start = resume_on_fit_start
    return training_module, callbacks, model_class, original_on_fit_start


def run_piper_mps_fit(piper_argv: Sequence[str]) -> None:
    """Run Piper with a scoped, pinned-checkpoint weights-only override."""

    torch = importlib.import_module("torch")
    warm_start = _single_value(piper_argv, WARM_START_ARGUMENT)
    resume = _single_value(piper_argv, RESUME_ARGUMENT)
    warm_start_mode = warm_start not in (None, "null")
    resume_mode = resume is not None
    if warm_start_mode == resume_mode:
        raise _error(
            "tts.f1_mps.training_checkpoint_mode_invalid",
            "Training requires exactly one warm-start or resume checkpoint",
        )
    if warm_start_mode:
        checkpoint = validate_piper_mps_checkpoint(
            piper_argv,
            torch.serialization.get_unsafe_globals_in_checkpoint,
        )
    else:
        checkpoint = validate_piper_mps_resume_checkpoint(
            piper_argv,
            torch.serialization.get_unsafe_globals_in_checkpoint,
        )
    original_torch_load = torch.load

    def pinned_weights_only_load(source: object, *args: object, **kwargs: object) -> object:
        try:
            candidate = Path(os.fspath(source)).expanduser().absolute()
        except TypeError:
            candidate = None
        if warm_start_mode and candidate == checkpoint.absolute():
            kwargs["weights_only"] = True
        return original_torch_load(source, *args, **kwargs)

    (
        training_module,
        original_callbacks,
        model_class,
        original_on_fit_start,
    ) = _prepare_piper_main(piper_argv)
    previous_argv = sys.argv
    sys.argv = ["piper.train", *piper_argv]
    torch.load = pinned_weights_only_load
    try:
        with torch.serialization.safe_globals([PosixPath]):
            training_module.main()
    finally:
        if model_class is not None:
            model_class.on_fit_start = original_on_fit_start
        training_module._DEFAULT_CALLBACKS = list(original_callbacks)
        torch.load = original_torch_load
        sys.argv = previous_argv
