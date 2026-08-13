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
PROTECTED_F1_ARGUMENT = "--jarvis-protected-f1"
PROTECTED_F1_MAX_STEPS = 200
PROTECTED_F1_CHECKPOINT_INTERVAL = 100
PROTECTED_F1_LEARNING_RATE = 2e-6
PROTECTED_F1_DISCRIMINATOR_LEARNING_RATE = 1e-6
PROTECTED_F1_FROZEN_MODULE_PARAMETERS = {
    "enc_p": 6_341_760,
    "flow": 7_102_080,
    "dp": 1_317_168,
}
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


def _protected_f1_arguments(piper_argv: Sequence[str]) -> tuple[bool, list[str]]:
    count = sum(value == PROTECTED_F1_ARGUMENT for value in piper_argv)
    if count > 1:
        raise _error(
            "tts.f1_mps.protected_argument_invalid",
            "Protected F1 training argument is duplicated",
        )
    return count == 1, [value for value in piper_argv if value != PROTECTED_F1_ARGUMENT]


def _validate_protected_f1_profile(piper_argv: Sequence[str]) -> None:
    expected = {
        "--trainer.accelerator": "mps",
        "--trainer.devices": "1",
        "--trainer.precision": "32-true",
        "--trainer.max_steps": str(PROTECTED_F1_MAX_STEPS),
        "--trainer.limit_val_batches": "0",
        "--trainer.num_sanity_val_steps": "0",
        "--data.batch_size": "1",
        "--data.num_workers": "0",
        "--model.learning_rate": str(PROTECTED_F1_LEARNING_RATE),
        "--model.learning_rate_d": str(PROTECTED_F1_DISCRIMINATOR_LEARNING_RATE),
        "--model.mos_metric": "none",
    }
    if any(
        _single_value(piper_argv, key) != value for key, value in expected.items()
    ):
        raise _error(
            "tts.f1_mps.protected_profile_invalid",
            "Protected F1 training profile changed",
        )
    forbidden = {
        "--trainer.fast_dev_run",
        "--trainer.limit_train_batches",
        "--trainer.max_epochs",
        "--trainer.overfit_batches",
    }
    if any(value in forbidden for value in piper_argv):
        raise _error(
            "tts.f1_mps.protected_profile_invalid",
            "Protected F1 training cannot use an epoch budget",
        )


def freeze_protected_f1_stack(
    instance: object,
    *,
    expected_parameters: dict[str, int] | None = None,
) -> dict[str, int]:
    """Freeze the exact inference alignment stack or fail before training."""

    expected = (
        PROTECTED_F1_FROZEN_MODULE_PARAMETERS
        if expected_parameters is None
        else expected_parameters
    )
    model_g = getattr(instance, "model_g", None)
    observed: dict[str, int] = {}
    modules: list[object] = []
    for name, expected_count in expected.items():
        module = getattr(model_g, name, None)
        parameters = getattr(module, "parameters", None)
        if module is None or not callable(parameters):
            raise _error(
                "tts.f1_mps.protected_stack_invalid",
                "Protected F1 inference stack changed",
            )
        try:
            count = sum(int(parameter.numel()) for parameter in parameters())
        except (AttributeError, TypeError, ValueError) as exc:
            raise _error(
                "tts.f1_mps.protected_stack_invalid",
                "Protected F1 inference stack could not be inspected",
            ) from exc
        if count != expected_count:
            raise _error(
                "tts.f1_mps.protected_stack_invalid",
                "Protected F1 inference stack parameter count changed",
            )
        observed[name] = count
        modules.append(module)
    for module in modules:
        requires_grad = getattr(module, "requires_grad_", None)
        evaluate = getattr(module, "eval", None)
        if not callable(requires_grad) or not callable(evaluate):
            raise _error(
                "tts.f1_mps.protected_stack_invalid",
                "Protected F1 inference stack cannot be frozen",
            )
        requires_grad(False)
        evaluate()
    return observed


def _prepare_piper_main(
    piper_argv: Sequence[str],
    *,
    protected_f1: bool = False,
) -> tuple[object, tuple[object, ...], object | None, dict[str, object]]:
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
    originals: dict[str, object] = {}
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
        originals["on_fit_start"] = original_on_fit_start

    if protected_f1:
        model_class = getattr(training_module, "VitsModel", None)
        original_configure = getattr(model_class, "configure_optimizers", None)
        original_on_fit_start = getattr(model_class, "on_fit_start", None)
        original_on_train_batch_start = getattr(
            model_class, "on_train_batch_start", None
        )
        if (
            model_class is None
            or not callable(original_configure)
            or not callable(original_on_fit_start)
            or not callable(original_on_train_batch_start)
        ):
            training_module._DEFAULT_CALLBACKS = list(callbacks)
            raise _error(
                "tts.f1_mps.protected_hook_invalid",
                "Pinned Piper protected-training hooks changed",
            )
        checkpoint_module = importlib.import_module("lightning.pytorch.callbacks")
        checkpoint_class = getattr(checkpoint_module, "ModelCheckpoint", None)
        if checkpoint_class is None:
            training_module._DEFAULT_CALLBACKS = list(callbacks)
            raise _error(
                "tts.f1_mps.protected_checkpoint_invalid",
                "Protected F1 checkpoint callback is unavailable",
            )
        training_module._DEFAULT_CALLBACKS = [
            checkpoint_class(
                monitor=None,
                save_top_k=-1,
                save_last=True,
                save_weights_only=False,
                filename="step={step}",
                auto_insert_metric_name=False,
                every_n_train_steps=PROTECTED_F1_CHECKPOINT_INTERVAL,
                every_n_epochs=None,
                save_on_train_epoch_end=False,
                enable_version_counter=False,
            )
        ]

        def protected_configure_optimizers(instance: object) -> object:
            freeze_protected_f1_stack(instance)
            return original_configure(instance)

        def protected_on_fit_start(instance: object) -> object:
            result = original_on_fit_start(instance)
            freeze_protected_f1_stack(instance)
            return result

        def protected_on_train_batch_start(
            instance: object, *args: object, **kwargs: object
        ) -> object:
            result = original_on_train_batch_start(instance, *args, **kwargs)
            freeze_protected_f1_stack(instance)
            return result

        originals = {
            "configure_optimizers": original_configure,
            "on_fit_start": original_on_fit_start,
            "on_train_batch_start": original_on_train_batch_start,
        }
        model_class.configure_optimizers = protected_configure_optimizers
        model_class.on_fit_start = protected_on_fit_start
        model_class.on_train_batch_start = protected_on_train_batch_start
    return training_module, callbacks, model_class, originals


def run_piper_mps_fit(piper_argv: Sequence[str]) -> None:
    """Run Piper with a scoped, pinned-checkpoint weights-only override."""

    torch = importlib.import_module("torch")
    protected_f1, piper_arguments = _protected_f1_arguments(piper_argv)
    warm_start = _single_value(piper_arguments, WARM_START_ARGUMENT)
    resume = _single_value(piper_arguments, RESUME_ARGUMENT)
    warm_start_mode = warm_start not in (None, "null")
    resume_mode = resume is not None
    if warm_start_mode == resume_mode:
        raise _error(
            "tts.f1_mps.training_checkpoint_mode_invalid",
            "Training requires exactly one warm-start or resume checkpoint",
        )
    if protected_f1:
        if not warm_start_mode:
            raise _error(
                "tts.f1_mps.protected_warmstart_required",
                "Protected F1 training requires the pinned warm-start",
            )
        _validate_protected_f1_profile(piper_arguments)
    if warm_start_mode:
        checkpoint = validate_piper_mps_checkpoint(
            piper_arguments,
            torch.serialization.get_unsafe_globals_in_checkpoint,
        )
    else:
        checkpoint = validate_piper_mps_resume_checkpoint(
            piper_arguments,
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
        original_methods,
    ) = _prepare_piper_main(piper_arguments, protected_f1=protected_f1)
    previous_argv = sys.argv
    sys.argv = ["piper.train", *piper_arguments]
    torch.load = pinned_weights_only_load
    try:
        with torch.serialization.safe_globals([PosixPath]):
            training_module.main()
    finally:
        if model_class is not None:
            for name, method in original_methods.items():
                setattr(model_class, name, method)
        training_module._DEFAULT_CALLBACKS = list(original_callbacks)
        torch.load = original_torch_load
        sys.argv = previous_argv
