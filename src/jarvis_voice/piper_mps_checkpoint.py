"""Strict safe-global bridge for the pinned Piper warm-start checkpoint."""

from __future__ import annotations

import hashlib
import importlib
import os
import runpy
import sys
from pathlib import Path, PosixPath
from typing import Callable, Sequence

from .errors import BackendError
from .macos.compact_greek_tts_pilot import PIPER_CHECKPOINT_ARTIFACTS


UnsafeGlobalsProbe = Callable[[str], Sequence[str]]
EXPECTED_UNSAFE_GLOBALS = frozenset({"pathlib.PosixPath"})
WARM_START_SPEC = PIPER_CHECKPOINT_ARTIFACTS[-1]
WARM_START_ARGUMENT = "--model.warmstart_ckpt"


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


def run_piper_mps_fit(piper_argv: Sequence[str]) -> None:
    """Run Piper with a scoped, pinned-checkpoint weights-only override."""

    torch = importlib.import_module("torch")
    checkpoint = validate_piper_mps_checkpoint(
        piper_argv,
        torch.serialization.get_unsafe_globals_in_checkpoint,
    )
    original_torch_load = torch.load

    def pinned_weights_only_load(source: object, *args: object, **kwargs: object) -> object:
        try:
            candidate = Path(os.fspath(source)).expanduser().absolute()
        except TypeError:
            candidate = None
        if candidate == checkpoint.absolute():
            kwargs["weights_only"] = True
        return original_torch_load(source, *args, **kwargs)

    previous_argv = sys.argv
    sys.argv = ["piper.train", *piper_argv]
    torch.load = pinned_weights_only_load
    try:
        with torch.serialization.safe_globals([PosixPath]):
            runpy.run_module("piper.train", run_name="__main__", alter_sys=True)
    finally:
        torch.load = original_torch_load
        sys.argv = previous_argv
