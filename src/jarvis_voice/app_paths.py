"""Per-user application paths and frozen-bundle boundary checks."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import BackendError


@dataclass(frozen=True, slots=True)
class ApplicationPaths:
    root: Path
    config: Path
    models: Path
    logs: Path
    state: Path

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "ApplicationPaths":
        values = os.environ if environment is None else environment
        local_app_data = values.get("LOCALAPPDATA", "").strip()
        if not local_app_data:
            raise BackendError(
                "application.local_app_data_missing",
                "LOCALAPPDATA is required for the Windows application",
                subsystem="application",
            )
        base = Path(local_app_data).expanduser()
        if not base.is_absolute():
            raise BackendError(
                "application.local_app_data_invalid",
                "LOCALAPPDATA must be an absolute per-user path",
                subsystem="application",
            )
        root = base / "JARVIS Voice"
        return cls(root=root, config=root / "config", models=root / "models", logs=root / "logs", state=root / "state")

    @property
    def default_config_file(self) -> Path:
        return self.config / "voice.json"

    @property
    def event_log_file(self) -> Path:
        return self.logs / "voice-events.jsonl"

    def ensure(self) -> None:
        for directory in (self.root, self.config, self.models, self.logs, self.state):
            try:
                directory.mkdir(parents=True, exist_ok=True)
                os.chmod(directory, 0o700)
            except OSError as exc:
                raise BackendError(
                    "application.directory_unavailable",
                    "A per-user application directory is unavailable",
                    subsystem="application",
                ) from exc


def frozen_bundle_root() -> Path | None:
    """Return the immutable onedir root, never the mutable application root."""

    if not getattr(sys, "frozen", False):
        return None
    return Path(sys.executable).resolve().parent


def frozen_library_directory() -> Path | None:
    if not getattr(sys, "frozen", False):
        return None
    extracted = getattr(sys, "_MEIPASS", None)
    return Path(extracted).resolve() if extracted else frozen_bundle_root()


def require_external_mutable_path(path: str | Path, bundle_root: str | Path | None = None) -> Path:
    """Reject config/model/log paths that point into the frozen application."""

    candidate = Path(path).expanduser().resolve(strict=False)
    immutable = Path(bundle_root).resolve(strict=False) if bundle_root is not None else frozen_bundle_root()
    if immutable is not None and (candidate == immutable or immutable in candidate.parents):
        raise BackendError(
            "packaging.mutable_path_inside_bundle",
            "Mutable application data must be outside the installed bundle",
            subsystem="packaging",
        )
    return candidate
