"""Composition root for the Windows tray and the existing voice runtime."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable

from .app_paths import ApplicationPaths, frozen_bundle_root, require_external_mutable_path
from .config import VoiceConfig, load_voice_config
from .errors import BackendError
from .lifecycle import LifecycleStatus, VoiceLifecycleController
from .safe_log import SafeEventLogger
from .windows.single_instance import SingleInstanceGuard
from .windows.tray import TrayCommand, WindowsTrayIcon


def _open_directory(path: Path) -> None:  # pragma: no cover - native Windows shell only
    if not hasattr(os, "startfile"):
        raise BackendError("application.shell_open_windows_only", "Opening an application folder requires Windows", subsystem="application")
    os.startfile(str(path))  # type: ignore[attr-defined]


class VoiceTrayApplication:
    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        paths: ApplicationPaths | None = None,
        runtime_factory: Callable[[], object] | None = None,
        tray_factory: Callable[..., WindowsTrayIcon] = WindowsTrayIcon,
        single_instance: SingleInstanceGuard | None = None,
        open_directory: Callable[[Path], None] = _open_directory,
    ) -> None:
        self.paths = paths or ApplicationPaths.from_environment()
        selected_config = Path(config_path) if config_path is not None else self.paths.default_config_file
        self.config_path = require_external_mutable_path(selected_config, frozen_bundle_root())
        self._runtime_factory = runtime_factory or self._create_runtime
        self._tray_factory = tray_factory
        self._single_instance = single_instance or SingleInstanceGuard()
        self._open_directory = open_directory
        self._tray: WindowsTrayIcon | None = None
        self._logger: SafeEventLogger | None = None
        self._lifecycle: VoiceLifecycleController | None = None

    def _create_runtime(self) -> object:
        config = load_voice_config(self.config_path)
        if config.whisper_model_directory:
            require_external_mutable_path(config.whisper_model_directory, frozen_bundle_root())
        if config.vad.artifact.path:
            require_external_mutable_path(config.vad.artifact.path, frozen_bundle_root())
        from .runtime import WindowsVoiceRuntime
        return WindowsVoiceRuntime(config)

    def _status_text(self) -> str:
        assert self._lifecycle is not None
        snapshot = self._lifecycle.snapshot
        labels = {
            LifecycleStatus.STOPPED: "Stopped",
            LifecycleStatus.STARTING: "Starting",
            LifecycleStatus.RUNNING: "Running",
            LifecycleStatus.STOPPING: "Stopping",
            LifecycleStatus.ERROR: f"Error ({snapshot.error_code or 'runtime.failure'})",
        }
        return f"JARVIS Voice — {labels[snapshot.status]}"

    @staticmethod
    def _background(target: Callable[[], object], name: str) -> None:
        threading.Thread(target=target, name=name, daemon=True).start()

    def _open_safely(self, path: Path) -> None:
        try:
            self._open_directory(path)
        except BackendError as exc:
            if self._logger is not None:
                self._logger.write("voice.shell.error", status="error", code=exc.code)
        except Exception:
            if self._logger is not None:
                self._logger.write("voice.shell.error", status="error", code="application.shell_open_failed")

    def _handle_command(self, command: TrayCommand) -> None:
        assert self._lifecycle is not None
        if command == TrayCommand.START:
            self._lifecycle.start()
        elif command == TrayCommand.STOP:
            self._background(self._lifecycle.stop, "jarvis-tray-stop")
        elif command == TrayCommand.OPEN_CONFIG:
            self._background(lambda: self._open_safely(self.paths.config), "jarvis-open-config")
        elif command == TrayCommand.OPEN_LOGS:
            self._background(lambda: self._open_safely(self.paths.logs), "jarvis-open-logs")
        elif command == TrayCommand.EXIT:
            self._background(self._shutdown_and_close, "jarvis-tray-exit")

    def _shutdown_and_close(self) -> None:
        assert self._lifecycle is not None
        self._lifecycle.shutdown()
        if self._tray is not None:
            self._tray.request_close()

    def run(self) -> None:
        self.paths.ensure()
        application = VoiceConfig().application
        try:
            application = load_voice_config(self.config_path).application
        except BackendError:
            # A missing/invalid config is represented as a runtime error in the
            # tray; safe lifecycle defaults keep the recovery UI available.
            pass
        self._logger = SafeEventLogger(
            self.paths.event_log_file,
            maximum_bytes=application.log_maximum_bytes,
            backup_count=application.log_backup_count,
        )
        self._lifecycle = VoiceLifecycleController(
            self._runtime_factory,
            logger=self._logger,
            shutdown_timeout=application.shutdown_timeout_seconds,
        )
        self._single_instance.acquire()
        try:
            self._tray = self._tray_factory(command_handler=self._handle_command, status_provider=self._status_text)
            self._lifecycle.start()
            self._tray.run()
        finally:
            self._lifecycle.shutdown()
            self._single_instance.release()
