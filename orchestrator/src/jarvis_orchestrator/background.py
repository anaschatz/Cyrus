"""Per-user background service manager for macOS launchd and Windows tasks."""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path
from xml.sax.saxutils import escape

from .operations import OperationalError, ProcessLock, operational_paths


MACOS_LABEL = "com.jarvis.orchestrator"
WINDOWS_TASK_NAME = "JARVIS-Orchestrator"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _runner_source(config_path: Path) -> str:
    root = _repository_root()
    sources = (
        root / "orchestrator" / "src",
        root / "src",
        root / "inference" / "src",
        root / "anamnesis" / "src",
        root / "research" / "src",
    )
    return (
        "from __future__ import annotations\n"
        "import sys\n"
        f"for value in {tuple(str(item) for item in sources)!r}:\n"
        "    if value not in sys.path:\n"
        "        sys.path.insert(0, value)\n"
        "from jarvis_orchestrator.cli import main\n"
        f"raise SystemExit(main(['--config', {str(config_path)!r}, 'serve']))\n"
    )


class BackgroundServiceManager:
    def __init__(self, *, config_path: str | Path, database_path: str | Path) -> None:
        self.config_path = Path(config_path).resolve()
        self.database_path = Path(database_path).resolve()
        if not self.config_path.is_file() or not self.config_path.is_absolute():
            raise OperationalError("background_config_unavailable")
        self.paths = operational_paths(self.database_path)
        self.runner = self.paths["state"] / "orchestrator-background-runner.py"
        self.windows_xml = self.paths["state"] / "orchestrator-task.xml"

    @property
    def macos_plist(self) -> Path:
        return Path.home() / "Library" / "LaunchAgents" / f"{MACOS_LABEL}.plist"

    def _write_runner(self) -> None:
        self.paths["state"].mkdir(parents=True, exist_ok=True)
        self.runner.write_text(_runner_source(self.config_path), encoding="utf-8")
        if os.name != "nt":
            self.runner.chmod(0o600)

    @staticmethod
    def _run(command: list[str], *, allowed_fail: bool = False) -> bool:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )
        if completed.returncode != 0 and not allowed_fail:
            raise OperationalError("background_manager_command_failed")
        return completed.returncode == 0

    def _write_macos_plist(self) -> None:
        self.macos_plist.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "Label": MACOS_LABEL,
            "ProgramArguments": [sys.executable, str(self.runner)],
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "ProcessType": "Background",
            "ThrottleInterval": 5,
            "WorkingDirectory": str(_repository_root()),
            "StandardOutPath": "/dev/null",
            "StandardErrorPath": "/dev/null",
        }
        with self.macos_plist.open("wb") as destination:
            plistlib.dump(document, destination, sort_keys=True)
        self.macos_plist.chmod(0o600)

    def _write_windows_xml(self) -> None:
        command = escape(str(sys.executable))
        arguments = escape(f'"{self.runner}"')
        document = f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers><LogonTrigger><Enabled>true</Enabled></LogonTrigger></Triggers>
  <Principals><Principal id="Author"><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RestartOnFailure><Interval>PT1M</Interval><Count>3</Count></RestartOnFailure>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author"><Exec><Command>{command}</Command><Arguments>{arguments}</Arguments><WorkingDirectory>{escape(str(_repository_root()))}</WorkingDirectory></Exec></Actions>
</Task>'''
        self.windows_xml.write_text(document, encoding="utf-16")

    def _is_running(self) -> bool:
        probe = ProcessLock(self.paths["lock"])
        try:
            probe.acquire()
        except OperationalError:
            return True
        else:
            probe.release()
            return False

    def install(self) -> dict[str, object]:
        self._write_runner()
        self.paths["stop"].unlink(missing_ok=True)
        if sys.platform == "darwin":
            self._write_macos_plist()
            domain = f"gui/{os.getuid()}"
            self._run(["launchctl", "bootout", domain, str(self.macos_plist)], allowed_fail=True)
            self._run(["launchctl", "bootstrap", domain, str(self.macos_plist)])
        elif os.name == "nt":  # pragma: no cover - exercised on Windows CI
            self._write_windows_xml()
            self._run(
                ["schtasks", "/Create", "/TN", WINDOWS_TASK_NAME, "/XML", str(self.windows_xml), "/F"]
            )
            self._run(["schtasks", "/Run", "/TN", WINDOWS_TASK_NAME])
        else:
            raise OperationalError("background_platform_unsupported")
        return self.status()

    def start(self) -> dict[str, object]:
        self.paths["stop"].unlink(missing_ok=True)
        if sys.platform == "darwin":
            if not self.macos_plist.is_file():
                raise OperationalError("background_not_installed")
            domain = f"gui/{os.getuid()}"
            self._run(["launchctl", "bootstrap", domain, str(self.macos_plist)], allowed_fail=True)
            self._run(["launchctl", "kickstart", f"{domain}/{MACOS_LABEL}"], allowed_fail=True)
        elif os.name == "nt":  # pragma: no cover
            self._run(["schtasks", "/Run", "/TN", WINDOWS_TASK_NAME])
        else:
            raise OperationalError("background_platform_unsupported")
        return self._wait_for_state(running=True)

    def stop(self, *, timeout_seconds: float = 10.0) -> dict[str, object]:
        self.paths["state"].mkdir(parents=True, exist_ok=True)
        self.paths["stop"].write_bytes(b"stop\n")
        return self._wait_for_state(running=False, timeout_seconds=timeout_seconds)

    def _wait_for_state(
        self, *, running: bool, timeout_seconds: float = 10.0
    ) -> dict[str, object]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._is_running() is running:
                return self.status()
            time.sleep(0.1)
        raise OperationalError("background_state_timeout")

    def uninstall(self) -> dict[str, object]:
        if self._is_running():
            self.stop()
        if sys.platform == "darwin":
            domain = f"gui/{os.getuid()}"
            self._run(["launchctl", "bootout", domain, str(self.macos_plist)], allowed_fail=True)
            self.macos_plist.unlink(missing_ok=True)
        elif os.name == "nt":  # pragma: no cover
            self._run(["schtasks", "/Delete", "/TN", WINDOWS_TASK_NAME, "/F"], allowed_fail=True)
            self.windows_xml.unlink(missing_ok=True)
        else:
            raise OperationalError("background_platform_unsupported")
        self.runner.unlink(missing_ok=True)
        self.paths["stop"].unlink(missing_ok=True)
        return self.status()

    def status(self) -> dict[str, object]:
        if sys.platform == "darwin":
            installed = self.macos_plist.is_file() and self.runner.is_file()
        elif os.name == "nt":  # pragma: no cover
            installed = self._run(
                ["schtasks", "/Query", "/TN", WINDOWS_TASK_NAME], allowed_fail=True
            )
        else:
            installed = False
        return {
            "installed": installed,
            "platform_supported": sys.platform == "darwin" or os.name == "nt",
            "running": self._is_running(),
            "version": "background-service/1",
        }


__all__ = ["BackgroundServiceManager", "MACOS_LABEL", "WINDOWS_TASK_NAME"]
