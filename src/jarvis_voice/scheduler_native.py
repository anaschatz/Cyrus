"""Qualified native notification and lock-state adapters for S2."""

from __future__ import annotations

import base64
import hashlib
import json
import plistlib
import os
import queue
import re
import select
import stat
import subprocess
import sys
import threading
import time
import uuid
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .scheduler_client import select_notification_presentation
from .scheduler_callback import MAX_CALLBACK_BYTES


MAX_NATIVE_COMMAND_BYTES = 32_768  # Matches the signed helper's maximumInputBytes.


WINDOWS_NOTIFICATION_ADAPTER_SHA256 = hashlib.sha256(
    b"jarvis-windows-winrt-toast-adapter/1"
).hexdigest()
MACOS_NOTIFICATION_ADAPTER_SHA256 = hashlib.sha256(
    b"jarvis-macos-nsusernotification-adapter/1"
).hexdigest()
MACOS_SCHEDULER_HOST_ADAPTER_SHA256 = hashlib.sha256(
    b"jarvis-macos-signed-scheduler-host-adapter/1"
).hexdigest()
WINDOWS_LOCK_PROVIDER_SHA256 = hashlib.sha256(
    b"jarvis-windows-input-desktop-lock-provider/1"
).hexdigest()
MACOS_LOCK_PROVIDER_SHA256 = hashlib.sha256(
    b"jarvis-macos-ioreg-lock-provider/1"
).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeDispatchResult:
    accepted: bool
    reason_code: str
    ambiguous: bool = False


class NativeNotificationAdapter(Protocol):
    qualification_sha256: str

    def post(self, *, notification_id: str, title: str, body: str, silent: bool) -> NativeDispatchResult: ...
    def remove(self, notification_id: str) -> NativeDispatchResult: ...


class LockStateProvider(Protocol):
    qualification_sha256: str

    def state(self) -> str: ...


Runner = Callable[..., subprocess.CompletedProcess[bytes]]


def _run_bytes(command: list[str], *, input: bytes, timeout: float) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        input=input,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


class MacOSNotificationAdapter:
    qualification_sha256 = MACOS_NOTIFICATION_ADAPTER_SHA256

    def __init__(self, *, runner: Runner = _run_bytes) -> None:
        self._runner = runner

    @staticmethod
    def _script(operation: str, payload: dict[str, Any]) -> bytes:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        script = f"""ObjC.import('Foundation');
const p = {data};
const center = $.NSUserNotificationCenter.defaultUserNotificationCenter;
const delivered = center.deliveredNotifications.js;
for (const item of delivered) {{
  if (ObjC.unwrap(item.identifier) === p.notification_id) center.removeDeliveredNotification(item);
}}
if ({json.dumps(operation)} === 'post') {{
  const note = $.NSUserNotification.alloc.init;
  note.identifier = p.notification_id;
  note.title = p.title;
  note.informativeText = p.body;
  note.soundName = p.silent ? null : $.NSUserNotificationDefaultSoundName;
  center.deliverNotification(note);
}}
"""
        return script.encode("utf-8")

    def _invoke(self, operation: str, payload: dict[str, Any]) -> NativeDispatchResult:
        try:
            completed = self._runner(
                ["/usr/bin/osascript", "-l", "JavaScript"],
                input=self._script(operation, payload),
                timeout=5.0,
            )
        except subprocess.TimeoutExpired:
            return NativeDispatchResult(False, "native_timeout", ambiguous=True)
        except OSError:
            return NativeDispatchResult(False, "adapter_unavailable")
        if completed.returncode == 0:
            return NativeDispatchResult(True, "native_api_accepted")
        return NativeDispatchResult(False, "permission_denied")

    def post(self, *, notification_id: str, title: str, body: str, silent: bool) -> NativeDispatchResult:
        return self._invoke("post", {
            "notification_id": notification_id,
            "title": title,
            "body": body,
            "silent": bool(silent),
        })

    def remove(self, notification_id: str) -> NativeDispatchResult:
        return self._invoke("remove", {
            "notification_id": notification_id,
            "title": "",
            "body": "",
            "silent": True,
        })


class MacOSSchedulerHostAdapter:
    """Persistent signed helper owner for notifications and native callbacks."""

    qualification_sha256 = MACOS_SCHEDULER_HOST_ADAPTER_SHA256

    def __init__(
        self,
        *,
        helper_path: str,
        helper_sha256: str,
        helper_code_identity_sha256: str,
        popen: Callable[..., Any] = subprocess.Popen,
        run: Callable[..., Any] = subprocess.run,
        timeout_seconds: float = 5.0,
    ) -> None:
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not 0 < timeout_seconds <= threading.TIMEOUT_MAX):
            raise ValueError("scheduler helper timeout invalid")
        self._timeout = float(timeout_seconds)
        self.helper_path = Path(helper_path)
        self._validate_helper(helper_sha256, helper_code_identity_sha256, run)
        self._key = os.urandom(32)
        self._command_lock = threading.Lock()
        self._receipts: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=64)
        self._callbacks: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._closed = False
        self._process = popen(
            [os.fspath(self.helper_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
            bufsize=0,
        )
        with ExitStack() as initialization:
            initialization.callback(self.close)
            if self._process.stdin is None or self._process.stdout is None:
                raise ValueError("scheduler native helper pipes unavailable")
            self._reader = threading.Thread(target=self._read_output, daemon=True)
            self._reader.start()
            initialized = self._command({
                "type": "initialize",
                "callback_key": base64.b64encode(self._key).decode("ascii"),
            })
            if not initialized.accepted:
                raise ValueError("scheduler native helper initialization failed")
            initialization.pop_all()

    def _validate_helper(self, expected_sha: str, expected_identity: str, run: Callable[..., Any]) -> None:
        from jarvis_orchestrator.macos_code_identity import code_identity_sha256

        helper = self.helper_path
        if not helper.is_absolute() or helper.is_symlink():
            raise ValueError("scheduler native helper path invalid")
        selected = helper.stat()
        if (
            not stat.S_ISREG(selected.st_mode)
            or selected.st_uid != os.getuid()
            or not os.access(helper, os.X_OK)
            or hashlib.sha256(helper.read_bytes()).hexdigest() != expected_sha
        ):
            raise ValueError("scheduler native helper artifact invalid")
        verified = run(
            ["/usr/bin/codesign", "--verify", "--strict", os.fspath(helper)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=self._timeout,
        )
        identity = run(
            ["/usr/bin/codesign", "--display", "--requirements", "-", os.fspath(helper)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=self._timeout,
        )
        if (
            verified.returncode != 0
            or identity.returncode != 0
            or code_identity_sha256(identity.stdout) != expected_identity
        ):
            raise ValueError("scheduler native helper identity invalid")

    def _read_output(self) -> None:
        assert self._process.stdout is not None
        discarding = False
        while not self._closed:
            try:
                raw = self._process.stdout.readline(MAX_CALLBACK_BYTES + 1)
            except (OSError, ValueError):
                return
            if not raw:
                return
            if discarding or len(raw) > MAX_CALLBACK_BYTES or not raw.endswith(b"\n"):
                discarding = not raw.endswith(b"\n")
                continue
            try:
                value = json.loads(raw)
            except (ValueError, RecursionError):
                continue
            if isinstance(value, dict) and value.get("type") == "command.receipt":
                # A flooded receipt queue must not block callback parsing or shutdown.
                with suppress(queue.Full):
                    self._receipts.put_nowait(value)
            elif isinstance(value, dict) and value.get("contract") == "jarvis.scheduler-native-callback/1":
                # Retain the action until the consumer makes room; allow shutdown to interrupt.
                while not self._closed:
                    try:
                        self._callbacks.put(raw.rstrip(b"\r\n"), timeout=0.1)
                        break
                    except queue.Full:
                        continue

    def _command(self, payload: dict[str, Any]) -> NativeDispatchResult:
        deadline = time.monotonic() + self._timeout
        if not self._command_lock.acquire(timeout=self._timeout):
            return NativeDispatchResult(False, "native_timeout")
        try:
            request_id = str(uuid.uuid4())
            message = {**payload, "request_id": request_id}
            raw = json.dumps(message, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode() + b"\n"
            if len(raw) - 1 > MAX_NATIVE_COMMAND_BYTES:
                return NativeDispatchResult(False, "native_command_too_large")
            if self._closed or self._process.poll() is not None:
                return NativeDispatchResult(False, "adapter_unavailable")
            try:
                assert self._process.stdin is not None
                descriptor = self._process.stdin.fileno()
                os.set_blocking(descriptor, False)
                pending = memoryview(raw)
                while pending:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not select.select([], [descriptor], [], remaining)[1]:
                        raise TimeoutError
                    try:
                        written = os.write(descriptor, pending)
                    except BlockingIOError:
                        continue
                    if written <= 0:
                        raise BrokenPipeError
                    pending = pending[written:]
            except TimeoutError:
                # A partial JSON line cannot safely be followed by another command.
                self._closed = True
                return NativeDispatchResult(False, "native_timeout", ambiguous=True)
            except (OSError, ValueError):
                self._closed = True
                return NativeDispatchResult(False, "adapter_unavailable", ambiguous=True)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return NativeDispatchResult(False, "native_timeout", ambiguous=True)
                try:
                    receipt = self._receipts.get(timeout=remaining)
                except queue.Empty:
                    continue
                if receipt.get("request_id") != request_id:
                    # Commands are serialized; other IDs are late or unsolicited receipts.
                    continue
                expected = {"type", "request_id", "accepted", "ambiguous", "reason_code"}
                if (set(receipt) != expected or receipt["type"] != "command.receipt"
                        or type(receipt["accepted"]) is not bool or type(receipt["ambiguous"]) is not bool
                        or (receipt["accepted"] and receipt["ambiguous"])
                        or not isinstance(receipt["reason_code"], str)
                        or re.fullmatch(r"[a-z0-9_]{1,64}", receipt["reason_code"]) is None):
                    return NativeDispatchResult(False, "adapter_invalid_receipt", ambiguous=True)
                return NativeDispatchResult(receipt["accepted"], receipt["reason_code"], receipt["ambiguous"])
        finally:
            self._command_lock.release()

    @property
    def callback_key(self) -> bytes:
        return self._key

    def drain_callbacks(self) -> list[bytes]:
        callbacks: list[bytes] = []
        for _ in range(self._callbacks.qsize()):
            try:
                callbacks.append(self._callbacks.get_nowait())
            except queue.Empty:
                break
        return callbacks

    def post(self, *, notification_id: str, title: str, body: str, silent: bool) -> NativeDispatchResult:
        return self._command({
            "type": "post", "notification_id": notification_id,
            "title": title, "body": body, "silent": bool(silent),
        })

    def remove(self, notification_id: str) -> NativeDispatchResult:
        return self._command({"type": "remove", "notification_id": notification_id})

    def close(self) -> None:
        self._closed = True
        process = getattr(self, "_process", None)
        if process is None:
            return
        try:
            if process.stdin is not None:
                with suppress(OSError):
                    process.stdin.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        finally:
            reader = getattr(self, "_reader", None)
            if reader is not None and reader.is_alive():
                reader.join(timeout=2.0)
            if process.stdout is not None:
                process.stdout.close()


class WindowsNotificationAdapter:
    qualification_sha256 = WINDOWS_NOTIFICATION_ADAPTER_SHA256

    def __init__(self, *, runner: Runner = _run_bytes, app_id: str = "JARVIS.Voice") -> None:
        self._runner = runner
        self._app_id = app_id

    def _invoke(self, operation: str, payload: dict[str, Any]) -> NativeDispatchResult:
        encoded = base64.b64encode(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).decode()
        script = r"""
$ErrorActionPreference = 'Stop'
$p = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__PAYLOAD__')) | ConvertFrom-Json
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
$history = [Windows.UI.Notifications.ToastNotificationManager]::History
if ($p.operation -eq 'remove') {
  $history.Remove($p.tag, $p.group, $p.app_id)
  exit 0
}
[Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null
$xml = [Windows.Data.Xml.Dom.XmlDocument]::new()
$xml.LoadXml('<toast><visual><binding template="ToastGeneric"><text></text><text></text></binding></visual></toast>')
if ([bool]$p.silent) {
  $audio = $xml.CreateElement('audio')
  $audio.SetAttribute('silent', 'true')
  $xml.DocumentElement.AppendChild($audio) > $null
}
$nodes = $xml.GetElementsByTagName('text')
$nodes[0].AppendChild($xml.CreateTextNode([string]$p.title)) > $null
$nodes[1].AppendChild($xml.CreateTextNode([string]$p.body)) > $null
$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
$toast.Tag = $p.tag
$toast.Group = $p.group
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($p.app_id).Show($toast)
""".replace("__PAYLOAD__", encoded)
        try:
            completed = self._runner(
                ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", "-"],
                input=script.encode("utf-8"),
                timeout=8.0,
            )
        except subprocess.TimeoutExpired:
            return NativeDispatchResult(False, "native_timeout", ambiguous=True)
        except OSError:
            return NativeDispatchResult(False, "adapter_unavailable")
        if completed.returncode == 0:
            return NativeDispatchResult(True, "native_api_accepted")
        return NativeDispatchResult(False, "permission_denied")

    def _payload(
        self,
        notification_id: str,
        operation: str,
        title: str = "",
        body: str = "",
        *,
        silent: bool = True,
    ) -> dict[str, Any]:
        return {
            "operation": operation,
            "app_id": self._app_id,
            "tag": notification_id[:16],
            "group": notification_id[16:32],
            "title": title,
            "body": body,
            "silent": silent,
        }

    def post(self, *, notification_id: str, title: str, body: str, silent: bool) -> NativeDispatchResult:
        return self._invoke(
            "post", self._payload(notification_id, "post", title, body, silent=bool(silent))
        )

    def remove(self, notification_id: str) -> NativeDispatchResult:
        return self._invoke("remove", self._payload(notification_id, "remove"))


class MacOSLockStateProvider:
    qualification_sha256 = MACOS_LOCK_PROVIDER_SHA256

    def __init__(self, *, runner: Runner = _run_bytes) -> None:
        self._runner = runner

    def state(self) -> str:
        def find_lock(value: Any) -> bool | None:
            if isinstance(value, dict):
                for name in ("CGSSessionScreenIsLocked", "IOConsoleLocked"):
                    if isinstance(value.get(name), bool):
                        return value[name]
                for nested in value.values():
                    found = find_lock(nested)
                    if found is not None:
                        return found
            elif isinstance(value, list):
                for nested in value:
                    found = find_lock(nested)
                    if found is not None:
                        return found
            return None

        try:
            completed = self._runner(
                ["/usr/sbin/ioreg", "-n", "Root", "-d1", "-a"], input=b"", timeout=2.0
            )
            if completed.returncode != 0:
                return "unknown"
            value = plistlib.loads(completed.stdout)
            locked = find_lock(value)
            return "locked" if locked is True else "unlocked" if locked is False else "unknown"
        except (OSError, subprocess.TimeoutExpired, ValueError, plistlib.InvalidFileException):
            return "unknown"


class WindowsLockStateProvider:
    qualification_sha256 = WINDOWS_LOCK_PROVIDER_SHA256

    def __init__(self, *, user32: Any | None = None, get_last_error: Callable[[], int] | None = None) -> None:
        if user32 is None:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.OpenInputDesktop.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            user32.OpenInputDesktop.restype = wintypes.HANDLE
            user32.CloseDesktop.argtypes = [wintypes.HANDLE]
            user32.CloseDesktop.restype = wintypes.BOOL
            get_last_error = ctypes.get_last_error
        self._user32 = user32
        self._get_last_error = get_last_error or (lambda: 0)

    def state(self) -> str:
        try:
            desktop = self._user32.OpenInputDesktop(0, False, 0x0100)
            if desktop:
                self._user32.CloseDesktop(desktop)
                return "unlocked"
            # Access denied is the documented signal that the input desktop is
            # unavailable to this user session. Every other failure is unknown
            # and must take the stricter private-notification path.
            return "locked" if self._get_last_error() == 5 else "unknown"
        except Exception:
            return "unknown"


class QualifiedNativeSchedulerDelivery:
    def __init__(self, *, adapter: NativeNotificationAdapter, lock_state: LockStateProvider,
                 qualified_full_body_opt_in: bool = False) -> None:
        self.adapter = adapter
        self.lock_state = lock_state
        self.qualified_full_body_opt_in = qualified_full_body_opt_in

    def post(self, payload: dict[str, Any]) -> NativeDispatchResult:
        presentation = select_notification_presentation(
            payload,
            lock_state=self.lock_state.state(),
            qualified_full_body_opt_in=self.qualified_full_body_opt_in,
        )
        return self.adapter.post(
            notification_id=payload["notification_id"],
            title=presentation["title"],
            body=presentation["body"],
            silent=bool(payload["silent"]),
        )

    def remove(self, notification_id: str) -> NativeDispatchResult:
        return self.adapter.remove(notification_id)


def platform_native_components(platform_name: str | None = None, *, config: Any | None = None) -> tuple[NativeNotificationAdapter, LockStateProvider]:
    selected = platform_name or sys.platform
    if selected == "darwin":
        if config is not None and all((
            config.native_helper_path,
            config.native_helper_sha256,
            config.native_helper_code_identity_sha256,
        )):
            return MacOSSchedulerHostAdapter(
                helper_path=config.native_helper_path,
                helper_sha256=config.native_helper_sha256,
                helper_code_identity_sha256=config.native_helper_code_identity_sha256,
            ), MacOSLockStateProvider()
        return MacOSNotificationAdapter(), MacOSLockStateProvider()
    if selected == "win32":
        return WindowsNotificationAdapter(), WindowsLockStateProvider()
    raise ValueError("scheduler native platform unsupported")


def load_qualified_native_delivery(config: Any, *, platform_name: str | None = None):
    """Load both signed contracts before constructing an authority-bearing adapter."""

    from pathlib import Path

    from .scheduler_compatibility import load_scheduler_compatibility
    from .scheduler_native_qualification import load_native_scheduler_qualification
    from .scheduler_callback_qualification import load_scheduler_callback_qualification

    selected = platform_name or sys.platform
    compatibility = load_scheduler_compatibility(
        Path(config.compatibility_manifest_path), config.compatibility_manifest_sha256
    )
    qualification = load_native_scheduler_qualification(
        Path(config.native_qualification_path),
        config.native_qualification_sha256,
        platform_name=selected,
        compatibility=compatibility,
    )
    callback_qualification = None
    if config.callback_qualification_path is not None:
        callback_qualification = load_scheduler_callback_qualification(
            Path(config.callback_qualification_path),
            config.callback_qualification_sha256,
            platform_name=selected,
        )
        if (
            callback_qualification.helper_sha256 != config.native_helper_sha256
            or callback_qualification.helper_code_identity_sha256
            != config.native_helper_code_identity_sha256
            or callback_qualification.adapter_sha256 != MACOS_SCHEDULER_HOST_ADAPTER_SHA256
            or "scheduler_notification_actions" not in compatibility.capabilities
        ):
            raise ValueError("scheduler callback qualification binding mismatch")
    elif "scheduler_notification_actions" in compatibility.capabilities:
        raise ValueError("scheduler callback qualification missing")
    adapter, lock_state = platform_native_components(selected, config=config)
    with ExitStack() as initialization:
        close_adapter = getattr(adapter, "close", None)
        if close_adapter is not None:
            initialization.callback(close_adapter)
        if (
            adapter.qualification_sha256 != qualification.notification_adapter_sha256
            or lock_state.qualification_sha256 != qualification.lock_state_provider_sha256
        ):
            raise ValueError("scheduler native artifact digest mismatch")
        delivery = QualifiedNativeSchedulerDelivery(
            adapter=adapter,
            lock_state=lock_state,
            qualified_full_body_opt_in=bool(config.full_body_when_locked),
        )
        initialization.pop_all()
        return compatibility, qualification, delivery, callback_qualification
