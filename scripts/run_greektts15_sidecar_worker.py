#!/usr/bin/env python3
"""Private JSONL worker for the source-profile GreekTTS F1 sidecar."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jarvis_voice.backends import SpeechUnit  # noqa: E402
from jarvis_voice.errors import BackendError  # noqa: E402
from jarvis_voice.macos.greektts15_runtime import (  # noqa: E402
    GreekTtsF1Backend,
    IsolatedSoundDevicePcmPlayback,
    MlxGreekTtsStreamingEngine,
    load_digest_bound_greektts_candidate,
)
from jarvis_voice.macos.greektts15_sidecar import MAXIMUM_MESSAGE_BYTES  # noqa: E402


def _read_message() -> dict[str, object] | None:
    raw = sys.stdin.buffer.readline(MAXIMUM_MESSAGE_BYTES + 1)
    if not raw:
        return None
    if len(raw) > MAXIMUM_MESSAGE_BYTES or not raw.endswith(b"\n"):
        raise BackendError("tts.greektts15.worker_protocol_invalid", "GreekTTS worker command is invalid", subsystem="tts")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.worker_protocol_invalid", "GreekTTS worker command is invalid", subsystem="tts") from exc
    if not isinstance(value, dict):
        raise BackendError("tts.greektts15.worker_protocol_invalid", "GreekTTS worker command is invalid", subsystem="tts")
    return value


def main() -> int:
    protocol = sys.stdout.buffer
    sys.stdout = sys.stderr
    output_lock = threading.Lock()

    def emit(value: dict[str, object]) -> None:
        encoded = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with output_lock:
            protocol.write(encoded)
            protocol.flush()

    backend: GreekTtsF1Backend | None = None
    monitor_stop = threading.Event()
    try:
        init = _read_message()
        required = {
            "type", "model_root", "approval_manifest_path", "approval_manifest_sha256",
            "repository_root", "warm_on_start", "queue_capacity", "maximum_text_bytes",
        }
        if init is None or set(init) != required or init.get("type") != "init":
            raise BackendError("tts.greektts15.worker_protocol_invalid", "GreekTTS worker initialization is invalid", subsystem="tts")
        if not all(isinstance(init.get(name), str) for name in (
            "model_root", "approval_manifest_path", "approval_manifest_sha256", "repository_root"
        )) or not isinstance(init.get("warm_on_start"), bool):
            raise BackendError("tts.greektts15.worker_protocol_invalid", "GreekTTS worker initialization is invalid", subsystem="tts")
        queue_capacity = init.get("queue_capacity")
        maximum_text_bytes = init.get("maximum_text_bytes")
        if not isinstance(queue_capacity, int) or not isinstance(maximum_text_bytes, int):
            raise BackendError("tts.greektts15.worker_protocol_invalid", "GreekTTS worker bounds are invalid", subsystem="tts")
        candidate = load_digest_bound_greektts_candidate(
            str(init["approval_manifest_path"]),
            str(init["approval_manifest_sha256"]),
            str(init["repository_root"]),
        )
        engine = MlxGreekTtsStreamingEngine(model_root=str(init["model_root"]), candidate=candidate)

        def unit_event(kind: str, unit: SpeechUnit) -> None:
            emit({"type": kind, "response_id": unit.response_id, "epoch": unit.epoch, "unit_index": unit.unit_index})

        backend = GreekTtsF1Backend(
            engine=engine,
            playback_factory=IsolatedSoundDevicePcmPlayback,
            on_started=lambda unit: unit_event("started", unit),
            on_completed=lambda unit: unit_event("completed", unit),
            queue_capacity=queue_capacity,
            maximum_text_bytes=maximum_text_bytes,
            warm_on_start=bool(init["warm_on_start"]),
            warm_timeout_seconds=120.0,
        )
        emit({"type": "ready", "warm_load_ms": backend.warm_load_ms})

        def monitor() -> None:
            while not monitor_stop.wait(0.02):
                error = backend.last_error if backend is not None else None
                if error is not None:
                    emit({"type": "error", "code": error.code})
                    return

        monitor_thread = threading.Thread(target=monitor, name="jarvis-greektts-sidecar-monitor", daemon=True)
        monitor_thread.start()
        while True:
            command = _read_message()
            if command is None:
                break
            kind = command.get("type")
            if kind == "enqueue" and set(command) == {"type", "response_id", "epoch", "unit_index", "locale", "text"}:
                if (
                    not isinstance(command.get("response_id"), str)
                    or not isinstance(command.get("epoch"), int)
                    or isinstance(command.get("epoch"), bool)
                    or int(command["epoch"]) < 1
                    or not isinstance(command.get("unit_index"), int)
                    or isinstance(command.get("unit_index"), bool)
                    or int(command["unit_index"]) < 1
                    or not isinstance(command.get("locale"), str)
                    or not isinstance(command.get("text"), str)
                ):
                    raise BackendError("tts.greektts15.worker_protocol_invalid", "GreekTTS worker unit is invalid", subsystem="tts")
                backend.enqueue(
                    SpeechUnit(
                        response_id=str(command["response_id"]),
                        epoch=int(command["epoch"]),
                        unit_index=int(command["unit_index"]),
                        locale=str(command["locale"]),
                        text=str(command["text"]),
                    )
                )
            elif (
                kind == "cancel"
                and set(command) == {"type", "epoch"}
                and isinstance(command.get("epoch"), int)
                and not isinstance(command.get("epoch"), bool)
                and int(command["epoch"]) >= 1
            ):
                backend.cancel(int(command["epoch"]), "sidecar_cancel")
            elif kind == "close" and set(command) == {"type"}:
                break
            else:
                raise BackendError("tts.greektts15.worker_protocol_invalid", "GreekTTS worker command is invalid", subsystem="tts")
        monitor_stop.set()
        backend.close()
        emit({"type": "closed"})
        return 0
    except BackendError as exc:
        emit({"type": "error", "code": exc.code})
        return 2
    except Exception:
        emit({"type": "error", "code": "tts.greektts15.worker_failed"})
        return 2
    finally:
        monitor_stop.set()
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
