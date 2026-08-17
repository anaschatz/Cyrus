#!/usr/bin/env python3
from __future__ import annotations

import json
import sys


def emit(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


deferred = None
for raw in sys.stdin:
    message = json.loads(raw)
    kind = message.get("type")
    if kind == "init":
        emit({"type": "ready", "warm_load_ms": 12.5})
    elif kind == "enqueue":
        identity = {
            "response_id": message["response_id"],
            "epoch": message["epoch"],
            "unit_index": message["unit_index"],
        }
        if message["text"] == "defer":
            deferred = identity
        elif message["text"] == "fail":
            emit({"type": "error", "code": "tts.greektts15.fake_failure"})
        elif message["text"] == "completed_only":
            emit({"type": "completed", **identity})
        elif message["text"] == "duplicate_started":
            emit({"type": "started", **identity})
            emit({"type": "started", **identity})
        else:
            emit({"type": "started", **identity})
            emit({"type": "completed", **identity})
    elif kind == "cancel":
        if deferred is not None:
            emit({"type": "started", **deferred})
            emit({"type": "completed", **deferred})
            deferred = None
    elif kind == "close":
        emit({"type": "closed"})
        break
