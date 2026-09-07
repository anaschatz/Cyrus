#!/usr/bin/env python3
"""One bounded, content-free native MLX generation smoke."""

from __future__ import annotations

import argparse
from contextlib import closing
import json
import threading
import time
from pathlib import Path

from jarvis_inference.contracts import normalize_request
from jarvis_inference.mlx_engine import MlxWorkerEngine
from jarvis_inference.manifest import load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(prog="smoke-macos-inference")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.manifest)
        with closing(MlxWorkerEngine(manifest)) as engine:
            engine.start(time.monotonic() + 600.0)
            request = normalize_request(
                {
                    "model": "jarvis-local",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Απάντησε μόνο με τη λέξη ναι. /no_think",
                        }
                    ],
                    "max_completion_tokens": 24,
                    "temperature": 0,
                    "top_p": 1,
                },
                alias=manifest.alias,
                capabilities=manifest.capabilities,
                context_window=manifest.context_window,
            )
            deadline = time.monotonic() + 300.0
            prepared = engine.prepare(request, deadline)
            events = list(engine.generate(prepared, threading.Event(), deadline))
            engine.cleanup(deadline)
            text = "".join(event.text for event in events if event.kind == "text")
            finals = [event for event in events if event.kind == "final"]
            ready = bool(
                text
                and len(finals) == 1
                and finals[0].finish_reason in {"stop", "length"}
                and type(finals[0].completion_tokens) is int
                and 0 < finals[0].completion_tokens <= request.max_completion_tokens
            )
        print(
            json.dumps(
                {
                    "ready": ready,
                    "code": "smoke.macos_inference_generated"
                    if ready
                    else "smoke.macos_inference_invalid",
                    "prompt_tokens": len(prepared.token_ids),
                    "completion_tokens": finals[0].completion_tokens
                    if len(finals) == 1
                    else None,
                    "text_events": sum(event.kind == "text" for event in events),
                    "greek_output": any(
                        "\u0370" <= character <= "\u03ff" for character in text.lower()
                    ),
                    "content_emitted": False,
                    "network_used": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0 if ready else 2
    except BaseException:
        print(
            json.dumps(
                {
                    "ready": False,
                    "code": "smoke.macos_inference_failed",
                    "content_emitted": False,
                    "network_used": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
