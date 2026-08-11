#!/usr/bin/env python3
"""Run one fixed bilingual lifecycle-memory diagnostic against local Ollama."""

from __future__ import annotations

import json
import urllib.request

from jarvis_orchestrator.anamnesis_shadow import (
    LIFECYCLE_DECISION_SCHEMA,
    LIFECYCLE_SYSTEM_PROMPT,
)


MODEL = "qwen3.5:9b-q4_K_M"
CASES = (
    ("en_ignore", "What time is it?", (), "ignore"),
    ("en_create", "Remember that I prefer tea.", (), "create"),
    (
        "en_replace",
        "Correction: I prefer coffee instead.",
        ({"index": 0, "text": "Remember that I prefer tea.", "language": "en"},),
        "replace_0",
    ),
    (
        "en_cancel",
        "Forget my drink preference.",
        ({"index": 0, "text": "Remember that I prefer coffee.", "language": "en"},),
        "cancel_0",
    ),
    ("el_ignore", "Τι ώρα είναι;", (), "ignore"),
    ("el_create", "Να θυμάσαι ότι προτιμώ τσάι.", (), "create"),
    (
        "el_replace",
        "Διόρθωση: προτιμώ καφέ.",
        ({"index": 0, "text": "Να θυμάσαι ότι προτιμώ τσάι.", "language": "el"},),
        "replace_0",
    ),
    (
        "el_cancel",
        "Ξέχασε την προτίμησή μου για ρόφημα.",
        ({"index": 0, "text": "Να θυμάσαι ότι προτιμώ καφέ.", "language": "el"},),
        "cancel_0",
    ),
)


def _compact(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> int:
    results = []
    for case_id, text, active, expected in CASES:
        language = case_id[:2]
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": LIFECYCLE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _compact(
                        {
                            "current": {"text": text, "language": language},
                            "active_memories": list(active),
                        }
                    ),
                },
            ],
            "stream": False,
            "think": False,
            "format": LIFECYCLE_DECISION_SCHEMA,
            "options": {
                "temperature": 0,
                "top_p": 1,
                "num_ctx": 4096,
                "num_predict": 32,
            },
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=_compact(body).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            value = json.loads(response.read())
        raw_content = value["message"].get("content")
        try:
            content = json.loads(raw_content) if isinstance(raw_content, str) else None
        except (json.JSONDecodeError, TypeError):
            content = None
        actual = (
            content.get("decision")
            if isinstance(content, dict) and set(content) == {"decision"}
            else None
        )
        results.append(
            {
                "case": case_id,
                "expected": expected,
                "actual": actual,
                "pass": actual == expected,
                "done_reason": value.get("done_reason"),
                "prompt_tokens": value.get("prompt_eval_count"),
                "output_tokens": value.get("eval_count"),
                "content_bytes": (
                    len(raw_content.encode("utf-8"))
                    if isinstance(raw_content, str)
                    else None
                ),
            }
        )
    print(
        _compact(
            {
                "model": MODEL,
                "passed": sum(item["pass"] for item in results),
                "total": len(results),
                "results": results,
            }
        )
    )
    return 0 if all(item["pass"] for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
