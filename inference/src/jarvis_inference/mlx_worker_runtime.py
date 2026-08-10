"""Pinned MLX-LM worker entrypoint.

This process owns warm weights, but creates generation/Outlines state anew for
every command. It communicates only over inherited stdio pipes and never opens a
network listener.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from typing import Any


def emit(value: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def strict_object(line: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    def nonfinite(_: str) -> None:
        raise ValueError

    value = json.loads(line, object_pairs_hook=pairs, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError
    return value


def main() -> int:
    model = tokenizer = None
    prepared_tokens: tuple[int, ...] | None = None
    for line in sys.stdin:
        try:
            command = strict_object(line)
            operation = command.get("op")
            if operation == "init":
                import mlx_lm
                import outlines
                model_dir = os.environ["JARVIS_MLX_MODEL_DIR"]
                model, tokenizer = mlx_lm.load(model_dir)
                emit({"event": "ready"})
            elif operation == "prepare":
                prompt = command["prompt"]
                if not isinstance(prompt, str) or tokenizer is None:
                    raise ValueError
                token_ids = tokenizer.encode(prompt, add_special_tokens=True)
                prepared_tokens = tuple(int(token) for token in token_ids)
                emit({"event": "prepared", "token_ids": list(token_ids)})
            elif operation == "generate":
                supplied = command.get("token_ids")
                if model is None or tokenizer is None or not isinstance(supplied, list) or tuple(supplied) != prepared_tokens:
                    raise ValueError
                schema = command.get("schema")
                accepted_count = 0
                finish_reason: str | None = None
                generated_ids: list[int] = []
                stop_token_sequences = [
                    tuple(tokenizer.encode(stop, add_special_tokens=False))
                    for stop in command.get("stop", [])
                ]
                from mlx_lm.sample_utils import make_sampler
                sampler = make_sampler(command["temperature"], top_p=command["top_p"])
                if schema is None:
                    import mlx_lm
                    iterator = mlx_lm.stream_generate(
                        model, tokenizer, prompt=list(prepared_tokens),
                        max_tokens=command["max_tokens"],
                        sampler=sampler,
                    )
                else:
                    # The Outlines adapter is constructed per request so no
                    # parser/FSM state is reusable by the next request.
                    import outlines
                    outline_model = outlines.from_mlxlm(model, tokenizer)
                    iterator = outline_model.stream(
                        list(prepared_tokens), output_type=schema,
                        max_tokens=command["max_tokens"],
                        sampler=sampler,
                    )
                for response in iterator:
                    text = response if isinstance(response, str) else getattr(response, "text", None)
                    if not isinstance(text, str):
                        raise RuntimeError("worker response has no text")
                    if text:
                        emit({"event": "text", "text": text})
                    token = getattr(response, "token", None)
                    if not isinstance(token, int) or isinstance(token, bool):
                        raise RuntimeError("runtime cannot expose accepted token IDs")
                    accepted_count += 1
                    generated_ids.append(token)
                    candidate = getattr(response, "finish_reason", None)
                    if candidate in {"stop", "length"}:
                        finish_reason = candidate
                    if any(sequence and tuple(generated_ids[-len(sequence):]) == sequence for sequence in stop_token_sequences):
                        finish_reason = "stop"
                        break
                if finish_reason is None:
                    raise RuntimeError("runtime cannot expose an exact finish reason")
                emit({"event": "done", "completion_tokens": accepted_count, "finish_reason": finish_reason})
                iterator = None
                response = None
                if schema is not None:
                    outline_model = None
                prepared_tokens = None
            elif operation == "cleanup":
                prepared_tokens = None
                try:
                    import mlx.core as mx
                    mx.synchronize()
                    mx.clear_cache()
                except (ImportError, AttributeError):
                    pass
                gc.collect()
                emit({"event": "idle"})
            elif operation == "shutdown":
                return 0
            else:
                raise ValueError
        except BaseException:
            # Never echo prompts, schemas, argument values or exception text.
            emit({"event": "error", "code": "worker_failure"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
