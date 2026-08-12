from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .strict_json import canonical_bytes, parse_body


PREFLIGHT_SUITE_VERSION = "jarvis-model-profile-preflight-v1"
FALLBACK_SYSTEM_PROMPT = (
    "You are the JARVIS local fallback. Answer in the user's language. "
    "Follow every requested word, sentence, and output-format limit exactly. "
    "Be concise and factually accurate. Never claim to execute tools or actions. "
    "If a request requires tools, research, private memory, or the primary model, "
    "reply exactly PRIMARY_REQUIRED."
)


class PreflightFailure(RuntimeError):
    """A content-free profile preflight failure."""


@dataclass(frozen=True, slots=True)
class Fixture:
    case_id: str
    language: str
    prompt: str
    max_tokens: int
    exact: str | None = None
    required_groups: tuple[tuple[str, ...], ...] = ()
    forbidden_groups: tuple[tuple[str, ...], ...] = ()
    max_words: int | None = None
    category: str = "preflight"


FIXTURES = (
    Fixture("el_exact", "el", "Απάντησε μόνο με τη λέξη: Αθήνα", 16, exact="Αθήνα"),
    Fixture("el_explain", "el", "Σε μία σύντομη πρόταση εξήγησε σωστά γιατί ο ουρανός φαίνεται μπλε.", 128,
            required_groups=(("μπλε",), ("σκεδ", "διαχ", "rayleigh", "ράιλι"), ("ατμόσφ", "αέρα"))),
    Fixture("el_summary", "el", "Σύνοψισε σε έως πέντε λέξεις: Ο Γιάννης πήγε στην αγορά για να αγοράσει ψωμί και γάλα.", 32,
            required_groups=(("ψωμί",), ("γάλα",)), max_words=5),
    Fixture("en_exact", "en", "Reply with only this word: Paris", 16, exact="Paris"),
    Fixture("en_explain", "en", "In one short sentence, correctly explain why the sky appears blue.", 128,
            required_groups=(("blue",), ("scatter",), ("atmosphere", "air"))),
    Fixture("en_summary", "en", "Summarize in at most five words: John went to the market to buy bread and milk.", 32,
            required_groups=(("bread",), ("milk",)), max_words=5),
)


def fixture_digest() -> str:
    value = {
        "system_prompt": FALLBACK_SYSTEM_PROMPT,
        "fixtures": [
            {
                "case_id": item.case_id,
                "language": item.language,
                "prompt": item.prompt,
                "max_tokens": item.max_tokens,
                "exact": item.exact,
                "required_groups": [list(group) for group in item.required_groups],
                "forbidden_groups": [list(group) for group in item.forbidden_groups],
                "max_words": item.max_words,
                "category": item.category,
            }
            for item in FIXTURES
        ],
    }
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _word_count(text: str) -> int:
    return len(re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", text, flags=re.UNICODE))


def score_output(fixture: Fixture, text: str, finish_reason: str | None) -> dict[str, bool]:
    lowered = text.casefold().strip()
    exact_ok = fixture.exact is None or text.strip() == fixture.exact
    # Exact outputs may intentionally be language-neutral numbers, timestamps,
    # identifiers, or PRIMARY_REQUIRED. In those cases the exact expectation is
    # the language/content contract; requiring a Greek or Latin letter would
    # create a false failure.
    language_ok = exact_ok if fixture.exact is not None else (
        any("\u0370" <= character <= "\u03ff" for character in lowered)
        if fixture.language == "el"
        else any("a" <= character <= "z" for character in lowered)
    )
    required_ok = all(any(candidate.casefold() in lowered for candidate in group) for group in fixture.required_groups)
    forbidden_ok = not any(any(candidate.casefold() in lowered for candidate in group) for group in fixture.forbidden_groups)
    word_limit_ok = fixture.max_words is None or _word_count(text) <= fixture.max_words
    completed_ok = finish_reason == "stop"
    return {
        "language_ok": language_ok,
        "exact_ok": exact_ok,
        "required_content_ok": required_ok,
        "forbidden_content_ok": forbidden_ok,
        "word_limit_ok": word_limit_ok,
        "completed_ok": completed_ok,
        "quality_pass": language_ok and exact_ok and required_ok and forbidden_ok and word_limit_ok and completed_ok,
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1 or any(not math.isfinite(item) for item in values):
        raise ValueError("invalid percentile sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


class LoopbackClient:
    def __init__(
        self,
        base_url: str,
        bearer: str,
        *,
        expected_capabilities: dict[str, bool] | None = None,
    ):
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port is None
            or parsed.path.rstrip("/") != "/v1"
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise PreflightFailure("invalid loopback base URL")
        if len(bearer.encode("utf-8")) < 32:
            raise PreflightFailure("invalid private bearer")
        self.host = parsed.hostname
        self.port = parsed.port
        self.bearer = bearer
        self.expected_capabilities = expected_capabilities or {
            "text": True,
            "stream": True,
            "tools": False,
            "json_object": False,
            "json_schema": False,
        }

    def _connection(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=180.0)

    def _headers(self, body: bytes | None = None) -> dict[str, str]:
        result = {"Authorization": "Bearer " + self.bearer, "Host": f"127.0.0.1:{self.port}"}
        if body is not None:
            result.update({"Content-Type": "application/json", "Content-Length": str(len(body))})
        return result

    def readiness(self) -> None:
        connection = self._connection()
        try:
            connection.request("GET", "/readyz", headers={"Host": f"127.0.0.1:{self.port}"})
            response = connection.getresponse()
            raw = response.read()
            if response.status != 200 or parse_body(raw) != {"ready": True}:
                raise PreflightFailure("gateway is not ready")
        finally:
            connection.close()

    def capabilities(self) -> None:
        connection = self._connection()
        try:
            connection.request("GET", "/jarvis/v1/capabilities", headers=self._headers())
            response = connection.getresponse()
            value = parse_body(response.read())
            if (
                response.status != 200
                or not isinstance(value, dict)
                or value.get("model") != "jarvis-local"
                or value.get("capabilities") != self.expected_capabilities
            ):
                raise PreflightFailure("gateway capabilities do not match local-fallback")
        finally:
            connection.close()

    def stream(self, fixture: Fixture) -> dict[str, Any]:
        body = canonical_bytes({
            "model": "jarvis-local",
            "messages": [
                {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                {"role": "user", "content": fixture.prompt},
            ],
            "max_completion_tokens": fixture.max_tokens,
            "temperature": 0,
            "top_p": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
        })
        connection = self._connection()
        started = time.perf_counter()
        first_content: float | None = None
        parts: list[str] = []
        usage: dict[str, Any] | None = None
        finish_reason: str | None = None
        done = False
        try:
            connection.request("POST", "/v1/chat/completions", body=body, headers=self._headers(body))
            response = connection.getresponse()
            if response.status != 200 or response.getheader("Content-Type") != "text/event-stream; charset=utf-8":
                raise PreflightFailure("stream request failed")
            while True:
                raw = response.readline()
                if not raw:
                    break
                if not raw.startswith(b"data: "):
                    if raw.strip():
                        raise PreflightFailure("invalid SSE framing")
                    continue
                payload = raw[6:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                event = parse_body(payload)
                if not isinstance(event, dict):
                    raise PreflightFailure("invalid SSE event")
                choices = event.get("choices")
                if choices == []:
                    candidate = event.get("usage")
                    if isinstance(candidate, dict):
                        usage = candidate
                    continue
                if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                    raise PreflightFailure("invalid SSE choice")
                choice = choices[0]
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    raise PreflightFailure("invalid SSE delta")
                content = delta.get("content")
                if isinstance(content, str) and content:
                    if first_content is None:
                        first_content = time.perf_counter()
                    parts.append(content)
                candidate_reason = choice.get("finish_reason")
                if candidate_reason is not None:
                    finish_reason = candidate_reason
        finally:
            ended = time.perf_counter()
            connection.close()
        if first_content is None or not done or not isinstance(usage, dict):
            raise PreflightFailure("incomplete SSE response")
        completion_tokens = usage.get("completion_tokens")
        if type(completion_tokens) is not int or completion_tokens <= 0:
            raise PreflightFailure("invalid completion token accounting")
        text = "".join(parts).strip()
        score = score_output(fixture, text, finish_reason)
        decode_seconds = max(0.000001, ended - first_content)
        return {
            "case_id": fixture.case_id,
            "language": fixture.language,
            "category": fixture.category,
            "protocol_pass": True,
            **score,
            "finish_reason": finish_reason,
            "ttft_seconds": first_content - started,
            "end_to_end_seconds": ended - started,
            "completion_tokens": completion_tokens,
            "decode_tokens_per_second": completion_tokens / decode_seconds,
            "output_utf8_bytes": len(text.encode("utf-8")),
        }


def run_preflight(client: LoopbackClient, *, warmups: int, repetitions: int) -> dict[str, Any]:
    if not 1 <= warmups <= 20 or not 1 <= repetitions <= 20:
        raise ValueError("preflight counts are out of bounds")
    client.readiness()
    client.capabilities()
    warmup_fixture = Fixture("warmup", "en", "Reply with only this word: ready", 16, exact="ready")
    warmup_seconds: list[float] = []
    for _ in range(warmups):
        started = time.perf_counter()
        client.stream(warmup_fixture)
        warmup_seconds.append(time.perf_counter() - started)
    results = [client.stream(fixture) for _ in range(repetitions) for fixture in FIXTURES]
    ttft = [float(item["ttft_seconds"]) for item in results]
    end_to_end = [float(item["end_to_end_seconds"]) for item in results]
    decode = [float(item["decode_tokens_per_second"]) for item in results]
    metrics = {
        "ttft_p50_seconds": statistics.median(ttft),
        "ttft_p95_seconds": percentile(ttft, 0.95),
        "end_to_end_p95_seconds": percentile(end_to_end, 0.95),
        "decode_p5_tokens_per_second": percentile(decode, 0.05),
        "decode_p50_tokens_per_second": statistics.median(decode),
    }
    quality_passes = sum(bool(item["quality_pass"]) for item in results)
    thresholds = {
        "all_quality_cases": quality_passes == len(results),
        "ttft_p95_at_most_2_seconds": metrics["ttft_p95_seconds"] <= 2.0,
        "end_to_end_p95_at_most_8_seconds": metrics["end_to_end_p95_seconds"] <= 8.0,
        "decode_p5_at_least_20_tokens_per_second": metrics["decode_p5_tokens_per_second"] >= 20.0,
    }
    return {
        "report_schema_version": 1,
        "suite_version": PREFLIGHT_SUITE_VERSION,
        "authoritative_qualification": False,
        "profile_id": "local-fallback",
        "public_model_alias": "jarvis-local",
        "fixture_digest": fixture_digest(),
        "warmups": warmups,
        "repetitions": repetitions,
        "sample_count": len(results),
        "warmup_seconds": warmup_seconds,
        "quality": {"passed": quality_passes, "total": len(results)},
        "metrics": metrics,
        "thresholds": thresholds,
        "cases": results,
        "status": "preflight_pass" if all(thresholds.values()) else "preflight_failed",
    }


def _bearer_from_config(path: Path) -> str:
    value = parse_body(path.read_bytes())
    bearer = value.get("inference_bearer_token") if isinstance(value, dict) else None
    if not isinstance(bearer, str) or len(bearer.encode("utf-8")) < 32:
        raise PreflightFailure("orchestrator config has no valid private bearer")
    return bearer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run-profile-preflight")
    parser.add_argument("--base-url", default="http://127.0.0.1:8766/v1")
    parser.add_argument("--orchestrator-config", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        report = run_preflight(
            LoopbackClient(args.base_url, _bearer_from_config(args.orchestrator_config)),
            warmups=args.warmups,
            repetitions=args.repetitions,
        )
    except (OSError, ValueError, PreflightFailure):
        print(json.dumps({"status": "preflight_error", "authoritative_qualification": False}, separators=(",", ":")))
        return 2
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0 if report["status"] == "preflight_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
