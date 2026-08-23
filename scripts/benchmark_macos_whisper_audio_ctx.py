#!/usr/bin/env python3
"""Benchmark whisper.cpp per-request audio_ctx without changing the model.

The benchmark is deliberately offline and compares every candidate against the
full-context decoder on the same deterministic, transcript-labelled corpus.
It reports latency as well as reference edit distance so a faster profile is
never promoted merely because it is fast.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import math
import socket
import statistics
import subprocess
import sys
import time
import unicodedata
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Fixture:
    path: Path
    reference: str
    duration_seconds: float


def _normalise(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    return " ".join(
        "".join(character for character in folded if unicodedata.category(character)[0] in {"L", "N", "Z"}).split()
    )


def _canonical_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, 1):
        current = [left_index]
        for right_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _error_counts(reference: str, hypothesis: str) -> tuple[int, int, int, int]:
    normal_reference = _normalise(reference)
    normal_hypothesis = _normalise(hypothesis)
    reference_words = normal_reference.split()
    hypothesis_words = normal_hypothesis.split()
    return (
        _distance(reference_words, hypothesis_words),
        max(1, len(reference_words)),
        _distance(list(normal_reference.replace(" ", "")), list(normal_hypothesis.replace(" ", ""))),
        max(1, len(normal_reference.replace(" ", ""))),
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _fixtures(metadata: Path, audio_root: Path, count: int) -> list[Fixture]:
    available: list[Fixture] = []
    with metadata.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream, delimiter="|"):
            if len(row) != 2:
                continue
            path = audio_root / row[0]
            if not path.is_file():
                continue
            with wave.open(str(path), "rb") as source:
                duration = source.getnframes() / source.getframerate()
            if 1.0 <= duration <= 12.0:
                available.append(Fixture(path, row[1], duration))
    if len(available) < count:
        raise RuntimeError(f"corpus has only {len(available)} eligible fixtures")
    available.sort(key=lambda fixture: (fixture.duration_seconds, fixture.path.name))
    if count == 1:
        return [available[len(available) // 2]]
    selected = [available[round(index * (len(available) - 1) / (count - 1))] for index in range(count)]
    if len({fixture.path for fixture in selected}) != count:
        raise RuntimeError("deterministic fixture selection produced duplicates")
    return selected


def _resample(path: Path, ffmpeg: str) -> bytes:
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _multipart(wav: bytes, boundary: str, audio_ctx: int) -> bytes:
    marker = boundary.encode("ascii")
    parts: list[bytes] = []

    def field(name: str, value: str) -> None:
        parts.extend(
            [
                b"--" + marker + b"\r\n",
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"),
                value.encode("ascii"),
                b"\r\n",
            ]
        )

    for name, value in (
        ("response_format", "verbose_json"),
        ("language", "auto"),
        ("temperature", "0.0"),
        ("audio_ctx", str(audio_ctx)),
    ):
        field(name, value)
    parts.extend(
        [
            b"--" + marker + b"\r\n",
            b'Content-Disposition: form-data; name="file"; filename="turn.wav"\r\n',
            b"Content-Type: audio/wav\r\n\r\n",
            wav,
            b"\r\n--" + marker + b"--\r\n",
        ]
    )
    return b"".join(parts)


def _request(port: int, request_path: str, wav: bytes, audio_ctx: int) -> tuple[float, str, str]:
    boundary = "cyrus-" + uuid.uuid4().hex
    body = _multipart(wav, boundary, audio_ctx)
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    started = time.perf_counter()
    connection.request(
        "POST",
        request_path + "/inference",
        body=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    response = connection.getresponse()
    raw = response.read()
    elapsed = time.perf_counter() - started
    connection.close()
    if response.status != 200:
        raise RuntimeError(f"whisper.cpp returned {response.status}: {raw[:200]!r}")
    payload = json.loads(raw)
    return elapsed, str(payload.get("text", "")).strip(), str(payload.get("language", ""))


def _wait_ready(process: subprocess.Popen[bytes], port: int, request_path: str) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("whisper.cpp sidecar exited during startup")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.25)
            connection.request("GET", request_path + "/")
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return
        except OSError:
            pass
        time.sleep(0.025)
    raise RuntimeError("whisper.cpp startup timed out")


def _summarise(records: list[dict[str, object]], contexts: Iterable[int]) -> dict[str, object]:
    baseline_by_fixture = {str(item["fixture"]): item for item in records if item["audio_ctx"] == 0}
    summary: dict[str, object] = {}
    for context in contexts:
        selected = [item for item in records if item["audio_ctx"] == context]
        latencies = [float(item["latency_seconds"]) for item in selected]
        word_errors = sum(int(item["word_errors"]) for item in selected)
        reference_words = sum(int(item["reference_words"]) for item in selected)
        character_errors = sum(int(item["character_errors"]) for item in selected)
        reference_characters = sum(int(item["reference_characters"]) for item in selected)
        changed = raw_changed = degraded = improved = 0
        if context:
            for item in selected:
                baseline = baseline_by_fixture[str(item["fixture"])]
                changed += _normalise(str(item["text"])) != _normalise(str(baseline["text"]))
                raw_changed += _canonical_text(str(item["text"])) != _canonical_text(str(baseline["text"]))
                degraded += int(item["word_errors"]) > int(baseline["word_errors"])
                improved += int(item["word_errors"]) < int(baseline["word_errors"])
        summary[str(context)] = {
            "count": len(selected),
            "latency_mean_seconds": round(statistics.fmean(latencies), 3),
            "latency_p50_seconds": round(statistics.median(latencies), 3),
            "latency_p95_seconds": round(_percentile(latencies, 0.95), 3),
            "word_error_rate": round(word_errors / reference_words, 6),
            "character_error_rate": round(character_errors / reference_characters, 6),
            "exact_reference_matches": sum(int(item["word_errors"]) == 0 for item in selected),
            "changed_from_full_context": changed,
            "raw_changed_from_full_context": raw_changed,
            "degraded_from_full_context": degraded,
            "improved_from_full_context": improved,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--audio-contexts", type=int, nargs="+", default=[0, 384, 512, 768])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sample_count < 1 or args.sample_count > 500:
        parser.error("--sample-count must be between 1 and 500")
    contexts = list(dict.fromkeys(args.audio_contexts))
    if not contexts or contexts[0] != 0 or any(context < 0 or context > 1500 for context in contexts):
        parser.error("--audio-contexts must start with 0 and stay between 0 and 1500")
    for artifact in (args.server, args.model, args.metadata, args.audio_root):
        if not artifact.exists():
            parser.error(f"missing local artifact: {artifact}")

    fixtures = _fixtures(args.metadata, args.audio_root, args.sample_count)
    resampled = {fixture.path.name: _resample(fixture.path, args.ffmpeg) for fixture in fixtures}
    port = _available_port()
    request_path = "/audio-ctx-bench-" + uuid.uuid4().hex
    process = subprocess.Popen(
        [
            str(args.server),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--request-path", request_path,
            "--model", str(args.model),
            "--threads", "4",
            "--language", "auto",
            "--max-context", "0",
            "--flash-attn",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    records: list[dict[str, object]] = []
    try:
        _wait_ready(process, port, request_path)
        total = len(fixtures) * len(contexts)
        for fixture_index, fixture in enumerate(fixtures):
            order = contexts[fixture_index % len(contexts) :] + contexts[: fixture_index % len(contexts)]
            for context in order:
                elapsed, text, language = _request(port, request_path, resampled[fixture.path.name], context)
                word_errors, reference_words, character_errors, reference_characters = _error_counts(fixture.reference, text)
                record: dict[str, object] = {
                    "fixture": fixture.path.name,
                    "duration_seconds": round(fixture.duration_seconds, 3),
                    "audio_ctx": context,
                    "latency_seconds": round(elapsed, 3),
                    "language": language,
                    "text": text,
                    "reference": fixture.reference,
                    "word_errors": word_errors,
                    "reference_words": reference_words,
                    "character_errors": character_errors,
                    "reference_characters": reference_characters,
                }
                records.append(record)
                print(json.dumps({"completed": len(records), "total": total, **record}, ensure_ascii=False), flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    report = {
        "version": "jarvis-whisper-audio-ctx-benchmark/1",
        "model": str(args.model),
        "sample_count": len(fixtures),
        "audio_contexts": contexts,
        "summary": _summarise(records, contexts),
        "records": records,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"summary": report["summary"]}, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
