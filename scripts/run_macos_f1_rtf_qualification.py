#!/usr/bin/env python3
"""Measure the approved F1 incremental runtime in bounded, content-free batches."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jarvis_voice.backends import SpeechUnit  # noqa: E402
from jarvis_voice.errors import BackendError  # noqa: E402
from jarvis_voice.macos.greektts15_pilot import QUALITY_FIXTURES, QUALITY_MAX_MEAN_REALTIME_FACTOR  # noqa: E402
from jarvis_voice.macos.greektts15_runtime import (  # noqa: E402
    MlxGreekTtsStreamingEngine,
    PcmChunk,
    load_digest_bound_greektts_candidate,
)
from jarvis_voice.qualification import write_report  # noqa: E402


MAXIMUM_SAMPLES_PER_PROCESS = 2
SAMPLE_RATE = 24_000


def _bounded_sample_count(value: str) -> int:
    count = int(value)
    if not 1 <= count <= MAXIMUM_SAMPLES_PER_PROCESS:
        raise argparse.ArgumentTypeError("sample count must be 1 or 2")
    return count


def measure_rtf_batch(
    engine: object,
    fixtures: Sequence[tuple[str, str]],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    if not 1 <= len(fixtures) <= MAXIMUM_SAMPLES_PER_PROCESS:
        raise BackendError("tts.greektts15.rtf_batch_invalid", "RTF batch size is invalid", subsystem="tts")
    warm_load_ms = float(engine.warm())
    samples: list[dict[str, object]] = []
    try:
        for index, (fixture_id, text) in enumerate(fixtures, start=1):
            cancelled = threading.Event()
            started = clock()
            first_pcm_at: float | None = None
            pcm_bytes = 0
            for chunk in engine.stream(SpeechUnit(f"rtf-{fixture_id}", index, 1, "el-GR", text), cancelled):
                if not isinstance(chunk, PcmChunk) or chunk.sample_rate != SAMPLE_RATE or chunk.channels != 1:
                    raise BackendError("tts.greektts15.rtf_chunk_invalid", "RTF runtime emitted invalid PCM", subsystem="tts")
                if first_pcm_at is None:
                    first_pcm_at = clock()
                pcm_bytes += len(chunk.pcm16)
            completed = clock()
            if first_pcm_at is None or pcm_bytes <= 0 or pcm_bytes % 2:
                raise BackendError("tts.greektts15.rtf_audio_missing", "RTF runtime emitted no PCM", subsystem="tts")
            audio_seconds = pcm_bytes / 2.0 / SAMPLE_RATE
            synthesis_seconds = completed - started
            samples.append(
                {
                    "fixture_id": fixture_id,
                    "audio_duration_ms": round(audio_seconds * 1000.0, 3),
                    "synthesis_ms": round(synthesis_seconds * 1000.0, 3),
                    "time_to_first_pcm_ms": round((first_pcm_at - started) * 1000.0, 3),
                    "realtime_factor": round(synthesis_seconds / audio_seconds, 4),
                }
            )
    finally:
        engine.cancel()
    mean_rtf = sum(float(sample["realtime_factor"]) for sample in samples) / len(samples)
    return {
        "ready": mean_rtf <= QUALITY_MAX_MEAN_REALTIME_FACTOR,
        "code": (
            "tts.greektts15.rtf_within_limit"
            if mean_rtf <= QUALITY_MAX_MEAN_REALTIME_FACTOR
            else "tts.greektts15.rtf_limit_exceeded"
        ),
        "warm_load_ms": round(warm_load_ms, 3),
        "mean_realtime_factor": round(mean_rtf, 4),
        "mean_realtime_factor_limit": QUALITY_MAX_MEAN_REALTIME_FACTOR,
        "samples": samples,
        "raw_audio_persisted": False,
        "network_used": False,
        "installation_performed": False,
        "maximum_samples_per_process": MAXIMUM_SAMPLES_PER_PROCESS,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--approval-manifest", type=Path, required=True)
    parser.add_argument("--approval-manifest-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--fixture-start", type=int, choices=range(len(QUALITY_FIXTURES)), required=True)
    parser.add_argument("--sample-count", type=_bounded_sample_count, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        stop = args.fixture_start + args.sample_count
        if stop > len(QUALITY_FIXTURES):
            raise BackendError("tts.greektts15.rtf_batch_invalid", "RTF fixture range is invalid", subsystem="tts")
        candidate = load_digest_bound_greektts_candidate(
            args.approval_manifest.resolve(),
            args.approval_manifest_sha256,
            args.repository_root.resolve(),
        )
        engine = MlxGreekTtsStreamingEngine(model_root=args.model_root.resolve(), candidate=candidate)
        report = measure_rtf_batch(engine, QUALITY_FIXTURES[args.fixture_start:stop])
    except BackendError as exc:
        report = {
            "ready": False,
            "code": exc.code,
            "network_used": False,
            "installation_performed": False,
            "maximum_samples_per_process": MAXIMUM_SAMPLES_PER_PROCESS,
        }
        if exc.safe_context:
            report["context"] = dict(exc.safe_context)
    write_report(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
