#!/usr/bin/env python3
"""Transcribe the three GreekTTS listening candidates with local Whisper."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jarvis_voice.errors import BackendError  # noqa: E402
from jarvis_voice.macos.greektts15_pilot import QUALITY_FIXTURES  # noqa: E402
from jarvis_voice.macos.greektts15_training import (  # noqa: E402
    ASR_CANDIDATE_LABELS,
    score_candidate_transcripts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--lr-5e-7", required=True)
    parser.add_argument("--lr-1e-6", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    model_dir = Path(args.model_dir).expanduser()
    required = {"config.json", "model.bin", "tokenizer.json", "vocabulary.txt"}
    if not model_dir.is_absolute() or not model_dir.is_dir() or any(not (model_dir / name).is_file() for name in required):
        raise SystemExit("invalid local Whisper model directory")
    candidate_dirs = {
        "baseline": Path(args.baseline).expanduser(),
        "lr-5e-7": Path(args.lr_5e_7).expanduser(),
        "lr-1e-6": Path(args.lr_1e_6).expanduser(),
    }
    if set(candidate_dirs) != ASR_CANDIDATE_LABELS or any(not path.is_absolute() or not path.is_dir() for path in candidate_dirs.values()):
        raise SystemExit("invalid candidate directories")
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    from faster_whisper import WhisperModel

    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True)
    transcripts: dict[str, dict[str, str]] = {}
    details: dict[str, list[dict[str, object]]] = {}
    for label, directory in candidate_dirs.items():
        transcripts[label] = {}
        details[label] = []
        for index, (fixture_id, _) in enumerate(QUALITY_FIXTURES, start=1):
            audio = directory / f"{index:02d}-{fixture_id}.wav"
            if audio.is_symlink() or not audio.is_file():
                raise SystemExit(f"missing fixture: {audio}")
            segments, info = model.transcribe(
                str(audio),
                language="el",
                task="transcribe",
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            realized = list(segments)
            text = " ".join(segment.text.strip() for segment in realized).strip()
            transcripts[label][fixture_id] = text
            details[label].append(
                {
                    "fixture_id": fixture_id,
                    "language_probability": round(float(info.language_probability), 6),
                    "mean_log_probability": round(sum(segment.avg_logprob for segment in realized) / max(len(realized), 1), 6),
                }
            )
    try:
        report = score_candidate_transcripts(dict(QUALITY_FIXTURES), transcripts)
    except BackendError as exc:
        report = {"ready": False, "code": exc.code}
    report["asr"] = {
        "backend": "faster-whisper",
        "model": "Systran/faster-whisper-base",
        "device": "cpu",
        "compute_type": "int8",
        "language": "el",
        "network_used": False,
        "details": details,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
