#!/usr/bin/env python3
"""Score native F1, GreekTTS baseline, and the conservative blend offline."""

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
from jarvis_voice.macos.greektts15_pilot import PRONUNCIATION_REFERENCES  # noqa: E402
from jarvis_voice.macos.greektts15_training import score_pronunciation_transcripts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--blend", required=True)
    parser.add_argument(
        "--candidate-label",
        choices=(
            "blend-25pct",
            "blend-37_5pct",
            "lr-2_5e-7",
            "target-freeze-output-layers",
            "attention-only",
        ),
        default="blend-25pct",
    )
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    model_dir = Path(args.model_dir).expanduser()
    dataset = Path(args.dataset_root).expanduser()
    baseline = Path(args.baseline).expanduser()
    blend = Path(args.blend).expanduser()
    required_model_files = {"config.json", "model.bin", "tokenizer.json", "vocabulary.txt"}
    if (
        not model_dir.is_absolute()
        or not model_dir.is_dir()
        or any(not (model_dir / name).is_file() for name in required_model_files)
        or not dataset.is_absolute()
        or not baseline.is_absolute()
        or not blend.is_absolute()
    ):
        raise SystemExit("invalid offline pronunciation inputs")
    os.environ.update(
        {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
    )
    from faster_whisper import WhisperModel

    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True)
    references = {item.fixture_id: item.text for item in PRONUNCIATION_REFERENCES}
    transcripts = {label: {} for label in ("native", "baseline", args.candidate_label)}
    details = {label: [] for label in transcripts}
    directories = {"baseline": baseline, args.candidate_label: blend}
    for index, item in enumerate(PRONUNCIATION_REFERENCES, start=1):
        audio_files = {
            "native": dataset / "audio" / item.filename,
            "baseline": directories["baseline"] / f"{index:02d}-{item.fixture_id}.wav",
            args.candidate_label: directories[args.candidate_label] / f"{index:02d}-{item.fixture_id}.wav",
        }
        for label, audio in audio_files.items():
            if audio.is_symlink() or not audio.is_file():
                raise SystemExit(f"missing pronunciation fixture: {audio}")
            segments, info = model.transcribe(
                str(audio),
                language="el",
                task="transcribe",
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            realized = list(segments)
            transcripts[label][item.fixture_id] = " ".join(segment.text.strip() for segment in realized).strip()
            details[label].append(
                {
                    "fixture_id": item.fixture_id,
                    "language_probability": round(float(info.language_probability), 6),
                    "mean_log_probability": round(
                        sum(segment.avg_logprob for segment in realized) / max(len(realized), 1),
                        6,
                    ),
                }
            )
    try:
        report = score_pronunciation_transcripts(references, transcripts)
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
