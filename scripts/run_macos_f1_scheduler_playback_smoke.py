#!/usr/bin/env python3
"""Run one real F1 scheduler-playback unit through the isolated macOS TTS path."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jarvis_voice.backends import SpeechUnit  # noqa: E402
from jarvis_voice.config import VoiceConfig, load_voice_config  # noqa: E402
from jarvis_voice.errors import BackendError  # noqa: E402
from jarvis_voice.macos.native_run import create_macos_calibration_tts  # noqa: E402
from jarvis_voice.qualification import write_report  # noqa: E402
from jarvis_voice.scheduler_playback import SchedulerPlaybackController, SchedulerPlaybackRecord  # noqa: E402
from jarvis_voice.scheduler_protocol import SchedulerEventFactory  # noqa: E402
from jarvis_voice.state import VoiceStateMachine  # noqa: E402


class OneUnitJournal:
    def __init__(self) -> None:
        self.record = SchedulerPlaybackRecord("a" * 64, 1, "el-GR", "Η φωνή λειτουργεί καθαρά.")
        self.state = "STAGED"

    def next_staged_playback(self):
        return self.record if self.state == "STAGED" else None

    def set_playback_state(self, notification_id, unit_index, state):
        if (notification_id, unit_index) != (self.record.notification_id, self.record.unit_index):
            return False
        allowed = {"PLAYING": "STAGED", "COMPLETED": "PLAYING", "SKIPPED": "PLAYING"}
        if allowed.get(state) != self.state:
            return False
        self.state = state
        return True

    def recover_interrupted_playback(self):
        return []

    def cancel_notification_playback(self, notification_id):
        if notification_id != self.record.notification_id or self.state not in {"STAGED", "PLAYING"}:
            return []
        self.state = "CANCELLED"
        return [self.record.unit_index]


def qualification_config(args: argparse.Namespace) -> VoiceConfig:
    config = load_voice_config(args.config)
    greek = replace(
        config.macos.greektts15,
        model_root=str(args.model_root.resolve()),
        approval_manifest_path=str(args.approval_manifest.resolve()),
        approval_manifest_sha256=args.approval_manifest_sha256,
        repository_root=str(args.repository_root.resolve()),
        warm_on_start=True,
    )
    return replace(config, macos=replace(config.macos, tts_backend="greektts15_f1", greektts15=greek))


def run_scheduler_playback_smoke(
    config: VoiceConfig,
    *,
    timeout_seconds: float = 120.0,
    tts_builder=create_macos_calibration_tts,
    clock=time.perf_counter,
) -> dict[str, object]:
    if timeout_seconds <= 0:
        raise BackendError("qualification.scheduler_playback_config_invalid", "Scheduler smoke timeout is invalid", subsystem="qualification")
    callbacks: queue.Queue[tuple[str, SpeechUnit, float]] = queue.Queue()
    tts = tts_builder(
        config,
        on_started=lambda unit: callbacks.put_nowait(("started", unit, clock())),
        on_completed=lambda unit: callbacks.put_nowait(("completed", unit, clock())),
    )
    journal = OneUnitJournal()
    state = VoiceStateMachine()
    controller = SchedulerPlaybackController(
        journal=journal,
        tts=tts,
        factory=SchedulerEventFactory(str(uuid.uuid4())),
        reserve_epoch=state.reserve_output_epoch,
    )
    receipts: list[str] = []
    pumped_at = clock()
    started_at: float | None = None
    completed_at: float | None = None
    try:
        controller.pump(allowed=True)
        deadline = clock() + timeout_seconds
        while clock() < deadline and completed_at is None:
            try:
                kind, unit, observed_at = callbacks.get(timeout=min(0.1, max(0.001, deadline - clock())))
            except queue.Empty:
                backend_error = getattr(tts, "last_error", None)
                if isinstance(backend_error, BackendError):
                    raise BackendError(
                        "qualification.scheduler_playback_tts_failed",
                        "Scheduler playback TTS failed",
                        subsystem="qualification",
                        safe_context={"backend_error_code": backend_error.code},
                    ) from backend_error
                continue
            handled, emitted = controller.on_started(unit) if kind == "started" else controller.on_completed(unit)
            if not handled:
                raise BackendError("qualification.scheduler_playback_callback_unowned", "Scheduler callback was not owned", subsystem="qualification")
            receipts.extend(str(item.payload["state"]) for item in emitted)
            if kind == "started":
                started_at = observed_at
            else:
                completed_at = observed_at
        if started_at is None or completed_at is None or receipts != ["PLAYING", "COMPLETED"] or journal.state != "COMPLETED":
            raise BackendError("qualification.scheduler_playback_timeout", "Scheduler playback did not complete", subsystem="qualification")
        return {
            "ready": True,
            "code": "qualification.scheduler_playback_completed",
            "queue_to_started_ms": round((started_at - pumped_at) * 1000.0, 3),
            "started_to_completed_ms": round((completed_at - started_at) * 1000.0, 3),
            "receipts": receipts,
            "journal_state": journal.state,
            "process_isolated_playback": True,
            "raw_audio_persisted": False,
            "network_used": False,
            "installation_performed": False,
        }
    finally:
        tts.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--approval-manifest", type=Path, required=True)
    parser.add_argument("--approval-manifest-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_scheduler_playback_smoke(qualification_config(args))
    except BackendError as exc:
        report = {"ready": False, "code": exc.code, "network_used": False, "installation_performed": False}
        if exc.safe_context:
            report["context"] = dict(exc.safe_context)
    write_report(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
