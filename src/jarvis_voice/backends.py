"""Backend boundaries for the pure core and platform adapters."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .protocol import Envelope


@dataclass(frozen=True, slots=True)
class LanguageMetadata:
    code: str
    score: float


@dataclass(frozen=True, slots=True)
class AsrScore:
    value: float
    raw_avg_logprob: float
    method: str = "exp_weighted_avg_logprob"
    calibrated: bool = False


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    text: str
    language: LanguageMetadata
    asr_score: AsrScore

    def to_payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": {"code": self.language.code, "score": self.language.score},
            "asr_score": {
                "value": self.asr_score.value,
                "method": self.asr_score.method,
                "raw_avg_logprob": self.asr_score.raw_avg_logprob,
                "calibrated": self.asr_score.calibrated,
            },
        }


@dataclass(frozen=True, slots=True)
class SpeechUnit:
    response_id: str
    epoch: int
    unit_index: int
    locale: str
    text: str


@dataclass(frozen=True, slots=True)
class AudioChunk:
    samples: array
    sample_rate: int


@dataclass(frozen=True, slots=True)
class VadResult:
    speech_detected: bool
    audio: AudioChunk
    speech_start_sample: int | None = None
    speech_end_sample: int | None = None


@runtime_checkable
class AsrBackend(Protocol):
    def transcribe(self, audio: object) -> TranscriptResult: ...


@runtime_checkable
class TtsBackend(Protocol):
    def enqueue(self, unit: SpeechUnit) -> None: ...
    def cancel(self, new_epoch: int, reason: str) -> None: ...
    @property
    def is_idle(self) -> bool: ...


@runtime_checkable
class OrchestratorTransport(Protocol):
    def send(self, event: Envelope) -> None: ...
    def receive(self) -> Envelope | None: ...


@runtime_checkable
class PttSource(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...


@runtime_checkable
class AudioCapture(Protocol):
    def begin_turn(self) -> None: ...
    def end_turn(self) -> AudioChunk: ...
    def abort_turn(self) -> None: ...


@runtime_checkable
class VadBackend(Protocol):
    def trim(self, audio: AudioChunk) -> VadResult: ...

