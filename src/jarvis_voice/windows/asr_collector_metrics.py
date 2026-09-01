"""Content-free metric helpers for the native Windows ASR collector."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..errors import BackendError
from .asr_collector_contract import CollectorFailure


MAX_REGRESSION_FRACTION = 0.15


class ForcedCudaFailure:
    """Collector-local trigger for the production CPU fallback route.

    This is an injected qualification failure.  It deliberately does not
    simulate or claim actual VRAM exhaustion.
    """

    residency = "turn_scoped"

    def transcribe(self, _audio: object) -> Any:
        raise BackendError(
            "cuda.insufficient_free_vram",
            "Injected qualification fallback trigger",
            subsystem="asr",
        )

    def cancel_current(self) -> None:
        return None

    def close(self) -> None:
        return None


def word_errors(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Compute deterministic Levenshtein word errors with bounded memory."""

    if not reference or len(reference) > 128 or len(hypothesis) > 1_024:
        raise CollectorFailure("wer_input_invalid")
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_word in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + int(reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1]


def within_regression(baseline: float, coexistence: float) -> bool:
    return coexistence <= baseline * (1.0 + MAX_REGRESSION_FRACTION)


__all__ = ["ForcedCudaFailure", "within_regression", "word_errors"]
