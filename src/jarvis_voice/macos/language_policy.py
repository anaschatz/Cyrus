"""Conservative bilingual language stabilization for a macOS voice session."""

from __future__ import annotations

import re

from ..backends import LanguageMetadata, TranscriptResult


_GREEK_LETTER = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")
_ENGLISH_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")


class MacSessionLanguagePolicy:
    """Keep short or uncertain turns from flipping the response voice."""

    def __init__(
        self,
        *,
        default_language: str = "el",
        english_switch_score: float = 0.90,
        english_minimum_words: int = 2,
        english_minimum_letters: int = 8,
    ) -> None:
        self.current_language = default_language
        self.english_switch_score = english_switch_score
        self.english_minimum_words = english_minimum_words
        self.english_minimum_letters = english_minimum_letters

    @staticmethod
    def _with_language(
        result: TranscriptResult,
        code: str,
        score: float,
    ) -> TranscriptResult:
        return TranscriptResult(
            text=result.text,
            language=LanguageMetadata(code, score),
            asr_score=result.asr_score,
        )

    def stabilize(self, result: TranscriptResult) -> TranscriptResult:
        detected = result.language.code.lower()
        score = result.language.score

        # Greek script is unambiguous enough to restore the Greek session even
        # when language detection on a short utterance is unstable.
        if _GREEK_LETTER.search(result.text):
            self.current_language = "el"
            return self._with_language(
                result,
                "el",
                score if detected == "el" else 0.0,
            )

        words = _ENGLISH_WORD.findall(result.text)
        english_letters = sum(len(word.replace("'", "").replace("’", "")) for word in words)
        clear_english = (
            detected == "en"
            and score >= self.english_switch_score
            and len(words) >= self.english_minimum_words
            and english_letters >= self.english_minimum_letters
        )
        if clear_english:
            self.current_language = "en"
            return result

        if detected == self.current_language:
            return result
        # A zero language score makes it explicit that this is the session
        # default, not a fabricated ASR confidence value.
        return self._with_language(result, self.current_language, 0.0)
