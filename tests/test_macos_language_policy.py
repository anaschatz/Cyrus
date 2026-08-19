from __future__ import annotations

import unittest

from jarvis_voice.backends import AsrScore, LanguageMetadata, TranscriptResult
from jarvis_voice.macos.language_policy import MacSessionLanguagePolicy


def transcript(text: str, language: str, score: float) -> TranscriptResult:
    return TranscriptResult(
        text=text,
        language=LanguageMetadata(language, score),
        asr_score=AsrScore(0.8, -0.2),
    )


class MacSessionLanguagePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = MacSessionLanguagePolicy()

    def test_defaults_short_english_detection_to_greek(self) -> None:
        result = self.policy.stabilize(transcript("One", "en", 0.99))

        self.assertEqual(result.language, LanguageMetadata("el", 0.0))
        self.assertEqual(self.policy.current_language, "el")

    def test_defaults_uncertain_english_phrase_to_greek(self) -> None:
        result = self.policy.stabilize(transcript("Good morning", "en", 0.70))

        self.assertEqual(result.language, LanguageMetadata("el", 0.0))

    def test_switches_on_clear_high_confidence_english_phrase(self) -> None:
        result = self.policy.stabilize(transcript("Good morning Jarvis", "en", 0.96))

        self.assertEqual(result.language, LanguageMetadata("en", 0.96))
        self.assertEqual(self.policy.current_language, "en")

    def test_keeps_english_session_for_short_followup(self) -> None:
        self.policy.stabilize(transcript("Good morning Jarvis", "en", 0.96))

        result = self.policy.stabilize(transcript("Yes", "en", 0.61))

        self.assertEqual(result.language, LanguageMetadata("en", 0.61))

    def test_greek_script_restores_greek_session(self) -> None:
        self.policy.stabilize(transcript("Good morning Jarvis", "en", 0.96))

        result = self.policy.stabilize(transcript("Ναι", "en", 0.95))

        self.assertEqual(result.language, LanguageMetadata("el", 0.0))
        self.assertEqual(self.policy.current_language, "el")


if __name__ == "__main__":
    unittest.main()
