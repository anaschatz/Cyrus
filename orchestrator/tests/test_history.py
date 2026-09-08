from __future__ import annotations

import unittest

from jarvis_orchestrator.history import canonical_messages_to_pydantic


class CanonicalHistoryTests(unittest.TestCase):
    def test_completed_pairs_preserve_text_and_separate_current_prompt(self) -> None:
        history, prompt = canonical_messages_to_pydantic((
            {"role": "user", "content": "Προηγούμενη ερώτηση"},
            {"role": "assistant", "content": "Προηγούμενη απάντηση"},
            {"role": "user", "content": "Νέα ερώτηση"},
        ))
        self.assertEqual([message.parts[0].content for message in history],
                         ["Προηγούμενη ερώτηση", "Προηγούμενη απάντηση"])
        self.assertEqual(prompt, "Νέα ερώτηση")

    def test_current_prompt_requires_text_like_the_history(self) -> None:
        for final in ({"role": "user"}, *({"role": "user", "content": value}
                                         for value in (None, True, 1, [], {}))):
            with self.subTest(final=final):
                with self.assertRaisesRegex(ValueError, "content must be text"):
                    canonical_messages_to_pydantic((final,))


if __name__ == "__main__":
    unittest.main()
