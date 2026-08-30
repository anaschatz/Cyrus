from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from jarvis_orchestrator.domain import OrchestratorError
from jarvis_orchestrator.llm_context import (
    validate_buffered_model_text,
    validate_stream_fragment,
)


class LlmOutputValidationTests(unittest.TestCase):
    def test_stream_rejects_complete_reasoning_tags_before_publication(self) -> None:
        with self.assertRaisesRegex(
            OrchestratorError,
            "local_structural_reasoning_leak",
        ):
            validate_stream_fragment("<think>private")

    def test_stream_rejects_trailing_tag_prefix_before_next_chunk(self) -> None:
        for fragment in ("<", "answer <t", "answer <thi", "answer </think"):
            with self.subTest(fragment=fragment), self.assertRaisesRegex(
                OrchestratorError,
                "local_structural_reasoning_leak",
            ):
                validate_stream_fragment(fragment)

    def test_buffered_output_rejects_unfinished_reasoning_marker(self) -> None:
        with self.assertRaisesRegex(
            OrchestratorError,
            "cloud_structural_reasoning_leak",
        ):
            validate_buffered_model_text("A final answer followed by <think")

    def test_plain_final_answer_remains_valid(self) -> None:
        self.assertEqual(
            validate_buffered_model_text("A concise grounded answer."),
            "A concise grounded answer.",
        )
        validate_stream_fragment("A concise grounded answer.")


if __name__ == "__main__":
    unittest.main()
