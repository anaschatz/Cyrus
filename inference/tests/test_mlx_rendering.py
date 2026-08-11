from __future__ import annotations

import unittest

from jarvis_inference.contracts import Message, NormalizedRequest, ResponseFormat
from jarvis_inference.rendering import render_mlx_qwen3_prompt


class MlxQwen3RenderingTests(unittest.TestCase):
    def request(self) -> NormalizedRequest:
        return NormalizedRequest(
            model="jarvis-local",
            messages=(
                Message("system", "system text"),
                Message("user", "user text"),
            ),
            max_completion_tokens=64,
            stream=True,
            include_usage=True,
            temperature=0,
            top_p=1,
            stop=(),
            tools=(),
            tool_choice="none",
            response_format=ResponseFormat("text"),
        )

    def test_exact_non_thinking_chatml_boundary(self) -> None:
        self.assertEqual(
            render_mlx_qwen3_prompt(self.request()),
            "<|im_start|>system\nsystem text<|im_end|>\n"
            "<|im_start|>user\nuser text<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

    def test_non_text_profile_fails_closed(self) -> None:
        request = self.request()
        invalid = NormalizedRequest(
            request.model,
            request.messages,
            request.max_completion_tokens,
            request.stream,
            request.include_usage,
            request.temperature,
            request.top_p,
            request.stop,
            request.tools,
            request.tool_choice,
            ResponseFormat("json_object"),
        )
        with self.assertRaises(ValueError):
            render_mlx_qwen3_prompt(invalid)


if __name__ == "__main__":
    unittest.main()
