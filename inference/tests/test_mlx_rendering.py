from __future__ import annotations

import unittest

from jarvis_inference.contracts import Message, NormalizedRequest, ResponseFormat, ToolCall, ToolDefinition
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

    def test_json_schema_is_rendered_deterministically(self) -> None:
        request = self.request()
        structured = NormalizedRequest(
            request.model,
            request.messages,
            request.max_completion_tokens,
            False,
            False,
            request.temperature,
            request.top_p,
            request.stop,
            request.tools,
            request.tool_choice,
            ResponseFormat(
                "json_schema",
                "result",
                None,
                {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            ),
        )
        self.assertEqual(
            render_mlx_qwen3_prompt(structured),
            "<|im_start|>system\nsystem text\n\n"
            "Output only one JSON object conforming exactly to this schema. Do not add Markdown or prose.\n"
            '{"additionalProperties":false,"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"}'
            "<|im_end|>\n<|im_start|>user\nuser text<|im_end|>\n<|im_start|>assistant\n",
        )

    def test_tool_definition_and_history_match_qwen_boundary(self) -> None:
        request = self.request()
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        }
        tool = ToolDefinition("get_weather", "Current weather", schema)
        call = ToolCall("call_1", "get_weather", '{"city":"Athens"}', {"city": "Athens"})
        messages = request.messages + (
            Message("assistant", None, (call,)),
            Message("tool", "sunny", tool_call_id="call_1"),
            Message("user", "Summarize it"),
        )
        tools_request = NormalizedRequest(
            request.model,
            messages,
            request.max_completion_tokens,
            False,
            False,
            request.temperature,
            request.top_p,
            request.stop,
            (tool,),
            ("named", "get_weather"),
            request.response_format,
        )
        rendered = render_mlx_qwen3_prompt(tools_request)
        self.assertIn("# Tools", rendered)
        self.assertIn('Tool policy: get_weather.', rendered)
        self.assertIn(
            '<tool_call>\n{"arguments":{"city":"Athens"},"name":"get_weather"}\n</tool_call>',
            rendered,
        )
        self.assertIn("<tool_response>\nsunny\n</tool_response>", rendered)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n"))


if __name__ == "__main__":
    unittest.main()
