from __future__ import annotations

import unittest

from jarvis_inference.contracts import Capabilities, normalize_request
from jarvis_inference.gemma4 import (
    GemmaToolCallError,
    parse_gemma4_tool_call,
    render_gemma4_prompt,
)


CAPABILITIES = Capabilities(
    text=True,
    stream=True,
    tools=True,
    json_object=True,
    json_schema=True,
)
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
TOOL_DECLARATION = (
    '<|tool>declaration:get_weather{description:<|"|>Current weather<|"|>,'
    'parameters:{properties:{city:{type:<|"|>STRING<|"|>}},'
    'required:[<|"|>city<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>'
)
GENERATION_PRIMER = "<|turn>model\n<|channel>thought\n<channel|>"


def request(**updates):
    raw = {
        "model": "jarvis-local",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_completion_tokens": 64,
    }
    raw.update(updates)
    return normalize_request(
        raw,
        alias="jarvis-local",
        capabilities=CAPABILITIES,
        context_window=4096,
    )


class Gemma4RenderingTests(unittest.TestCase):
    def test_plain_chat_and_system_match_authoritative_template(self) -> None:
        self.assertEqual(
            render_gemma4_prompt(request(messages=[{"role": "user", "content": " Hello "}])),
            "<bos><|turn>user\nHello<turn|>\n" + GENERATION_PRIMER,
        )
        self.assertEqual(
            render_gemma4_prompt(request(messages=[
                {"role": "system", "content": " You are concise. "},
                {"role": "user", "content": " Hello "},
            ])),
            "<bos><|turn>system\nYou are concise.<turn|>\n"
            "<|turn>user\nHello<turn|>\n" + GENERATION_PRIMER,
        )

    def test_json_schema_policy_is_deterministic(self) -> None:
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        }
        rendered = render_gemma4_prompt(request(
            stream=False,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema, "strict": True},
            },
        ))
        self.assertEqual(
            rendered,
            "<bos><|turn>system\n"
            "Return exactly one JSON object that validates against this JSON Schema. "
            "Do not use Markdown or add prose.\n"
            '{"additionalProperties":false,"properties":{"city":{"type":"string"}},'
            '"required":["city"],"type":"object"}'
            "<turn|>\n<|turn>user\nHello<turn|>\n" + GENERATION_PRIMER,
        )

    def test_auto_named_required_and_none_tool_choices_are_explicit(self) -> None:
        auto = render_gemma4_prompt(request(tools=[TOOL], tool_choice="auto"))
        self.assertEqual(
            auto,
            "<bos><|turn>system\n" + TOOL_DECLARATION + "<turn|>\n"
            "<|turn>user\nHello<turn|>\n" + GENERATION_PRIMER,
        )

        named = render_gemma4_prompt(request(
            tools=[TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        ))
        self.assertEqual(
            named,
            '<bos><|turn>system\nCall only the declared function "get_weather" in this turn.'
            + TOOL_DECLARATION + "<turn|>\n<|turn>user\nHello<turn|>\n"
            + GENERATION_PRIMER,
        )

        required = render_gemma4_prompt(request(tools=[TOOL], tool_choice="required"))
        self.assertIn("Call exactly one declared function in this turn." + TOOL_DECLARATION, required)

        none = render_gemma4_prompt(request(tools=[TOOL], tool_choice="none"))
        self.assertEqual(
            none,
            "<bos><|turn>system\nDo not call a function in this turn; answer directly.<turn|>\n"
            "<|turn>user\nHello<turn|>\n" + GENERATION_PRIMER,
        )
        self.assertNotIn("<|tool>", none)

    def test_assistant_tool_history_and_response_match_authoritative_template(self) -> None:
        messages = [
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Athens"}'},
                }],
            },
            {"role": "tool", "content": "sunny", "tool_call_id": "call_1"},
        ]
        rendered = render_gemma4_prompt(request(messages=messages, tools=[TOOL]))
        self.assertEqual(
            rendered,
            "<bos><|turn>system\n" + TOOL_DECLARATION + "<turn|>\n"
            "<|turn>user\nweather<turn|>\n<|turn>model\n"
            '<|tool_call>call:get_weather{city:<|"|>Athens<|"|>}<tool_call|>'
            '<|tool_response>response:get_weather{value:<|"|>sunny<|"|>}<tool_response|>',
        )

        continued = render_gemma4_prompt(request(
            messages=messages + [{"role": "user", "content": "summarize"}],
            tools=[TOOL],
        ))
        self.assertTrue(continued.endswith(
            "<tool_response|><turn|>\n<|turn>user\nsummarize<turn|>\n" + GENERATION_PRIMER
        ))


class Gemma4ToolParserTests(unittest.TestCase):
    def test_parses_one_native_call_and_canonicalizes_arguments(self) -> None:
        name, arguments = parse_gemma4_tool_call(
            '<|tool_call>call:get_weather{meta:{lang:<|"|>el<|"|>},city:<|"|>Athens<|"|>,'
            'days:2,flags:[true,null]}<tool_call|>'
        )
        self.assertEqual(name, "get_weather")
        self.assertEqual(
            arguments,
            '{"city":"Athens","days":2,"flags":[true,null],"meta":{"lang":"el"}}',
        )

    def test_rejects_malformed_extra_or_ambiguous_output(self) -> None:
        malformed = (
            "plain text",
            'prose <|tool_call>call:get_weather{city:<|"|>Athens<|"|>}<tool_call|>',
            '<|tool_call>call:get_weather{city:<|"|>Athens<|"|>}<tool_call|> trailing',
            '<|tool_call>call:get_weather{city:<|"|>Athens<|"|>,city:<|"|>Rome<|"|>}<tool_call|>',
            '<|tool_call>call:get_weather{city:<|"|>Athens}<tool_call|>',
            '<|tool_call>call:get_weather{city:<|"|>Athens<|"|>,}<tool_call|>',
            '<|tool_call>call:get_weather{}<tool_call|><|tool_call>call:get_weather{}<tool_call|>',
        )
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(GemmaToolCallError):
                parse_gemma4_tool_call(value)


if __name__ == "__main__":
    unittest.main()
