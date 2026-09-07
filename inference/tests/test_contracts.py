from __future__ import annotations

import unittest

from jarvis_inference.contracts import Capabilities, normalize_request, validate_generated_tool, validate_structured_content
from jarvis_inference.errors import ApplianceError
from jarvis_inference.strict_json import parse_body


OBJECT_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


class StrictJsonTests(unittest.TestCase):
    def test_duplicate_keys_rejected_at_nested_depth(self) -> None:
        with self.assertRaises(ApplianceError) as caught:
            parse_body(b'{"outer":{"model":"a","model":"b"}}')
        self.assertEqual(caught.exception.code, "duplicate_json_key")

    def test_nonfinite_and_bom_rejected(self) -> None:
        with self.assertRaises(ApplianceError):
            parse_body(b'{"x":NaN}')
        with self.assertRaises(ApplianceError) as caught:
            parse_body(b'\xef\xbb\xbf{}')
        self.assertEqual(caught.exception.code, "invalid_json_encoding")


class RequestContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.capabilities = Capabilities(True, True, True, True, True)

    def base(self) -> dict:
        return {"model": "jarvis-local", "messages": [{"role": "user", "content": "hello"}]}

    def test_exact_json_schema_shape(self) -> None:
        raw = self.base()
        raw["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "result", "schema": OBJECT_SCHEMA, "strict": True},
        }
        request = normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)
        self.assertEqual(request.response_format.type, "json_schema")
        with self.assertRaises(ApplianceError) as caught:
            validate_structured_content("[" * 2000 + "0" + "]" * 2000, request)
        self.assertEqual(caught.exception.status, 502)
        for strict in (False, None):
            raw["response_format"]["json_schema"]["strict"] = strict
            with self.assertRaises(ApplianceError):
                normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)
        del raw["response_format"]["json_schema"]["strict"]
        with self.assertRaises(ApplianceError):
            normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)

    def test_historical_arguments_accept_noncompact_json(self) -> None:
        raw = self.base()
        raw["tools"] = [{"type": "function", "function": {
            "name": "weather", "parameters": OBJECT_SCHEMA, "strict": True,
        }}]
        raw["messages"] = [
            {"role": "user", "content": "weather"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_0123456789abcdef0123456789abcdef", "type": "function",
                "function": {"name": "weather", "arguments": "{ \"city\": \"Athens\" }"},
            }]},
            {"role": "tool", "content": "sunny", "tool_call_id": "call_0123456789abcdef0123456789abcdef"},
        ]
        request = normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)
        self.assertEqual(request.messages[1].tool_calls[0].arguments, '{"city":"Athens"}')
        for depth in (100, 2000):
            raw["messages"][1]["tool_calls"][0]["function"]["arguments"] = "[" * depth + "0" + "]" * depth
            with self.subTest(depth=depth), self.assertRaises(ApplianceError) as caught:
                normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)
            self.assertEqual((caught.exception.status, caught.exception.code), (400, "invalid_tool_arguments"))

    def test_unknown_fields_and_developer_rejected(self) -> None:
        raw = self.base()
        raw["seed"] = 1
        with self.assertRaises(ApplianceError) as caught:
            normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)
        self.assertEqual(caught.exception.code, "unsupported_parameter")
        raw = self.base()
        raw["messages"] = [{"role": "developer", "content": "hidden"}]
        with self.assertRaises(ApplianceError) as caught:
            normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)
        self.assertEqual(caught.exception.code, "unsupported_parameter")

    def test_max_tokens_alias_is_exclusive(self) -> None:
        raw = self.base()
        raw.update({"max_tokens": 4, "max_completion_tokens": 4})
        with self.assertRaises(ApplianceError):
            normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)

    def test_nonstring_roles_are_rejected_as_client_errors(self) -> None:
        for role in ([], {}, None, True, 1):
            raw = self.base()
            raw["messages"][0]["role"] = role
            with self.subTest(role=role), self.assertRaises(ApplianceError) as caught:
                normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)
            self.assertEqual(caught.exception.status, 400)

    def test_supported_scalar_fields_and_stream_options(self) -> None:
        raw = self.base()
        raw.update({
            "max_tokens": 9, "stream": True, "stream_options": {"include_usage": True},
            "n": 1, "parallel_tool_calls": False, "temperature": 0,
            "top_p": 0.5, "stop": ["A", "Β"], "response_format": {"type": "text"},
        })
        request = normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)
        self.assertEqual(request.max_completion_tokens, 9)
        self.assertTrue(request.stream and request.include_usage)
        self.assertEqual((request.temperature, request.top_p, request.stop), (0.0, 0.5, ("A", "Β")))

    def test_boolean_n_and_nested_unknown_fields_are_rejected(self) -> None:
        for update in (
            {"n": True},
            {"parallel_tool_calls": True},
            {"stream_options": {"include_usage": True, "extra": 1}, "stream": True},
            {"response_format": {"type": "json_object", "extra": 1}},
            {"messages": [{"role": "user", "content": "x", "name": "y"}]},
        ):
            raw = self.base(); raw.update(update)
            with self.subTest(update=update), self.assertRaises(ApplianceError):
                normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)

    def test_exact_tool_definition_and_named_choice(self) -> None:
        raw = self.base()
        raw["tools"] = [{"type": "function", "function": {
            "name": "get_weather", "description": "weather", "parameters": OBJECT_SCHEMA, "strict": True,
        }}]
        raw["tool_choice"] = {"type": "function", "function": {"name": "get_weather"}}
        request = normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)
        self.assertEqual(request.tool_choice, ("named", "get_weather"))
        raw["tools"][0]["function"]["strict"] = False
        with self.assertRaises(ApplianceError):
            normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)

    def test_tool_history_must_be_unique_and_resolved(self) -> None:
        raw = self.base()
        raw["tools"] = [{"type": "function", "function": {
            "name": "weather", "parameters": OBJECT_SCHEMA, "strict": True,
        }}]
        call = {
            "id": "same", "type": "function",
            "function": {"name": "weather", "arguments": '{"city":"Athens"}'},
        }
        raw["messages"] = [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "content": "sunny", "tool_call_id": "same"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "content": "sunny", "tool_call_id": "same"},
        ]
        with self.assertRaises(ApplianceError):
            normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)
        raw["messages"] = raw["messages"][:1]
        with self.assertRaises(ApplianceError):
            normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)

    def test_model_type_and_content_parts_fail_strictly(self) -> None:
        raw = self.base(); raw["model"] = 1
        with self.assertRaises(ApplianceError) as caught:
            normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)
        self.assertEqual(caught.exception.code, "invalid_parameter")
        raw = self.base(); raw["messages"][0]["content"] = [{"type": "text", "text": "x"}]
        with self.assertRaises(ApplianceError):
            normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)

    def test_generated_numeric_tool_arguments_are_canonicalized(self) -> None:
        raw = self.base()
        raw["tools"] = [{"type": "function", "function": {
            "name": "set_value",
            "parameters": {"type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"], "additionalProperties": False},
            "strict": True,
        }}]
        raw["tool_choice"] = "required"
        request = normalize_request(raw, alias="jarvis-local", capabilities=self.capabilities, context_window=4096)
        self.assertEqual(validate_generated_tool("set_value", '{"value":1.50}', request), '{"value":1.5}')
        with self.assertRaises(ApplianceError) as caught:
            validate_generated_tool("set_value", "[" * 2000 + "0" + "]" * 2000, request)
        self.assertEqual(caught.exception.status, 502)
