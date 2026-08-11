from __future__ import annotations

import unittest

from jarvis_inference.mlx_worker_runtime import structured_logits_processor


class FakeOutlines:
    def __init__(self) -> None:
        self.received = None
        self.output_type = type(
            "OutputType",
            (),
            {"schema": "canonical schema", "whitespace_pattern": "spaces"},
        )()
        self.wrapped_model = object()

    def json_schema(self, schema):
        self.received = schema
        return self.output_type

    def from_mlxlm(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        return self.wrapped_model


class MlxWorkerRuntimeTests(unittest.TestCase):
    def test_schema_dict_builds_processor_without_retokenizing_input(self) -> None:
        outlines = FakeOutlines()
        processor = object()
        calls = []

        def factory(*args):
            calls.append(args)
            return processor

        model = object()
        tokenizer = object()
        schema = {
            "type": "object",
            "properties": {"operation": {"type": "string"}},
            "required": ["operation"],
            "additionalProperties": False,
        }

        result = structured_logits_processor(
            outlines,
            factory,
            model,
            tokenizer,
            schema,
        )

        self.assertIs(result, processor)
        self.assertIs(outlines.received, schema)
        self.assertIs(outlines.model, model)
        self.assertIs(outlines.tokenizer, tokenizer)
        self.assertEqual(
            calls,
            [(None, outlines.wrapped_model, "canonical schema", "spaces")],
        )


if __name__ == "__main__":
    unittest.main()
