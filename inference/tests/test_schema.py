from __future__ import annotations

import unittest

from jarvis_inference.errors import ApplianceError
from jarvis_inference.schema_subset import normalize_schema, validate_instance


class SchemaTests(unittest.TestCase):
    def test_local_ref_is_inlined_and_validated(self) -> None:
        schema = normalize_schema({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {"city": {"type": "string", "minLength": 1}},
            "type": "object",
            "properties": {"city": {"$ref": "#/$defs/city"}},
            "required": ["city"],
            "additionalProperties": False,
        })
        self.assertNotIn("$ref", schema["properties"]["city"])
        validate_instance({"city": "Athens"}, schema)
        with self.assertRaises(ApplianceError):
            validate_instance({"city": ""}, schema)

    def test_external_ref_regex_and_open_object_rejected(self) -> None:
        cases = [
            {"type": "object", "properties": {"x": {"$ref": "https://example/x"}}, "required": ["x"], "additionalProperties": False},
            {"type": "object", "properties": {"x": {"type": "string", "pattern": ".*"}}, "required": ["x"], "additionalProperties": False},
            {"type": "object", "properties": {}, "required": [], "additionalProperties": True},
        ]
        for schema in cases:
            with self.subTest(schema=schema), self.assertRaises(ApplianceError):
                normalize_schema(schema)

    def test_unreferenced_defs_and_inapplicable_keywords_are_not_ignored(self) -> None:
        cases = [
            {
                "$defs": {"hidden": {"type": "string", "pattern": ".*"}},
                "type": "object", "properties": {}, "required": [], "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"x": {"type": "string", "properties": {}}},
                "required": ["x"], "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"x": {"type": "string", "enum": [1]}},
                "required": ["x"], "additionalProperties": False,
            },
        ]
        for schema in cases:
            with self.subTest(schema=schema), self.assertRaises(ApplianceError):
                normalize_schema(schema)
