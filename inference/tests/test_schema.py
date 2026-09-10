from __future__ import annotations

import unittest

from jarvis_inference.errors import ApplianceError
from jarvis_inference.schema_subset import normalize_schema, validate_instance


class SchemaTests(unittest.TestCase):
    def test_nullable_scalar_keeps_type_enum_bounds_and_closed_union(self):
        def document(node):
            return {"type": "object", "properties": {"x": node}, "required": ["x"], "additionalProperties": False}

        schema = normalize_schema(document({"anyOf": [
            {"type": "integer", "minimum": 1, "enum": [1, 2]}, {"type": "null"},
        ]}))
        for value in (None, 1, 2):
            validate_instance({"x": value}, schema)
        for value in (True, "1", 0, 3, []):
            with self.subTest(value=value), self.assertRaises(ApplianceError):
                validate_instance({"x": value}, schema)
        for node in (
            {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            {"anyOf": [{"type": "null"}, {"type": "string"}]},
            {"anyOf": [{"type": "string", "pattern": ".*"}, {"type": "null"}]},
            {"anyOf": [{"type": "string"}, {"type": "null"}], "const": 1},
            {"anyOf": [document({"type": "string"}), {"type": "null"}]},
        ):
            with self.subTest(node=node), self.assertRaises(ApplianceError):
                normalize_schema(document(node))

    def test_total_enum_budget_remains_bounded_separately_from_each_enum(self):
        for count, accepted in ((2, True), (3, False)):
            properties = {f"x{i}": {"type": "integer", "enum": list(range(64))} for i in range(count)}
            document = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
            if accepted:
                normalize_schema(document)
            else:
                with self.assertRaises(ApplianceError):
                    normalize_schema(document)
        with self.assertRaises(ApplianceError):
            normalize_schema({"type": "object", "properties": {"x": {"type": "integer", "enum": list(range(65))}},
                              "required": ["x"], "additionalProperties": False})

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
