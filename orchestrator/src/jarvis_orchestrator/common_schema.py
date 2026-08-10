"""Frozen JSON Schema transformer for the local inference CommonSchema."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

try:  # The dependency is intentionally optional for dependency-free core tests.
    from pydantic_ai import JsonSchemaTransformer as _PydanticTransformer
except ImportError:  # pragma: no cover - exercised when the pinned extra is installed
    class _PydanticTransformer:  # type: ignore[no-redef]
        def __init__(self, schema: dict[str, Any], **_: Any) -> None:
            self.schema = schema

        def transform(self, schema: dict[str, Any] | None = None) -> dict[str, Any]:
            return schema or self.schema


ALLOWED = frozenset(
    {
        "$schema",
        "$defs",
        "$ref",
        "title",
        "description",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "enum",
        "const",
        "default",
        "format",
    }
)
FORBIDDEN = frozenset({"anyOf", "oneOf", "allOf", "not", "if", "then", "else"})


class CommonSchemaError(ValueError):
    pass


def transform_common_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    source = deepcopy(dict(schema))
    definitions = source.get("$defs", {})
    if not isinstance(definitions, dict):
        raise CommonSchemaError("$defs must be an object")

    def walk(node: Any, stack: tuple[str, ...] = ()) -> dict[str, Any]:
        if not isinstance(node, dict):
            raise CommonSchemaError("every schema node must be an object")
        if set(node) & FORBIDDEN:
            raise CommonSchemaError("schema composition is unsupported")
        unknown = set(node) - ALLOWED
        if unknown:
            raise CommonSchemaError(f"unsupported keyword: {sorted(unknown)[0]}")
        if "$ref" in node:
            if set(node) != {"$ref"}:
                raise CommonSchemaError("references cannot have siblings")
            prefix = "#/$defs/"
            reference = node["$ref"]
            if not isinstance(reference, str) or not reference.startswith(prefix):
                raise CommonSchemaError("only local direct references are supported")
            name = reference[len(prefix) :]
            if name in stack or name not in definitions:
                raise CommonSchemaError("recursive or missing reference")
            return walk(definitions[name], stack + (name,))
        schema_type = node.get("type")
        if isinstance(schema_type, list) or schema_type is None:
            raise CommonSchemaError("nullable unions and missing scalar types are unsupported")
        if schema_type not in {"object", "array", "string", "number", "integer", "boolean", "null"}:
            raise CommonSchemaError("unsupported schema type")
        result: dict[str, Any] = {}
        if "$schema" in node:
            result["$schema"] = node["$schema"]
        for key in ("title", "description"):
            if key in node:
                result[key] = node[key]
        result["type"] = schema_type
        if schema_type == "object":
            properties = node.get("properties")
            if not isinstance(properties, dict):
                raise CommonSchemaError("object properties are required")
            result["properties"] = {name: walk(child, stack) for name, child in properties.items()}
            result["required"] = list(properties)
            result["additionalProperties"] = False
        elif schema_type == "array":
            if "items" not in node:
                raise CommonSchemaError("array items are required")
            result["items"] = walk(node["items"], stack)
            for key in ("minItems", "maxItems"):
                if key in node:
                    result[key] = node[key]
        elif schema_type == "string":
            for key in ("minLength", "maxLength"):
                if key in node:
                    result[key] = node[key]
        elif schema_type in {"number", "integer"}:
            for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                if key in node:
                    result[key] = node[key]
        for key in ("enum", "const"):
            if key in node:
                result[key] = deepcopy(node[key])
        return result

    transformed = walk(source)
    if transformed.get("type") != "object":
        raise CommonSchemaError("root schema must be object")
    return transformed


class JarvisCommonSchemaTransformer(_PydanticTransformer):
    """Pydantic-compatible transformer class, passed to model profiles as a class."""

    def walk(self) -> dict[str, Any]:
        selected = getattr(self, "schema", None)
        if not isinstance(selected, dict):
            raise CommonSchemaError("schema is required")
        return transform_common_schema(selected)

    def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
        # ``walk`` is deliberately authoritative; Pydantic's recursive walker
        # can otherwise simplify unions before JARVIS gets a chance to reject
        # them. This method satisfies the public abstract transformer contract.
        return deepcopy(schema)


def route_decision_schema(tool_names: tuple[str, ...] = ()) -> dict[str, Any]:
    return transform_common_schema(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "route": {"type": "string", "enum": ["local", "memory_local", "research_local", "cloud", "tool_local"]},
                "memory_query": {"type": "string", "maxLength": 4096},
                "research_query": {"type": "string", "maxLength": 4096},
                "tool_name": {"type": "string", "enum": ["none", *tool_names]},
                "response_language": {"type": "string", "enum": ["el", "en", "default"]},
            },
            "required": ["route", "memory_query", "research_query", "tool_name", "response_language"],
            "additionalProperties": False,
        }
    )
