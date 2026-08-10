from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .errors import invalid
from .limits import (
    DESCRIPTION_BYTES,
    ENUM_BYTES,
    ENUM_VALUES,
    OBJECT_PROPERTIES,
    SCHEMA_BYTES,
    SCHEMA_DEFS,
    SCHEMA_DEPTH,
    SCHEMA_NODES,
    SCHEMA_PROPERTIES,
)
from .strict_json import canonical_bytes

DIALECT = "https://json-schema.org/draft/2020-12/schema"
ALLOWED = {
    "$schema", "$defs", "$ref", "title", "description", "type",
    "properties", "required", "additionalProperties", "items",
    "minItems", "maxItems", "minLength", "maxLength", "minimum",
    "maximum", "exclusiveMinimum", "exclusiveMaximum", "enum", "const",
}
SCALAR_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


@dataclass(slots=True)
class _Budget:
    nodes: int = 0
    properties: int = 0
    enum_values: int = 0
    enum_bytes: int = 0


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def normalize_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise invalid("invalid_schema", "A schema must be a JSON object.", "schema")
    if len(canonical_bytes(schema)) > SCHEMA_BYTES:
        raise invalid("schema_too_large", "The schema exceeds the size limit.", "schema")
    defs = schema.get("$defs", {})
    if not isinstance(defs, dict) or len(defs) > SCHEMA_DEFS:
        raise invalid("invalid_schema", "The schema definitions are invalid.", "schema")
    if any(not isinstance(name, str) or not name or "/" in name or "~" in name for name in defs):
        raise invalid("invalid_schema", "Schema definition names are invalid.", "schema")
    budget = _Budget()
    visited: set[str] = set()
    normalized = _walk(schema, defs, budget, 1, (), visited, root=True)
    # Definitions cannot be used as a hiding place for unsupported keywords or
    # external/recursive references.  Validate even definitions not reached by
    # the root schema.
    for name in defs:
        if name not in visited:
            visited.add(name)
            _walk(defs[name], defs, budget, 1, (name,), visited)
    if normalized.get("type") != "object":
        raise invalid("invalid_schema", "The root schema type must be object.", "schema")
    return normalized


def _walk(
    node: Any,
    defs: dict[str, Any],
    budget: _Budget,
    depth: int,
    refs: tuple[str, ...],
    visited: set[str],
    *,
    root: bool = False,
) -> dict[str, Any]:
    if not isinstance(node, dict) or depth > SCHEMA_DEPTH:
        raise invalid("invalid_schema", "The schema shape or depth is invalid.", "schema")
    budget.nodes += 1
    if budget.nodes > SCHEMA_NODES:
        raise invalid("invalid_schema", "The schema exceeds the node limit.", "schema")
    unknown = set(node) - ALLOWED
    if unknown:
        raise invalid("unsupported_parameter", "The schema contains an unsupported keyword.", "schema")
    if "$ref" in node:
        if set(node) != {"$ref"}:
            raise invalid("invalid_schema", "A local reference cannot have sibling keywords.", "schema")
        ref = node["$ref"]
        prefix = "#/$defs/"
        if not isinstance(ref, str) or not ref.startswith(prefix) or "/" in ref[len(prefix):]:
            raise invalid("invalid_schema", "Only direct local $defs references are supported.", "schema")
        name = ref[len(prefix):]
        if name not in defs or name in refs:
            raise invalid("invalid_schema", "The local schema reference is missing or recursive.", "schema")
        visited.add(name)
        return _walk(defs[name], defs, budget, depth, refs + (name,), visited)
    if "$schema" in node and (not root or node["$schema"] != DIALECT):
        raise invalid("invalid_schema", "The schema dialect is not supported.", "schema")
    if "$defs" in node and not root:
        raise invalid("invalid_schema", "$defs is allowed only at the schema root.", "schema")
    result: dict[str, Any] = {}
    for annotation in ("title", "description"):
        if annotation in node:
            value = node[annotation]
            if not isinstance(value, str) or len(value.encode()) > DESCRIPTION_BYTES:
                raise invalid("invalid_schema", "A schema annotation is invalid.", "schema")
            result[annotation] = value
    schema_type = node.get("type")
    if not isinstance(schema_type, str) or schema_type not in SCALAR_TYPES:
        raise invalid("invalid_schema", "Every schema node requires one scalar type.", "schema")
    common = {"type", "title", "description", "enum", "const"}
    by_type = {
        "object": {"properties", "required", "additionalProperties"},
        "array": {"items", "minItems", "maxItems"},
        "string": {"minLength", "maxLength"},
        "number": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"},
        "integer": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"},
        "boolean": set(),
        "null": set(),
    }
    meta = ({"$schema", "$defs"} if root else set())
    if set(node) - common - by_type[schema_type] - meta:
        raise invalid("invalid_schema", "A schema keyword is not valid for its declared type.", "schema")
    result["type"] = schema_type
    if schema_type == "object":
        properties = node.get("properties")
        required = node.get("required")
        if not isinstance(properties, dict) or len(properties) > OBJECT_PROPERTIES:
            raise invalid("invalid_schema", "Object properties are invalid.", "schema")
        if node.get("additionalProperties") is not False:
            raise invalid("invalid_schema", "Every object must set additionalProperties to false.", "schema")
        if not isinstance(required, list) or any(not isinstance(x, str) for x in required):
            raise invalid("invalid_schema", "Object required must be an array of property names.", "schema")
        if len(set(required)) != len(required) or not set(required).issubset(properties):
            raise invalid("invalid_schema", "Object required entries are invalid.", "schema")
        budget.properties += len(properties)
        if budget.properties > SCHEMA_PROPERTIES:
            raise invalid("invalid_schema", "The schema exceeds the property limit.", "schema")
        if any(not isinstance(key, str) for key in properties):
            raise invalid("invalid_schema", "Object property names are invalid.", "schema")
        result["properties"] = {k: _walk(v, defs, budget, depth + 1, refs, visited) for k, v in properties.items()}
        result["required"] = list(required)
        result["additionalProperties"] = False
    elif schema_type == "array":
        if "items" not in node:
            raise invalid("invalid_schema", "Array schemas require items.", "schema")
        result["items"] = _walk(node["items"], defs, budget, depth + 1, refs, visited)
        for key in ("minItems", "maxItems"):
            if key in node:
                if not _integer(node[key]) or node[key] < 0:
                    raise invalid("invalid_schema", "Array bounds must be non-negative integers.", "schema")
                result[key] = node[key]
        if result.get("minItems", 0) > result.get("maxItems", 2**31):
            raise invalid("invalid_schema", "Array bounds are inconsistent.", "schema")
    elif schema_type == "string":
        for key in ("minLength", "maxLength"):
            if key in node:
                if not _integer(node[key]) or node[key] < 0:
                    raise invalid("invalid_schema", "String bounds must be non-negative integers.", "schema")
                result[key] = node[key]
        if result.get("minLength", 0) > result.get("maxLength", 2**31):
            raise invalid("invalid_schema", "String bounds are inconsistent.", "schema")
    elif schema_type in {"number", "integer"}:
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if key in node:
                value = node[key]
                if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
                    raise invalid("invalid_schema", "Numeric bounds are invalid.", "schema")
                result[key] = value
        lower = result.get("minimum", result.get("exclusiveMinimum"))
        upper = result.get("maximum", result.get("exclusiveMaximum"))
        if lower is not None and upper is not None and lower > upper:
            raise invalid("invalid_schema", "Numeric bounds are inconsistent.", "schema")
    for keyword in ("enum", "const"):
        if keyword in node:
            if keyword == "enum":
                values = node[keyword]
                if not isinstance(values, list) or not values or len(values) > ENUM_VALUES:
                    raise invalid("invalid_schema", "Schema enum is invalid.", "schema")
                if len({canonical_bytes(value) for value in values}) != len(values):
                    raise invalid("invalid_schema", "Schema enum values must be unique.", "schema")
                budget.enum_values += len(values)
                budget.enum_bytes += len(canonical_bytes(values))
                if budget.enum_values > ENUM_VALUES or budget.enum_bytes > ENUM_BYTES:
                    raise invalid("invalid_schema", "Schema enum exceeds its limits.", "schema")
            result[keyword] = deepcopy(node[keyword])
    try:
        if "enum" in result:
            for value in result["enum"]:
                validate_instance(value, result)
        if "const" in result:
            validate_instance(result["const"], result)
    except Exception as exc:
        raise invalid("invalid_schema", "Schema enum or const values do not match the declared type.", "schema") from exc
    return result


def validate_instance(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    kind = schema["type"]
    valid = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float, Decimal)) and not isinstance(value, bool),
        "integer": (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, Decimal) and value == value.to_integral()),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }[kind]
    if not valid:
        raise invalid("schema_validation_failed", "A JSON value does not match the declared schema.", path)
    if "enum" in schema and value not in schema["enum"]:
        raise invalid("schema_validation_failed", "A JSON value is outside the declared enum.", path)
    if "const" in schema and value != schema["const"]:
        raise invalid("schema_validation_failed", "A JSON value does not match the declared constant.", path)
    if kind == "object":
        props = schema["properties"]
        if set(value) - set(props) or not set(schema["required"]).issubset(value):
            raise invalid("schema_validation_failed", "A JSON object has missing or unknown properties.", path)
        for key, child in value.items():
            validate_instance(child, props[key], path + "." + key)
    elif kind == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 2**31):
            raise invalid("schema_validation_failed", "A JSON array violates its size bounds.", path)
        for index, child in enumerate(value):
            validate_instance(child, schema["items"], f"{path}[{index}]")
    elif kind == "string":
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", 2**31):
            raise invalid("schema_validation_failed", "A JSON string violates its length bounds.", path)
    elif kind in {"number", "integer"}:
        if "minimum" in schema and value < schema["minimum"]:
            raise invalid("schema_validation_failed", "A JSON number violates its lower bound.", path)
        if "maximum" in schema and value > schema["maximum"]:
            raise invalid("schema_validation_failed", "A JSON number violates its upper bound.", path)
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise invalid("schema_validation_failed", "A JSON number violates its exclusive lower bound.", path)
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            raise invalid("schema_validation_failed", "A JSON number violates its exclusive upper bound.", path)
