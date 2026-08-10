"""Stable component pins included in every Anamnesis system fingerprint."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from pydantic import BaseModel

from anamnesis.memory import (
    CompilerStateView,
    ExecutionRecord,
    FactRevision,
    IntentRevision,
    MemoryDelta,
    MemorySelection,
    Occurrence,
)
from anamnesis.schema import MemoryView

# Bump the relevant identifier whenever its behavior changes. The experiment
# manifest also pins the git commit; these identifiers make component drift an
# explicit part of the per-system configuration fingerprint.
ANAMNESIS_REDUCER_VERSION = "anamnesis.reducer.v0.1.0"
ANAMNESIS_TRIGGER_ENGINE_VERSION = "anamnesis.trigger-engine.v0.1.0"
ANAMNESIS_RENDERER_VERSION = "anamnesis.memory-view-renderer.v0.1.0"
ANAMNESIS_MEMORY_SCHEMA_VERSION = "anamnesis.memory-schema.v0.1.0"
_HISTORICAL_V1_MEMORY_SCHEMA_SHA256 = (
    "cde6c640e9514300eade7dd5eee2e1011992a6e6174124bbd30c41c5c4a4da53"
)

# Architecture revision v2 is the sole current executable identity. The
# original constants remain byte-stable historical metadata for W1-W3 artifact
# interpretation at their source commits; current cells always fingerprint v2.
ANAMNESIS_REDUCER_V2_VERSION = "anamnesis.reducer.v0.2.0"
ANAMNESIS_TRIGGER_ENGINE_V2_VERSION = "anamnesis.trigger-engine.v0.2.0"
ANAMNESIS_RENDERER_V2_VERSION = "anamnesis.memory-view-renderer.v0.2.0"
ANAMNESIS_MEMORY_SCHEMA_V2_VERSION = "anamnesis.memory-schema.v0.2.0"
ANAMNESIS_COMPILER_STATE_V2_VERSION = "anamnesis.compiler-state.v0.2.0"

_MEMORY_SCHEMA_MODELS: tuple[type[BaseModel], ...] = (
    MemoryDelta,
    FactRevision,
    IntentRevision,
    Occurrence,
    ExecutionRecord,
    MemorySelection,
    MemoryView,
)

_MEMORY_SCHEMA_V2_MODELS: tuple[type[BaseModel], ...] = (
    *_MEMORY_SCHEMA_MODELS,
    CompilerStateView,
)


@lru_cache(maxsize=1)
def memory_schema_sha256() -> str:
    """Hash the complete reducer input/state/view schema bundle."""

    schema_bundle = {
        model.__name__: model.model_json_schema() for model in _MEMORY_SCHEMA_MODELS
    }
    serialized = json.dumps(schema_bundle, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


@lru_cache(maxsize=1)
def memory_schema_v2_sha256() -> str:
    """Hash the v2 reducer state plus the closed compiler-facing projection."""

    schema_bundle = {
        model.__name__: model.model_json_schema() for model in _MEMORY_SCHEMA_V2_MODELS
    }
    serialized = json.dumps(schema_bundle, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def historical_anamnesis_runtime_contract_v1() -> dict[str, str]:
    """Return v1 metadata for interpreting artifacts from their source commit.

    This is deliberately not the executable runtime identity.  The only
    in-process reducer implementation is architecture v2, so current task and
    fallback fingerprints must use :func:`anamnesis_runtime_contract` below.
    """

    return {
        "memory_schema_version": "anamnesis.memory-schema.v0.1.0",
        "memory_schema_sha256": _HISTORICAL_V1_MEMORY_SCHEMA_SHA256,
        "reducer_version": "anamnesis.reducer.v0.1.0",
        "trigger_engine_version": "anamnesis.trigger-engine.v0.1.0",
        "renderer_version": "anamnesis.memory-view-renderer.v0.1.0",
    }


def anamnesis_runtime_contract_v2() -> dict[str, str]:
    """Return the architecture-v2 deterministic-memory component pins."""

    return {
        "memory_schema_version": ANAMNESIS_MEMORY_SCHEMA_V2_VERSION,
        "memory_schema_sha256": memory_schema_v2_sha256(),
        "reducer_version": ANAMNESIS_REDUCER_V2_VERSION,
        "trigger_engine_version": ANAMNESIS_TRIGGER_ENGINE_V2_VERSION,
        "renderer_version": ANAMNESIS_RENDERER_V2_VERSION,
        "compiler_state_version": ANAMNESIS_COMPILER_STATE_V2_VERSION,
    }


def anamnesis_runtime_contract() -> dict[str, str]:
    """Return the current executable deterministic-memory contract."""

    return anamnesis_runtime_contract_v2()


__all__ = [
    "ANAMNESIS_MEMORY_SCHEMA_VERSION",
    "ANAMNESIS_COMPILER_STATE_V2_VERSION",
    "ANAMNESIS_REDUCER_VERSION",
    "ANAMNESIS_REDUCER_V2_VERSION",
    "ANAMNESIS_RENDERER_VERSION",
    "ANAMNESIS_RENDERER_V2_VERSION",
    "ANAMNESIS_TRIGGER_ENGINE_VERSION",
    "ANAMNESIS_TRIGGER_ENGINE_V2_VERSION",
    "ANAMNESIS_MEMORY_SCHEMA_V2_VERSION",
    "anamnesis_runtime_contract",
    "anamnesis_runtime_contract_v2",
    "historical_anamnesis_runtime_contract_v1",
    "memory_schema_sha256",
    "memory_schema_v2_sha256",
]
