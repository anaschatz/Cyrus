from __future__ import annotations

import asyncio
import re

import pytest

import anamnesis.runtime_contract as runtime_contract
from anamnesis.baselines import AnamnesisMemoryStrategy
from anamnesis.inspect_adapter import _system_config_sha256
from anamnesis.io import load_scenarios
from anamnesis.memory import DeterministicCompiler
from anamnesis.runner import DecisionCall, DecisionRequest, run_scenario
from anamnesis.schema import Decision


class _EmptyDecisionModel:
    name = "provider/frozen-snapshot"

    async def decide(self, request: DecisionRequest) -> DecisionCall:
        return DecisionCall(decision=Decision())


def _anamnesis_system_hash() -> str:
    return _system_config_sha256(
        system="anamnesis",
        model="provider/frozen-snapshot",
        top_k=5,
        embedding_model="BAAI/bge-small-en-v1.5",
        pricing_config_sha256="a" * 64,
    )


def test_anamnesis_runtime_contract_has_explicit_component_pins() -> None:
    contract = runtime_contract.anamnesis_runtime_contract()

    assert set(contract) == {
        "memory_schema_version",
        "memory_schema_sha256",
        "reducer_version",
        "trigger_engine_version",
        "renderer_version",
        "compiler_state_version",
    }
    assert re.fullmatch(r"[0-9a-f]{64}", contract["memory_schema_sha256"])
    assert contract["memory_schema_sha256"] == (
        runtime_contract.memory_schema_v2_sha256()
    )
    for key in (
        "memory_schema_version",
        "reducer_version",
        "trigger_engine_version",
        "renderer_version",
        "compiler_state_version",
    ):
        assert contract[key].startswith("anamnesis.")


@pytest.mark.parametrize(
    "version_name,contract_key",
    [
        ("ANAMNESIS_MEMORY_SCHEMA_V2_VERSION", "memory_schema_version"),
        ("ANAMNESIS_REDUCER_V2_VERSION", "reducer_version"),
        ("ANAMNESIS_TRIGGER_ENGINE_V2_VERSION", "trigger_engine_version"),
        ("ANAMNESIS_RENDERER_V2_VERSION", "renderer_version"),
        ("ANAMNESIS_COMPILER_STATE_V2_VERSION", "compiler_state_version"),
    ],
)
def test_component_version_drift_changes_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
    version_name: str,
    contract_key: str,
) -> None:
    before = runtime_contract.anamnesis_runtime_contract()
    monkeypatch.setattr(runtime_contract, version_name, "anamnesis.drift.v9")
    after = runtime_contract.anamnesis_runtime_contract()

    assert after[contract_key] != before[contract_key]


@pytest.mark.parametrize(
    "version_name",
    [
        "ANAMNESIS_MEMORY_SCHEMA_V2_VERSION",
        "ANAMNESIS_REDUCER_V2_VERSION",
        "ANAMNESIS_TRIGGER_ENGINE_V2_VERSION",
        "ANAMNESIS_RENDERER_V2_VERSION",
        "ANAMNESIS_COMPILER_STATE_V2_VERSION",
    ],
)
def test_frozen_system_hash_detects_component_version_drift(
    monkeypatch: pytest.MonkeyPatch,
    version_name: str,
) -> None:
    manifest_pin = _anamnesis_system_hash()
    monkeypatch.setattr(runtime_contract, version_name, "anamnesis.drift.v9")

    assert _anamnesis_system_hash() != manifest_pin


def test_frozen_system_hash_detects_memory_schema_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_pin = _anamnesis_system_hash()
    monkeypatch.setattr(runtime_contract, "memory_schema_v2_sha256", lambda: "f" * 64)

    assert _anamnesis_system_hash() != manifest_pin


def test_architecture_v2_contract_is_additive_and_explicit() -> None:
    legacy = runtime_contract.historical_anamnesis_runtime_contract_v1()
    contract = runtime_contract.anamnesis_runtime_contract_v2()

    assert set(contract) == {*legacy, "compiler_state_version"}
    assert contract["memory_schema_version"] == (
        runtime_contract.ANAMNESIS_MEMORY_SCHEMA_V2_VERSION
    )
    assert contract["memory_schema_sha256"] == (
        runtime_contract.memory_schema_v2_sha256()
    )
    assert contract != legacy
    assert legacy["memory_schema_sha256"] == (
        "cde6c640e9514300eade7dd5eee2e1011992a6e6174124bbd30c41c5c4a4da53"
    )
    assert runtime_contract.anamnesis_runtime_contract() == contract
    assert re.fullmatch(r"[0-9a-f]{64}", contract["memory_schema_sha256"])


@pytest.mark.parametrize(
    "version_name,contract_key",
    [
        ("ANAMNESIS_MEMORY_SCHEMA_V2_VERSION", "memory_schema_version"),
        ("ANAMNESIS_REDUCER_V2_VERSION", "reducer_version"),
        ("ANAMNESIS_TRIGGER_ENGINE_V2_VERSION", "trigger_engine_version"),
        ("ANAMNESIS_RENDERER_V2_VERSION", "renderer_version"),
        ("ANAMNESIS_COMPILER_STATE_V2_VERSION", "compiler_state_version"),
    ],
)
def test_architecture_v2_component_drift_changes_only_v2_contract(
    monkeypatch: pytest.MonkeyPatch,
    version_name: str,
    contract_key: str,
) -> None:
    legacy = runtime_contract.historical_anamnesis_runtime_contract_v1()
    before = runtime_contract.anamnesis_runtime_contract_v2()

    monkeypatch.setattr(runtime_contract, version_name, "anamnesis.drift.v9")

    after = runtime_contract.anamnesis_runtime_contract_v2()
    assert after[contract_key] != before[contract_key]
    assert runtime_contract.anamnesis_runtime_contract() == after
    assert runtime_contract.historical_anamnesis_runtime_contract_v1() == legacy


def test_anamnesis_component_drift_does_not_change_simple_baseline_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = {
        "system": "full_context",
        "model": "provider/frozen-snapshot",
        "top_k": 5,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "pricing_config_sha256": "a" * 64,
    }
    before = _system_config_sha256(**kwargs)
    monkeypatch.setattr(
        runtime_contract,
        "ANAMNESIS_REDUCER_V2_VERSION",
        "anamnesis.drift.v9",
    )

    assert _system_config_sha256(**kwargs) == before


def test_unpinned_runner_fallback_hash_detects_reducer_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_scenarios("eval/scenarios/smoke.jsonl")[0]

    def run_hash() -> str:
        run = asyncio.run(
            run_scenario(
                scenario=scenario,
                strategy=AnamnesisMemoryStrategy(DeterministicCompiler({})),
                model=_EmptyDecisionModel(),
            )
        )
        return run.system_config_sha256

    before = run_hash()
    monkeypatch.setattr(
        runtime_contract,
        "ANAMNESIS_REDUCER_V2_VERSION",
        "anamnesis.drift.v9",
    )

    assert run_hash() != before
