from __future__ import annotations

import asyncio
import json
import runpy
from types import SimpleNamespace

import pytest
from inspect_ai.model import ModelOutput, ModelUsage

from anamnesis.inspect_adapter import (
    SCENARIO_METADATA_KEY,
    SCENARIO_SHA256_METADATA_KEY,
    _decision_schema,
    _logical_input_tokens,
    _memory_delta_schema,
    _supports_strict_schema,
    _TaskHostedWarmup,
    _usage_from_output,
    _verify_official_openai_endpoint,
    scenario_record_to_sample,
    scenario_solver,
)
from anamnesis.io import canonical_sha256
from anamnesis.schema import Scenario, Usage


def first_record() -> dict[str, object]:
    with open("eval/scenarios/smoke.jsonl", encoding="utf-8") as handle:
        return json.loads(next(handle))


def test_inspect_sample_keeps_gold_only_in_target() -> None:
    sample = scenario_record_to_sample(first_record())
    gold = Scenario.model_validate_json(sample.target)
    observable = sample.metadata[SCENARIO_METADATA_KEY]

    assert gold.expected_actions
    assert set(observable) == {"id", "events"}
    assert "expected_actions" not in observable
    assert "forbidden_actions" not in observable
    assert "title" not in observable
    assert "description" not in observable
    assert "tags" not in observable
    assert all("supersedes" not in event for event in observable["events"])
    assert all(
        set(event) == {"id", "at", "kind", "text"} for event in observable["events"]
    )
    assert sample.metadata[SCENARIO_SHA256_METADATA_KEY] == canonical_sha256(gold)


def test_logical_input_tokens_include_cache_traffic() -> None:
    usage = ModelUsage(
        input_tokens=100,
        output_tokens=20,
        input_tokens_cache_read=30,
        input_tokens_cache_write=10,
        total_cost=0.05,
    )
    assert _logical_input_tokens(usage) == 140

    converted = _usage_from_output(
        ModelOutput(model="provider/model", completion="{}", usage=usage)
    )
    assert converted.input_tokens == 140
    assert converted.uncached_input_tokens == 100
    assert converted.cache_read_input_tokens == 30
    assert converted.cache_write_input_tokens == 10
    assert converted.output_tokens == 20
    assert converted.cost_usd == 0.05


def test_hosted_warmup_runs_once_and_keeps_auditable_setup_usage() -> None:
    class FakeDecisionModel:
        name = "openai/frozen-snapshot"

        def __init__(self) -> None:
            self.calls = 0

        async def complete_structured(self, *, prompt, response_schema):
            self.calls += 1
            return (
                ModelOutput(
                    model=self.name,
                    completion='{"actions":[]}',
                    usage=ModelUsage(
                        input_tokens=12,
                        output_tokens=3,
                        total_cost=0.0004,
                    ),
                ),
                7.5,
            )

    async def exercise():
        coordinator = _TaskHostedWarmup()
        model = FakeDecisionModel()
        first, first_performed = await coordinator.ensure(model)  # type: ignore[arg-type]
        second, second_performed = await coordinator.ensure(model)  # type: ignore[arg-type]
        return model, first, first_performed, second, second_performed

    model, first, first_performed, second, second_performed = asyncio.run(exercise())

    assert model.calls == 1
    assert first_performed is True
    assert second_performed is False
    assert first == second
    assert first.usage.input_tokens == 12
    assert first.usage.cost_usd == 0.0004
    assert first.included_in_headline is False


def test_logical_input_tokens_require_a_complete_cache_breakdown() -> None:
    with pytest.raises(ValueError, match="cache breakdown"):
        Usage(input_tokens=1)


def test_solver_rejects_invalid_repetition() -> None:
    with pytest.raises(ValueError, match="repetition"):
        scenario_solver("no_memory", repetition=0)


def test_measured_vector_solver_requires_exact_embedding_revision() -> None:
    with pytest.raises(ValueError, match="exact embedding revision"):
        scenario_solver("vector_rag")


def test_strict_schema_is_enabled_only_for_supported_providers() -> None:
    assert _supports_strict_schema("openai/model")
    assert _supports_strict_schema("mistral/model")
    assert not _supports_strict_schema("mockllm/model")


def test_official_openai_endpoint_guard_rejects_environment_proxy() -> None:
    official = SimpleNamespace(
        api=SimpleNamespace(
            client=SimpleNamespace(base_url="https://api.openai.com/v1/")
        )
    )
    _verify_official_openai_endpoint(
        official,
        "openai/gpt-4.1-mini-2025-04-14",
    )

    proxy = SimpleNamespace(
        api=SimpleNamespace(client=SimpleNamespace(base_url="https://proxy.invalid/v1"))
    )
    with pytest.raises(ValueError, match="official OpenAI endpoint"):
        _verify_official_openai_endpoint(
            proxy,
            "openai/gpt-4.1-mini-2025-04-14",
        )


def test_sealed_all_dataset_fails_closed_without_frozen_manifest() -> None:
    module = runpy.run_path("eval/anamnesis_eval.py")
    with pytest.raises(ValueError, match="frozen final manifest"):
        module["_scenario_dataset"]("all")
    with pytest.raises(ValueError, match="frozen final manifest"):
        module["_scenario_dataset"](
            "all", manifest_path="eval/experiment_manifest.template.json"
        )


def test_measured_task_freezes_generation_cache_and_connection_policy() -> None:
    module = runpy.run_path("eval/anamnesis_eval.py")
    task = module["no_memory"](seed=101)

    assert task.config.temperature == 0.0
    assert task.config.seed == 101
    assert task.config.cache is False
    assert task.config.max_retries == 0
    assert task.config.max_connections == 1
    assert task.config.adaptive_connections is False

    preflight = module["model_preflight"]()
    assert preflight.config.max_retries == 0


@pytest.mark.parametrize("factory", [_decision_schema, _memory_delta_schema])
def test_openai_wire_schemas_are_closed_required_and_fully_constrained(
    factory,
) -> None:
    def assert_constrained(schema, path: str = "root") -> None:
        if schema.properties is not None:
            assert schema.type == "object", path
            assert schema.additionalProperties is False, path
            assert set(schema.required or []) == set(schema.properties), path
            for name, child in schema.properties.items():
                assert_constrained(child, f"{path}.{name}")
        if schema.items is not None:
            assert_constrained(schema.items, f"{path}[]")
        for index, child in enumerate(schema.anyOf or []):
            assert_constrained(child, f"{path}.anyOf[{index}]")
        assert not (
            schema.type is None and schema.properties is None and schema.anyOf is None
        ), path

    response_schema = factory("openai/frozen-snapshot")

    assert response_schema.strict is True
    assert_constrained(response_schema.json_schema)
    properties = response_schema.json_schema.properties or {}
    if factory is _decision_schema:
        kind = properties["actions"].items.properties["kind"]
    else:
        creates = properties["intent_creates"].items
        kind = creates.properties["action_template"].properties["kind"]
    assert kind.enum == ["reminder"]
