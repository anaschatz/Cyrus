"""Two-call live compatibility gate for the additive aligned vLLM schema."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from anamnesis.openmemory_vllm import (
    HttpOperatorVllmProbe,
    VllmDecisionAudit,
    VllmDecisionRuntimePin,
    VllmOpenMemoryAlignedDecisionModel,
    build_openmemory_vllm_user_envelope,
    openmemory_vllm_aligned_decision_contract_sha256,
    openmemory_vllm_aligned_schema_sha256,
)
from anamnesis.openmemory_vllm_run import REPO_ROOT, _verify_source_commit
from anamnesis.runner import DecisionRequest
from anamnesis.schema import Decision, ObservableEvent, StrictModel, Usage
from anamnesis.vllm_runtime import (
    OpenAIExternalVllmClient,
    VllmProbeSnapshot,
    canonical_json_sha256,
    verify_vllm_artifact,
)

BASE_PIN_PATH = REPO_ROOT / "eval/openmemory/vllm_v4_runtime.pin.json"
FIXTURE_PATH = REPO_ROOT / "eval/openmemory/vllm_v5_compatibility.json"
PIN_PATH = REPO_ROOT / "eval/openmemory/vllm_v5_compatibility.pin.json"


class _FrozenStrictModel(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VllmV5Case(_FrozenStrictModel):
    id: str = Field(pattern=r"^omv5_[a-z0-9_]+$")
    event: ObservableEvent
    retrospective_recall: tuple[str, ...]
    expected_mode: Literal["emit", "no_action"]
    expected_action_key: str | None
    expected_evidence_event_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_expectation(self) -> Self:
        if self.retrospective_recall:
            raise ValueError("v5 compatibility cases must not use recall content")
        if self.expected_mode == "emit":
            if self.expected_action_key != self.event.id:
                raise ValueError("emit compatibility action key must be the event")
            if self.expected_evidence_event_ids != (self.event.id,):
                raise ValueError("emit compatibility evidence must be the event")
        elif self.expected_action_key is not None or self.expected_evidence_event_ids:
            raise ValueError("no-action compatibility case cannot expect action fields")
        return self


class VllmV5Fixture(_FrozenStrictModel):
    schema_version: Literal[1] = 1
    purpose: Literal["openmemory_vllm_v5_schema_alignment_compatibility"]
    hypothesis_test_eligible: Literal[False] = False
    cases: tuple[VllmV5Case, VllmV5Case]

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if tuple(case.expected_mode for case in self.cases) != ("emit", "no_action"):
            raise ValueError("v5 compatibility order must be emit then no_action")
        return self


class VllmV5Pin(_FrozenStrictModel):
    schema_version: Literal[1] = 1
    purpose: Literal["openmemory_vllm_v5_schema_alignment_compatibility"]
    hypothesis_test_eligible: Literal[False] = False
    parent_source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    base_runtime_pin_path: Literal["eval/openmemory/vllm_v4_runtime.pin.json"]
    base_runtime_pin_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_path: Literal["eval/openmemory/vllm_v5_compatibility.json"]
    fixture_raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_url: Literal["http://127.0.0.1:18001/v1"]
    served_model: Literal["anamnesis-openmemory-v5"]
    server_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: Literal[2]
    expected_model_calls: Literal[2]
    seed: Literal[101]
    temperature: Literal[0.0]
    max_tokens: Literal[256]
    retry_or_repair_calls: Literal[0]
    stopping_rule: Literal["one_ordered_emit_then_no_action_compatibility_run"]


class VllmV5CaseResult(_FrozenStrictModel):
    case_id: str
    expected_mode: Literal["emit", "no_action"]
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Decision
    usage: Usage
    latency_ms: float = Field(ge=0)
    raw_completion: str | None
    audit: VllmDecisionAudit
    semantic_passed: bool


class OpenMemoryVllmV5Run(_FrozenStrictModel):
    schema_version: Literal[1] = 1
    purpose: Literal["openmemory_vllm_v5_schema_alignment_compatibility"] = (
        "openmemory_vllm_v5_schema_alignment_compatibility"
    )
    hypothesis_test_eligible: Literal[False] = False
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pin_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: tuple[VllmV5CaseResult, VllmV5CaseResult]
    usage: Usage
    passed: bool

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if tuple(case.expected_mode for case in self.cases) != ("emit", "no_action"):
            raise ValueError("v5 result order differs from frozen fixture")
        expected_usage = self.cases[0].usage.plus(self.cases[1].usage)
        if self.usage != expected_usage:
            raise ValueError("v5 aggregate usage differs from two calls")
        expected_passed = all(
            case.semantic_passed and case.audit.validation.accepted
            for case in self.cases
        )
        if self.passed != expected_passed:
            raise ValueError("v5 passed flag differs from frozen compatibility gate")
        return self


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _load_inputs() -> tuple[VllmV5Pin, VllmV5Fixture, VllmDecisionRuntimePin]:
    pin = VllmV5Pin.model_validate_json(PIN_PATH.read_text(encoding="utf-8"))
    if _sha256(BASE_PIN_PATH) != pin.base_runtime_pin_sha256:
        raise ValueError("v5 base runtime pin bytes differ")
    if _sha256(FIXTURE_PATH) != pin.fixture_raw_sha256:
        raise ValueError("v5 compatibility fixture bytes differ")
    raw_fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if _canonical_sha256(raw_fixture) != pin.fixture_canonical_sha256:
        raise ValueError("v5 compatibility fixture semantics differ")
    fixture = VllmV5Fixture.model_validate(raw_fixture)
    if pin.decision_contract_sha256 != (
        openmemory_vllm_aligned_decision_contract_sha256()
    ):
        raise ValueError("v5 aligned decision contract differs")
    if pin.response_schema_sha256 != openmemory_vllm_aligned_schema_sha256():
        raise ValueError("v5 aligned response schema differs")
    base = VllmDecisionRuntimePin.model_validate_json(
        BASE_PIN_PATH.read_text(encoding="utf-8")
    )
    raw_runtime = base.model_dump(mode="json") | {
        "base_url": pin.base_url,
        "served_model": pin.served_model,
        "server_config_sha256": pin.server_config_sha256,
        "decision_contract_sha256": pin.decision_contract_sha256,
        "response_schema_sha256": pin.response_schema_sha256,
        "max_tokens": pin.max_tokens,
    }
    runtime = VllmDecisionRuntimePin.model_validate(raw_runtime)
    return pin, fixture, runtime


def _server_config(runtime: VllmDecisionRuntimePin) -> dict[str, object]:
    return {
        "artifact_manifest_sha256": runtime.artifact.manifest_sha256,
        "generation_config": "vllm",
        "host": "127.0.0.1",
        "log_deltas": False,
        "log_outputs": False,
        "max_model_len": 4096,
        "max_num_seqs": 1,
        "multimodal_mode": "text-only-compat",
        "paged_attention": True,
        "port": 18001,
        "served_model": runtime.served_model,
        "speculative_decoding": False,
        "structured_output_backend": "xgrammar",
    }


def _declared_probe(runtime: VllmDecisionRuntimePin) -> VllmProbeSnapshot:
    return VllmProbeSnapshot(
        health_ok=False,
        base_url=runtime.base_url,
        vllm_version=runtime.vllm_server_version,
        model_ids=(runtime.served_model,),
        model_artifact_manifest_sha256=runtime.artifact.manifest_sha256,
        server_config=_server_config(runtime),
        runtime_packages={item.name: item.version for item in runtime.runtime_packages},
        structured_output_backend="xgrammar",
        generation_config="vllm",
        max_model_len=runtime.max_model_len,
        max_num_seqs=1,
        speculative_decoding=False,
    )


def _semantic_passed(case: VllmV5Case, decision: Decision) -> bool:
    if case.expected_mode == "no_action":
        return not decision.actions
    if len(decision.actions) != 1:
        return False
    action = decision.actions[0]
    return (
        action.action_key == case.expected_action_key
        and tuple(action.evidence_event_ids) == case.expected_evidence_event_ids
        and len(str(action.payload["subject"]).split()) >= 2
    )


async def run_v5_compatibility(
    *, artifact_root: Path, api_key: str, source_commit: str
) -> OpenMemoryVllmV5Run:
    _verify_source_commit(REPO_ROOT, source_commit)
    pin, fixture, runtime = _load_inputs()
    if canonical_json_sha256(_server_config(runtime)) != pin.server_config_sha256:
        raise ValueError("v5 server configuration differs from pin")
    verify_vllm_artifact(artifact_root, runtime.artifact)
    client = OpenAIExternalVllmClient(
        base_url=runtime.base_url,
        api_key=api_key,
        request_timeout_seconds=runtime.request_timeout_seconds,
    )
    model = VllmOpenMemoryAlignedDecisionModel(
        pin=runtime,
        api_key=api_key,
        artifact_root=artifact_root,
        client=client,
        probe=HttpOperatorVllmProbe(
            declared=_declared_probe(runtime),
            api_key=api_key,
            timeout_seconds=runtime.request_timeout_seconds,
        ),
    )
    results: list[VllmV5CaseResult] = []
    try:
        for case in fixture.cases:
            prompt = build_openmemory_vllm_user_envelope(
                now=case.event.at.isoformat(),
                current_event_id=case.event.id,
                context_events=[case.event],
                decision_history=[],
                memory_view=None,
                retrospective_recall=case.retrospective_recall,
            )
            call = await model.decide(DecisionRequest(event=case.event, prompt=prompt))
            results.append(
                VllmV5CaseResult(
                    case_id=case.id,
                    expected_mode=case.expected_mode,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    decision=call.decision,
                    usage=call.usage,
                    latency_ms=call.latency_ms,
                    raw_completion=call.raw_completion,
                    audit=model.audits[-1],
                    semantic_passed=_semantic_passed(case, call.decision),
                )
            )
    finally:
        await client.aclose()
    cases = (results[0], results[1])
    return OpenMemoryVllmV5Run(
        source_commit=source_commit,
        fixture_sha256=pin.fixture_raw_sha256,
        pin_sha256=_sha256(PIN_PATH),
        cases=cases,
        usage=cases[0].usage.plus(cases[1].usage),
        passed=all(
            case.semantic_passed and case.audit.validation.accepted for case in cases
        ),
    )


def _output_path(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (
        REPO_ROOT / "results/runs/local/openmemory_vllm_v5_compatibility"
    ).resolve()
    if not resolved.is_relative_to(allowed) or resolved.suffix != ".json":
        raise ValueError("v5 output must be JSON under its frozen run folder")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(
        run_v5_compatibility(
            artifact_root=args.artifact_root,
            api_key=args.api_key,
            source_commit=args.source_commit,
        )
    )
    output = _output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OpenMemoryVllmV5Run",
    "VllmV5Case",
    "VllmV5CaseResult",
    "VllmV5Fixture",
    "VllmV5Pin",
    "run_v5_compatibility",
]
