"""Prospective paired validation of source-grounded action canonicalization."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from anamnesis.action_canonicalizer import (
    CanonicalizationChange,
    canonicalize_immediate_decision,
)
from anamnesis.baselines import FastEmbedVectorizer
from anamnesis.openmemory_diagnostic import ExpectedRecallDecision
from anamnesis.openmemory_recall import OpenMemoryRecallIndex
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
from anamnesis.openmemory_vllm_v5 import BASE_PIN_PATH
from anamnesis.openmemory_vllm_v6 import (
    EMBEDDING_ARTIFACT_SHA256,
    EMBEDDING_REVISION,
    LocalVectorMemoryClient,
    _correct,
)
from anamnesis.runner import DecisionRequest
from anamnesis.schema import Decision, ObservableEvent, StrictModel, Usage
from anamnesis.vllm_runtime import (
    OpenAIExternalVllmClient,
    VllmProbeSnapshot,
    canonical_json_sha256,
    verify_vllm_artifact,
)

FIXTURE_PATH = REPO_ROOT / "eval/openmemory/real_memory_canonicalizer.v1.json"
PIN_PATH = REPO_ROOT / "eval/openmemory/real_memory_canonicalizer.v1.pin.json"
CANONICALIZER_PATH = REPO_ROOT / "src/anamnesis/action_canonicalizer.py"


class _Frozen(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class V7Memory(_Frozen):
    id: str = Field(pattern=r"^omr2_mem_[a-z0-9_]+$")
    content: str = Field(min_length=1, max_length=4096)


class V7Case(_Frozen):
    id: str = Field(pattern=r"^omr2_[a-z0-9_]+$")
    event: ObservableEvent
    memories: tuple[V7Memory, ...]
    expected_retrieved_memory_ids: tuple[str, ...]
    expected: ExpectedRecallDecision
    helpful_opportunity: bool

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        ids = tuple(item.id for item in self.memories)
        if len(ids) != len(set(ids)) or any(
            item not in ids for item in self.expected_retrieved_memory_ids
        ):
            raise ValueError("v7 memory identity is invalid")
        if len(self.expected_retrieved_memory_ids) > 1:
            raise ValueError("v7 freezes top_k=1")
        return self


class V7Fixture(_Frozen):
    schema_version: Literal[1] = 1
    purpose: Literal["openmemory_vllm_canonicalized_memory_v1"]
    hypothesis_test_eligible: Literal[False] = False
    cases: tuple[V7Case, ...]

    @model_validator(mode="after")
    def validate_fixture(self) -> Self:
        if len(self.cases) != 6 or sum(x.helpful_opportunity for x in self.cases) != 3:
            raise ValueError("v7 requires six cases and three helpful opportunities")
        return self


class V7Pin(_Frozen):
    schema_version: Literal[1] = 1
    purpose: Literal["openmemory_vllm_canonicalized_memory_v1"]
    hypothesis_test_eligible: Literal[False] = False
    fixture_raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_runtime_pin_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonicalizer_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_url: Literal["http://127.0.0.1:18003/v1"]
    served_model: Literal["anamnesis-openmemory-v7"]
    server_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_revision: Literal[EMBEDDING_REVISION]
    embedding_artifact_sha256: Literal[EMBEDDING_ARTIFACT_SHA256]
    case_count: Literal[6]
    top_k: Literal[1]
    expected_model_calls: Literal[12]
    seed: Literal[101]
    temperature: Literal[0.0]
    max_tokens: Literal[256]
    retries_repairs_cache: Literal[0]
    stopping_rule: Literal["one_fresh_paired_canonicalizer_run"]


class V7Arm(_Frozen):
    arm: Literal["baseline", "recall"]
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_decision: Decision
    canonical_decision: Decision
    changes: tuple[CanonicalizationChange, ...]
    correct: bool
    usage: Usage
    latency_ms: float = Field(ge=0)
    audit: VllmDecisionAudit


class V7CaseResult(_Frozen):
    case_id: str
    retrieved_memory_ids: tuple[str, ...]
    retrieved_contents: tuple[str, ...]
    retrieval_correct: bool
    baseline: V7Arm
    recall: V7Arm


class V7Metrics(_Frozen):
    baseline_correct: int
    recall_correct: int
    helpful_gain: int
    safety_regressions: int
    retrieval_correct: int
    accepted_calls: int
    canonicalizer_changes: int
    gate_passed: bool


class V7Run(_Frozen):
    schema_version: Literal[1] = 1
    purpose: Literal["openmemory_vllm_canonicalized_memory_v1"] = (
        "openmemory_vllm_canonicalized_memory_v1"
    )
    hypothesis_test_eligible: Literal[False] = False
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pin_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonicalizer_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: tuple[V7CaseResult, ...]
    metrics: V7Metrics
    usage: Usage
    passed: bool

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        usage = Usage()
        for case in self.cases:
            usage = usage.plus(case.baseline.usage).plus(case.recall.usage)
        if len(self.cases) != 6 or usage != self.usage:
            raise ValueError("v7 matrix or usage differs")
        if self.passed != self.metrics.gate_passed:
            raise ValueError("v7 pass flag differs")
        return self


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        "port": 18003,
        "served_model": runtime.served_model,
        "speculative_decoding": False,
        "structured_output_backend": "xgrammar",
    }


def _load_inputs() -> tuple[V7Pin, V7Fixture, VllmDecisionRuntimePin]:
    pin = V7Pin.model_validate_json(PIN_PATH.read_text())
    raw = json.loads(FIXTURE_PATH.read_text())
    checks = (
        (_sha256(FIXTURE_PATH), pin.fixture_raw_sha256),
        (canonical_json_sha256(raw), pin.fixture_canonical_sha256),
        (_sha256(BASE_PIN_PATH), pin.base_runtime_pin_sha256),
        (_sha256(CANONICALIZER_PATH), pin.canonicalizer_source_sha256),
        (
            openmemory_vllm_aligned_decision_contract_sha256(),
            pin.decision_contract_sha256,
        ),
        (openmemory_vllm_aligned_schema_sha256(), pin.response_schema_sha256),
    )
    if any(actual != expected for actual, expected in checks):
        raise ValueError("v7 frozen input differs")
    base = VllmDecisionRuntimePin.model_validate_json(BASE_PIN_PATH.read_text())
    runtime = VllmDecisionRuntimePin.model_validate(
        base.model_dump(mode="json")
        | {
            "base_url": pin.base_url,
            "served_model": pin.served_model,
            "server_config_sha256": pin.server_config_sha256,
            "decision_contract_sha256": pin.decision_contract_sha256,
            "response_schema_sha256": pin.response_schema_sha256,
            "max_tokens": pin.max_tokens,
        }
    )
    return pin, V7Fixture.model_validate(raw), runtime


def _probe(runtime: VllmDecisionRuntimePin) -> VllmProbeSnapshot:
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
        max_model_len=4096,
        max_num_seqs=1,
        speculative_decoding=False,
    )


async def run_v7(
    *, artifact_root: Path, embedding_snapshot: Path, api_key: str, source_commit: str
) -> V7Run:
    _verify_source_commit(REPO_ROOT, source_commit)
    pin, fixture, runtime = _load_inputs()
    if canonical_json_sha256(_server_config(runtime)) != pin.server_config_sha256:
        raise ValueError("v7 server configuration differs")
    verify_vllm_artifact(artifact_root, runtime.artifact)
    vectorizer = FastEmbedVectorizer(
        revision=EMBEDDING_REVISION, snapshot_path=embedding_snapshot
    )
    vectorizer.warmup()
    if vectorizer.artifact_sha256 != EMBEDDING_ARTIFACT_SHA256:
        raise ValueError("v7 embedding artifact differs")
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
            declared=_probe(runtime),
            api_key=api_key,
            timeout_seconds=runtime.request_timeout_seconds,
        ),
    )
    results: list[V7CaseResult] = []
    try:
        for case in fixture.cases:
            memory_client = LocalVectorMemoryClient(vectorizer)
            index = OpenMemoryRecallIndex(
                namespace=case.id, user_id="prospective-test", client=memory_client
            )
            for memory in case.memories:
                await index.add(memory.content, metadata={"fixture_id": memory.id})
            recalled = await index.search(case.event.text, limit=1)
            contents = tuple(match.content for match in recalled.matches)
            arms: list[V7Arm] = []
            for arm, recall in (("baseline", None), ("recall", contents)):
                prompt = build_openmemory_vllm_user_envelope(
                    now=case.event.at.isoformat(),
                    current_event_id=case.event.id,
                    context_events=[case.event],
                    decision_history=[],
                    memory_view=None,
                    retrospective_recall=recall,
                )
                call = await model.decide(
                    DecisionRequest(event=case.event, prompt=prompt)
                )
                normalized = canonicalize_immediate_decision(
                    event=case.event,
                    retrospective_recall=recall,
                    decision=call.decision,
                )
                arms.append(
                    V7Arm(
                        arm=arm,
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        raw_decision=call.decision,
                        canonical_decision=normalized.decision,
                        changes=normalized.changes,
                        correct=_correct(case.expected, normalized.decision),
                        usage=call.usage,
                        latency_ms=call.latency_ms,
                        audit=model.audits[-1],
                    )
                )
            results.append(
                V7CaseResult(
                    case_id=case.id,
                    retrieved_memory_ids=memory_client.last_search_ids,
                    retrieved_contents=contents,
                    retrieval_correct=(
                        memory_client.last_search_ids
                        == case.expected_retrieved_memory_ids
                    ),
                    baseline=arms[0],
                    recall=arms[1],
                )
            )
    finally:
        await client.aclose()
    baseline_correct = sum(item.baseline.correct for item in results)
    recall_correct = sum(item.recall.correct for item in results)
    helpful_gain = sum(
        case.helpful_opportunity
        and not result.baseline.correct
        and result.recall.correct
        for case, result in zip(fixture.cases, results, strict=True)
    )
    safety_regressions = sum(
        not case.helpful_opportunity
        and result.baseline.correct
        and not result.recall.correct
        for case, result in zip(fixture.cases, results, strict=True)
    )
    retrieval_correct = sum(item.retrieval_correct for item in results)
    accepted = sum(
        item.baseline.audit.validation.accepted + item.recall.audit.validation.accepted
        for item in results
    )
    changes = sum(
        len(item.baseline.changes) + len(item.recall.changes) for item in results
    )
    passed = (
        retrieval_correct == 6
        and accepted == 12
        and helpful_gain >= 2
        and safety_regressions == 0
        and recall_correct > baseline_correct
    )
    metrics = V7Metrics(
        baseline_correct=baseline_correct,
        recall_correct=recall_correct,
        helpful_gain=helpful_gain,
        safety_regressions=safety_regressions,
        retrieval_correct=retrieval_correct,
        accepted_calls=accepted,
        canonicalizer_changes=changes,
        gate_passed=passed,
    )
    usage = Usage()
    for result in results:
        usage = usage.plus(result.baseline.usage).plus(result.recall.usage)
    return V7Run(
        source_commit=source_commit,
        fixture_sha256=pin.fixture_raw_sha256,
        pin_sha256=_sha256(PIN_PATH),
        canonicalizer_source_sha256=pin.canonicalizer_source_sha256,
        cases=tuple(results),
        metrics=metrics,
        usage=usage,
        passed=passed,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--embedding-snapshot", type=Path, required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(
        run_v7(
            artifact_root=args.artifact_root,
            embedding_snapshot=args.embedding_snapshot,
            api_key=args.api_key,
            source_commit=args.source_commit,
        )
    )
    output = args.output.resolve()
    allowed = (REPO_ROOT / "results/runs/local/openmemory_vllm_v7").resolve()
    if not output.is_relative_to(allowed) or output.suffix != ".json":
        raise ValueError("v7 output must be JSON under its run directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2) + "\n")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
