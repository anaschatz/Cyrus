"""One-shot paired test of real local indexed recall through the OpenMemory boundary."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import ConfigDict, Field, model_validator

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
from anamnesis.runner import DecisionRequest
from anamnesis.schema import Decision, ObservableEvent, StrictModel, Usage
from anamnesis.vllm_runtime import (
    OpenAIExternalVllmClient,
    VllmProbeSnapshot,
    canonical_json_sha256,
    verify_vllm_artifact,
)

FIXTURE_PATH = REPO_ROOT / "eval/openmemory/real_memory_diagnostic.v1.json"
PIN_PATH = REPO_ROOT / "eval/openmemory/real_memory_diagnostic.v1.pin.json"
EMBEDDING_REVISION = "52398278842ec682c6f32300af41344b1c0b0bb2"
EMBEDDING_ARTIFACT_SHA256 = (
    "d435d05b3411502ad9a280cc9ac0157f7bcd9f176df2fdc8971f788a121a02d7"
)


class _Frozen(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MemoryRecord(_Frozen):
    id: str = Field(pattern=r"^omr1_mem_[a-z0-9_]+$")
    content: str = Field(min_length=1, max_length=4096)


class RealMemoryCase(_Frozen):
    id: str = Field(pattern=r"^omr1_[a-z0-9_]+$")
    event: ObservableEvent
    memories: tuple[MemoryRecord, ...]
    expected_retrieved_memory_ids: tuple[str, ...]
    expected: ExpectedRecallDecision
    helpful_opportunity: bool

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        ids = tuple(item.id for item in self.memories)
        if len(ids) != len(set(ids)):
            raise ValueError("memory IDs must be unique")
        if any(item not in ids for item in self.expected_retrieved_memory_ids):
            raise ValueError("expected retrieval references an unknown memory")
        if len(self.expected_retrieved_memory_ids) > 1:
            raise ValueError("v6 freezes top_k=1")
        return self


class RealMemoryFixture(_Frozen):
    schema_version: Literal[1] = 1
    purpose: Literal["openmemory_vllm_real_indexed_memory_v1"]
    hypothesis_test_eligible: Literal[False] = False
    cases: tuple[RealMemoryCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_fixture(self) -> Self:
        if len(self.cases) != 8:
            raise ValueError("v6 requires exactly eight fresh cases")
        if sum(case.helpful_opportunity for case in self.cases) != 4:
            raise ValueError("v6 requires four helpful opportunities")
        return self


class RealMemoryPin(_Frozen):
    schema_version: Literal[1] = 1
    purpose: Literal["openmemory_vllm_real_indexed_memory_v1"]
    hypothesis_test_eligible: Literal[False] = False
    fixture_raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_runtime_pin_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_url: Literal["http://127.0.0.1:18002/v1"]
    served_model: Literal["anamnesis-openmemory-v6"]
    server_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_revision: Literal[EMBEDDING_REVISION]
    embedding_artifact_sha256: Literal[EMBEDDING_ARTIFACT_SHA256]
    case_count: Literal[8]
    top_k: Literal[1]
    expected_model_calls: Literal[16]
    seed: Literal[101]
    temperature: Literal[0.0]
    max_tokens: Literal[256]
    retries_repairs_cache: Literal[0]
    stopping_rule: Literal["one_paired_baseline_then_recall_run"]


class ArmResult(_Frozen):
    arm: Literal["baseline", "recall"]
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Decision
    correct: bool
    usage: Usage
    latency_ms: float = Field(ge=0)
    audit: VllmDecisionAudit


class CaseResult(_Frozen):
    case_id: str
    retrieved_memory_ids: tuple[str, ...]
    retrieved_contents: tuple[str, ...]
    retrieval_correct: bool
    baseline: ArmResult
    recall: ArmResult


class RealMemoryMetrics(_Frozen):
    baseline_correct: int
    recall_correct: int
    helpful_gain: int
    safety_regressions: int
    retrieval_correct: int
    accepted_calls: int
    gate_passed: bool


class RealMemoryRun(_Frozen):
    schema_version: Literal[1] = 1
    purpose: Literal["openmemory_vllm_real_indexed_memory_v1"] = (
        "openmemory_vllm_real_indexed_memory_v1"
    )
    hypothesis_test_eligible: Literal[False] = False
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pin_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_artifact_sha256: Literal[EMBEDDING_ARTIFACT_SHA256]
    cases: tuple[CaseResult, ...]
    metrics: RealMemoryMetrics
    usage: Usage
    passed: bool

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if len(self.cases) != 8:
            raise ValueError("v6 result requires eight cases")
        usages = Usage()
        for case in self.cases:
            usages = usages.plus(case.baseline.usage).plus(case.recall.usage)
        if usages != self.usage or self.passed != self.metrics.gate_passed:
            raise ValueError("v6 aggregate mismatch")
        return self


class LocalVectorMemoryClient:
    """Real local add/search/get/delete client using the pinned FastEmbed model."""

    is_local = True
    mode = "local"
    embedding_provider = "fastembed"

    def __init__(self, vectorizer: FastEmbedVectorizer) -> None:
        self.vectorizer = vectorizer
        self.rows: dict[str, dict[str, object]] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.last_search_ids: tuple[str, ...] = ()

    def add(
        self, content: str, *, user_id: str, metadata: dict[str, object]
    ) -> dict[str, object]:
        memory_id = str(metadata["fixture_id"])
        self.rows[memory_id] = {
            "id": memory_id,
            "content": content,
            "user_id": user_id,
            "metadata": dict(metadata),
        }
        self.vectors[memory_id] = self.vectorizer.embed_documents([content])[0]
        return {"id": memory_id}

    def search(
        self, query: str, *, user_id: str, limit: int
    ) -> list[dict[str, object]]:
        candidates = [
            (key, row) for key, row in self.rows.items() if row["user_id"] == user_id
        ]
        if not candidates:
            self.last_search_ids = ()
            return []
        query_vector = self.vectorizer.embed_query(query)
        ranked = sorted(
            (
                (
                    float(
                        np.dot(vector, query_vector)
                        / (np.linalg.norm(vector) * np.linalg.norm(query_vector))
                    ),
                    key,
                    row,
                )
                for key, row in candidates
                for vector in (self.vectors[key],)
            ),
            key=lambda item: (-item[0], item[1]),
        )[:limit]
        self.last_search_ids = tuple(item[1] for item in ranked)
        return [
            dict(item[2]) | {"score": max(0.0, min(1.0, item[0]))} for item in ranked
        ]

    def get(self, memory_id: str) -> dict[str, object] | None:
        return self.rows.get(memory_id)

    def delete(self, memory_id: str) -> None:
        self.rows.pop(memory_id, None)
        self.vectors.pop(memory_id, None)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return canonical_json_sha256(value)


def _load_inputs() -> tuple[RealMemoryPin, RealMemoryFixture, VllmDecisionRuntimePin]:
    pin = RealMemoryPin.model_validate_json(PIN_PATH.read_text())
    if _sha256(FIXTURE_PATH) != pin.fixture_raw_sha256:
        raise ValueError("v6 fixture bytes differ")
    raw = json.loads(FIXTURE_PATH.read_text())
    if _canonical(raw) != pin.fixture_canonical_sha256:
        raise ValueError("v6 fixture semantics differ")
    if _sha256(BASE_PIN_PATH) != pin.base_runtime_pin_sha256:
        raise ValueError("v6 base runtime pin differs")
    if (
        pin.decision_contract_sha256
        != openmemory_vllm_aligned_decision_contract_sha256()
        or pin.response_schema_sha256 != openmemory_vllm_aligned_schema_sha256()
    ):
        raise ValueError("v6 aligned decision contract differs")
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
    return pin, RealMemoryFixture.model_validate(raw), runtime


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
        "port": 18002,
        "served_model": runtime.served_model,
        "speculative_decoding": False,
        "structured_output_backend": "xgrammar",
    }


def _probe(runtime: VllmDecisionRuntimePin) -> VllmProbeSnapshot:
    return VllmProbeSnapshot(
        health_ok=False,
        base_url=runtime.base_url,
        vllm_version=runtime.vllm_server_version,
        model_ids=(runtime.served_model,),
        model_artifact_manifest_sha256=runtime.artifact.manifest_sha256,
        server_config=_server_config(runtime),
        runtime_packages={x.name: x.version for x in runtime.runtime_packages},
        structured_output_backend="xgrammar",
        generation_config="vllm",
        max_model_len=4096,
        max_num_seqs=1,
        speculative_decoding=False,
    )


def _correct(expected: ExpectedRecallDecision, decision: Decision) -> bool:
    if expected.mode == "no_action":
        return not decision.actions
    if len(decision.actions) != 1:
        return False
    action = decision.actions[0]
    return (
        action.action_key == expected.action_key
        and action.payload == expected.payload
        and tuple(action.evidence_event_ids) == expected.evidence_event_ids
    )


async def run_real_memory(
    *, artifact_root: Path, embedding_snapshot: Path, api_key: str, source_commit: str
) -> RealMemoryRun:
    _verify_source_commit(REPO_ROOT, source_commit)
    pin, fixture, runtime = _load_inputs()
    if canonical_json_sha256(_server_config(runtime)) != pin.server_config_sha256:
        raise ValueError("v6 server configuration differs")
    verify_vllm_artifact(artifact_root, runtime.artifact)
    vectorizer = FastEmbedVectorizer(
        revision=EMBEDDING_REVISION, snapshot_path=embedding_snapshot
    )
    vectorizer.warmup()
    if vectorizer.artifact_sha256 != EMBEDDING_ARTIFACT_SHA256:
        raise ValueError("v6 embedding artifact differs")
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
    results: list[CaseResult] = []
    try:
        for case in fixture.cases:
            memory_client = LocalVectorMemoryClient(vectorizer)
            index = OpenMemoryRecallIndex(
                namespace=case.id, user_id="paired-test", client=memory_client
            )
            for memory in case.memories:
                await index.add(memory.content, metadata={"fixture_id": memory.id})
            recalled = await index.search(case.event.text, limit=1)
            contents = tuple(match.content for match in recalled.matches)
            arms: list[ArmResult] = []
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
                arms.append(
                    ArmResult(
                        arm=arm,
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        decision=call.decision,
                        correct=_correct(case.expected, call.decision),
                        usage=call.usage,
                        latency_ms=call.latency_ms,
                        audit=model.audits[-1],
                    )
                )
            results.append(
                CaseResult(
                    case_id=case.id,
                    retrieved_memory_ids=memory_client.last_search_ids,
                    retrieved_contents=contents,
                    retrieval_correct=memory_client.last_search_ids
                    == case.expected_retrieved_memory_ids,
                    baseline=arms[0],
                    recall=arms[1],
                )
            )
    finally:
        await client.aclose()
    baseline_correct = sum(x.baseline.correct for x in results)
    recall_correct = sum(x.recall.correct for x in results)
    helpful_gain = sum(
        c.helpful_opportunity and not r.baseline.correct and r.recall.correct
        for c, r in zip(fixture.cases, results, strict=True)
    )
    safety_regressions = sum(
        (not c.helpful_opportunity) and r.baseline.correct and not r.recall.correct
        for c, r in zip(fixture.cases, results, strict=True)
    )
    retrieval_correct = sum(x.retrieval_correct for x in results)
    accepted = sum(
        x.baseline.audit.validation.accepted + x.recall.audit.validation.accepted
        for x in results
    )
    passed = (
        retrieval_correct == 8
        and accepted == 16
        and helpful_gain >= 1
        and safety_regressions == 0
        and recall_correct > baseline_correct
    )
    metrics = RealMemoryMetrics(
        baseline_correct=baseline_correct,
        recall_correct=recall_correct,
        helpful_gain=helpful_gain,
        safety_regressions=safety_regressions,
        retrieval_correct=retrieval_correct,
        accepted_calls=accepted,
        gate_passed=passed,
    )
    usage = Usage()
    for result in results:
        usage = usage.plus(result.baseline.usage).plus(result.recall.usage)
    return RealMemoryRun(
        source_commit=source_commit,
        fixture_sha256=pin.fixture_raw_sha256,
        pin_sha256=_sha256(PIN_PATH),
        embedding_artifact_sha256=EMBEDDING_ARTIFACT_SHA256,
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
        run_real_memory(
            artifact_root=args.artifact_root,
            embedding_snapshot=args.embedding_snapshot,
            api_key=args.api_key,
            source_commit=args.source_commit,
        )
    )
    output = args.output.resolve()
    allowed = (REPO_ROOT / "results/runs/local/openmemory_vllm_v6").resolve()
    if not output.is_relative_to(allowed) or output.suffix != ".json":
        raise ValueError("v6 output must be JSON under its run directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2) + "\n")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
