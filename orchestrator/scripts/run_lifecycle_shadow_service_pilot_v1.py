#!/usr/bin/env python3
"""Run the frozen lifecycle gate through the complete JARVIS service boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import uuid
from collections import Counter
from collections.abc import AsyncIterator
from pathlib import Path

from jarvis_orchestrator.anamnesis_shadow import (
    DETERMINISTIC_FIRST_LOCAL_LIFECYCLE_WRITER_V3_CONTRACT_SHA256,
    DETERMINISTIC_LIFECYCLE_GATE_V3_CONTRACT_SHA256,
    LOCAL_LIFECYCLE_WRITER_CONTRACT_SHA256,
    AnamnesisShadowAdapter,
    DeterministicFirstLifecycleWriter,
    DeterministicLifecycleGateV3,
    LifecycleDecision,
    LifecycleOperation,
    reconstruct_active_shadow_memories,
)
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import ModelChunk, ModelUsage, TaskState
from jarvis_orchestrator.model import BudgetedModelPort
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.store import OrchestratorStore


PROTOCOL_SHA256 = "00669e1ff12b84475ad65d5af0bfff49f6152bef7c55309fe83e3747c9c5f012"
DATASET_SHA256 = "320ea83b76bc91d302713846961d8e8b5ce5d47a9abda16698ed90f5f53942ae"
MANIFEST_SHA256 = "c5a24b801b75cae24274a43f6014e3d4538711fa6eabfeabcf012e6dab878de5"
GATE_SHA256 = "18d80ac7e750625cb22990979d64d868b799d4fce59eda6e09c72fa0f8644a33"
WRITER_SHA256 = "fddfab1bccd089b941ae0e8cc2f28a2b65567525cee82c9e55751818deac2b1b"
EXPECTED_SCENARIOS = 8
EXPECTED_TURNS = 30
EXPECTED_FALLBACK_CALLS = 2
EXPECTED_OPERATION_COUNTS = {
    "cancel": 3,
    "create": 7,
    "ignore": 17,
    "replace": 3,
}
EXPECTED_LANGUAGE_COUNTS = {"el": 11, "en": 19}
ROW_KEYS = {"id", "turns"}
TURN_KEYS = {
    "language",
    "text",
    "expected_operation",
    "expected_active_count",
    "expected_fallback",
}


class _StaticAssistantModel:
    def __init__(self) -> None:
        self.requests = []

    async def stream(self, request) -> AsyncIterator[ModelChunk]:
        self.requests.append(request)
        yield ModelChunk("Έτοιμο.", ModelUsage(input_tokens=3, output_tokens=2))


class _OfflineIgnoreFallback:
    """Observe fallback routing without making a lifecycle model call."""

    contract_sha256 = LOCAL_LIFECYCLE_WRITER_CONTRACT_SHA256

    def __init__(self) -> None:
        self.calls = []

    async def decide(self, observation, active_memories) -> LifecycleDecision:
        self.calls.append((observation, active_memories))
        return LifecycleDecision(LifecycleOperation.IGNORE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cases() -> list[dict[str, object]]:
    root = Path(__file__).resolve().parents[1] / "eval"
    protocol = root / "lifecycle_shadow_service_pilot_v1.protocol.json"
    dataset = root / "lifecycle_shadow_service_pilot_v1.jsonl"
    manifest_path = root / "lifecycle_shadow_service_pilot_v1.manifest.json"
    for path, expected in (
        (protocol, PROTOCOL_SHA256),
        (dataset, DATASET_SHA256),
        (manifest_path, MANIFEST_SHA256),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"frozen service-pilot artifact changed: {path.name}")
    manifest = json.loads(manifest_path.read_text())
    if manifest != {
        "category_counts": EXPECTED_OPERATION_COUNTS,
        "dataset_path": "orchestrator/eval/lifecycle_shadow_service_pilot_v1.jsonl",
        "dataset_sha256": DATASET_SHA256,
        "fallback_turns": EXPECTED_FALLBACK_CALLS,
        "gate_contract_sha256": GATE_SHA256,
        "language_counts": EXPECTED_LANGUAGE_COUNTS,
        "protocol_path": (
            "orchestrator/eval/lifecycle_shadow_service_pilot_v1.protocol.json"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "scenario_count": EXPECTED_SCENARIOS,
        "turn_count": EXPECTED_TURNS,
    }:
        raise RuntimeError("service-pilot manifest contract changed")
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    if len(rows) != EXPECTED_SCENARIOS or len({row.get("id") for row in rows}) != len(
        rows
    ):
        raise RuntimeError("service-pilot scenario identity mismatch")
    turns = []
    for row in rows:
        if set(row) != ROW_KEYS or not isinstance(row["turns"], list):
            raise RuntimeError("service-pilot scenario shape is invalid")
        for turn in row["turns"]:
            if not isinstance(turn, dict) or set(turn) != TURN_KEYS:
                raise RuntimeError("service-pilot turn shape is invalid")
            if turn["language"] not in {"en", "el"}:
                raise RuntimeError("service-pilot language is invalid")
            if turn["expected_operation"] not in {"upsert", "cancel", "ignore"}:
                raise RuntimeError("service-pilot operation is invalid")
            if (
                not isinstance(turn["expected_active_count"], int)
                or turn["expected_active_count"] < 0
            ):
                raise RuntimeError("service-pilot active count is invalid")
            turns.append(turn)
    operation_counts = Counter(
        {
            "upsert": 0,
            "cancel": 0,
            "ignore": 0,
        }
    )
    for turn in turns:
        operation_counts[str(turn["expected_operation"])] += 1
    if {
        "create": operation_counts["upsert"] - EXPECTED_OPERATION_COUNTS["replace"],
        "replace": EXPECTED_OPERATION_COUNTS["replace"],
        "cancel": operation_counts["cancel"],
        "ignore": operation_counts["ignore"],
    } != EXPECTED_OPERATION_COUNTS:
        raise RuntimeError("service-pilot operation counts changed")
    if Counter(str(turn["language"]) for turn in turns) != EXPECTED_LANGUAGE_COUNTS:
        raise RuntimeError("service-pilot language counts changed")
    if (
        len(turns) != EXPECTED_TURNS
        or sum(bool(turn["expected_fallback"]) for turn in turns)
        != EXPECTED_FALLBACK_CALLS
    ):
        raise RuntimeError("service-pilot turn counts changed")
    return rows


async def _run_scenario(row: dict[str, object]) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temporary:
        store = OrchestratorStore(Path(temporary) / "orchestrator.sqlite3")
        try:
            fallback = _OfflineIgnoreFallback()
            writer = DeterministicFirstLifecycleWriter(
                fallback,
                gate=DeterministicLifecycleGateV3(),
            )
            assistant = _StaticAssistantModel()
            config = OrchestratorConfig.from_test_mapping(
                {
                    "database_path": str(store.path),
                    "control_bearer_token": "control-private",
                    "inference_bearer_token": "inference-private",
                    "memory_shadow_enabled": True,
                }
            )
            service = OrchestratorService(
                config,
                store,
                BudgetedModelPort(store, assistant),
                memory=AnamnesisShadowAdapter(store, writer),
            )
            failures: list[dict[str, object]] = []
            conversation_id = None
            fallback_before = 0
            for index, turn in enumerate(row["turns"]):
                namespace = uuid.UUID("604af2b7-ccae-42f2-a95b-0c106b123f93")
                event_id = str(uuid.uuid5(namespace, f"{row['id']}:{index}:event"))
                turn_id = str(uuid.uuid5(namespace, f"{row['id']}:{index}:turn"))
                admitted = store.admit_transcript(
                    event_id=event_id,
                    turn_id=turn_id,
                    payload={
                        "text": turn["text"],
                        "language": {"code": turn["language"], "score": 1.0},
                    },
                ).task
                assert admitted is not None
                conversation_id = admitted.conversation_id
                completed = await service.execute_task(admitted.task_id)
                audit = store.memory_shadow_audit(admitted.task_id)
                active = reconstruct_active_shadow_memories(
                    store.memory_shadow_history(admitted.conversation_id)
                )
                fallback_after = len(fallback.calls)
                checks = {
                    "completed": completed.state is TaskState.COMPLETED,
                    "audit_present": audit is not None,
                    "operation": (
                        audit is not None
                        and audit["event_type"] == "memory.shadow.proposed"
                        and audit["payload"]["proposal"]["operation"]
                        == turn["expected_operation"]
                    ),
                    "active_count": len(active) == turn["expected_active_count"],
                    "fallback": (
                        fallback_after - fallback_before
                        == int(bool(turn["expected_fallback"]))
                    ),
                    "non_authoritative": (
                        audit is not None
                        and audit["payload"]["authoritative_state_changed"] is False
                    ),
                    "content_free_audit": (
                        audit is not None
                        and str(turn["text"]) not in json.dumps(audit, ensure_ascii=False)
                    ),
                    "authoritative_records_zero": (
                        store.memory_record_count(admitted.conversation_id) == 0
                    ),
                    "no_recall_prompt": (
                        "Retrospective memory JSON"
                        not in str(assistant.requests[-1].messages)
                    ),
                }
                failed = sorted(name for name, passed in checks.items() if not passed)
                if failed:
                    failures.append(
                        {
                            "turn_index": index,
                            "failed_checks": failed,
                        }
                    )
                fallback_before = fallback_after
            assert conversation_id is not None
            canonical = store.canonical_conversation(conversation_id)
            if len(canonical.turns) != len(row["turns"]):
                failures.append(
                    {
                        "turn_index": None,
                        "failed_checks": ["canonical_turn_count"],
                    }
                )
            return {
                "id": row["id"],
                "passed": not failures,
                "turns": len(row["turns"]),
                "fallback_calls": len(fallback.calls),
                "assistant_model_calls": len(assistant.requests),
                "final_active_count": len(
                    reconstruct_active_shadow_memories(
                        store.memory_shadow_history(conversation_id)
                    )
                ),
                "authoritative_records": store.memory_record_count(conversation_id),
                "failures": failures,
            }
        finally:
            store.close()


def evaluate() -> dict[str, object]:
    if DETERMINISTIC_LIFECYCLE_GATE_V3_CONTRACT_SHA256 != GATE_SHA256:
        raise RuntimeError("lifecycle gate v3 contract changed")
    if DETERMINISTIC_FIRST_LOCAL_LIFECYCLE_WRITER_V3_CONTRACT_SHA256 != WRITER_SHA256:
        raise RuntimeError("deterministic-first v3 writer contract changed")
    scenarios = asyncio.run(_evaluate_all(_load_cases()))
    passed = sum(bool(item["passed"]) for item in scenarios)
    fallback_calls = sum(int(item["fallback_calls"]) for item in scenarios)
    assistant_calls = sum(int(item["assistant_model_calls"]) for item in scenarios)
    authoritative_records = sum(
        int(item["authoritative_records"]) for item in scenarios
    )
    promotion_passed = (
        passed == EXPECTED_SCENARIOS
        and fallback_calls == EXPECTED_FALLBACK_CALLS
        and assistant_calls == EXPECTED_TURNS
        and authoritative_records == 0
    )
    return {
        "diagnostic": "jarvis.lifecycle-shadow-service-pilot.v1",
        "run_date": "2026-08-10",
        "hypothesis_test_eligible": False,
        "authoritative_memory_enabled": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset_sha256": DATASET_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "gate_contract_sha256": GATE_SHA256,
        "writer_contract_sha256": WRITER_SHA256,
        "scenario_passed": passed,
        "scenario_total": EXPECTED_SCENARIOS,
        "turn_passed": sum(
            int(item["turns"]) for item in scenarios if item["passed"]
        ),
        "turn_total": EXPECTED_TURNS,
        "offline_fallback_calls": fallback_calls,
        "assistant_stub_calls": assistant_calls,
        "lifecycle_model_calls": 0,
        "provider_api_cost_usd": 0.0,
        "authoritative_memory_records": authoritative_records,
        "failures": [
            {"id": item["id"], "failures": item["failures"]}
            for item in scenarios
            if not item["passed"]
        ],
        "promotion_decision": (
            "service_shadow_pilot_pass"
            if promotion_passed
            else "service_shadow_pilot_rejected"
        ),
    }


async def _evaluate_all(rows):
    return [await _run_scenario(row) for row in rows]


def main() -> int:
    result = evaluate()
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if result["promotion_decision"] == "service_shadow_pilot_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
