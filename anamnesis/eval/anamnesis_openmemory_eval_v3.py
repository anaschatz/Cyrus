"""Single-attempt immediate-action OpenMemory recall diagnostic v3."""

from __future__ import annotations

import hashlib
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from anamnesis.cli import _verify_git_state
from anamnesis.local_runtime import LOCAL_W3_M2_OLLAMA_MODEL, LocalInspectDecisionModel
from anamnesis.openmemory_diagnostic import (
    OpenMemoryPairedRun,
    build_openmemory_immediate_decision_prompt,
    load_openmemory_diagnostic,
    openmemory_diagnostic_sha256,
    openmemory_immediate_decision_contract,
    run_openmemory_decision_diagnostic,
)
from eval.anamnesis_local_eval import (
    ACTIVE_LOCAL_W3_M2_PRICING_SHA256,
    _require_ollama_models_dir,
    _verify_installed_w3_m2_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "eval" / "openmemory" / "decision_diagnostic.v3.json"
ARTIFACT_RAW_SHA256 = "9bc0e73e8b7b2299ea83ff630379447c9acfe83424d588cf612fb99de76f2cd9"
ARTIFACT_CANONICAL_SHA256 = (
    "36ed70646b4f9f0fc78605d148599cc75dac8691d47ef8b531f722f2a73fb146"
)
DECISION_CONTRACT_SHA256 = (
    "1505cfc4df8be3812d6e7f0ef53a1245d2ec82e3865b0e709476f9001d21754e"
)
TASK_VERSION = "openmemory-immediate-action-diagnostic.local.v0.3"
TASK_PURPOSE = "paired_openmemory_immediate_action_diagnostic_v3"
SAMPLE_ID = "openmemory-immediate-action-diagnostic-v3"
STORE_KEY = "openmemory_immediate_action_diagnostic_v3"
TRANSPORT_FIELD = "reasoning_effort=none"


def _load_frozen_artifact():
    if hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() != ARTIFACT_RAW_SHA256:
        raise ValueError("OpenMemory diagnostic v3 bytes differ from the frozen pin")
    artifact = load_openmemory_diagnostic(ARTIFACT_PATH)
    if openmemory_diagnostic_sha256(artifact) != ARTIFACT_CANONICAL_SHA256:
        raise ValueError(
            "OpenMemory diagnostic v3 semantics differ from the frozen pin"
        )
    return artifact


def _verify_decision_contract() -> None:
    actual = hashlib.sha256(
        openmemory_immediate_decision_contract().encode()
    ).hexdigest()
    if actual != DECISION_CONTRACT_SHA256:
        raise ValueError("OpenMemory immediate-action contract differs from its pin")


@solver
def openmemory_immediate_action_v3_solver() -> Solver:
    artifact = _load_frozen_artifact()
    _verify_decision_contract()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if str(state.model) != LOCAL_W3_M2_OLLAMA_MODEL:
            raise ValueError(
                "OpenMemory diagnostic v3 requires the pinned local 9B model"
            )
        model = LocalInspectDecisionModel(state, generate)
        result = await run_openmemory_decision_diagnostic(
            artifact,
            model=model,
            prompt_builder=build_openmemory_immediate_decision_prompt,
        )
        state = model.state
        serialized = result.model_dump(mode="json")
        state.metadata[STORE_KEY] = serialized
        state.store.set(STORE_KEY, serialized)
        state.output = ModelOutput.from_content(
            model=model.name,
            content=result.model_dump_json(),
        )
        return state

    return solve


@scorer(metrics=[])
def openmemory_immediate_action_v3_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        result = OpenMemoryPairedRun.model_validate_json(state.output.completion)
        raw_complete = all(
            not call.parse_error
            and call.raw_completion is not None
            and call.usage_complete
            and call.cost_complete
            and call.usage.cost_usd == 0.0
            for call in result.calls
        )
        passed = result.metrics.gate_passed and raw_complete
        return Score(
            value=1 if passed else 0,
            answer="pass" if passed else "fail",
            explanation=(
                "v3 paired immediate-action recall gate and accounting passed"
                if passed
                else "v3 paired immediate-action recall gate or accounting failed"
            ),
        )

    return score


@task
def local_openmemory_immediate_action_diagnostic_v3(
    ollama_models_dir: str | None = None,
    source_commit: str | None = None,
    seed: int = 101,
) -> Task:
    """Run the frozen fresh 8x2 v3 matrix once, without retry or repair."""

    if seed != 101:
        raise ValueError("OpenMemory diagnostic v3 requires seed 101 exactly")
    if source_commit is None:
        raise ValueError("OpenMemory diagnostic v3 requires an exact source_commit")
    _verify_git_state(source_commit)
    models_dir = _require_ollama_models_dir(ollama_models_dir)
    verified_model_bytes = _verify_installed_w3_m2_model(models_dir)
    _load_frozen_artifact()
    _verify_decision_contract()
    return Task(
        dataset=[
            Sample(
                id=SAMPLE_ID,
                input=(
                    "Run the frozen paired OpenMemory immediate-action v3 "
                    "diagnostic once."
                ),
                target="pass",
            )
        ],
        solver=openmemory_immediate_action_v3_solver(),
        scorer=openmemory_immediate_action_v3_scorer(),
        config=GenerateConfig(
            temperature=0.0,
            seed=seed,
            cache=False,
            max_retries=0,
            max_connections=1,
            adaptive_connections=False,
            extra_body={"reasoning_effort": "none"},
        ),
        version=TASK_VERSION,
        metadata={
            "purpose": TASK_PURPOSE,
            "hypothesis_test_eligible": False,
            "diagnostic_only": True,
            "source_commit": source_commit,
            "artifact_raw_sha256": ARTIFACT_RAW_SHA256,
            "artifact_canonical_sha256": ARTIFACT_CANONICAL_SHA256,
            "decision_contract_sha256": DECISION_CONTRACT_SHA256,
            "pricing_config_sha256": ACTIVE_LOCAL_W3_M2_PRICING_SHA256,
            "verified_model_bytes": verified_model_bytes,
            "call_order": "per_case_baseline_then_recall",
            "expected_model_calls": 16,
            "openmemory_online_writes": False,
            "openmemory_usage_complete": False,
            "transport_field": TRANSPORT_FIELD,
            "intervention": "dedicated_immediate_action_decision_contract",
            "fresh_case_version": "v3",
        },
    )


__all__ = [
    "local_openmemory_immediate_action_diagnostic_v3",
    "openmemory_immediate_action_v3_scorer",
    "openmemory_immediate_action_v3_solver",
]
