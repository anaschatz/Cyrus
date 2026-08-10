"""Single-attempt no-thinking OpenMemory recall diagnostic v2."""

from __future__ import annotations

import hashlib
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from anamnesis.cli import _verify_git_state
from anamnesis.local_runtime import (
    LOCAL_W3_M2_OLLAMA_MODEL,
    LocalInspectDecisionModel,
    build_local_decision_prompt,
)
from anamnesis.openmemory_diagnostic import (
    OpenMemoryPairedRun,
    load_openmemory_diagnostic,
    openmemory_diagnostic_sha256,
    run_openmemory_decision_diagnostic,
)
from eval.anamnesis_local_eval import (
    ACTIVE_LOCAL_W3_M2_PRICING_SHA256,
    _require_ollama_models_dir,
    _verify_installed_w3_m2_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "eval" / "openmemory" / "decision_diagnostic.v2.json"
ARTIFACT_RAW_SHA256 = "18d69eec94c35c2b750d2ad75f03db8056881405aaeb7a2838fb36d26593de20"
ARTIFACT_CANONICAL_SHA256 = (
    "7ce91e19d9ca13e6244ea5917c7a3a4a8e499af458b534f90127abedd2bcea61"
)
TASK_VERSION = "openmemory-decision-diagnostic.local.v0.2"
TASK_PURPOSE = "paired_openmemory_recall_decision_diagnostic_v2"
SAMPLE_ID = "openmemory-decision-diagnostic-v2"
STORE_KEY = "openmemory_decision_diagnostic_v2"
TRANSPORT_FIELD = "reasoning_effort=none"


def _load_frozen_artifact():
    if hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() != ARTIFACT_RAW_SHA256:
        raise ValueError("OpenMemory diagnostic v2 bytes differ from the frozen pin")
    artifact = load_openmemory_diagnostic(ARTIFACT_PATH)
    if openmemory_diagnostic_sha256(artifact) != ARTIFACT_CANONICAL_SHA256:
        raise ValueError(
            "OpenMemory diagnostic v2 semantics differ from the frozen pin"
        )
    return artifact


@solver
def openmemory_decision_diagnostic_v2_solver() -> Solver:
    artifact = _load_frozen_artifact()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if str(state.model) != LOCAL_W3_M2_OLLAMA_MODEL:
            raise ValueError(
                "OpenMemory diagnostic v2 requires the pinned local 9B model"
            )
        model = LocalInspectDecisionModel(state, generate)
        result = await run_openmemory_decision_diagnostic(
            artifact,
            model=model,
            prompt_builder=build_local_decision_prompt,
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
def openmemory_decision_diagnostic_v2_scorer() -> Scorer:
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
                "v2 paired recall gate and raw local-model accounting passed"
                if passed
                else "v2 paired recall gate or raw local-model accounting failed"
            ),
        )

    return score


@task
def local_openmemory_decision_diagnostic_v2(
    ollama_models_dir: str | None = None,
    source_commit: str | None = None,
    seed: int = 101,
) -> Task:
    """Run the frozen fresh 8x2 v2 matrix once, without retry or repair."""

    if seed != 101:
        raise ValueError("OpenMemory diagnostic v2 requires seed 101 exactly")
    if source_commit is None:
        raise ValueError("OpenMemory diagnostic v2 requires an exact source_commit")
    _verify_git_state(source_commit)
    models_dir = _require_ollama_models_dir(ollama_models_dir)
    verified_model_bytes = _verify_installed_w3_m2_model(models_dir)
    _load_frozen_artifact()
    return Task(
        dataset=[
            Sample(
                id=SAMPLE_ID,
                input="Run the frozen paired OpenMemory decision diagnostic v2 once.",
                target="pass",
            )
        ],
        solver=openmemory_decision_diagnostic_v2_solver(),
        scorer=openmemory_decision_diagnostic_v2_scorer(),
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
            "pricing_config_sha256": ACTIVE_LOCAL_W3_M2_PRICING_SHA256,
            "verified_model_bytes": verified_model_bytes,
            "call_order": "per_case_baseline_then_recall",
            "expected_model_calls": 16,
            "openmemory_online_writes": False,
            "openmemory_usage_complete": False,
            "transport_field": TRANSPORT_FIELD,
            "parent_cell": "openmemory-decision-diagnostic-v1",
            "fresh_case_version": "v2",
        },
    )


__all__ = [
    "local_openmemory_decision_diagnostic_v2",
    "openmemory_decision_diagnostic_v2_scorer",
    "openmemory_decision_diagnostic_v2_solver",
]
