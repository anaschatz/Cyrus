"""Single-cell local paired decision diagnostic for OpenMemory recall."""

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
ARTIFACT_PATH = REPO_ROOT / "eval" / "openmemory" / "decision_diagnostic.v1.json"
ARTIFACT_RAW_SHA256 = "a1541939dc977ddf233395318ac8470ca17d0bb39ef3284fbd65411edf89e36a"
ARTIFACT_CANONICAL_SHA256 = (
    "b8da030f0e632c5e85523e75ba9ff948c85950435f8d55ca1b0aa3381e830126"
)
TASK_VERSION = "openmemory-decision-diagnostic.local.v0.1"
TASK_PURPOSE = "paired_openmemory_recall_decision_diagnostic"
SAMPLE_ID = "openmemory-decision-diagnostic-v1"
STORE_KEY = "openmemory_decision_diagnostic_v1"


def _load_frozen_artifact():
    if hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() != (ARTIFACT_RAW_SHA256):
        raise ValueError("OpenMemory diagnostic bytes differ from the frozen pin")
    artifact = load_openmemory_diagnostic(ARTIFACT_PATH)
    if openmemory_diagnostic_sha256(artifact) != ARTIFACT_CANONICAL_SHA256:
        raise ValueError("OpenMemory diagnostic semantics differ from the frozen pin")
    return artifact


@solver
def openmemory_decision_diagnostic_solver() -> Solver:
    artifact = _load_frozen_artifact()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if str(state.model) != LOCAL_W3_M2_OLLAMA_MODEL:
            raise ValueError("OpenMemory diagnostic requires the pinned local 9B model")
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
def openmemory_decision_diagnostic_scorer() -> Scorer:
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
                "paired recall gate and all raw local-model accounting passed"
                if passed
                else "paired recall gate or raw local-model accounting failed"
            ),
        )

    return score


@task
def local_openmemory_decision_diagnostic(
    ollama_models_dir: str | None = None,
    source_commit: str | None = None,
    seed: int = 101,
) -> Task:
    """Run one frozen 8x2 diagnostic matrix, without retry or repair."""

    if seed != 101:
        raise ValueError("OpenMemory diagnostic requires seed 101 exactly")
    if source_commit is None:
        raise ValueError("OpenMemory diagnostic requires an exact source_commit")
    _verify_git_state(source_commit)
    models_dir = _require_ollama_models_dir(ollama_models_dir)
    verified_model_bytes = _verify_installed_w3_m2_model(models_dir)
    _load_frozen_artifact()
    return Task(
        dataset=[
            Sample(
                id=SAMPLE_ID,
                input="Run the frozen paired OpenMemory decision diagnostic once.",
                target="pass",
            )
        ],
        solver=openmemory_decision_diagnostic_solver(),
        scorer=openmemory_decision_diagnostic_scorer(),
        config=GenerateConfig(
            temperature=0.0,
            seed=seed,
            cache=False,
            max_retries=0,
            max_connections=1,
            adaptive_connections=False,
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
        },
    )


__all__ = [
    "local_openmemory_decision_diagnostic",
    "openmemory_decision_diagnostic_scorer",
    "openmemory_decision_diagnostic_solver",
]
