"""Inspect AI task entry points for three baselines and Anamnesis v0."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, json_dataset
from inspect_ai.model import GenerateConfig

from anamnesis.cli import _verify_file_artifact, _verify_git_state
from anamnesis.experiment import ExperimentManifest
from anamnesis.inspect_adapter import (
    SystemName,
    hosted_warmup_prompt_sha256,
    hosted_warmup_schema_sha256,
    model_preflight_sample,
    model_preflight_scorer,
    model_preflight_solver,
    scenario_record_to_sample,
    scenario_run_scorer,
    scenario_solver,
)
from anamnesis.io import (
    dataset_sha256,
    load_scenarios,
    require_preregistered_final_dataset,
)
from anamnesis.preflight import (
    MODEL_PREFLIGHT_PURPOSE,
    MODEL_PREFLIGHT_TASK_VERSION,
    validate_model_preflight_artifact,
)
from anamnesis.prompts import memory_compiler_contract, prompt_contract

SCENARIO_DIRECTORY = Path(__file__).resolve().parent / "scenarios"
DatasetSplit = Literal["smoke", "development", "all"]
DATASETS = {
    "smoke": ("smoke.jsonl", "anamnesis-smoke-v0"),
    "development": ("dev.jsonl", "anamnesis-development-v0"),
    "all": ("all.jsonl", "anamnesis-all-v0"),
}


@task
def model_preflight(seed: int = 101) -> Task:
    """Synthetic hosted-model compatibility gate, not an evaluation run."""

    return Task(
        dataset=[model_preflight_sample()],
        solver=model_preflight_solver(),
        scorer=model_preflight_scorer(),
        config=GenerateConfig(
            temperature=0.0,
            seed=seed,
            cache=False,
            max_retries=0,
            max_connections=1,
            adaptive_connections=False,
        ),
        version=MODEL_PREFLIGHT_TASK_VERSION,
        metadata={"purpose": MODEL_PREFLIGHT_PURPOSE},
    )


def _scenario_dataset(
    split: DatasetSplit,
    *,
    manifest_path: str | None = None,
    final_manifest: ExperimentManifest | None = None,
) -> Dataset:
    filename, name = DATASETS[split]
    dataset_path = SCENARIO_DIRECTORY / filename
    if split == "all":
        final_manifest = final_manifest or _validated_final_manifest(
            dataset_path, manifest_path
        )
    return json_dataset(
        str(dataset_path),
        sample_fields=scenario_record_to_sample,
        name=name,
    )


def _validated_final_manifest(
    dataset_path: Path,
    manifest_path: str | None,
) -> ExperimentManifest:
    if manifest_path is None:
        raise ValueError("dataset='all' requires a frozen final manifest")
    return _validated_experiment_manifest(
        dataset_path,
        manifest_path,
        expected_phase="final",
    )


def _validated_experiment_manifest(
    dataset_path: Path,
    manifest_path: str | None,
    *,
    expected_phase: Literal["baseline", "final"],
) -> ExperimentManifest:
    if manifest_path is None:
        raise ValueError(f"measured {expected_phase} task requires a frozen manifest")
    manifest = ExperimentManifest.model_validate_json(
        Path(manifest_path).read_text(encoding="utf-8")
    )
    if manifest.status != "frozen" or manifest.phase != expected_phase:
        raise ValueError(f"task requires a frozen {expected_phase} manifest")
    if Path(manifest.dataset.path).resolve() != dataset_path.resolve():
        raise ValueError("final manifest points to a different dataset")
    for name, artifact in (
        ("model.pricing", manifest.model.pricing),
        ("model.preflight", manifest.model.preflight),
        ("dependency_lock", manifest.dependency_lock),
        ("research_contract", manifest.research_contract),
        ("architecture_contract", manifest.architecture_contract),
    ):
        _verify_file_artifact(name, artifact)
    validate_model_preflight_artifact(
        manifest.model.preflight,
        model_name=manifest.model.snapshot or "",
        pricing=manifest.model.pricing,
    )
    _verify_git_state(manifest.git_commit)
    decision_hash = hashlib.sha256(prompt_contract().encode()).hexdigest()
    if manifest.decision_prompt_sha256 != decision_hash:
        raise ValueError("final manifest decision prompt differs from runtime")
    if expected_phase == "final":
        compiler_hash = hashlib.sha256(memory_compiler_contract().encode()).hexdigest()
        if manifest.memory_compiler_sha256 != compiler_hash:
            raise ValueError("final manifest memory compiler differs from runtime")
    scenarios = load_scenarios(dataset_path)
    if expected_phase == "final":
        require_preregistered_final_dataset(dataset_path, scenarios)
    if manifest.dataset.sha256 != dataset_sha256(scenarios):
        raise ValueError("final manifest dataset hash does not match all.jsonl")
    return manifest


def _system_task(
    system: SystemName,
    *,
    seed: int | None = None,
    repetition: int = 1,
    top_k: int = 5,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    embedding_repository: str = "qdrant/bge-small-en-v1.5-onnx-q",
    embedding_revision: str | None = None,
    embedding_snapshot_path: str | None = None,
    dataset: DatasetSplit = "development",
    manifest: str | None = None,
) -> Task:
    dataset_path = SCENARIO_DIRECTORY / DATASETS[dataset][0]
    frozen_manifest: ExperimentManifest | None = None
    if dataset == "all":
        frozen_manifest = _validated_final_manifest(dataset_path, manifest)
    elif manifest is not None:
        if dataset != "development":
            raise ValueError("frozen baseline manifests require dataset='development'")
        frozen_manifest = _validated_experiment_manifest(
            dataset_path,
            manifest,
            expected_phase="baseline",
        )
    if frozen_manifest is not None:
        if repetition < 1 or repetition > frozen_manifest.execution.repetitions:
            raise ValueError("repetition is outside the frozen manifest")
        expected_seed = frozen_manifest.execution.seeds[repetition - 1]
        if seed != expected_seed:
            raise ValueError(f"repetition {repetition} requires seed {expected_seed}")
        if system == "vector_rag":
            if (
                embedding_model != frozen_manifest.embedding.model
                or embedding_repository != frozen_manifest.embedding.repository
                or embedding_revision != frozen_manifest.embedding.revision
                or top_k != frozen_manifest.embedding.top_k
            ):
                raise ValueError(
                    "vector embedding pin differs from the frozen manifest"
                )
            embedding_snapshot_path = _require_local_embedding_snapshot(
                embedding_snapshot_path
            )
    manifest_sha256: str | None = None
    if frozen_manifest is not None and manifest is not None:
        manifest_bytes = Path(manifest).read_bytes()
        if ExperimentManifest.model_validate_json(manifest_bytes) != frozen_manifest:
            raise ValueError("frozen manifest changed during task construction")
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    return Task(
        dataset=_scenario_dataset(
            dataset,
            manifest_path=manifest,
            final_manifest=frozen_manifest if dataset == "all" else None,
        ),
        solver=scenario_solver(
            system,
            repetition=repetition,
            seed=seed,
            top_k=top_k,
            embedding_model=embedding_model,
            embedding_repository=embedding_repository,
            embedding_revision=embedding_revision,
            embedding_snapshot_path=embedding_snapshot_path,
            expected_model=(
                frozen_manifest.model.snapshot if frozen_manifest is not None else None
            ),
            expected_system_config_sha256=(
                frozen_manifest.system_config_sha256[system]
                if frozen_manifest is not None
                else None
            ),
            expected_embedding_artifact_sha256=(
                frozen_manifest.embedding.artifact_sha256
                if frozen_manifest is not None and system == "vector_rag"
                else None
            ),
            manifest_sha256=manifest_sha256,
            pricing_config_sha256=(
                frozen_manifest.model.pricing.sha256
                if frozen_manifest is not None
                else None
            ),
            pricing_config_path=(
                frozen_manifest.model.pricing.path
                if frozen_manifest is not None
                else None
            ),
        ),
        scorer=scenario_run_scorer(),
        config=GenerateConfig(
            temperature=0.0,
            seed=seed,
            cache=False,
            max_retries=0,
            max_connections=1,
            adaptive_connections=False,
        ),
        version="0.3",
        metadata={
            "system": system,
            "dataset": f"eval/scenarios/{dataset_path.name}",
            "dataset_split": dataset,
            "canonical_dataset_sha256": dataset_sha256(load_scenarios(dataset_path)),
            "dataset_file_sha256": hashlib.sha256(
                dataset_path.read_bytes()
            ).hexdigest(),
            "repetition": repetition,
            "manifest_sha256": manifest_sha256,
            "hosted_warmup_prompt_sha256": hosted_warmup_prompt_sha256(),
            "hosted_warmup_schema_sha256": hosted_warmup_schema_sha256(),
            "embedding_model": embedding_model if system == "vector_rag" else None,
            "embedding_repository": (
                embedding_repository if system == "vector_rag" else None
            ),
            "embedding_revision": (
                embedding_revision if system == "vector_rag" else None
            ),
        },
    )


def _require_local_embedding_snapshot(snapshot_path: str | None) -> str:
    """Require an explicit existing local artifact for a frozen vector run."""

    if not isinstance(snapshot_path, str) or not snapshot_path.strip():
        raise ValueError(
            "frozen vector_rag requires an explicit embedding_snapshot_path"
        )
    path = Path(snapshot_path)
    if not path.is_absolute():
        raise ValueError("embedding_snapshot_path must be an absolute local path")
    if not path.is_dir():
        raise ValueError("embedding_snapshot_path is not an existing directory")
    return str(path.resolve())


@task
def no_memory(
    seed: int | None = None,
    repetition: int = 1,
    dataset: DatasetSplit = "development",
    manifest: str | None = None,
) -> Task:
    """No-persistent-memory baseline over development scenarios by default."""

    return _system_task(
        "no_memory",
        seed=seed,
        repetition=repetition,
        dataset=dataset,
        manifest=manifest,
    )


@task
def full_context(
    seed: int | None = None,
    repetition: int = 1,
    dataset: DatasetSplit = "development",
    manifest: str | None = None,
) -> Task:
    """Full-context baseline over development scenarios by default."""

    return _system_task(
        "full_context",
        seed=seed,
        repetition=repetition,
        dataset=dataset,
        manifest=manifest,
    )


@task
def vector_rag(
    seed: int | None = None,
    repetition: int = 1,
    top_k: int = 5,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    embedding_repository: str = "qdrant/bge-small-en-v1.5-onnx-q",
    embedding_revision: str | None = None,
    embedding_snapshot_path: str | None = None,
    dataset: DatasetSplit = "development",
    manifest: str | None = None,
) -> Task:
    """Exact cosine top-k RAG over development scenarios by default."""

    return _system_task(
        "vector_rag",
        seed=seed,
        repetition=repetition,
        top_k=top_k,
        embedding_model=embedding_model,
        embedding_repository=embedding_repository,
        embedding_revision=embedding_revision,
        embedding_snapshot_path=embedding_snapshot_path,
        dataset=dataset,
        manifest=manifest,
    )


@task
def anamnesis(
    seed: int | None = None,
    repetition: int = 1,
    dataset: DatasetSplit = "development",
    manifest: str | None = None,
) -> Task:
    """Structured temporal/prospective memory over development by default."""

    return _system_task(
        "anamnesis",
        seed=seed,
        repetition=repetition,
        dataset=dataset,
        manifest=manifest,
    )
