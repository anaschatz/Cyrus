"""Inspect tasks for the isolated zero-provider-cost local diagnostic track."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, json_dataset
from inspect_ai.model import GenerateConfig, ModelInfo, set_model_info

from anamnesis.cli import _verify_git_state
from anamnesis.inspect_adapter import (
    scenario_record_to_sample,
    scenario_run_scorer,
)
from anamnesis.io import dataset_sha256, load_scenarios
from anamnesis.local_experiment import (
    LOCAL_MODEL_ARTIFACT_PATH,
    LOCAL_MODEL_ID,
    LOCAL_PRICING_PATH,
    LOCAL_PRICING_SHA256,
    LOCAL_W3_M2_MODEL_ARTIFACT_PATH,
    LOCAL_W3_M2_MODEL_ID,
    LOCAL_W3_M2_PRICING_PATH,
    LOCAL_W3_M2_PRICING_SHA256,
    LOCAL_W3_M2_PROTOCOL_PATH,
    LOCAL_W3_M2_PROTOCOL_SHA256,
    LOCAL_W3_M2_T1_PROTOCOL_PATH,
    LOCAL_W3_M2_T1_PROTOCOL_SHA256,
    LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_PATH,
    LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256,
    LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_PATH,
    LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256,
    LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_PATH,
    LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256,
    LocalExperimentManifest,
    load_ollama_artifact_pin,
    require_local_only_environment,
    validate_zero_api_pricing,
    verify_ollama_artifact,
    verify_static_local_inputs,
)
from anamnesis.local_preflight import (
    validate_local_preflight_artifact,
    validate_local_w2_preflight_artifact,
    validate_local_w3_preflight_artifact,
)
from anamnesis.local_runtime import (
    LOCAL_DECISION_VERSION,
    LOCAL_MODEL_PREFLIGHT_PURPOSE,
    LOCAL_MODEL_PREFLIGHT_TASK_VERSION,
    LOCAL_MODEL_PREFLIGHT_W2_PURPOSE,
    LOCAL_MODEL_PREFLIGHT_W2_TASK_VERSION,
    LOCAL_MODEL_PREFLIGHT_W3_M2_PURPOSE,
    LOCAL_MODEL_PREFLIGHT_W3_M2_T1_PURPOSE,
    LOCAL_MODEL_PREFLIGHT_W3_M2_T1_TASK_VERSION,
    LOCAL_MODEL_PREFLIGHT_W3_M2_TASK_VERSION,
    LOCAL_MODEL_PREFLIGHT_W3_PURPOSE,
    LOCAL_MODEL_PREFLIGHT_W3_TASK_VERSION,
    LOCAL_OLLAMA_CONTEXT_LENGTH,
    LOCAL_SCENARIO_TASK_VERSION,
    LOCAL_SCENARIO_W3_TASK_VERSION,
    LOCAL_W3_M2_OLLAMA_MANIFEST_SHA256,
    LOCAL_ZERO_MODEL_COST,
    LocalSystemName,
    local_decision_prompt_contract,
    local_decision_schema_contract,
    local_memory_compiler_prompt_contract,
    local_memory_compiler_schema_contract,
    local_memory_compiler_w2_prompt_contract,
    local_memory_compiler_w3_prompt_contract,
    local_model_preflight_sample,
    local_model_preflight_scorer,
    local_model_preflight_solver,
    local_model_preflight_w2_sample,
    local_model_preflight_w2_scorer,
    local_model_preflight_w2_solver,
    local_model_preflight_w3_m2_sample,
    local_model_preflight_w3_m2_scorer,
    local_model_preflight_w3_m2_solver,
    local_model_preflight_w3_m2_t1_sample,
    local_model_preflight_w3_m2_t1_scorer,
    local_model_preflight_w3_m2_t1_solver,
    local_model_preflight_w3_sample,
    local_model_preflight_w3_scorer,
    local_model_preflight_w3_solver,
    local_scenario_solver,
    local_system_config_sha256,
)
from anamnesis.local_wire import (
    LOCAL_MEMORY_COMPILER_W2_VERSION,
    LOCAL_MEMORY_COMPILER_W3_VERSION,
)
from anamnesis.oracle import (
    ORACLE_ANNOTATION_POLICY,
    ORACLE_ARTIFACT_PURPOSE,
    ORACLE_COMPILER_VERSION,
    ORACLE_SYSTEM_NAME,
    load_oracle_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIRECTORY = Path(__file__).resolve().parent / "scenarios"
LOCAL_MODEL_MANIFEST_RELATIVE = Path(
    "manifests/registry.ollama.ai/library/qwen3/4b-instruct"
)
LOCAL_W3_M2_MODEL_MANIFEST_RELATIVE = Path(
    "manifests/registry.ollama.ai/library/qwen3.5/9b-q4_K_M"
)

# Inspect's built-in model database does not include this exact Ollama tag.
# Inspect applies --model-cost-config before importing task modules, so that
# flag cannot register an absent Ollama tag. Validate the tracked zero-price
# bytes here, then register the exact model before task/model construction.
ACTIVE_LOCAL_PRICING_SHA256 = validate_zero_api_pricing(
    REPO_ROOT / LOCAL_PRICING_PATH,
    LOCAL_MODEL_ID,
)
if ACTIVE_LOCAL_PRICING_SHA256 != LOCAL_PRICING_SHA256:
    raise ValueError("tracked local pricing bytes differ from the pinned SHA-256")
set_model_info(
    LOCAL_MODEL_ID,
    ModelInfo(
        organization="Qwen/Ollama",
        model="qwen3:4b-instruct",
        snapshot="qwen3:4b-instruct",
        context_length=LOCAL_OLLAMA_CONTEXT_LENGTH,
        output_tokens=LOCAL_OLLAMA_CONTEXT_LENGTH,
        family="qwen3",
        cost=LOCAL_ZERO_MODEL_COST,
    ),
)

ACTIVE_LOCAL_W3_M2_PRICING_SHA256 = validate_zero_api_pricing(
    REPO_ROOT / LOCAL_W3_M2_PRICING_PATH,
    LOCAL_W3_M2_MODEL_ID,
)
if ACTIVE_LOCAL_W3_M2_PRICING_SHA256 != LOCAL_W3_M2_PRICING_SHA256:
    raise ValueError("tracked W3-M2 pricing bytes differ from the pinned SHA-256")
set_model_info(
    LOCAL_W3_M2_MODEL_ID,
    ModelInfo(
        organization="Qwen/Ollama",
        model="qwen3.5:9b-q4_K_M",
        snapshot="qwen3.5:9b-q4_K_M",
        context_length=LOCAL_OLLAMA_CONTEXT_LENGTH,
        output_tokens=LOCAL_OLLAMA_CONTEXT_LENGTH,
        family="qwen35",
        cost=LOCAL_ZERO_MODEL_COST,
    ),
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _scenario_dataset(path: Path, *, name: str) -> Dataset:
    return json_dataset(
        str(path),
        sample_fields=scenario_record_to_sample,
        name=name,
    )


def _require_ollama_models_dir(value: str | None) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("local tasks require an explicit ollama_models_dir")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("ollama_models_dir must be an absolute path")
    if not path.is_dir():
        raise ValueError("ollama_models_dir is not an existing directory")
    return path.resolve()


def _verify_installed_model(models_dir: Path) -> int:
    pin = load_ollama_artifact_pin(REPO_ROOT / LOCAL_MODEL_ARTIFACT_PATH)
    if pin.model != LOCAL_MODEL_ID:
        raise ValueError("tracked Ollama pin identifies a different local model")
    return verify_ollama_artifact(
        pin,
        manifest_path=models_dir / LOCAL_MODEL_MANIFEST_RELATIVE,
        blobs_dir=models_dir / "blobs",
    )


def _verify_installed_w3_m2_model(models_dir: Path) -> int:
    pin = load_ollama_artifact_pin(REPO_ROOT / LOCAL_W3_M2_MODEL_ARTIFACT_PATH)
    if pin.model != LOCAL_W3_M2_MODEL_ID:
        raise ValueError("tracked W3-M2 Ollama pin identifies a different model")
    return verify_ollama_artifact(
        pin,
        manifest_path=models_dir / LOCAL_W3_M2_MODEL_MANIFEST_RELATIVE,
        blobs_dir=models_dir / "blobs",
    )


def _require_local_embedding_snapshot(value: str | None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("local vector_rag requires embedding_snapshot_path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("embedding_snapshot_path must be an absolute path")
    if not path.is_dir():
        raise ValueError("embedding_snapshot_path is not an existing directory")
    return str(path.resolve())


def _repo_artifact_path(relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("local manifest artifact paths must remain repo-relative")
    path = (REPO_ROOT / candidate).resolve()
    if not path.is_relative_to(REPO_ROOT) or not path.is_file():
        raise ValueError(f"local manifest artifact does not exist: {relative}")
    return path


def _require_oracle_annotations_path(
    value: str | None,
    *,
    manifest_path: str,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("local oracle task requires oracle_annotations_path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    provided = candidate.resolve()
    expected = _repo_artifact_path(manifest_path)
    if provided != expected:
        raise ValueError("oracle_annotations_path differs from the frozen manifest pin")
    return expected


def _validated_local_manifest(
    manifest_path: str | None,
    *,
    system: LocalSystemName,
    seed: int | None,
    repetition: int,
    ollama_models_dir: str | None,
    required_phase: str | None = None,
) -> tuple[LocalExperimentManifest, str, Path]:
    if manifest_path is None:
        raise ValueError("local scenario tasks require a frozen local manifest")
    path = Path(manifest_path)
    if not path.is_file():
        raise ValueError("local manifest path does not exist")
    manifest_bytes = path.read_bytes()
    manifest = LocalExperimentManifest.model_validate_json(manifest_bytes)
    if manifest.status != "frozen":
        raise ValueError("local scenario tasks require a frozen local manifest")
    if required_phase is not None and manifest.phase != required_phase:
        raise ValueError(f"local task requires phase {required_phase}")
    if system not in manifest.systems:
        raise ValueError("system is outside the frozen local matrix")
    if repetition < 1 or repetition > manifest.execution.repetitions:
        raise ValueError("repetition is outside the frozen local manifest")
    expected_seed = manifest.execution.seeds[repetition - 1]
    if seed != expected_seed:
        raise ValueError(f"repetition {repetition} requires seed {expected_seed}")

    require_local_only_environment(os.environ)
    verify_static_local_inputs(manifest, repo_root=REPO_ROOT)
    models_dir = _require_ollama_models_dir(ollama_models_dir)
    _verify_installed_model(models_dir)
    preflight_path = _repo_artifact_path(manifest.model.preflight.path)
    if manifest.git_commit is None or manifest.model.pricing.sha256 is None:
        raise ValueError("frozen local manifest is missing git/pricing pins")
    if manifest.phase == "writer_diagnostic_w3":
        if manifest.preflight_fixture is None or manifest.preflight_protocol is None:
            raise ValueError("frozen W3 manifest omitted preflight fixture/protocol")
        fixture_path = _repo_artifact_path(manifest.preflight_fixture.path)
        protocol_path = _repo_artifact_path(manifest.preflight_protocol.path)
        validate_local_w3_preflight_artifact(
            manifest.model.preflight.model_copy(update={"path": str(preflight_path)}),
            fixture_artifact=manifest.preflight_fixture.model_copy(
                update={"path": str(fixture_path)}
            ),
            protocol_artifact=manifest.preflight_protocol.model_copy(
                update={"path": str(protocol_path)}
            ),
            expected_git_commit=manifest.git_commit,
            expected_pricing_sha256=manifest.model.pricing.sha256,
            seed=expected_seed,
        )
    elif manifest.phase == "writer_diagnostic_w2":
        if manifest.preflight_fixture is None:
            raise ValueError("frozen W2 manifest omitted preflight_fixture")
        fixture_path = _repo_artifact_path(manifest.preflight_fixture.path)
        validate_local_w2_preflight_artifact(
            manifest.model.preflight.model_copy(update={"path": str(preflight_path)}),
            fixture_artifact=manifest.preflight_fixture.model_copy(
                update={"path": str(fixture_path)}
            ),
            expected_git_commit=manifest.git_commit,
            expected_pricing_sha256=manifest.model.pricing.sha256,
            seed=expected_seed,
        )
    else:
        validate_local_preflight_artifact(
            manifest.model.preflight.model_copy(update={"path": str(preflight_path)}),
            expected_git_commit=manifest.git_commit,
            expected_pricing_sha256=manifest.model.pricing.sha256,
            seed=expected_seed,
        )
    _verify_git_state(manifest.git_commit)

    expected_hashes = {
        "decision_prompt_sha256": _sha256_text(local_decision_prompt_contract()),
        "decision_schema_sha256": _sha256_text(local_decision_schema_contract()),
    }
    if "anamnesis" in manifest.systems:
        compiler_prompt_contract_by_phase = {
            "smoke": local_memory_compiler_prompt_contract,
            "baseline": local_memory_compiler_prompt_contract,
            "oracle_smoke": local_memory_compiler_prompt_contract,
            "writer_diagnostic": local_memory_compiler_prompt_contract,
            "writer_diagnostic_w2": local_memory_compiler_w2_prompt_contract,
            "writer_diagnostic_w3": local_memory_compiler_w3_prompt_contract,
        }
        expected_hashes.update(
            memory_compiler_prompt_sha256=_sha256_text(
                compiler_prompt_contract_by_phase[manifest.phase]()
            ),
            memory_compiler_schema_sha256=_sha256_text(
                local_memory_compiler_schema_contract()
            ),
        )
    for field_name, expected in expected_hashes.items():
        if getattr(manifest, field_name) != expected:
            raise ValueError(f"local manifest {field_name} differs from runtime")

    compiler_variant_by_phase = {
        "smoke": "w1",
        "baseline": "w1",
        "oracle_smoke": "w1",
        "writer_diagnostic": "w1",
        "writer_diagnostic_w2": "w2",
        "writer_diagnostic_w3": "w3",
    }
    expected_system_hashes = {
        name: local_system_config_sha256(
            system=name,  # type: ignore[arg-type]
            top_k=manifest.embedding.top_k,
            embedding_model=manifest.embedding.model,
            embedding_repository=manifest.embedding.repository,
            embedding_revision=manifest.embedding.revision,
            embedding_artifact_sha256=(
                manifest.embedding.artifact_sha256 if name == "vector_rag" else None
            ),
            pricing_config_sha256=manifest.model.pricing.sha256,
            oracle_annotations_sha256=(
                manifest.oracle_annotations.sha256
                if name == ORACLE_SYSTEM_NAME
                and manifest.oracle_annotations is not None
                else None
            ),
            compiler_prompt_variant=compiler_variant_by_phase[manifest.phase],  # type: ignore[arg-type]
            w3_preflight_fixture_sha256=(
                manifest.preflight_fixture.sha256
                if manifest.phase == "writer_diagnostic_w3"
                and manifest.preflight_fixture is not None
                else None
            ),
            w3_preflight_protocol_sha256=(
                manifest.preflight_protocol.sha256
                if manifest.phase == "writer_diagnostic_w3"
                and manifest.preflight_protocol is not None
                else None
            ),
            w3_dataset_sha256=(
                manifest.dataset.sha256
                if manifest.phase == "writer_diagnostic_w3"
                else None
            ),
            w3_reference_sha256=(
                manifest.writer_reference.sha256
                if manifest.phase == "writer_diagnostic_w3"
                and manifest.writer_reference is not None
                else None
            ),
        )
        for name in manifest.systems
    }
    if manifest.system_config_sha256 != expected_system_hashes:
        raise ValueError("local manifest system hashes differ from runtime")

    dataset_path = _repo_artifact_path(manifest.dataset.path)
    scenarios = load_scenarios(dataset_path)
    if len(scenarios) != manifest.scenario_count:
        raise ValueError("local dataset count differs from the frozen manifest")
    if manifest.phase == "writer_diagnostic_w3":
        checkpoint_count = sum(len(scenario.events) for scenario in scenarios)
        compiler_call_count = sum(
            event.kind != "clock_tick"
            for scenario in scenarios
            for event in scenario.events
        )
        if checkpoint_count != 62 or compiler_call_count != 39:
            raise ValueError("local W3 dataset must contain exactly 62/39 calls")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    return manifest, manifest_sha256, dataset_path


@task
def local_model_preflight(seed: int = 101) -> Task:
    """Synthetic local semantic gate; never a scenario evaluation."""

    return Task(
        dataset=[local_model_preflight_sample()],
        solver=local_model_preflight_solver(),
        scorer=local_model_preflight_scorer(),
        config=GenerateConfig(
            temperature=0.0,
            seed=seed,
            cache=False,
            max_retries=0,
            max_connections=1,
            adaptive_connections=False,
        ),
        version=LOCAL_MODEL_PREFLIGHT_TASK_VERSION,
        metadata={
            "purpose": LOCAL_MODEL_PREFLIGHT_PURPOSE,
            "track": "local_zero_api_cost",
            "hypothesis_test_eligible": False,
            "pricing_config_sha256": ACTIVE_LOCAL_PRICING_SHA256,
        },
    )


@task
def local_model_preflight_w2(seed: int = 101) -> Task:
    """Frozen four-call W2 semantic gate; never a scenario evaluation."""

    if seed != 101:
        raise ValueError("local W2 preflight requires seed 101 exactly")
    fixture_path = _repo_artifact_path(LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_PATH)
    if hashlib.sha256(fixture_path.read_bytes()).hexdigest() != (
        LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256
    ):
        raise ValueError("tracked W2 preflight fixture differs from its pin")
    return Task(
        dataset=[local_model_preflight_w2_sample()],
        solver=local_model_preflight_w2_solver(str(fixture_path)),
        scorer=local_model_preflight_w2_scorer(),
        config=GenerateConfig(
            temperature=0.0,
            seed=seed,
            cache=False,
            max_retries=0,
            max_connections=1,
            adaptive_connections=False,
        ),
        version=LOCAL_MODEL_PREFLIGHT_W2_TASK_VERSION,
        metadata={
            "purpose": LOCAL_MODEL_PREFLIGHT_W2_PURPOSE,
            "track": "local_zero_api_cost",
            "hypothesis_test_eligible": False,
            "pricing_config_sha256": ACTIVE_LOCAL_PRICING_SHA256,
            "preflight_fixture_sha256": (LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256),
        },
    )


@task
def local_model_preflight_w3(seed: int = 101) -> Task:
    """Frozen nine-call W3 semantic gate; never a scenario evaluation."""

    if seed != 101:
        raise ValueError("local W3 preflight requires seed 101 exactly")
    fixture_path = _repo_artifact_path(LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_PATH)
    protocol_path = _repo_artifact_path(LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_PATH)
    if hashlib.sha256(fixture_path.read_bytes()).hexdigest() != (
        LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256
    ):
        raise ValueError("tracked W3 preflight fixture differs from its pin")
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != (
        LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256
    ):
        raise ValueError("tracked W3 preflight protocol differs from its pin")
    return Task(
        dataset=[local_model_preflight_w3_sample()],
        solver=local_model_preflight_w3_solver(str(fixture_path)),
        scorer=local_model_preflight_w3_scorer(),
        config=GenerateConfig(
            temperature=0.0,
            seed=seed,
            cache=False,
            max_retries=0,
            max_connections=1,
            adaptive_connections=False,
        ),
        version=LOCAL_MODEL_PREFLIGHT_W3_TASK_VERSION,
        metadata={
            "purpose": LOCAL_MODEL_PREFLIGHT_W3_PURPOSE,
            "track": "local_zero_api_cost",
            "hypothesis_test_eligible": False,
            "pricing_config_sha256": ACTIVE_LOCAL_PRICING_SHA256,
            "preflight_fixture_sha256": LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256,
            "preflight_protocol_sha256": LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256,
        },
    )


@task
def local_model_preflight_w3_m2(
    ollama_models_dir: str | None = None,
    seed: int = 101,
) -> Task:
    """One W3 model-only gate for the byte-pinned Qwen3.5 9B artifact."""

    if seed != 101:
        raise ValueError("local W3-M2 preflight requires seed 101 exactly")
    models_dir = _require_ollama_models_dir(ollama_models_dir)
    _verify_installed_w3_m2_model(models_dir)
    fixture_path = _repo_artifact_path(LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_PATH)
    protocol_path = _repo_artifact_path(LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_PATH)
    m2_protocol_path = _repo_artifact_path(LOCAL_W3_M2_PROTOCOL_PATH)
    if hashlib.sha256(fixture_path.read_bytes()).hexdigest() != (
        LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256
    ):
        raise ValueError("tracked W3 fixture differs from its frozen pin")
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != (
        LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256
    ):
        raise ValueError("tracked W3 protocol differs from its frozen pin")
    if hashlib.sha256(m2_protocol_path.read_bytes()).hexdigest() != (
        LOCAL_W3_M2_PROTOCOL_SHA256
    ):
        raise ValueError("tracked W3-M2 protocol differs from its frozen pin")
    return Task(
        dataset=[local_model_preflight_w3_m2_sample()],
        solver=local_model_preflight_w3_m2_solver(str(fixture_path)),
        scorer=local_model_preflight_w3_m2_scorer(),
        config=GenerateConfig(
            temperature=0.0,
            seed=seed,
            cache=False,
            max_retries=0,
            max_connections=1,
            adaptive_connections=False,
        ),
        version=LOCAL_MODEL_PREFLIGHT_W3_M2_TASK_VERSION,
        metadata={
            "purpose": LOCAL_MODEL_PREFLIGHT_W3_M2_PURPOSE,
            "track": "local_zero_api_cost",
            "hypothesis_test_eligible": False,
            "intervention": "model_only",
            "parent_prompt_cell": "W3",
            "pricing_config_sha256": ACTIVE_LOCAL_W3_M2_PRICING_SHA256,
            "preflight_fixture_sha256": LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256,
            "preflight_protocol_sha256": LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256,
            "model_artifact_sha256": LOCAL_W3_M2_OLLAMA_MANIFEST_SHA256,
            "model_only_protocol_sha256": LOCAL_W3_M2_PROTOCOL_SHA256,
        },
    )


@task
def local_model_preflight_w3_m2_t1(
    ollama_models_dir: str | None = None,
    seed: int = 101,
) -> Task:
    """One transport-only W3-M2 gate with Ollama thinking disabled."""

    if seed != 101:
        raise ValueError("local W3-M2-T1 preflight requires seed 101 exactly")
    models_dir = _require_ollama_models_dir(ollama_models_dir)
    _verify_installed_w3_m2_model(models_dir)
    fixture_path = _repo_artifact_path(LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_PATH)
    protocol_path = _repo_artifact_path(LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_PATH)
    m2_protocol_path = _repo_artifact_path(LOCAL_W3_M2_PROTOCOL_PATH)
    t1_protocol_path = _repo_artifact_path(LOCAL_W3_M2_T1_PROTOCOL_PATH)
    expected_pins = (
        (fixture_path, LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256),
        (protocol_path, LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256),
        (m2_protocol_path, LOCAL_W3_M2_PROTOCOL_SHA256),
        (t1_protocol_path, LOCAL_W3_M2_T1_PROTOCOL_SHA256),
    )
    if any(
        hashlib.sha256(path.read_bytes()).hexdigest() != expected
        for path, expected in expected_pins
    ):
        raise ValueError("tracked W3-M2-T1 input differs from its frozen pin")
    return Task(
        dataset=[local_model_preflight_w3_m2_t1_sample()],
        solver=local_model_preflight_w3_m2_t1_solver(str(fixture_path)),
        scorer=local_model_preflight_w3_m2_t1_scorer(),
        config=GenerateConfig(
            temperature=0.0,
            seed=seed,
            cache=False,
            max_retries=0,
            max_connections=1,
            adaptive_connections=False,
            extra_body={"reasoning_effort": "none"},
        ),
        version=LOCAL_MODEL_PREFLIGHT_W3_M2_T1_TASK_VERSION,
        metadata={
            "purpose": LOCAL_MODEL_PREFLIGHT_W3_M2_T1_PURPOSE,
            "track": "local_zero_api_cost",
            "hypothesis_test_eligible": False,
            "intervention": "transport_only",
            "parent_cell": "W3-M2",
            "transport_field": "reasoning_effort=none",
            "pricing_config_sha256": ACTIVE_LOCAL_W3_M2_PRICING_SHA256,
            "preflight_fixture_sha256": LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256,
            "preflight_protocol_sha256": LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256,
            "model_artifact_sha256": LOCAL_W3_M2_OLLAMA_MANIFEST_SHA256,
            "model_only_protocol_sha256": LOCAL_W3_M2_PROTOCOL_SHA256,
            "transport_protocol_sha256": LOCAL_W3_M2_T1_PROTOCOL_SHA256,
        },
    )


def _system_task(
    system: LocalSystemName,
    *,
    seed: int | None,
    repetition: int,
    manifest: str | None,
    ollama_models_dir: str | None,
    embedding_snapshot_path: str | None = None,
    oracle_annotations_path: str | None = None,
    required_phase: str | None = None,
) -> Task:
    frozen, manifest_sha256, dataset_path = _validated_local_manifest(
        manifest,
        system=system,
        seed=seed,
        repetition=repetition,
        ollama_models_dir=ollama_models_dir,
        required_phase=required_phase,
    )
    if system == "vector_rag":
        embedding_snapshot_path = _require_local_embedding_snapshot(
            embedding_snapshot_path
        )
    scenarios = load_scenarios(dataset_path)
    oracle_runtime_path: str | None = None
    oracle_annotations_sha256: str | None = None
    if system == ORACLE_SYSTEM_NAME:
        if frozen.oracle_annotations is None:
            raise ValueError("frozen oracle manifest omitted oracle_annotations")
        annotations_path = _require_oracle_annotations_path(
            oracle_annotations_path,
            manifest_path=frozen.oracle_annotations.path,
        )
        load_oracle_artifact(annotations_path, scenarios)
        oracle_runtime_path = str(annotations_path)
        oracle_annotations_sha256 = frozen.oracle_annotations.sha256
        if oracle_annotations_sha256 is None:
            raise ValueError("frozen oracle manifest omitted annotation SHA-256")
    elif oracle_annotations_path is not None:
        raise ValueError("oracle_annotations_path is only valid for the oracle task")
    compiler_prompt_variant_by_phase = {
        "smoke": "w1",
        "baseline": "w1",
        "oracle_smoke": "w1",
        "writer_diagnostic": "w1",
        "writer_diagnostic_w2": "w2",
        "writer_diagnostic_w3": "w3",
    }
    compiler_prompt_variant = compiler_prompt_variant_by_phase[frozen.phase]
    w2_preflight_fixture_path: str | None = None
    w3_preflight_fixture_path: str | None = None
    if frozen.phase == "writer_diagnostic_w2":
        if frozen.preflight_fixture is None:
            raise ValueError("frozen W2 manifest omitted preflight_fixture")
        w2_preflight_fixture_path = str(
            _repo_artifact_path(frozen.preflight_fixture.path)
        )
    elif frozen.phase == "writer_diagnostic_w3":
        if frozen.preflight_fixture is None or frozen.preflight_protocol is None:
            raise ValueError("frozen W3 manifest omitted preflight fixture/protocol")
        w3_preflight_fixture_path = str(
            _repo_artifact_path(frozen.preflight_fixture.path)
        )
    task_version = (
        LOCAL_SCENARIO_W3_TASK_VERSION
        if frozen.phase == "writer_diagnostic_w3"
        else LOCAL_SCENARIO_TASK_VERSION
    )
    return Task(
        dataset=_scenario_dataset(
            dataset_path,
            name=f"anamnesis-local-{frozen.phase}-v0",
        ),
        solver=local_scenario_solver(
            system,
            repetition=repetition,
            seed=seed,
            top_k=frozen.embedding.top_k,
            embedding_model=frozen.embedding.model,
            embedding_repository=frozen.embedding.repository,
            embedding_revision=frozen.embedding.revision,
            embedding_snapshot_path=embedding_snapshot_path,
            expected_model=frozen.model.snapshot,
            expected_system_config_sha256=frozen.system_config_sha256[system],
            expected_embedding_artifact_sha256=(
                frozen.embedding.artifact_sha256 if system == "vector_rag" else None
            ),
            manifest_sha256=manifest_sha256,
            pricing_config_sha256=frozen.model.pricing.sha256,
            oracle_annotations_path=oracle_runtime_path,
            oracle_annotations_sha256=oracle_annotations_sha256,
            compiler_prompt_variant=compiler_prompt_variant,
            w2_preflight_fixture_path=w2_preflight_fixture_path,
            w3_preflight_fixture_path=w3_preflight_fixture_path,
            w3_preflight_protocol_sha256=(
                frozen.preflight_protocol.sha256
                if frozen.phase == "writer_diagnostic_w3"
                and frozen.preflight_protocol is not None
                else None
            ),
            w3_dataset_sha256=(
                frozen.dataset.sha256
                if frozen.phase == "writer_diagnostic_w3"
                else None
            ),
            w3_reference_sha256=(
                frozen.writer_reference.sha256
                if frozen.phase == "writer_diagnostic_w3"
                and frozen.writer_reference is not None
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
        version=task_version,
        metadata={
            "track": frozen.track,
            "claim_scope": frozen.claim_scope,
            "hypothesis_test_eligible": frozen.hypothesis_test_eligible,
            "system": system,
            "dataset": frozen.dataset.path,
            "dataset_split": frozen.phase,
            "dataset_scenario_count": len(scenarios),
            "dataset_sample_ids": [scenario.id for scenario in scenarios],
            "canonical_dataset_sha256": dataset_sha256(scenarios),
            "repetition": repetition,
            "manifest_sha256": manifest_sha256,
            "live_semantic_preflight_required": True,
            "provider_api_cost_usd": 0.0,
            "pricing_config_sha256": ACTIVE_LOCAL_PRICING_SHA256,
            "electricity_measured": False,
            "decision_prompt_version": LOCAL_DECISION_VERSION,
            **(
                {
                    "compiler_mode": "llm",
                    "memory_compiler_prompt_version": (
                        LOCAL_MEMORY_COMPILER_W3_VERSION
                    ),
                    "preflight_fixture_sha256": (
                        LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256
                    ),
                    "preflight_protocol_sha256": (
                        LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256
                    ),
                    "setup_preflight_task": "local_model_preflight_w3",
                    "setup_preflight_model_calls": 9,
                    "setup_preflight_compiler_calls": 8,
                    "setup_preflight_decision_calls": 1,
                    "setup_preflight_usage_in_headline": False,
                    "same_model_for_compiler_and_decision": True,
                    "scenario_compiler_model_calls": 39,
                    "dataset_checkpoint_count": 62,
                    "intervention": "bundled-repair",
                }
                if frozen.phase == "writer_diagnostic_w3"
                else {}
            ),
            **(
                {
                    "compiler_mode": "llm",
                    "memory_compiler_prompt_version": (
                        LOCAL_MEMORY_COMPILER_W2_VERSION
                    ),
                    "preflight_fixture_sha256": (
                        LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256
                    ),
                    "setup_preflight_task": "local_model_preflight_w2",
                    "setup_preflight_model_calls": 4,
                    "setup_preflight_compiler_calls": 3,
                    "setup_preflight_decision_calls": 1,
                    "setup_preflight_usage_in_headline": False,
                    "same_model_for_compiler_and_decision": True,
                    "scenario_compiler_model_calls": sum(
                        event.kind != "clock_tick"
                        for scenario in scenarios
                        for event in scenario.events
                    ),
                }
                if frozen.phase == "writer_diagnostic_w2"
                else {}
            ),
            **(
                {
                    "compiler_mode": frozen.compiler_mode,
                    "gold_assisted": True,
                    "human_annotation_measured": False,
                    "oracle_artifact_purpose": ORACLE_ARTIFACT_PURPOSE,
                    "oracle_annotation_policy": ORACLE_ANNOTATION_POLICY,
                    "oracle_compiler_version": ORACLE_COMPILER_VERSION,
                    "oracle_annotations_path": frozen.oracle_annotations.path,
                    "oracle_annotations_sha256": oracle_annotations_sha256,
                    "oracle_token_scope": "decision_only_lower_bound",
                    "same_model_for_compiler_and_decision": (
                        frozen.model.same_model_for_compiler_and_decision
                    ),
                    "scenario_compiler_model_calls": 0,
                    "setup_preflight_includes_llm_compiler_call": True,
                    "setup_preflight_compiler_used_in_scenarios": False,
                }
                if system == ORACLE_SYSTEM_NAME
                and frozen.oracle_annotations is not None
                else {}
            ),
        },
    )


@task
def local_no_memory(
    seed: int | None = None,
    repetition: int = 1,
    manifest: str | None = None,
    ollama_models_dir: str | None = None,
) -> Task:
    return _system_task(
        "no_memory",
        seed=seed,
        repetition=repetition,
        manifest=manifest,
        ollama_models_dir=ollama_models_dir,
    )


@task
def local_full_context(
    seed: int | None = None,
    repetition: int = 1,
    manifest: str | None = None,
    ollama_models_dir: str | None = None,
) -> Task:
    return _system_task(
        "full_context",
        seed=seed,
        repetition=repetition,
        manifest=manifest,
        ollama_models_dir=ollama_models_dir,
    )


@task
def local_vector_rag(
    seed: int | None = None,
    repetition: int = 1,
    manifest: str | None = None,
    ollama_models_dir: str | None = None,
    embedding_snapshot_path: str | None = None,
) -> Task:
    return _system_task(
        "vector_rag",
        seed=seed,
        repetition=repetition,
        manifest=manifest,
        ollama_models_dir=ollama_models_dir,
        embedding_snapshot_path=embedding_snapshot_path,
    )


@task
def local_anamnesis(
    seed: int | None = None,
    repetition: int = 1,
    manifest: str | None = None,
    ollama_models_dir: str | None = None,
) -> Task:
    return _system_task(
        "anamnesis",
        seed=seed,
        repetition=repetition,
        manifest=manifest,
        ollama_models_dir=ollama_models_dir,
        required_phase="smoke",
    )


@task
def local_anamnesis_writer_diagnostic(
    seed: int | None = None,
    repetition: int = 1,
    manifest: str | None = None,
    ollama_models_dir: str | None = None,
) -> Task:
    return _system_task(
        "anamnesis",
        seed=seed,
        repetition=repetition,
        manifest=manifest,
        ollama_models_dir=ollama_models_dir,
        required_phase="writer_diagnostic",
    )


@task
def local_anamnesis_writer_diagnostic_w2(
    seed: int | None = None,
    repetition: int = 1,
    manifest: str | None = None,
    ollama_models_dir: str | None = None,
) -> Task:
    return _system_task(
        "anamnesis",
        seed=seed,
        repetition=repetition,
        manifest=manifest,
        ollama_models_dir=ollama_models_dir,
        required_phase="writer_diagnostic_w2",
    )


@task
def local_anamnesis_writer_diagnostic_w3(
    seed: int | None = None,
    repetition: int = 1,
    manifest: str | None = None,
    ollama_models_dir: str | None = None,
) -> Task:
    return _system_task(
        "anamnesis",
        seed=seed,
        repetition=repetition,
        manifest=manifest,
        ollama_models_dir=ollama_models_dir,
        required_phase="writer_diagnostic_w3",
    )


@task
def local_anamnesis_oracle_compiler(
    seed: int | None = None,
    repetition: int = 1,
    manifest: str | None = None,
    ollama_models_dir: str | None = None,
    oracle_annotations_path: str | None = None,
) -> Task:
    return _system_task(
        ORACLE_SYSTEM_NAME,
        seed=seed,
        repetition=repetition,
        manifest=manifest,
        ollama_models_dir=ollama_models_dir,
        oracle_annotations_path=oracle_annotations_path,
        required_phase="oracle_smoke",
    )
