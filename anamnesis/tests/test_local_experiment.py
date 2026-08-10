from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import anamnesis.local_experiment as local_experiment
from anamnesis.local_experiment import (
    LOCAL_BASE_URL,
    LOCAL_MODEL_ID,
    LOCAL_WRITER_REFERENCE_PATH,
    LOCAL_WRITER_REFERENCE_SHA256,
    LOCAL_WRITER_W2_DATASET_SHA256,
    LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_PATH,
    LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256,
    LOCAL_WRITER_W2_REFERENCE_PATH,
    LOCAL_WRITER_W2_REFERENCE_SHA256,
    LocalExperimentManifest,
    OllamaArtifactPin,
    require_local_only_environment,
    verify_ollama_artifact,
    verify_static_local_inputs,
)
from anamnesis.oracle import ORACLE_SYSTEM_NAME

TEMPLATE = Path("eval/local_experiment_manifest.template.json")
WRITER_TEMPLATE = Path("eval/local_writer_experiment_manifest.template.json")
WRITER_W2_TEMPLATE = Path("eval/local_writer_w2_experiment_manifest.template.json")
MODEL_PIN = Path("eval/ollama_qwen3_4b_instruct.pin.json")
HASH = "a" * 64
COMMIT = "b" * 40


def _template_raw() -> dict[str, object]:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def _oracle_raw(*, annotation_sha256: str | None = HASH) -> dict[str, object]:
    raw = _template_raw()
    raw["phase"] = "oracle_smoke"
    raw["compiler_mode"] = "oracle"
    raw["systems"] = [ORACLE_SYSTEM_NAME]
    raw["oracle_annotations"] = {
        "path": "eval/oracle/smoke_memory_deltas.v1.json",
        "sha256": annotation_sha256,
    }
    model = raw["model"]
    assert isinstance(model, dict)
    model["same_model_for_compiler_and_decision"] = False
    return raw


def _writer_raw(
    *, reference_sha256: str | None = LOCAL_WRITER_REFERENCE_SHA256
) -> dict[str, object]:
    raw = json.loads(WRITER_TEMPLATE.read_text(encoding="utf-8"))
    reference = raw["writer_reference"]
    assert isinstance(reference, dict)
    reference["sha256"] = reference_sha256
    return raw


def _writer_w2_raw() -> dict[str, object]:
    return json.loads(WRITER_W2_TEMPLATE.read_text(encoding="utf-8"))


def test_local_smoke_template_is_valid_and_static_inputs_match() -> None:
    manifest = LocalExperimentManifest.model_validate_json(
        TEMPLATE.read_text(encoding="utf-8")
    )

    assert manifest.track == "local_zero_api_cost"
    assert manifest.claim_scope == "diagnostic_development_only"
    assert manifest.hypothesis_test_eligible is False
    assert manifest.phase == "smoke"
    assert manifest.scenario_count == 10
    assert set(manifest.systems) == {
        "no_memory",
        "full_context",
        "vector_rag",
        "anamnesis",
    }
    assert manifest.model.snapshot == LOCAL_MODEL_ID
    assert manifest.model.provider.base_url == LOCAL_BASE_URL
    assert manifest.model.provider.cloud_disabled_environment == ("OLLAMA_NO_CLOUD=1")
    assert manifest.model.provider.server_bind_environment == (
        "OLLAMA_HOST=127.0.0.1:11434"
    )
    assert manifest.runtime.context_length_environment == ("OLLAMA_CONTEXT_LENGTH=4096")
    assert manifest.runtime.num_parallel_environment == "OLLAMA_NUM_PARALLEL=1"
    assert manifest.runtime.max_loaded_models_environment == (
        "OLLAMA_MAX_LOADED_MODELS=1"
    )
    assert manifest.runtime.process_attestation_endpoint == (
        "http://127.0.0.1:11434/api/ps"
    )
    assert manifest.runtime.process_attestation_required is True
    assert manifest.model.schema_constrained_output is True
    assert manifest.model.live_preflight_required is True
    assert manifest.cost_policy.provider_api_price_usd == 0.0
    assert manifest.cost_policy.electricity_measured is False

    verify_static_local_inputs(manifest, repo_root=Path.cwd())


def test_local_track_cannot_be_used_as_a_final_hypothesis_run() -> None:
    raw = _template_raw()
    raw["phase"] = "final"
    raw["hypothesis_test_eligible"] = True

    with pytest.raises(ValidationError):
        LocalExperimentManifest.model_validate(raw)


def test_local_oracle_smoke_is_a_distinct_single_system_phase() -> None:
    manifest = LocalExperimentManifest.model_validate(_oracle_raw())

    assert manifest.phase == "oracle_smoke"
    assert manifest.systems == [ORACLE_SYSTEM_NAME]
    assert manifest.scenario_count == 10
    assert manifest.dataset.path == "eval/scenarios/smoke.jsonl"
    assert manifest.execution.seeds == [101]
    assert manifest.oracle_annotations is not None
    assert manifest.oracle_annotations.sha256 == HASH


def test_local_writer_template_is_a_distinct_single_system_phase() -> None:
    manifest = LocalExperimentManifest.model_validate_json(
        WRITER_TEMPLATE.read_text(encoding="utf-8")
    )

    assert manifest.phase == "writer_diagnostic"
    assert manifest.systems == ["anamnesis"]
    assert manifest.scenario_count == 10
    assert manifest.dataset.path == "eval/scenarios/writer_diagnostic.v1.jsonl"
    assert manifest.execution.seeds == [101]
    assert manifest.execution.repetitions == 1
    assert manifest.compiler_mode == "llm"
    assert manifest.model.same_model_for_compiler_and_decision is True
    assert manifest.writer_reference is not None
    assert manifest.writer_reference.path == LOCAL_WRITER_REFERENCE_PATH
    assert manifest.writer_reference.sha256 == LOCAL_WRITER_REFERENCE_SHA256

    verify_static_local_inputs(manifest, repo_root=Path.cwd())


def test_local_writer_w2_template_pins_v3_fixture_and_reporter_reference() -> None:
    manifest = LocalExperimentManifest.model_validate(_writer_w2_raw())

    assert manifest.phase == "writer_diagnostic_w2"
    assert manifest.systems == ["anamnesis"]
    assert manifest.dataset.sha256 == LOCAL_WRITER_W2_DATASET_SHA256
    assert manifest.writer_reference is not None
    assert manifest.writer_reference.path == LOCAL_WRITER_W2_REFERENCE_PATH
    assert manifest.writer_reference.sha256 == LOCAL_WRITER_W2_REFERENCE_SHA256
    assert manifest.preflight_fixture is not None
    assert manifest.preflight_fixture.path == LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_PATH
    assert manifest.preflight_fixture.sha256 == LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256
    assert manifest.execution.warmup_policy == "frozen_w2_semantic_gate_c1_c2_c3_d1"

    verify_static_local_inputs(manifest, repo_root=Path.cwd())


def test_local_writer_w2_warmup_policy_is_phase_exclusive() -> None:
    w2 = _writer_w2_raw()
    execution = w2["execution"]
    assert isinstance(execution, dict)
    execution["warmup_policy"] = "one_unmeasured_call_per_schema"
    with pytest.raises(ValidationError, match="requires warmup_policy"):
        LocalExperimentManifest.model_validate(w2)

    w1 = _writer_raw()
    execution = w1["execution"]
    assert isinstance(execution, dict)
    execution["warmup_policy"] = "frozen_w2_semantic_gate_c1_c2_c3_d1"
    with pytest.raises(ValidationError, match="requires warmup_policy"):
        LocalExperimentManifest.model_validate(w1)


def test_local_writer_w2_fixture_is_required_and_phase_exclusive() -> None:
    missing = _writer_w2_raw()
    missing.pop("preflight_fixture")
    with pytest.raises(ValidationError, match="requires preflight_fixture"):
        LocalExperimentManifest.model_validate(missing)

    w1 = _writer_raw()
    w1["preflight_fixture"] = {
        "path": LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_PATH,
        "sha256": LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256,
    }
    with pytest.raises(ValidationError, match="only valid"):
        LocalExperimentManifest.model_validate(w1)


def test_local_oracle_annotations_are_required_only_for_oracle_smoke() -> None:
    missing = _oracle_raw()
    missing.pop("oracle_annotations")
    with pytest.raises(ValidationError, match="requires oracle_annotations"):
        LocalExperimentManifest.model_validate(missing)

    smoke = _template_raw()
    smoke["oracle_annotations"] = {
        "path": "eval/oracle/smoke_memory_deltas.v1.json",
        "sha256": HASH,
    }
    with pytest.raises(ValidationError, match="only valid"):
        LocalExperimentManifest.model_validate(smoke)


def test_local_writer_reference_is_required_only_for_writer_diagnostic() -> None:
    missing = _writer_raw()
    missing.pop("writer_reference")
    with pytest.raises(ValidationError, match="requires writer_reference"):
        LocalExperimentManifest.model_validate(missing)

    smoke = _template_raw()
    smoke["writer_reference"] = {
        "path": "eval/oracle/writer_diagnostic_memory_deltas.v1.json",
        "sha256": HASH,
    }
    with pytest.raises(ValidationError, match="only valid"):
        LocalExperimentManifest.model_validate(smoke)


def test_measured_static_verifier_does_not_resolve_writer_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = LocalExperimentManifest.model_validate(_writer_raw())
    repo_file = local_experiment._repo_file

    def guarded_repo_file(repo_root: Path, relative_path: str) -> Path:
        if relative_path == LOCAL_WRITER_REFERENCE_PATH:
            raise AssertionError("measured verifier resolved writer reference")
        return repo_file(repo_root, relative_path)

    monkeypatch.setattr(local_experiment, "_repo_file", guarded_repo_file)

    verify_static_local_inputs(manifest, repo_root=Path.cwd())


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("path", "eval/oracle/other.json", "writer_reference.path"),
        ("sha256", HASH, "writer_reference.sha256"),
        ("sha256", None, "writer_reference.sha256"),
    ],
)
def test_local_writer_reference_pin_cannot_drift(
    field: str, value: object, expected: str
) -> None:
    raw = _writer_raw()
    reference = raw["writer_reference"]
    assert isinstance(reference, dict)
    reference[field] = value

    with pytest.raises(ValidationError, match=expected):
        LocalExperimentManifest.model_validate(raw)


def test_compiler_mode_and_same_model_claim_are_phase_locked() -> None:
    oracle = _oracle_raw()
    oracle["compiler_mode"] = "llm"
    with pytest.raises(ValidationError, match="compiler_mode=oracle"):
        LocalExperimentManifest.model_validate(oracle)

    oracle = _oracle_raw()
    model = oracle["model"]
    assert isinstance(model, dict)
    model["same_model_for_compiler_and_decision"] = True
    with pytest.raises(
        ValidationError, match="same_model_for_compiler_and_decision=false"
    ):
        LocalExperimentManifest.model_validate(oracle)

    smoke = _template_raw()
    smoke["compiler_mode"] = "oracle"
    with pytest.raises(ValidationError, match="compiler_mode=llm"):
        LocalExperimentManifest.model_validate(smoke)

    smoke = _template_raw()
    smoke_model = smoke["model"]
    assert isinstance(smoke_model, dict)
    smoke_model["same_model_for_compiler_and_decision"] = False
    with pytest.raises(
        ValidationError, match="same_model_for_compiler_and_decision=true"
    ):
        LocalExperimentManifest.model_validate(smoke)

    baseline = _template_raw()
    baseline["phase"] = "baseline"
    baseline["scenario_count"] = 35
    baseline["systems"] = ["no_memory", "full_context", "vector_rag"]
    baseline["dataset"] = {
        "path": "eval/scenarios/dev.jsonl",
        "sha256": HASH,
    }
    baseline["compiler_mode"] = "oracle"
    baseline_model = baseline["model"]
    assert isinstance(baseline_model, dict)
    baseline_model["same_model_for_compiler_and_decision"] = False
    with pytest.raises(ValidationError, match="compiler_mode=llm"):
        LocalExperimentManifest.model_validate(baseline)

    writer = _writer_raw()
    writer["compiler_mode"] = "oracle"
    writer_model = writer["model"]
    assert isinstance(writer_model, dict)
    writer_model["same_model_for_compiler_and_decision"] = False
    with pytest.raises(ValidationError, match="compiler_mode=llm"):
        LocalExperimentManifest.model_validate(writer)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("systems", [ORACLE_SYSTEM_NAME, "anamnesis"]),
        ("scenario_count", 35),
        ("sealed_opened", True),
        ("dataset", {"path": "eval/scenarios/dev.jsonl", "sha256": HASH}),
    ],
)
def test_local_oracle_smoke_matrix_cannot_drift(field: str, value: object) -> None:
    raw = _oracle_raw()
    raw[field] = value

    with pytest.raises(ValidationError):
        LocalExperimentManifest.model_validate(raw)


def test_frozen_local_oracle_requires_annotation_hash() -> None:
    raw = _oracle_raw(annotation_sha256=None)
    raw["status"] = "frozen"

    with pytest.raises(ValidationError, match="oracle_annotations.sha256"):
        LocalExperimentManifest.model_validate(raw)


def test_frozen_local_writer_requires_reference_hash() -> None:
    raw = _writer_raw(reference_sha256=None)
    raw["status"] = "frozen"

    with pytest.raises(ValidationError, match="writer_reference.sha256"):
        LocalExperimentManifest.model_validate(raw)


def test_complete_frozen_local_oracle_manifest_needs_no_llm_compiler_contract() -> None:
    raw = _oracle_raw()
    raw["status"] = "frozen"
    raw["git_commit"] = COMMIT
    raw["decision_prompt_sha256"] = HASH
    raw["decision_schema_sha256"] = HASH
    raw["system_config_sha256"] = {ORACLE_SYSTEM_NAME: HASH}
    model = raw["model"]
    assert isinstance(model, dict)
    preflight = model["preflight"]
    assert isinstance(preflight, dict)
    preflight["sha256"] = HASH

    manifest = LocalExperimentManifest.model_validate(raw)

    assert manifest.status == "frozen"
    assert manifest.memory_compiler_prompt_sha256 is None
    assert manifest.memory_compiler_schema_sha256 is None


def test_local_route_cannot_be_changed_to_remote_or_cloud_enabled() -> None:
    raw = _template_raw()
    model = raw["model"]
    assert isinstance(model, dict)
    provider = model["provider"]
    assert isinstance(provider, dict)
    provider["base_url"] = "https://api.example.test/v1"
    provider["cloud_disabled_environment"] = "OLLAMA_NO_CLOUD=0"

    with pytest.raises(ValidationError):
        LocalExperimentManifest.model_validate(raw)


@pytest.mark.parametrize("field", ["serial_number", "hardware_uuid", "machine_id"])
def test_hardware_pin_rejects_identifying_fields(field: str) -> None:
    raw = _template_raw()
    hardware = raw["hardware"]
    assert isinstance(hardware, dict)
    hardware[field] = "sensitive-value"

    with pytest.raises(ValidationError):
        LocalExperimentManifest.model_validate(raw)


def test_local_only_environment_requires_exact_string_value() -> None:
    valid_environment = {
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_CONTEXT_LENGTH": "4096",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    }
    require_local_only_environment(valid_environment)

    for invalid_value in ("true", "0", ""):
        environment = dict(valid_environment)
        environment["OLLAMA_NO_CLOUD"] = invalid_value
        with pytest.raises(ValueError, match="OLLAMA_NO_CLOUD=1"):
            require_local_only_environment(environment)

    environment = dict(valid_environment)
    environment["OLLAMA_HOST"] = "0.0.0.0:11434"
    with pytest.raises(ValueError, match="OLLAMA_HOST=127.0.0.1:11434"):
        require_local_only_environment(environment)


def test_frozen_smoke_requires_preflight_and_all_runtime_fingerprints() -> None:
    raw = _template_raw()
    raw["status"] = "frozen"

    with pytest.raises(ValidationError, match="model.preflight.sha256"):
        LocalExperimentManifest.model_validate(raw)


def test_complete_frozen_smoke_manifest_is_accepted() -> None:
    raw = _template_raw()
    raw["status"] = "frozen"
    raw["git_commit"] = COMMIT
    raw["decision_prompt_sha256"] = HASH
    raw["decision_schema_sha256"] = HASH
    raw["memory_compiler_prompt_sha256"] = HASH
    raw["memory_compiler_schema_sha256"] = HASH
    raw["system_config_sha256"] = {
        "no_memory": HASH,
        "full_context": HASH,
        "vector_rag": HASH,
        "anamnesis": HASH,
    }
    model = raw["model"]
    assert isinstance(model, dict)
    preflight = model["preflight"]
    assert isinstance(preflight, dict)
    preflight["sha256"] = HASH

    manifest = LocalExperimentManifest.model_validate(raw)

    assert manifest.status == "frozen"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scenario_count", 11),
        ("sealed_opened", True),
        ("systems", ["no_memory", "full_context", "vector_rag"]),
    ],
)
def test_local_smoke_matrix_cannot_drift(field: str, value: object) -> None:
    raw = _template_raw()
    raw[field] = value

    with pytest.raises(ValidationError):
        LocalExperimentManifest.model_validate(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scenario_count", 11),
        ("sealed_opened", True),
        ("systems", ["no_memory", "anamnesis"]),
        ("dataset", {"path": "eval/scenarios/smoke.jsonl", "sha256": HASH}),
    ],
)
def test_local_writer_matrix_cannot_drift(field: str, value: object) -> None:
    raw = _writer_raw()
    raw[field] = value

    with pytest.raises(ValidationError):
        LocalExperimentManifest.model_validate(raw)


def test_tracked_model_pin_identifies_exact_manifest_and_model_blob() -> None:
    raw_text = MODEL_PIN.read_text(encoding="utf-8")
    pin = OllamaArtifactPin.model_validate_json(raw_text)

    assert pin.model == LOCAL_MODEL_ID
    assert pin.manifest_sha256 == (
        "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
    )
    model_blob = next(blob for blob in pin.blobs if blob.role == "model")
    assert model_blob.sha256 == (
        "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
    )
    assert model_blob.size_bytes == 2_497_280_480
    assert "/Users/" not in raw_text


def test_ollama_artifact_verifier_hashes_manifest_and_every_blob(
    tmp_path: Path,
) -> None:
    contents = {
        "config": b"config",
        "model": b"model",
        "template": b"template",
        "license": b"license",
        "params": b"params",
    }
    media_types = {
        "config": "application/vnd.docker.container.image.v1+json",
        "model": "application/vnd.ollama.image.model",
        "template": "application/vnd.ollama.image.template",
        "license": "application/vnd.ollama.image.license",
        "params": "application/vnd.ollama.image.params",
    }
    blobs_dir = tmp_path / "blobs"
    blobs_dir.mkdir()
    descriptors: dict[str, dict[str, object]] = {}
    pin_blobs: list[dict[str, object]] = []
    for role, content in contents.items():
        digest = hashlib.sha256(content).hexdigest()
        (blobs_dir / f"sha256-{digest}").write_bytes(content)
        descriptors[role] = {
            "mediaType": media_types[role],
            "digest": f"sha256:{digest}",
            "size": len(content),
        }
        pin_blobs.append(
            {
                "role": role,
                "media_type": media_types[role],
                "sha256": digest,
                "size_bytes": len(content),
            }
        )

    manifest_raw = {
        "schemaVersion": 2,
        "config": descriptors["config"],
        "layers": [
            descriptors[role] for role in ("model", "template", "license", "params")
        ],
    }
    manifest_path = tmp_path / "manifest"
    manifest_path.write_text(json.dumps(manifest_raw), encoding="utf-8")
    pin = OllamaArtifactPin(
        model=LOCAL_MODEL_ID,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        blobs=pin_blobs,
    )

    assert verify_ollama_artifact(
        pin, manifest_path=manifest_path, blobs_dir=blobs_dir
    ) == sum(map(len, contents.values()))

    model_blob = next(blob for blob in pin.blobs if blob.role == "model")
    (blobs_dir / f"sha256-{model_blob.sha256}").write_bytes(b"changed")
    with pytest.raises(ValueError, match="model blob size"):
        verify_ollama_artifact(pin, manifest_path=manifest_path, blobs_dir=blobs_dir)
