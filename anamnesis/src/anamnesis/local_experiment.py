"""Reproducibility contract for the zero-provider-cost local experiment track."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from anamnesis.experiment import (
    FINAL_SYSTEMS,
    SIMPLE_SYSTEMS,
    ArtifactPin,
    EmbeddingPin,
)
from anamnesis.oracle import ORACLE_SYSTEM_NAME
from anamnesis.schema import StrictModel

LOCAL_MODEL_ID = "ollama/qwen3:4b-instruct"
LOCAL_W3_M2_MODEL_ID = "ollama/qwen3.5:9b-q4_K_M"
LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
LOCAL_MODEL_ARTIFACT_PATH = "eval/ollama_qwen3_4b_instruct.pin.json"
LOCAL_W3_M2_MODEL_ARTIFACT_PATH = "eval/ollama_qwen3_5_9b_q4_k_m.pin.json"
LOCAL_PRICING_PATH = "eval/local_model_costs.json"
LOCAL_W3_M2_PRICING_PATH = "eval/local_model_costs_w3_m2.json"
LOCAL_W3_M2_PRICING_SHA256 = (
    "74f7321226f6fc71d9c8c88551653d8507d1a22a22ea4127226c91a5ce06267c"
)
LOCAL_W3_M2_PROTOCOL_PATH = "eval/preflight/local_writer_w3_m2.protocol.v1.json"
LOCAL_W3_M2_PROTOCOL_SHA256 = (
    "1b563651b0b95a9a258082c1016dbc997b4da53b3573b24be54dcac30cb82d0e"
)
LOCAL_W3_M2_T1_PROTOCOL_PATH = "eval/preflight/local_writer_w3_m2_t1.protocol.v1.json"
LOCAL_W3_M2_T1_PROTOCOL_SHA256 = (
    "ef02b6bb019c705de96f98330b6e0f14532993635b2b6dc88614f4d1014db09c"
)
LOCAL_PRICING_SHA256 = (
    "c185e2fad06d6bd2abaaf0be81a1720fc245555fa2a477c1b1bea558b28c2f74"
)
LOCAL_WRITER_REFERENCE_PATH = "eval/oracle/writer_diagnostic_memory_deltas.v1.json"
LOCAL_WRITER_REFERENCE_SHA256 = (
    "93c24d604b32c838d635f9c9ed4fea20f770da254f522db6962b6bc57a232057"
)
LOCAL_WRITER_W2_DATASET_PATH = "eval/scenarios/writer_diagnostic.v3.jsonl"
LOCAL_WRITER_W2_DATASET_SHA256 = (
    "34e2e8751bf32a3a2e29ac75d727f2b5cf73aaba13ccc9ba1d9fdf00bf7eaf4f"
)
LOCAL_WRITER_W2_REFERENCE_PATH = "eval/oracle/writer_diagnostic_memory_deltas.v3.json"
LOCAL_WRITER_W2_REFERENCE_SHA256 = (
    "7adb64eda15daf5351260933fbd0625fbc13c6899361735a9bf0ce13c063f857"
)
LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_PATH = "eval/preflight/local_writer_w2.v1.json"
LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256 = (
    "3b82128bab1d801d073118488aa4f0a0a662603b98325f5c9d7dad497f026057"
)
LOCAL_WRITER_W3_DATASET_PATH = "eval/scenarios/writer_diagnostic.v4.jsonl"
LOCAL_WRITER_W3_DATASET_SHA256 = (
    "6b2530cb9f3426c792500f07e854d7f31ad84081ac77104cb8032737234ff91c"
)
LOCAL_WRITER_W3_REFERENCE_PATH = "eval/oracle/writer_diagnostic_memory_deltas.v4.json"
LOCAL_WRITER_W3_REFERENCE_SHA256 = (
    "72308bb34bda758cc72dc651e3f0fd2fd2bd1bff820479e2cf0774ee8d66cf5c"
)
LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_PATH = "eval/preflight/local_writer_w3.v1.json"
LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256 = (
    "5628c3c1d7f8e1a5da43d6e567d55ac8e4fbabd8b9c4054325de6f4def1da30c"
)
LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_PATH = (
    "eval/preflight/local_writer_w3.protocol.v1.json"
)
LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256 = (
    "7f63c156a8af74ced2d5e5530b3e8083da95c7f46e14e1afafaaf864b3ce1915"
)
LOCAL_OLLAMA_VERSION = "0.31.1"

SHA256_PATTERN = r"^[0-9a-f]{64}$"
LOCAL_SMOKE_SYSTEMS = FINAL_SYSTEMS
LOCAL_ORACLE_SMOKE_SYSTEMS = {ORACLE_SYSTEM_NAME}
LOCAL_WRITER_DIAGNOSTIC_SYSTEMS = {"anamnesis"}
LOCAL_SMOKE_SEEDS = [101]


class OllamaBlobPin(StrictModel):
    """One content-addressed object referenced by an Ollama manifest."""

    role: Literal["config", "model", "template", "license", "params"]
    media_type: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=0)


class OllamaArtifactPin(StrictModel):
    """Portable identity for a local Ollama model without a machine path."""

    schema_version: Literal[1] = 1
    model: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    blobs: list[OllamaBlobPin] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_blob_set(self) -> Self:
        roles = [blob.role for blob in self.blobs]
        required = {"config", "model", "license", "params"}
        allowed = required | {"template"}
        if (
            not required.issubset(roles)
            or not set(roles).issubset(allowed)
            or len(roles) != len(set(roles))
        ):
            raise ValueError(
                "Ollama artifact must pin config, model, license, and params "
                "exactly once, with at most one template"
            )
        digests = [blob.sha256 for blob in self.blobs]
        if len(digests) != len(set(digests)):
            raise ValueError("Ollama artifact blob hashes must be unique")
        return self


class LocalProviderRoute(StrictModel):
    """The only provider route permitted for the local experiment track."""

    provider: Literal["ollama"] = "ollama"
    base_url: Literal["http://127.0.0.1:11434/v1"] = LOCAL_BASE_URL
    cloud_disabled_environment: Literal["OLLAMA_NO_CLOUD=1"] = "OLLAMA_NO_CLOUD=1"
    server_bind_environment: Literal["OLLAMA_HOST=127.0.0.1:11434"] = (
        "OLLAMA_HOST=127.0.0.1:11434"
    )
    api_key_required: Literal[False] = False


class LocalModelPin(StrictModel):
    """Exact local model, provider route, artifact and preflight inputs."""

    snapshot: Literal["ollama/qwen3:4b-instruct"] = LOCAL_MODEL_ID
    artifact: ArtifactPin
    schema_constrained_output: Literal[True] = True
    live_preflight_required: Literal[True] = True
    same_model_for_compiler_and_decision: bool = True
    provider: LocalProviderRoute = Field(default_factory=LocalProviderRoute)
    pricing: ArtifactPin
    preflight: ArtifactPin


class LocalHardwarePin(StrictModel):
    """Non-identifying hardware class used for comparable latency results."""

    architecture: Literal["arm64"] = "arm64"
    chip: Literal["Apple M3"] = "Apple M3"
    memory_bytes: Literal[17179869184] = 17179869184


class LocalRuntimePin(StrictModel):
    """Local inference runtime settings that may affect outputs or latency."""

    ollama_version: Literal["0.31.1"] = LOCAL_OLLAMA_VERSION
    inference_backend: Literal["Metal"] = "Metal"
    context_window_tokens: Literal[4096] = 4096
    context_length_environment: Literal["OLLAMA_CONTEXT_LENGTH=4096"] = (
        "OLLAMA_CONTEXT_LENGTH=4096"
    )
    num_parallel_environment: Literal["OLLAMA_NUM_PARALLEL=1"] = "OLLAMA_NUM_PARALLEL=1"
    max_loaded_models_environment: Literal["OLLAMA_MAX_LOADED_MODELS=1"] = (
        "OLLAMA_MAX_LOADED_MODELS=1"
    )
    process_attestation_endpoint: Literal["http://127.0.0.1:11434/api/ps"] = (
        "http://127.0.0.1:11434/api/ps"
    )
    process_attestation_required: Literal[True] = True
    local_only: Literal[True] = True
    setup_latency_reported_separately: Literal[True] = True


class LocalCostPolicy(StrictModel):
    """Scope of the zero-cost claim for local inference."""

    headline_metric: Literal["provider_api_cost_usd"] = "provider_api_cost_usd"
    provider_api_price_usd: Literal[0.0] = 0.0
    electricity_measured: Literal[False] = False
    hardware_amortization_measured: Literal[False] = False


class LocalExecutionPolicy(StrictModel):
    """Frozen execution settings for local diagnostic or measured matrices."""

    temperature: float = Field(default=0.0, ge=0, le=0)
    seeds: list[int] = Field(default_factory=lambda: list(LOCAL_SMOKE_SEEDS))
    repetitions: int = Field(default=1, ge=1, le=3)
    response_cache: Literal[False] = False
    max_samples: Literal[1] = 1
    concurrency: Literal[1] = 1
    max_retries: Literal[0] = 0
    structured_output_repair_calls: Literal[0] = 0
    log_model_api: Literal[True] = True
    warmup_policy: Literal[
        "one_unmeasured_call_per_schema",
        "frozen_w2_semantic_gate_c1_c2_c3_d1",
        "frozen_w3_semantic_gate_c1_to_c8_d1",
    ] = "one_unmeasured_call_per_schema"
    warmup_latency_in_headline: Literal[False] = False

    @model_validator(mode="after")
    def validate_seeds(self) -> Self:
        if len(self.seeds) != self.repetitions or len(set(self.seeds)) != len(
            self.seeds
        ):
            raise ValueError("execution seeds must be unique and match repetitions")
        return self


class LocalExperimentManifest(StrictModel):
    """A local-only experiment declaration kept separate from the hosted track."""

    schema_version: Literal[1] = 1
    track: Literal["local_zero_api_cost"] = "local_zero_api_cost"
    claim_scope: Literal["diagnostic_development_only"] = "diagnostic_development_only"
    hypothesis_test_eligible: Literal[False] = False
    status: Literal["draft", "frozen"] = "draft"
    phase: Literal[
        "smoke",
        "baseline",
        "oracle_smoke",
        "writer_diagnostic",
        "writer_diagnostic_w2",
        "writer_diagnostic_w3",
    ]
    compiler_mode: Literal["llm", "oracle"] = "llm"
    dataset: ArtifactPin
    scenario_count: int = Field(ge=1)
    sealed_opened: bool
    systems: list[str] = Field(min_length=1)
    model: LocalModelPin
    hardware: LocalHardwarePin = Field(default_factory=LocalHardwarePin)
    runtime: LocalRuntimePin = Field(default_factory=LocalRuntimePin)
    cost_policy: LocalCostPolicy = Field(default_factory=LocalCostPolicy)
    embedding: EmbeddingPin
    execution: LocalExecutionPolicy = Field(default_factory=LocalExecutionPolicy)
    git_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    dependency_lock: ArtifactPin
    research_contract: ArtifactPin
    architecture_contract: ArtifactPin
    oracle_annotations: ArtifactPin | None = None
    writer_reference: ArtifactPin | None = None
    preflight_fixture: ArtifactPin | None = None
    preflight_protocol: ArtifactPin | None = None
    decision_prompt_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    decision_schema_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    memory_compiler_prompt_sha256: str | None = Field(
        default=None, pattern=SHA256_PATTERN
    )
    memory_compiler_schema_sha256: str | None = Field(
        default=None, pattern=SHA256_PATTERN
    )
    system_config_sha256: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_local_matrix(self) -> Self:
        matrix = {
            "smoke": {
                "systems": LOCAL_SMOKE_SYSTEMS,
                "count": 10,
                "sealed": False,
                "seeds": LOCAL_SMOKE_SEEDS,
                "dataset": "eval/scenarios/smoke.jsonl",
            },
            "baseline": {
                "systems": SIMPLE_SYSTEMS,
                "count": 35,
                "sealed": False,
                "seeds": [101],
                "dataset": "eval/scenarios/dev.jsonl",
            },
            "oracle_smoke": {
                "systems": LOCAL_ORACLE_SMOKE_SYSTEMS,
                "count": 10,
                "sealed": False,
                "seeds": LOCAL_SMOKE_SEEDS,
                "dataset": "eval/scenarios/smoke.jsonl",
            },
            "writer_diagnostic": {
                "systems": LOCAL_WRITER_DIAGNOSTIC_SYSTEMS,
                "count": 10,
                "sealed": False,
                "seeds": LOCAL_SMOKE_SEEDS,
                "dataset": "eval/scenarios/writer_diagnostic.v1.jsonl",
            },
            "writer_diagnostic_w2": {
                "systems": LOCAL_WRITER_DIAGNOSTIC_SYSTEMS,
                "count": 10,
                "sealed": False,
                "seeds": LOCAL_SMOKE_SEEDS,
                "dataset": LOCAL_WRITER_W2_DATASET_PATH,
            },
            "writer_diagnostic_w3": {
                "systems": LOCAL_WRITER_DIAGNOSTIC_SYSTEMS,
                "count": 10,
                "sealed": False,
                "seeds": LOCAL_SMOKE_SEEDS,
                "dataset": LOCAL_WRITER_W3_DATASET_PATH,
            },
        }[self.phase]

        expected_systems = matrix["systems"]
        if set(self.systems) != expected_systems or len(self.systems) != len(
            expected_systems
        ):
            raise ValueError(
                f"local {self.phase} phase requires systems "
                f"{sorted(expected_systems)} exactly"
            )
        if self.scenario_count != matrix["count"]:
            raise ValueError(
                f"local {self.phase} phase requires {matrix['count']} scenarios"
            )
        if self.sealed_opened is not matrix["sealed"]:
            raise ValueError(
                f"sealed_opened must be {matrix['sealed']} for local {self.phase}"
            )
        if self.dataset.path != matrix["dataset"]:
            raise ValueError(
                f"local {self.phase} phase requires dataset {matrix['dataset']}"
            )
        if (
            self.phase == "writer_diagnostic_w2"
            and self.dataset.sha256 != LOCAL_WRITER_W2_DATASET_SHA256
        ):
            raise ValueError(
                "local writer_diagnostic_w2 dataset.sha256 must be "
                f"{LOCAL_WRITER_W2_DATASET_SHA256}"
            )
        if (
            self.phase == "writer_diagnostic_w3"
            and self.dataset.sha256 != LOCAL_WRITER_W3_DATASET_SHA256
        ):
            raise ValueError(
                "local writer_diagnostic_w3 dataset.sha256 must be "
                f"{LOCAL_WRITER_W3_DATASET_SHA256}"
            )
        if self.execution.seeds != matrix["seeds"]:
            raise ValueError(
                f"local {self.phase} phase requires seeds {matrix['seeds']} exactly"
            )
        if self.execution.repetitions != len(matrix["seeds"]):
            raise ValueError(
                f"local {self.phase} phase requires {len(matrix['seeds'])} repetitions"
            )
        expected_warmup_policy = {
            "smoke": "one_unmeasured_call_per_schema",
            "baseline": "one_unmeasured_call_per_schema",
            "oracle_smoke": "one_unmeasured_call_per_schema",
            "writer_diagnostic": "one_unmeasured_call_per_schema",
            "writer_diagnostic_w2": "frozen_w2_semantic_gate_c1_c2_c3_d1",
            "writer_diagnostic_w3": "frozen_w3_semantic_gate_c1_to_c8_d1",
        }[self.phase]
        if self.execution.warmup_policy != expected_warmup_policy:
            raise ValueError(
                f"local {self.phase} phase requires warmup_policy="
                f"{expected_warmup_policy}"
            )

        if self.model.artifact.path != LOCAL_MODEL_ARTIFACT_PATH:
            raise ValueError(
                f"local model artifact path must be {LOCAL_MODEL_ARTIFACT_PATH}"
            )
        if self.model.pricing.path != LOCAL_PRICING_PATH:
            raise ValueError(f"local pricing path must be {LOCAL_PRICING_PATH}")

        if self.phase == "oracle_smoke":
            if self.oracle_annotations is None:
                raise ValueError("local oracle_smoke phase requires oracle_annotations")
            if self.compiler_mode != "oracle":
                raise ValueError("local oracle_smoke requires compiler_mode=oracle")
            if self.model.same_model_for_compiler_and_decision:
                raise ValueError(
                    "local oracle_smoke requires "
                    "same_model_for_compiler_and_decision=false"
                )
        elif self.oracle_annotations is not None:
            raise ValueError("oracle_annotations are only valid for local oracle_smoke")
        elif self.compiler_mode != "llm":
            raise ValueError(f"local {self.phase} requires compiler_mode=llm")
        elif not self.model.same_model_for_compiler_and_decision:
            raise ValueError(
                f"local {self.phase} requires same_model_for_compiler_and_decision=true"
            )

        if self.phase in {
            "writer_diagnostic",
            "writer_diagnostic_w2",
            "writer_diagnostic_w3",
        }:
            expected_reference_path = {
                "writer_diagnostic": LOCAL_WRITER_REFERENCE_PATH,
                "writer_diagnostic_w2": LOCAL_WRITER_W2_REFERENCE_PATH,
                "writer_diagnostic_w3": LOCAL_WRITER_W3_REFERENCE_PATH,
            }[self.phase]
            expected_reference_sha256 = {
                "writer_diagnostic": LOCAL_WRITER_REFERENCE_SHA256,
                "writer_diagnostic_w2": LOCAL_WRITER_W2_REFERENCE_SHA256,
                "writer_diagnostic_w3": LOCAL_WRITER_W3_REFERENCE_SHA256,
            }[self.phase]
            if self.writer_reference is None:
                raise ValueError(f"local {self.phase} phase requires writer_reference")
            if self.writer_reference.path != expected_reference_path:
                raise ValueError(
                    f"local {self.phase} writer_reference.path must be "
                    f"{expected_reference_path}"
                )
            if self.writer_reference.sha256 != expected_reference_sha256:
                raise ValueError(
                    f"local {self.phase} writer_reference.sha256 must be "
                    f"{expected_reference_sha256}"
                )
        elif self.writer_reference is not None:
            raise ValueError(
                "writer_reference is only valid for local writer diagnostics"
            )

        if self.phase in {"writer_diagnostic_w2", "writer_diagnostic_w3"}:
            expected_fixture_path = {
                "writer_diagnostic_w2": LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_PATH,
                "writer_diagnostic_w3": LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_PATH,
            }[self.phase]
            expected_fixture_sha256 = {
                "writer_diagnostic_w2": LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256,
                "writer_diagnostic_w3": LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256,
            }[self.phase]
            if self.preflight_fixture is None:
                raise ValueError(f"local {self.phase} phase requires preflight_fixture")
            if self.preflight_fixture.path != expected_fixture_path:
                raise ValueError(
                    f"local {self.phase} preflight_fixture.path must be "
                    f"{expected_fixture_path}"
                )
            if self.preflight_fixture.sha256 != expected_fixture_sha256:
                raise ValueError(
                    f"local {self.phase} preflight_fixture.sha256 must be "
                    f"{expected_fixture_sha256}"
                )
        elif self.preflight_fixture is not None:
            raise ValueError(
                "preflight_fixture is only valid for local W2/W3 writer diagnostics"
            )

        if self.phase == "writer_diagnostic_w3":
            if self.preflight_protocol is None:
                raise ValueError(
                    "local writer_diagnostic_w3 phase requires preflight_protocol"
                )
            if self.preflight_protocol.path != LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_PATH:
                raise ValueError(
                    "local writer_diagnostic_w3 preflight_protocol.path must be "
                    f"{LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_PATH}"
                )
            if (
                self.preflight_protocol.sha256
                != LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256
            ):
                raise ValueError(
                    "local writer_diagnostic_w3 preflight_protocol.sha256 must be "
                    f"{LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256}"
                )
        elif self.preflight_protocol is not None:
            raise ValueError(
                "preflight_protocol is only valid for local writer_diagnostic_w3"
            )

        invalid_hashes = {
            name: digest
            for name, digest in self.system_config_sha256.items()
            if not _is_sha256(digest)
        }
        if invalid_hashes:
            raise ValueError(f"invalid system configuration hashes: {invalid_hashes}")

        if self.status == "frozen":
            missing = self._missing_freeze_inputs(expected_systems)
            if missing:
                raise ValueError(
                    "frozen local manifest is missing reproducibility inputs: "
                    + ", ".join(missing)
                )
        return self

    def _missing_freeze_inputs(self, expected_systems: set[str]) -> list[str]:
        missing: list[str] = []
        for name, artifact in (
            ("dataset", self.dataset),
            ("model.artifact", self.model.artifact),
            ("model.pricing", self.model.pricing),
            ("model.preflight", self.model.preflight),
            ("dependency_lock", self.dependency_lock),
            ("research_contract", self.research_contract),
            ("architecture_contract", self.architecture_contract),
        ):
            if artifact.sha256 is None:
                missing.append(f"{name}.sha256")
        if self.phase == "oracle_smoke":
            if self.oracle_annotations is None:
                missing.append("oracle_annotations")
            elif self.oracle_annotations.sha256 is None:
                missing.append("oracle_annotations.sha256")
        if self.phase in {
            "writer_diagnostic",
            "writer_diagnostic_w2",
            "writer_diagnostic_w3",
        }:
            if self.writer_reference is None:
                missing.append("writer_reference")
            elif self.writer_reference.sha256 is None:
                missing.append("writer_reference.sha256")
        if self.phase in {"writer_diagnostic_w2", "writer_diagnostic_w3"}:
            if self.preflight_fixture is None:
                missing.append("preflight_fixture")
            elif self.preflight_fixture.sha256 is None:
                missing.append("preflight_fixture.sha256")
        if self.phase == "writer_diagnostic_w3":
            if self.preflight_protocol is None:
                missing.append("preflight_protocol")
            elif self.preflight_protocol.sha256 is None:
                missing.append("preflight_protocol.sha256")
        for name in ("git_commit", "decision_prompt_sha256", "decision_schema_sha256"):
            if getattr(self, name) is None:
                missing.append(name)
        if "anamnesis" in expected_systems:
            if self.memory_compiler_prompt_sha256 is None:
                missing.append("memory_compiler_prompt_sha256")
            if self.memory_compiler_schema_sha256 is None:
                missing.append("memory_compiler_schema_sha256")
        if self.embedding.revision is None:
            missing.append("embedding.revision")
        if self.embedding.artifact_sha256 is None:
            missing.append("embedding.artifact_sha256")
        if set(self.system_config_sha256) != expected_systems:
            missing.append("system_config_sha256")
        return missing


def load_ollama_artifact_pin(path: Path) -> OllamaArtifactPin:
    """Load the portable model pin without accessing an Ollama server."""

    return OllamaArtifactPin.model_validate_json(path.read_text(encoding="utf-8"))


def verify_ollama_artifact(
    pin: OllamaArtifactPin,
    *,
    manifest_path: Path,
    blobs_dir: Path,
) -> int:
    """Verify the manifest and every referenced local blob; return bytes hashed."""

    if _sha256_file(manifest_path) != pin.manifest_sha256:
        raise ValueError("Ollama manifest SHA-256 does not match the tracked pin")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_descriptors = _ollama_descriptors(raw)
    expected_descriptors = [blob.model_dump() for blob in pin.blobs]
    if actual_descriptors != expected_descriptors:
        raise ValueError("Ollama manifest descriptors do not match the tracked pin")

    bytes_hashed = 0
    for blob in pin.blobs:
        blob_path = blobs_dir / f"sha256-{blob.sha256}"
        if not blob_path.is_file():
            raise ValueError(f"missing pinned Ollama blob for role {blob.role}")
        size = blob_path.stat().st_size
        if size != blob.size_bytes:
            raise ValueError(f"Ollama {blob.role} blob size does not match the pin")
        if _sha256_file(blob_path) != blob.sha256:
            raise ValueError(f"Ollama {blob.role} blob SHA-256 does not match the pin")
        bytes_hashed += size
    return bytes_hashed


def validate_zero_api_pricing(path: Path, model: str = LOCAL_MODEL_ID) -> str:
    """Require one exact all-zero Inspect pricing entry and return its file hash."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    zero_rates = {
        "input": 0.0,
        "output": 0.0,
        "input_cache_write": 0.0,
        "input_cache_read": 0.0,
    }
    if raw != {model: zero_rates}:
        raise ValueError("local pricing must contain one exact all-zero model entry")
    return _sha256_file(path)


def require_local_only_environment(environment: Mapping[str, str]) -> None:
    """Fail unless the Ollama process is local-only with the pinned context."""

    required = {
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_CONTEXT_LENGTH": "4096",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    }
    mismatches = {
        name: expected
        for name, expected in required.items()
        if environment.get(name) != expected
    }
    if mismatches:
        expected_values = ", ".join(
            f"{name}={value}" for name, value in sorted(mismatches.items())
        )
        raise ValueError(f"local runs require {expected_values}")


def verify_static_local_inputs(
    manifest: LocalExperimentManifest, *, repo_root: Path
) -> None:
    """Verify tracked local inputs whose bytes exist before a model run."""

    artifacts = (
        manifest.model.artifact,
        manifest.model.pricing,
        manifest.dependency_lock,
        manifest.research_contract,
        manifest.architecture_contract,
        manifest.dataset,
    )
    if manifest.preflight_fixture is not None:
        artifacts = (*artifacts, manifest.preflight_fixture)
    if manifest.preflight_protocol is not None:
        artifacts = (*artifacts, manifest.preflight_protocol)
    if manifest.oracle_annotations is not None:
        artifacts = (*artifacts, manifest.oracle_annotations)
    # writer_reference is a gold-derived reporter input. The manifest schema
    # locks its declaration, but this measured-input verifier deliberately does
    # not resolve, open, or hash the referenced bytes; the strict writer
    # reporter owns that validation.
    for artifact in artifacts:
        if artifact.sha256 is None:
            if manifest.status == "frozen":
                raise ValueError(f"frozen artifact has no SHA-256: {artifact.path}")
            continue
        path = _repo_file(repo_root, artifact.path)
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"tracked artifact SHA-256 mismatch: {artifact.path}")

    pin = load_ollama_artifact_pin(_repo_file(repo_root, LOCAL_MODEL_ARTIFACT_PATH))
    if pin.model != manifest.model.snapshot:
        raise ValueError("Ollama artifact model does not match manifest snapshot")
    pricing_digest = validate_zero_api_pricing(
        _repo_file(repo_root, LOCAL_PRICING_PATH), manifest.model.snapshot
    )
    if pricing_digest != manifest.model.pricing.sha256:
        raise ValueError("local pricing SHA-256 does not match manifest pin")


def _ollama_descriptors(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, dict):
        raise ValueError("Ollama manifest must be a JSON object")
    config = raw.get("config")
    layers = raw.get("layers")
    if not isinstance(config, dict) or not isinstance(layers, list):
        raise ValueError("Ollama manifest requires config and layers")

    descriptors: list[tuple[str, object]] = [("config", config)]
    role_by_media_type = {
        "application/vnd.ollama.image.model": "model",
        "application/vnd.ollama.image.template": "template",
        "application/vnd.ollama.image.license": "license",
        "application/vnd.ollama.image.params": "params",
    }
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("Ollama manifest layer must be an object")
        media_type = layer.get("mediaType")
        try:
            role = role_by_media_type[str(media_type)]
        except KeyError as error:
            raise ValueError(
                f"unsupported Ollama layer media type: {media_type}"
            ) from error
        descriptors.append((role, layer))

    normalized: list[dict[str, object]] = []
    for role, descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError("Ollama descriptor must be an object")
        digest = descriptor.get("digest")
        media_type = descriptor.get("mediaType")
        size = descriptor.get("size")
        if (
            not isinstance(digest, str)
            or not digest.startswith("sha256:")
            or not isinstance(media_type, str)
            or not isinstance(size, int)
        ):
            raise ValueError("invalid Ollama descriptor")
        normalized.append(
            {
                "role": role,
                "media_type": media_type,
                "sha256": digest.removeprefix("sha256:"),
                "size_bytes": size,
            }
        )
    return normalized


def _repo_file(repo_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("manifest artifact paths must stay relative to repo root")
    resolved_root = repo_root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("manifest artifact path escapes repo root")
    if not resolved.is_file():
        raise ValueError(f"manifest artifact does not exist: {relative_path}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "LOCAL_BASE_URL",
    "LOCAL_MODEL_ARTIFACT_PATH",
    "LOCAL_MODEL_ID",
    "LOCAL_W3_M2_MODEL_ARTIFACT_PATH",
    "LOCAL_W3_M2_MODEL_ID",
    "LOCAL_W3_M2_PRICING_PATH",
    "LOCAL_W3_M2_PRICING_SHA256",
    "LOCAL_W3_M2_PROTOCOL_PATH",
    "LOCAL_W3_M2_PROTOCOL_SHA256",
    "LOCAL_W3_M2_T1_PROTOCOL_PATH",
    "LOCAL_W3_M2_T1_PROTOCOL_SHA256",
    "LOCAL_OLLAMA_VERSION",
    "LOCAL_PRICING_PATH",
    "LOCAL_WRITER_W2_DATASET_PATH",
    "LOCAL_WRITER_W2_DATASET_SHA256",
    "LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_PATH",
    "LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256",
    "LOCAL_WRITER_W2_REFERENCE_PATH",
    "LOCAL_WRITER_W2_REFERENCE_SHA256",
    "LOCAL_WRITER_W3_DATASET_PATH",
    "LOCAL_WRITER_W3_DATASET_SHA256",
    "LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_PATH",
    "LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256",
    "LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_PATH",
    "LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256",
    "LOCAL_WRITER_W3_REFERENCE_PATH",
    "LOCAL_WRITER_W3_REFERENCE_SHA256",
    "LOCAL_PRICING_SHA256",
    "LOCAL_WRITER_REFERENCE_PATH",
    "LOCAL_WRITER_REFERENCE_SHA256",
    "LocalExperimentManifest",
    "OllamaArtifactPin",
    "load_ollama_artifact_pin",
    "require_local_only_environment",
    "validate_zero_api_pricing",
    "verify_ollama_artifact",
    "verify_static_local_inputs",
]
