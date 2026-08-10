"""Frozen experiment-manifest contract for reproducible measured runs."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from anamnesis.schema import StrictModel

SHA256_PATTERN = r"^[0-9a-f]{64}$"
GIT_COMMIT_PATTERN = r"^[0-9a-f]{40}$"

SIMPLE_SYSTEMS = {"no_memory", "full_context", "vector_rag"}
FINAL_SYSTEMS = SIMPLE_SYSTEMS | {"anamnesis"}
BASELINE_SEEDS = [101]
FINAL_SEEDS = [101, 202, 303]
PREREGISTERED_SEEDS = FINAL_SEEDS


class ArtifactPin(StrictModel):
    """A file or dataset and the digest of the exact bytes used."""

    path: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class ModelProviderArgs(StrictModel):
    """Exact provider route required by the v0 hosted-model protocol."""

    responses_api: Literal[False]


class ModelPin(StrictModel):
    """The hosted snapshot and pricing inputs used for all compared systems."""

    snapshot: str | None = Field(default=None, min_length=1)
    strict_structured_output: bool = True
    provider_args: ModelProviderArgs
    pricing: ArtifactPin
    preflight: ArtifactPin


class EmbeddingPin(StrictModel):
    """Immutable FastEmbed source and verified local snapshot content."""

    model: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    artifact_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    top_k: int = Field(default=5, ge=1)


class ExecutionPolicy(StrictModel):
    """Preregistered execution settings that must not drift among systems."""

    temperature: float = Field(default=0.0, ge=0, le=0)
    seeds: list[int] = Field(default_factory=lambda: list(BASELINE_SEEDS))
    repetitions: int = Field(default=1, ge=1, le=3)
    response_cache: bool = False
    max_samples: int = Field(default=1, ge=1, le=1)
    concurrency: int = Field(default=1, ge=1, le=1)
    max_retries: Literal[0] = 0
    log_model_api: Literal[True] = True
    warmup: Literal["prewarmed"] = "prewarmed"

    @model_validator(mode="after")
    def validate_seeds(self) -> Self:
        if len(self.seeds) != self.repetitions or len(set(self.seeds)) != len(
            self.seeds
        ):
            raise ValueError("execution seeds must be unique and match repetitions")
        return self


class ExperimentManifest(StrictModel):
    """A draft or frozen declaration of one comparable experiment matrix."""

    schema_version: Literal[1] = 1
    status: Literal["draft", "frozen"] = "draft"
    phase: Literal["baseline", "final"]
    dataset: ArtifactPin
    scenario_count: int = Field(ge=1)
    sealed_opened: bool
    systems: list[str] = Field(min_length=1)
    model: ModelPin
    embedding: EmbeddingPin
    execution: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    git_commit: str | None = Field(default=None, pattern=GIT_COMMIT_PATTERN)
    dependency_lock: ArtifactPin
    research_contract: ArtifactPin
    architecture_contract: ArtifactPin
    decision_prompt_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    memory_compiler_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    system_config_sha256: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        expected_systems = SIMPLE_SYSTEMS if self.phase == "baseline" else FINAL_SYSTEMS
        expected_count = 35 if self.phase == "baseline" else 50
        expected_sealed = self.phase == "final"
        expected_seeds = BASELINE_SEEDS if self.phase == "baseline" else FINAL_SEEDS

        if set(self.systems) != expected_systems or len(self.systems) != len(
            expected_systems
        ):
            raise ValueError(
                f"{self.phase} phase requires systems "
                f"{sorted(expected_systems)} exactly"
            )
        if self.scenario_count != expected_count:
            raise ValueError(f"{self.phase} phase requires {expected_count} scenarios")
        if self.sealed_opened is not expected_sealed:
            raise ValueError(
                f"sealed_opened must be {expected_sealed} for {self.phase} phase"
            )
        if self.execution.seeds != expected_seeds:
            raise ValueError(
                f"{self.phase} phase requires seeds {expected_seeds} exactly"
            )
        if self.execution.repetitions != len(expected_seeds):
            raise ValueError(
                f"{self.phase} phase requires {len(expected_seeds)} repetitions"
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
                    "frozen manifest is missing reproducibility inputs: "
                    + ", ".join(missing)
                )
        return self

    def _missing_freeze_inputs(self, expected_systems: set[str]) -> list[str]:
        missing: list[str] = []
        if not self.model.snapshot:
            missing.append("model.snapshot")
        if not self.model.strict_structured_output:
            missing.append("model.strict_structured_output")
        for name, artifact in (
            ("dataset", self.dataset),
            ("model.pricing", self.model.pricing),
            ("model.preflight", self.model.preflight),
            ("dependency_lock", self.dependency_lock),
            ("research_contract", self.research_contract),
            ("architecture_contract", self.architecture_contract),
        ):
            if artifact.sha256 is None:
                missing.append(f"{name}.sha256")
        if self.git_commit is None:
            missing.append("git_commit")
        if self.decision_prompt_sha256 is None:
            missing.append("decision_prompt_sha256")
        if self.phase == "final" and self.memory_compiler_sha256 is None:
            missing.append("memory_compiler_sha256")
        if self.embedding.revision is None:
            missing.append("embedding.revision")
        if self.embedding.artifact_sha256 is None:
            missing.append("embedding.artifact_sha256")
        if set(self.system_config_sha256) != expected_systems:
            missing.append("system_config_sha256")
        return missing


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
