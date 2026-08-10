"""Isolated, manually annotated oracle-compiler diagnostic support.

The oracle compiler is deliberately not an evaluated Anamnesis compiler.  It
provides a gold-consistent ceiling for the deterministic store, trigger engine,
renderer, and shared decision policy.  Its annotation artifact is bound to the
sanitized observable timeline and cannot be used by headline system factories.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from anamnesis.baselines import AnamnesisMemoryStrategy
from anamnesis.io import canonical_sha256, dataset_sha256
from anamnesis.memory import (
    CompilerCall,
    CompilerRequest,
    InMemoryAnamnesis,
    MemoryDelta,
)
from anamnesis.schema import RuntimeScenario, Scenario, StrictModel, Usage

ORACLE_SYSTEM_NAME = "anamnesis_oracle_compiler"
ORACLE_COMPILER_VERSION = "oracle.v1"
ORACLE_ARTIFACT_PURPOSE = "oracle_compiler_ceiling"
ORACLE_ANNOTATION_POLICY = "causal-observable-prefix-v1"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class OracleEventDelta(StrictModel):
    """One explicit compiler result bound to one sanitized observable event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    observable_event_sha256: str = Field(pattern=_SHA256_PATTERN)
    delta: MemoryDelta


class OracleScenarioDeltas(StrictModel):
    """All and only the compiler-invoking events for one scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    events: tuple[OracleEventDelta, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_events(self) -> Self:
        event_ids = [record.event_id for record in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError(f"oracle scenario {self.scenario_id} repeats an event ID")
        return self


class OracleCompilerArtifact(StrictModel):
    """Strict annotations for a diagnostic oracle-compiler ceiling."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    purpose: Literal["oracle_compiler_ceiling"] = ORACLE_ARTIFACT_PURPOSE
    claim_scope: Literal["diagnostic_development_only"] = "diagnostic_development_only"
    hypothesis_test_eligible: Literal[False] = False
    annotation_policy: Literal["causal-observable-prefix-v1"] = ORACLE_ANNOTATION_POLICY
    canonical_dataset_sha256: str = Field(pattern=_SHA256_PATTERN)
    scenarios: tuple[OracleScenarioDeltas, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_scenarios_and_events(self) -> Self:
        scenario_ids = [record.scenario_id for record in self.scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("oracle artifact repeats a scenario ID")
        event_ids = [
            event.event_id for scenario in self.scenarios for event in scenario.events
        ]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("oracle event IDs must be globally unique")
        return self

    def records_for(self, scenario: RuntimeScenario) -> tuple[OracleEventDelta, ...]:
        """Return records after exact observable coverage and hash validation."""

        matches = [
            record for record in self.scenarios if record.scenario_id == scenario.id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"oracle artifact does not identify scenario exactly once: "
                f"{scenario.id}"
            )
        records = matches[0].events
        observable_events = [
            event for event in scenario.events if event.kind != "clock_tick"
        ]
        expected_ids = [event.id for event in observable_events]
        actual_ids = [record.event_id for record in records]
        if actual_ids != expected_ids:
            raise ValueError(
                f"oracle event coverage/order differs for scenario {scenario.id}"
            )
        for event, record in zip(observable_events, records, strict=True):
            if record.observable_event_sha256 != canonical_sha256(event):
                raise ValueError(
                    f"oracle observable-event hash differs for {record.event_id}"
                )
        return records

    def validate_runtime_scenarios(self, scenarios: Sequence[RuntimeScenario]) -> None:
        """Require the artifact to cover one ordered sanitized dataset exactly."""

        expected_ids = [scenario.id for scenario in scenarios]
        actual_ids = [record.scenario_id for record in self.scenarios]
        if actual_ids != expected_ids:
            raise ValueError("oracle scenario coverage/order differs from dataset")
        for scenario in scenarios:
            self.records_for(scenario)


def load_oracle_artifact(
    path: str | Path,
    scenarios: Sequence[Scenario],
) -> OracleCompilerArtifact:
    """Load annotations and bind them to exact full and sanitized dataset bytes."""

    artifact = OracleCompilerArtifact.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )
    scenario_list = list(scenarios)
    if artifact.canonical_dataset_sha256 != dataset_sha256(scenario_list):
        raise ValueError("oracle artifact canonical dataset hash differs")
    artifact.validate_runtime_scenarios(
        [scenario.to_runtime() for scenario in scenario_list]
    )
    return artifact


class OracleCompiler:
    """Replay explicit per-event deltas with complete zero-provider accounting."""

    name = ORACLE_COMPILER_VERSION

    def __init__(
        self,
        artifact: OracleCompilerArtifact,
        scenario: RuntimeScenario,
    ) -> None:
        self._scenario = scenario
        self._records = artifact.records_for(scenario)
        self.requests: list[CompilerRequest] = []
        self.reset()

    def reset(self) -> None:
        self._next_record = 0
        self.requests = []

    @property
    def remaining_event_ids(self) -> tuple[str, ...]:
        return tuple(record.event_id for record in self._records[self._next_record :])

    def assert_complete(self) -> None:
        if self.remaining_event_ids:
            raise ValueError(
                "oracle compiler did not consume all annotated events: "
                f"{list(self.remaining_event_ids)}"
            )

    async def compile(self, request: CompilerRequest) -> CompilerCall:
        started = perf_counter()
        if request.event.kind == "clock_tick":
            raise ValueError("oracle compiler cannot be called for a clock tick")
        if self._next_record >= len(self._records):
            raise ValueError("oracle compiler received an extra observable event")
        record = self._records[self._next_record]
        if request.event.id != record.event_id:
            raise ValueError(
                "oracle compiler event order differs: "
                f"expected {record.event_id}, received {request.event.id}"
            )
        if canonical_sha256(request.event) != record.observable_event_sha256:
            raise ValueError(
                f"oracle compiler observable-event hash differs for {record.event_id}"
            )
        self._next_record += 1
        self.requests.append(request)
        raw = record.delta.model_dump_json()
        latency_ms = max(0.0, (perf_counter() - started) * 1000)
        return CompilerCall(
            delta=record.delta,
            usage=Usage(cost_usd=0.0),
            latency_ms=latency_ms,
            parse_error=False,
            raw_completion=raw,
            usage_complete=True,
            cost_complete=True,
        )


class OracleAnamnesisMemoryStrategy(AnamnesisMemoryStrategy):
    """Distinct diagnostic strategy that cannot masquerade as Anamnesis."""

    name = ORACLE_SYSTEM_NAME

    def __init__(
        self,
        compiler: OracleCompiler,
        memory: InMemoryAnamnesis | None = None,
    ) -> None:
        super().__init__(compiler=compiler, memory=memory)
        self.compiler = compiler

    def reset(self) -> None:
        self.compiler.reset()
        super().reset()


def oracle_artifact_sha256(artifact: OracleCompilerArtifact) -> str:
    """Fingerprint canonical annotation semantics independently of formatting."""

    return canonical_sha256(artifact)


__all__ = [
    "ORACLE_ANNOTATION_POLICY",
    "ORACLE_ARTIFACT_PURPOSE",
    "ORACLE_COMPILER_VERSION",
    "ORACLE_SYSTEM_NAME",
    "OracleAnamnesisMemoryStrategy",
    "OracleCompiler",
    "OracleCompilerArtifact",
    "OracleEventDelta",
    "OracleScenarioDeltas",
    "load_oracle_artifact",
    "oracle_artifact_sha256",
]
