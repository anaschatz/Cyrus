"""Pure, project-owned routing policy for M1C.

No provider client or Pydantic graph type is allowed in this module.  The
planner may classify a task, but only this deterministic policy can select a
model tier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping

from .jcs import domain_sha256, jcs_text


class CloudContractError(ValueError):
    pass


class TaskKind(StrEnum):
    SUMMARY = "summary"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    TOOL_DECISION = "tool_decision"
    QA = "qa"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    CODING = "coding"


class ContextScope(StrEnum):
    LATEST_TURN = "latest_turn"
    CONVERSATION = "conversation"
    LARGE = "large"


class OutputContract(StrEnum):
    PLAIN_TEXT = "plain_text"
    STRICT_SCHEMA = "strict_schema"
    TOOL_CHOICE = "tool_choice"


class RouteTarget(StrEnum):
    LOCAL = "local"
    DEEPSEEK = "deepseek"
    TERRA = "terra"
    CAPABILITY_FAILURE = "capability_failure"


class RouteReason(StrEnum):
    SELECTED_LOCAL = "selected_local"
    SELECTED_DEEPSEEK = "selected_deepseek"
    SELECTED_TERRA = "selected_terra"
    LOCAL_CONTEXT_GATE = "local_context_gate"
    LOCAL_LATENCY_GATE = "local_latency_gate"
    LOCAL_BACKEND_GATE = "local_backend_gate"
    LOCAL_STRUCTURAL_GATE = "local_structural_gate"
    CLOUD_PRIVACY_DENIED = "cloud_privacy_denied"
    CLOUD_APPROVAL_REQUIRED = "cloud_approval_required"
    CLOUD_BUDGET_EXHAUSTED = "cloud_budget_exhausted"
    CLOUD_UNQUALIFIED = "cloud_unqualified"
    CLOUD_UNAVAILABLE = "cloud_unavailable"
    CAPABILITY_MISMATCH = "capability_mismatch"
    ROUTE_LOCKED_AFTER_OUTPUT = "route_locked_after_output"
    LOCAL_ONLY_FLOW = "local_only_flow"


ALLOWED_DOMAINS = frozenset(
    {
        "general",
        "software",
        "mathematics",
        "science",
        "business",
        "finance",
        "legal",
        "medical",
        "creative",
    }
)
ALLOWED_LANGUAGES = frozenset({"el", "en", "default"})


@dataclass(frozen=True, slots=True)
class TaskProfileV1:
    version: str
    task_kind: TaskKind
    reasoning_depth: int
    constraint_count: int
    domains: tuple[str, ...]
    context_scope: ContextScope
    output_contract: OutputContract
    response_language: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskProfileV1":
        expected = {
            "version",
            "task_kind",
            "reasoning_depth",
            "constraint_count",
            "domains",
            "context_scope",
            "output_contract",
            "response_language",
        }
        if set(value) != expected:
            raise CloudContractError("task_profile_fields")
        raw_domains = value["domains"]
        if not isinstance(raw_domains, (list, tuple)):
            raise CloudContractError("task_profile_domains")
        profile = cls(
            version=str(value["version"]),
            task_kind=TaskKind(value["task_kind"]),
            reasoning_depth=value["reasoning_depth"],
            constraint_count=value["constraint_count"],
            domains=tuple(raw_domains),
            context_scope=ContextScope(value["context_scope"]),
            output_contract=OutputContract(value["output_contract"]),
            response_language=str(value["response_language"]),
        )
        return profile.validate()

    def validate(self) -> "TaskProfileV1":
        if self.version != "task-profile/1":
            raise CloudContractError("task_profile_version")
        if isinstance(self.reasoning_depth, bool) or not isinstance(self.reasoning_depth, int) or not 0 <= self.reasoning_depth <= 3:
            raise CloudContractError("reasoning_depth")
        if isinstance(self.constraint_count, bool) or not isinstance(self.constraint_count, int) or not 0 <= self.constraint_count <= 8:
            raise CloudContractError("constraint_count")
        if not 1 <= len(self.domains) <= 4 or len(set(self.domains)) != len(self.domains):
            raise CloudContractError("domains")
        if any(item not in ALLOWED_DOMAINS for item in self.domains):
            raise CloudContractError("domains")
        if self.response_language not in ALLOWED_LANGUAGES:
            raise CloudContractError("response_language")
        return self

    def canonical(self) -> dict[str, Any]:
        return {
            "constraint_count": self.constraint_count,
            "context_scope": self.context_scope.value,
            "domains": list(self.domains),
            "output_contract": self.output_contract.value,
            "reasoning_depth": self.reasoning_depth,
            "response_language": self.response_language,
            "task_kind": self.task_kind.value,
            "version": self.version,
        }

    @property
    def sha256(self) -> str:
        return domain_sha256("task-profile/1", self.canonical())


_BASE = {
    TaskKind.CLASSIFICATION: 0,
    TaskKind.EXTRACTION: 2,
    TaskKind.TOOL_DECISION: 2,
    TaskKind.SUMMARY: 4,
    TaskKind.QA: 8,
    TaskKind.ANALYSIS: 18,
    TaskKind.PLANNING: 22,
    TaskKind.CODING: 24,
}
_CONTEXT_WEIGHT = {ContextScope.LATEST_TURN: 0, ContextScope.CONVERSATION: 8, ContextScope.LARGE: 20}
_OUTPUT_WEIGHT = {OutputContract.PLAIN_TEXT: 0, OutputContract.TOOL_CHOICE: 2, OutputContract.STRICT_SCHEMA: 4}


def complexity_score(profile: TaskProfileV1) -> int:
    profile.validate()
    return min(
        100,
        _BASE[profile.task_kind]
        + 12 * profile.reasoning_depth
        + 3 * profile.constraint_count
        + 5 * (len(profile.domains) - 1)
        + _CONTEXT_WEIGHT[profile.context_scope]
        + _OUTPUT_WEIGHT[profile.output_contract],
    )


def is_advanced_reasoning(profile: TaskProfileV1) -> bool:
    return (
        profile.task_kind in {TaskKind.ANALYSIS, TaskKind.PLANNING, TaskKind.CODING}
        and profile.reasoning_depth >= 2
        and complexity_score(profile) >= 70
    )


@dataclass(frozen=True, slots=True)
class CandidateFactsV1:
    alias: str
    qualified: bool
    ready: bool
    circuit_open: bool
    capabilities: tuple[str, ...]
    context_window: int
    sample_count: int
    p95_ttft_ms: int
    p95_total_ms: int
    max_reserved_microusd: int
    stable_catalog_sha256: str = ""
    qualification_sha256: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateFactsV1":
        expected = {
            "alias",
            "qualified",
            "ready",
            "circuit_open",
            "capabilities",
            "context_window",
            "sample_count",
            "p95_ttft_ms",
            "p95_total_ms",
            "max_reserved_microusd",
            "stable_catalog_sha256",
            "qualification_sha256",
        }
        if set(value) != expected or not isinstance(value["capabilities"], (list, tuple)):
            raise CloudContractError("candidate_fields")
        return cls(
            alias=value["alias"],
            qualified=value["qualified"],
            ready=value["ready"],
            circuit_open=value["circuit_open"],
            capabilities=tuple(value["capabilities"]),
            context_window=value["context_window"],
            sample_count=value["sample_count"],
            p95_ttft_ms=value["p95_ttft_ms"],
            p95_total_ms=value["p95_total_ms"],
            max_reserved_microusd=value["max_reserved_microusd"],
            stable_catalog_sha256=value["stable_catalog_sha256"],
            qualification_sha256=value["qualification_sha256"],
        ).validate()

    def validate(self) -> "CandidateFactsV1":
        if any(not isinstance(value, bool) for value in (self.qualified, self.ready, self.circuit_open)):
            raise CloudContractError("candidate_boolean_fact")
        integers = (
            self.context_window,
            self.sample_count,
            self.p95_ttft_ms,
            self.p95_total_ms,
            self.max_reserved_microusd,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in integers):
            raise CloudContractError("candidate_numeric_fact")
        if (
            not isinstance(self.alias, str)
            or not self.alias
            or any(not isinstance(item, str) or not item for item in self.capabilities)
            or len(set(self.capabilities)) != len(self.capabilities)
        ):
            raise CloudContractError("candidate_fact")
        for digest in (self.stable_catalog_sha256, self.qualification_sha256):
            if digest and (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                raise CloudContractError("candidate_digest")
        return self


@dataclass(frozen=True, slots=True)
class RoutingFactsV1:
    version: str
    task_id: str
    epoch: int
    profile_sha256: str
    estimated_input_tokens: int
    reserved_output_tokens: int
    remaining_active_ms: int
    remaining_absolute_ms: int
    durable_output_started: bool
    privacy_eligible: bool
    contains_memory: bool
    contains_research: bool
    contains_tool_result: bool
    extension_ready: bool
    connection_ready: bool
    approval_status: str
    cloud_dispatch_count: int
    monthly_remaining_microusd: int
    flow_kind: str
    local_failure: str
    local: CandidateFactsV1
    deepseek: CandidateFactsV1
    terra: CandidateFactsV1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RoutingFactsV1":
        expected = {
            "version",
            "task_id",
            "epoch",
            "profile_sha256",
            "estimated_input_tokens",
            "reserved_output_tokens",
            "remaining_active_ms",
            "remaining_absolute_ms",
            "durable_output_started",
            "privacy_eligible",
            "contains_memory",
            "contains_research",
            "contains_tool_result",
            "extension_ready",
            "connection_ready",
            "approval_status",
            "cloud_dispatch_count",
            "monthly_remaining_microusd",
            "flow_kind",
            "local_failure",
            "local",
            "deepseek",
            "terra",
        }
        if set(value) != expected:
            raise CloudContractError("routing_facts_fields")
        for name in ("local", "deepseek", "terra"):
            if not isinstance(value[name], Mapping):
                raise CloudContractError("routing_candidate")
        return cls(
            version=value["version"],
            task_id=value["task_id"],
            epoch=value["epoch"],
            profile_sha256=value["profile_sha256"],
            estimated_input_tokens=value["estimated_input_tokens"],
            reserved_output_tokens=value["reserved_output_tokens"],
            remaining_active_ms=value["remaining_active_ms"],
            remaining_absolute_ms=value["remaining_absolute_ms"],
            durable_output_started=value["durable_output_started"],
            privacy_eligible=value["privacy_eligible"],
            contains_memory=value["contains_memory"],
            contains_research=value["contains_research"],
            contains_tool_result=value["contains_tool_result"],
            extension_ready=value["extension_ready"],
            connection_ready=value["connection_ready"],
            approval_status=value["approval_status"],
            cloud_dispatch_count=value["cloud_dispatch_count"],
            monthly_remaining_microusd=value["monthly_remaining_microusd"],
            flow_kind=value["flow_kind"],
            local_failure=value["local_failure"],
            local=CandidateFactsV1.from_mapping(value["local"]),
            deepseek=CandidateFactsV1.from_mapping(value["deepseek"]),
            terra=CandidateFactsV1.from_mapping(value["terra"]),
        ).validate()

    def validate(self) -> "RoutingFactsV1":
        if self.version != "routing-facts/1" or self.epoch < 0:
            raise CloudContractError("routing_facts_version")
        integers = (
            self.estimated_input_tokens,
            self.reserved_output_tokens,
            self.remaining_active_ms,
            self.remaining_absolute_ms,
            self.cloud_dispatch_count,
            self.monthly_remaining_microusd,
        )
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in integers):
            raise CloudContractError("routing_numeric_fact")
        booleans = (
            self.durable_output_started,
            self.privacy_eligible,
            self.contains_memory,
            self.contains_research,
            self.contains_tool_result,
            self.extension_ready,
            self.connection_ready,
        )
        if any(not isinstance(value, bool) for value in booleans):
            raise CloudContractError("routing_boolean_fact")
        if self.approval_status not in {"absent", "pending", "approved", "denied"}:
            raise CloudContractError("approval_status")
        if self.flow_kind not in {"local", "memory", "research", "tool"}:
            raise CloudContractError("flow_kind")
        if self.local_failure not in {"", "context", "latency", "backend", "structural"}:
            raise CloudContractError("local_failure")
        if len(self.profile_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.profile_sha256):
            raise CloudContractError("profile_sha256")
        self.local.validate()
        self.deepseek.validate()
        self.terra.validate()
        return self

    def canonical(self) -> dict[str, Any]:
        def candidate(item: CandidateFactsV1) -> dict[str, Any]:
            raw = asdict(item)
            raw["capabilities"] = list(item.capabilities)
            return raw

        raw = asdict(self)
        raw["local"] = candidate(self.local)
        raw["deepseek"] = candidate(self.deepseek)
        raw["terra"] = candidate(self.terra)
        return raw

    @property
    def sha256(self) -> str:
        return domain_sha256("routing-facts/1", self.canonical())


@dataclass(frozen=True, slots=True)
class RoutePlanV1:
    version: str
    task_id: str
    epoch: int
    profile_sha256: str
    routing_facts_sha256: str
    complexity_score: int
    advanced_reasoning: bool
    target: RouteTarget
    reason: RouteReason

    def canonical(self) -> dict[str, Any]:
        return {
            "advanced_reasoning": self.advanced_reasoning,
            "complexity_score": self.complexity_score,
            "epoch": self.epoch,
            "profile_sha256": self.profile_sha256,
            "reason": self.reason.value,
            "routing_facts_sha256": self.routing_facts_sha256,
            "target": self.target.value,
            "task_id": self.task_id,
            "version": self.version,
        }

    @property
    def sha256(self) -> str:
        return domain_sha256("route-plan/1", self.canonical())

    def canonical_json(self) -> str:
        return jcs_text(self.canonical())


def _required_capability(profile: TaskProfileV1) -> str:
    if profile.output_contract == OutputContract.STRICT_SCHEMA:
        return "structured_output"
    if profile.output_contract == OutputContract.TOOL_CHOICE:
        return "tools"
    return "text"


def _base_candidate_eligible(candidate: CandidateFactsV1, profile: TaskProfileV1, facts: RoutingFactsV1) -> bool:
    required = _required_capability(profile)
    return (
        candidate.qualified
        and candidate.ready
        and not candidate.circuit_open
        and required in candidate.capabilities
        and facts.estimated_input_tokens + facts.reserved_output_tokens <= candidate.context_window
        and candidate.p95_total_ms + 5000 <= min(facts.remaining_active_ms, facts.remaining_absolute_ms)
    )


def local_eligible(profile: TaskProfileV1, facts: RoutingFactsV1) -> tuple[bool, RouteReason]:
    candidate = facts.local
    if not candidate.ready or candidate.circuit_open or _required_capability(profile) not in candidate.capabilities:
        return False, RouteReason.CAPABILITY_MISMATCH
    if facts.estimated_input_tokens > min(4096, candidate.context_window):
        return False, RouteReason.LOCAL_CONTEXT_GATE
    if candidate.sample_count >= 20 and (candidate.p95_ttft_ms > 2500 or candidate.p95_total_ms > 12000):
        return False, RouteReason.LOCAL_LATENCY_GATE
    if candidate.p95_total_ms + 5000 > min(facts.remaining_active_ms, facts.remaining_absolute_ms):
        return False, RouteReason.LOCAL_LATENCY_GATE
    if facts.local_failure:
        return False, RouteReason(f"local_{facts.local_failure}_gate")
    return True, RouteReason.SELECTED_LOCAL


def cloud_eligible(candidate: CandidateFactsV1, profile: TaskProfileV1, facts: RoutingFactsV1) -> tuple[bool, RouteReason]:
    if facts.durable_output_started:
        return False, RouteReason.ROUTE_LOCKED_AFTER_OUTPUT
    if facts.contains_memory or facts.contains_research or facts.contains_tool_result or not facts.privacy_eligible:
        return False, RouteReason.CLOUD_PRIVACY_DENIED
    if not facts.extension_ready or facts.approval_status == "denied":
        return False, RouteReason.CLOUD_APPROVAL_REQUIRED
    if facts.cloud_dispatch_count != 0 or facts.monthly_remaining_microusd < candidate.max_reserved_microusd:
        return False, RouteReason.CLOUD_BUDGET_EXHAUSTED
    if not candidate.qualified:
        return False, RouteReason.CLOUD_UNQUALIFIED
    if not _base_candidate_eligible(candidate, profile, facts):
        return False, RouteReason.CLOUD_UNAVAILABLE
    if facts.approval_status in {"absent", "pending"}:
        return True, RouteReason.CLOUD_APPROVAL_REQUIRED
    if not facts.connection_ready:
        return False, RouteReason.CLOUD_APPROVAL_REQUIRED
    return True, RouteReason.SELECTED_LOCAL


class DeterministicModelRouterV1:
    """Frozen pure selection policy.  It performs no I/O and has no clock."""

    @staticmethod
    def decide(profile: TaskProfileV1, facts: RoutingFactsV1) -> RoutePlanV1:
        profile.validate()
        facts.validate()
        if facts.profile_sha256 != profile.sha256:
            raise CloudContractError("profile_hash_mismatch")
        score = complexity_score(profile)
        advanced = is_advanced_reasoning(profile)

        target = RouteTarget.CAPABILITY_FAILURE
        reason = RouteReason.CAPABILITY_MISMATCH
        local_ok, local_reason = local_eligible(profile, facts)

        if facts.flow_kind != "local":
            target, reason = (RouteTarget.LOCAL, RouteReason.LOCAL_ONLY_FLOW) if local_ok else (target, local_reason)
        elif facts.durable_output_started:
            target, reason = RouteTarget.LOCAL, RouteReason.ROUTE_LOCKED_AFTER_OUTPUT
        else:
            desired = RouteTarget.LOCAL if score <= 39 else (RouteTarget.TERRA if score >= 70 and advanced else RouteTarget.DEEPSEEK)
            if facts.local_failure and desired == RouteTarget.LOCAL:
                desired = RouteTarget.DEEPSEEK
            if desired == RouteTarget.LOCAL:
                target, reason = (RouteTarget.LOCAL, RouteReason.SELECTED_LOCAL) if local_ok else (target, local_reason)
            else:
                candidate = facts.terra if desired == RouteTarget.TERRA else facts.deepseek
                cloud_ok, cloud_reason = cloud_eligible(candidate, profile, facts)
                if cloud_ok:
                    target = desired
                    reason = (
                        cloud_reason
                        if cloud_reason == RouteReason.CLOUD_APPROVAL_REQUIRED
                        else RouteReason.SELECTED_TERRA if desired == RouteTarget.TERRA else RouteReason.SELECTED_DEEPSEEK
                    )
                elif local_ok:
                    target, reason = RouteTarget.LOCAL, cloud_reason
                else:
                    reason = cloud_reason

        return RoutePlanV1(
            version="route-plan/1",
            task_id=facts.task_id,
            epoch=facts.epoch,
            profile_sha256=profile.sha256,
            routing_facts_sha256=facts.sha256,
            complexity_score=score,
            advanced_reasoning=advanced,
            target=target,
            reason=reason,
        )
