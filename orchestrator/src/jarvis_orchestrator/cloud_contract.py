"""Immutable M1C endpoint, dispatch, approval and cost contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from typing import Any, Mapping
import uuid
from datetime import datetime, timezone

from .jcs import domain_sha256, jcs_text


DEEPSEEK_MODEL = "deepseek/deepseek-v4-flash"
TERRA_MODEL = "openai/gpt-5.6-terra"
MONTHLY_HARD_CAP_MICROUSD = 5_000_000
TERRA_INPUT_FLOOR_PER_M = Decimal("2.50")
TERRA_OUTPUT_FLOOR_PER_M = Decimal("15")


class CloudBindingError(ValueError):
    pass


def _digest(value: str, field: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CloudBindingError(field)
    return value


def _uuid4(value: str, field: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise CloudBindingError(field) from exc
    if parsed.version != 4 or str(parsed) != value.lower():
        raise CloudBindingError(field)
    return value.lower()


def normalize_decimal(value: Decimal | str) -> str:
    selected = value if isinstance(value, Decimal) else Decimal(value)
    if selected < 0 or not selected.is_finite():
        raise CloudBindingError("decimal")
    text = format(selected.normalize(), "f")
    return "0" if text in {"-0", ""} else text


def reserve_cost_microusd(
    *,
    prompt_cap: int,
    completion_cap: int,
    prompt_rate_per_m: Decimal | str,
    completion_rate_per_m: Decimal | str,
    request_fee_usd: Decimal | str = "0",
) -> int:
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (prompt_cap, completion_cap)):
        raise CloudBindingError("token_cap")
    raw = (
        Decimal(request_fee_usd)
        + Decimal(prompt_cap) * Decimal(prompt_rate_per_m) / Decimal(1_000_000)
        + Decimal(completion_cap) * Decimal(completion_rate_per_m) / Decimal(1_000_000)
    )
    return int((raw * Decimal("1.10") * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING))


@dataclass(frozen=True, slots=True)
class QualifiedEndpointV1:
    model_canonical_id: str
    provider_endpoint_tag: str
    quantization: str
    supported_parameters: tuple[str, ...]
    qualification_version: str
    qualification_sha256: str
    context_limit: int
    output_limit: int
    prompt_rate_per_m: str
    completion_rate_per_m: str
    request_rate: str
    runtime_provider_name: str

    def validate(self) -> "QualifiedEndpointV1":
        if self.model_canonical_id not in {DEEPSEEK_MODEL, TERRA_MODEL}:
            raise CloudBindingError("model")
        if not self.provider_endpoint_tag or not self.quantization or not self.qualification_version or not self.runtime_provider_name:
            raise CloudBindingError("endpoint_identity")
        if len(self.supported_parameters) != len(set(self.supported_parameters)):
            raise CloudBindingError("supported_parameters")
        if self.context_limit <= 0 or self.output_limit <= 0:
            raise CloudBindingError("limits")
        _digest(self.qualification_sha256, "qualification_sha256")
        for rate in (self.prompt_rate_per_m, self.completion_rate_per_m, self.request_rate):
            normalize_decimal(rate)
        if Decimal(self.request_rate) != 0:
            raise CloudBindingError("request_rate")
        required = (
            {"max_tokens", "reasoning", "temperature", "top_p"}
            if self.model_canonical_id == DEEPSEEK_MODEL
            else {"max_completion_tokens", "reasoning"}
        )
        if not required.issubset(self.supported_parameters):
            raise CloudBindingError("supported_parameters")
        return self

    @property
    def parameter_profile_sha256(self) -> str:
        return domain_sha256(
            "supported-parameter-profile/1",
            {"parameters": sorted(self.supported_parameters)},
        )

    @property
    def stable_catalog_sha256(self) -> str:
        self.validate()
        return domain_sha256(
            "stable-endpoint-catalog/1",
            {
                "completion_rate_per_m": normalize_decimal(self.completion_rate_per_m),
                "context_limit": self.context_limit,
                "model_canonical_id": self.model_canonical_id,
                "output_limit": self.output_limit,
                "parameter_profile_sha256": self.parameter_profile_sha256,
                "prompt_rate_per_m": normalize_decimal(self.prompt_rate_per_m),
                "provider_endpoint_tag": self.provider_endpoint_tag,
                "qualification_sha256": self.qualification_sha256,
                "qualification_version": self.qualification_version,
                "quantization": self.quantization,
                "request_rate": normalize_decimal(self.request_rate),
                "runtime_provider_name": self.runtime_provider_name,
                "supported_parameters": sorted(self.supported_parameters),
                "zdr": True,
            },
        )


@dataclass(frozen=True, slots=True)
class EndpointHealthV1:
    model_canonical_id: str
    provider_endpoint_tag: str
    status: int
    uptime_last_30m_basis_points: int
    p90_latency_ms: int
    observed_at: str
    age_ms: int

    def eligible(self) -> bool:
        latency_limit = 15_000 if self.model_canonical_id == DEEPSEEK_MODEL else 45_000
        return (
            self.status == 0
            and self.uptime_last_30m_basis_points >= 9900
            and 0 <= self.age_ms <= 60_000
            and 0 <= self.p90_latency_ms <= latency_limit
        )

    @property
    def health_sha256(self) -> str:
        return domain_sha256(
            "endpoint-health/1",
            {
                "age_ms": self.age_ms,
                "model_canonical_id": self.model_canonical_id,
                "observed_at": self.observed_at,
                "p90_latency_ms": self.p90_latency_ms,
                "provider_endpoint_tag": self.provider_endpoint_tag,
                "status": self.status,
                "uptime_last_30m_basis_points": self.uptime_last_30m_basis_points,
            },
        )


@dataclass(frozen=True, slots=True)
class OpenRouterRevalidationV1:
    endpoint: QualifiedEndpointV1
    health: EndpointHealthV1
    key_remaining_microusd: int
    usage_monthly_microusd: int
    limit_reset: str
    key_limit_microusd: int = 5_000_000
    key_policy_verified: bool = True


@dataclass(frozen=True, slots=True)
class CloudModelResultV1:
    text: str
    generation_id: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    actual_microusd: int
    receipt: Mapping[str, Any]


def provider_object(endpoint: QualifiedEndpointV1) -> dict[str, Any]:
    endpoint.validate()
    return {
        "allow_fallbacks": False,
        "data_collection": "deny",
        "max_price": {
            "completion": normalize_decimal(endpoint.completion_rate_per_m),
            "prompt": normalize_decimal(endpoint.prompt_rate_per_m),
            "request": "0",
        },
        "only": [endpoint.provider_endpoint_tag],
        "order": [endpoint.provider_endpoint_tag],
        "quantizations": [endpoint.quantization],
        "require_parameters": True,
        "zdr": True,
    }


def build_dispatch_body(
    endpoint: QualifiedEndpointV1,
    *,
    messages: tuple[Mapping[str, str], ...],
    output_token_limit: int,
) -> dict[str, Any]:
    endpoint.validate()
    if output_token_limit < 256 or output_token_limit > endpoint.output_limit:
        raise CloudBindingError("output_token_limit")
    normalized_messages: list[dict[str, str]] = []
    for message in messages:
        if set(message) != {"role", "content"} or message["role"] not in {"system", "user", "assistant"}:
            raise CloudBindingError("message")
        normalized_messages.append({"content": message["content"], "role": message["role"]})
    body: dict[str, Any] = {
        "messages": normalized_messages,
        "model": endpoint.model_canonical_id,
        "provider": provider_object(endpoint),
        "stream": False,
    }
    if endpoint.model_canonical_id == DEEPSEEK_MODEL:
        body.update(
            {
                "max_tokens": output_token_limit,
                "reasoning": {"enabled": False},
                "temperature": 0,
                "top_p": 1,
            }
        )
    else:
        body.update(
            {
                "max_completion_tokens": output_token_limit,
                "reasoning": {"effort": "medium"},
            }
        )
    return body


@dataclass(frozen=True, slots=True)
class CloudApprovalBindingV1:
    conversation_id: str
    task_id: str
    epoch: int
    expires_at: str
    egress_sha256: str
    route_plan_sha256: str
    dispatch_body_sha256: str
    model_canonical_id: str
    provider_endpoint_tag: str
    quantization: str
    parameter_profile_sha256: str
    qualification_version: str
    qualification_sha256: str
    stable_catalog_sha256: str
    privacy_policy_sha256: str
    billable_prompt_tokens: int
    input_token_limit: int
    output_token_limit: int
    max_reserved_microusd: int
    included_turn_ids: tuple[str, ...]

    def canonical(self) -> dict[str, Any]:
        _uuid4(self.conversation_id, "conversation_id")
        _uuid4(self.task_id, "task_id")
        for turn_id in self.included_turn_ids:
            _uuid4(turn_id, "included_turn_ids")
        try:
            expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise CloudBindingError("expires_at") from exc
        if expiry.tzinfo is None or expiry.utcoffset() != timezone.utc.utcoffset(expiry):
            raise CloudBindingError("expires_at")
        numeric = (
            self.epoch,
            self.billable_prompt_tokens,
            self.input_token_limit,
            self.output_token_limit,
            self.max_reserved_microusd,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in numeric):
            raise CloudBindingError("numeric_binding_field")
        if self.model_canonical_id == DEEPSEEK_MODEL:
            if (
                not 1 <= self.billable_prompt_tokens <= self.input_token_limit <= 128_000
                or not 256 <= self.output_token_limit <= 2048
                or self.max_reserved_microusd > 20_000
            ):
                raise CloudBindingError("deepseek_limits")
        elif self.model_canonical_id == TERRA_MODEL:
            if (
                not 1 <= self.billable_prompt_tokens <= self.input_token_limit <= 60_000
                or not 256 <= self.output_token_limit <= 2000
                or self.max_reserved_microusd > 200_000
            ):
                raise CloudBindingError("terra_limits")
        else:
            raise CloudBindingError("model_canonical_id")
        return {
            "billable_prompt_tokens": self.billable_prompt_tokens,
            "conversation_id": self.conversation_id,
            "dispatch_body_sha256": _digest(self.dispatch_body_sha256, "dispatch_body_sha256"),
            "egress_sha256": _digest(self.egress_sha256, "egress_sha256"),
            "epoch": self.epoch,
            "expires_at": self.expires_at,
            "included_turn_ids": list(self.included_turn_ids),
            "input_token_limit": self.input_token_limit,
            "max_reserved_microusd": self.max_reserved_microusd,
            "model_canonical_id": self.model_canonical_id,
            "output_token_limit": self.output_token_limit,
            "parameter_profile_sha256": _digest(self.parameter_profile_sha256, "parameter_profile_sha256"),
            "privacy_policy_sha256": _digest(self.privacy_policy_sha256, "privacy_policy_sha256"),
            "provider_endpoint_tag": self.provider_endpoint_tag,
            "qualification_sha256": _digest(self.qualification_sha256, "qualification_sha256"),
            "qualification_version": self.qualification_version,
            "quantization": self.quantization,
            "route_plan_sha256": _digest(self.route_plan_sha256, "route_plan_sha256"),
            "stable_catalog_sha256": _digest(self.stable_catalog_sha256, "stable_catalog_sha256"),
            "task_id": self.task_id,
            "version": "cloud-approval-binding/1",
        }

    @property
    def sha256(self) -> str:
        return domain_sha256("cloud-approval-binding/1", self.canonical())

    def canonical_json(self) -> str:
        return jcs_text(self.canonical())
