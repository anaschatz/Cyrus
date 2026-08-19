"""Content-free entry gate for the first owner-only macOS beta session."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .config import OrchestratorConfig
from .private_state import PrivateStateReportV1, audit_orchestrator_state
from .stack_config import PrivateStackConfig

if TYPE_CHECKING:
    from jarvis_voice.private_state import VoicePrivateStateReportV1


BETA_PREFLIGHT_VERSION = "jarvis-macos-beta-preflight/3"
VOICE_PROBE_SCOPE = "current_process_identity"
_GENERIC_VOICE_BLOCKER = "voice.environment_not_ready"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REASON_CODE_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_.-")


def _valid_reason_code(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and value[0].isalpha()
        and all(character in _REASON_CODE_CHARACTERS for character in value)
    )


def _voice_blocker_codes(doctor: Mapping[str, Any]) -> list[str]:
    """Extract only stable codes from a Voice doctor report.

    Paths, device identifiers and exception strings are deliberately ignored.
    A malformed or unexpectedly shaped report fails closed behind one generic
    content-free reason code.
    """

    if doctor.get("ready") is True:
        return []
    checks = doctor.get("checks")
    if not isinstance(checks, Mapping):
        return [_GENERIC_VOICE_BLOCKER]
    pending: list[Mapping[str, Any]] = [checks]
    visited: set[int] = set()
    blockers: set[str] = set()
    malformed_blocker = False
    while pending and len(visited) < 256:
        item = pending.pop()
        identity = id(item)
        if identity in visited:
            continue
        visited.add(identity)
        if item.get("status") in {"blocked", "inconclusive"}:
            code = item.get("code")
            if _valid_reason_code(code):
                blockers.add(str(code))
            else:
                malformed_blocker = True
        for child in item.values():
            if isinstance(child, Mapping):
                pending.append(child)
    if pending or malformed_blocker or not blockers:
        blockers.add(_GENERIC_VOICE_BLOCKER)
    return sorted(blockers)


def _feature_scope(
    config: OrchestratorConfig,
    *,
    memory_crypto_state_sha256: str | None,
) -> tuple[bool, list[dict[str, Any]]]:
    memory_qualified = not config.memory_enabled or (
        isinstance(memory_crypto_state_sha256, str)
        and _SHA256.fullmatch(memory_crypto_state_sha256) is not None
    )
    disabled = {
        "cloud": (
            not config.cloud_enabled
            and not config.cloud_terra_enabled
            and config.cloud_secret_file is None
            and config.cloud_qualification_file is None
            and config.cloud_release_qualification_file is None
            and not config.cloud_release_qualification_sha256
        ),
        "research": (
            not config.research_enabled
            and not config.research_browser_fallback
            and not config.research_bearer_token
            and not config.research_model_sha256
            and not config.research_profile_sha256
            and not config.research_qualification_sha256
            and not config.research_browser_qualification_sha256
        ),
        "tools": (
            not config.tools_enabled
            and config.tool_roots is None
            and not config.tool_app_ids
            and not config.tool_policy_sha256
            and not config.scheduler_contract_sha256
            and not config.scheduler_qualification_manifest_sha256
            and config.action_host_socket_path is None
            and config.action_host_token_file is None
            and config.action_host_qualification_file is None
            and not config.action_host_qualification_sha256
        ),
        "scheduler_generation_and_delivery": (
            config.scheduler_compiler_qualification_file is None
            and not config.scheduler_compiler_qualification_sha256
            and not config.voice_scheduler_compatibility_sha256
            and not config.voice_scheduler_callback_qualification_sha256
        ),
        "prospective_memory": not config.memory_shadow_enabled,
    }
    capabilities: list[dict[str, Any]] = [
        {
            "desired": "READY",
            "effective": (
                "READY"
                if config.memory_enabled and memory_qualified
                else "BLOCKED"
                if config.memory_enabled
                else "DISABLED"
            ),
            "name": "memory.retrospective",
            "reason_code": (
                "memory_crypto_runtime_qualified"
                if config.memory_enabled and memory_qualified
                else "memory_crypto_runtime_unqualified"
                if config.memory_enabled
                else "disabled_by_configuration"
            ),
        }
    ]
    for name, ready in disabled.items():
        capabilities.append(
            {
                "desired": "DISABLED",
                "effective": "DISABLED" if ready else "BLOCKED",
                "name": name,
                "reason_code": (
                    "disabled_by_safe_beta_scope"
                    if ready
                    else "disabled_capability_has_runtime_residue"
                ),
            }
        )
    return all(disabled.values()) and memory_qualified, capabilities


def _live_memory_crypto_state(config: OrchestratorConfig) -> str | None:
    """Explicit preflight-only native attestation; never used by request health."""

    if not config.memory_enabled:
        return None
    try:
        from .cli import _memory_crypto_identity_state

        return _memory_crypto_identity_state(config)
    except Exception:  # noqa: BLE001 - one content-free blocked capability.
        return None


def evaluate_macos_beta_preflight(
    stack: PrivateStackConfig,
    *,
    doctor_report: Mapping[str, Any] | None = None,
    private_state_report: PrivateStateReportV1 | None = None,
    voice_private_state_report: VoicePrivateStateReportV1 | None = None,
) -> dict[str, Any]:
    """Evaluate static/native prerequisites without starting any service.

    This gate intentionally qualifies only a technical integration beta.  The
    current stack schema carries appliance safety evidence, but not a passing
    bilingual conversational-quality report, so daily-owner readiness remains
    false until that separate artifact is added and verified.
    """

    from jarvis_voice.config import load_voice_config
    from jarvis_voice.macos.doctor import macos_doctor_report
    from jarvis_voice.private_state import audit_voice_journal_state

    orchestrator = OrchestratorConfig.load(stack.orchestrator_config_path)
    voice = load_voice_config(stack.voice_config_path)
    doctor = dict(doctor_report or macos_doctor_report(voice))
    private_state = private_state_report or audit_orchestrator_state(
        orchestrator.database_path
    )
    voice_private_state = voice_private_state_report or audit_voice_journal_state()
    selected_memory_state = _live_memory_crypto_state(orchestrator)
    safe_scope, capabilities = _feature_scope(
        orchestrator,
        memory_crypto_state_sha256=selected_memory_state,
    )
    memory_crypto_qualified = not orchestrator.memory_enabled or (
        isinstance(selected_memory_state, str)
        and _SHA256.fullmatch(selected_memory_state) is not None
    )
    selected_voice_profile = (
        voice.macos.asr_backend == "faster_whisper_cpu_smoke"
        and voice.macos.tts_backend == "system_say"
        and voice.ptt_key in {"CONTROL_OPTION", "RIGHT_OPTION"}
    )
    voice_blockers = _voice_blocker_codes(doctor)
    checks = {
        "deployment_identity_bound": bool(stack.sha256),
        "inference_appliance_qualified": True,
        "memory_crypto_qualified": memory_crypto_qualified,
        "owner_only_state": (private_state.ready and voice_private_state.ready),
        "safe_feature_scope": safe_scope,
        "selected_voice_profile": selected_voice_profile,
        "voice_environment": doctor.get("ready") is True,
    }
    technical_ready = all(checks.values())
    identity = stack.deployment_identity()
    return {
        "capabilities": capabilities,
        "checks": checks,
        "content_captured": False,
        "deployment_sha256": identity["deployment_sha256"],
        "daily_owner_beta_ready": False,
        "daily_owner_blocker": "bilingual_conversational_quality_not_qualified",
        "network_used": False,
        "memory_crypto_state_sha256": (
            selected_memory_state if orchestrator.memory_enabled else None
        ),
        "platform": "macos-arm64",
        "production_ready": False,
        "scope": "technical-integration",
        "technical_beta_preflight_ready": technical_ready,
        "version": BETA_PREFLIGHT_VERSION,
        "voice_environment_blockers": voice_blockers,
        "voice_environment_probe_scope": VOICE_PROBE_SCOPE,
    }


__all__ = ["BETA_PREFLIGHT_VERSION", "evaluate_macos_beta_preflight"]
