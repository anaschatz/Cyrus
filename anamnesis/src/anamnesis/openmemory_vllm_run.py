"""Single-attempt external-vLLM execution for the frozen OpenMemory v4 cell."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
from functools import reduce
from pathlib import Path
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from anamnesis.openmemory_diagnostic import (
    OpenMemoryPairedRun,
    load_openmemory_diagnostic,
    openmemory_diagnostic_sha256,
    run_openmemory_decision_diagnostic,
)
from anamnesis.openmemory_vllm import (
    HttpOperatorVllmProbe,
    VllmDecisionAudit,
    VllmDecisionRuntimePin,
    VllmOpenMemoryDecisionModel,
    build_openmemory_vllm_user_envelope,
)
from anamnesis.runner import DecisionRequest
from anamnesis.schema import Decision, ObservableEvent, StrictModel, Usage
from anamnesis.vllm_runtime import (
    OpenAIExternalVllmClient,
    VllmProbeSnapshot,
    canonical_json_sha256,
    verify_vllm_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "eval/openmemory/decision_diagnostic.v4.json"
PIN_PATH = REPO_ROOT / "eval/openmemory/vllm_v4_runtime.pin.json"
PREFLIGHT_PATH = REPO_ROOT / "eval/openmemory/vllm_v4_preflight.json"
DATASET_RAW_SHA256 = "9f4fb7bdf000858c769b0702acb5585e0ef8e67eb7709bcfa2c8d83c5fbdd0d9"
DATASET_CANONICAL_SHA256 = (
    "30dc9cbdae1b399e5ca3de58b1efb8fe9c7ae448f6d38e7a2a27b727086ac524"
)
PREFLIGHT_RAW_SHA256 = (
    "1172ca39801ff19ad06ff066c4efa431e7b0dd5f4bf0c205650f5a44cad5409c"
)
PREFLIGHT_CANONICAL_SHA256 = (
    "ac0ec879b8f95e01f31fe647917d1c2ed797100ecd1be91a2c65a83b80deb81d"
)
EXPECTED_CALLS = 17


class _FrozenStrictModel(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VllmV4Preflight(_FrozenStrictModel):
    schema_version: Literal[1] = 1
    purpose: Literal["openmemory_vllm_v4_schema_canary"]
    hypothesis_test_eligible: Literal[False] = False
    event: ObservableEvent
    retrospective_recall: tuple[str, ...]
    expected_mode: Literal["no_action"]


class VllmV4CanaryResult(_FrozenStrictModel):
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Decision
    usage: Usage
    latency_ms: float = Field(ge=0)
    raw_completion: str | None
    accepted: bool
    semantic_passed: bool


class OpenMemoryVllmV4Run(_FrozenStrictModel):
    schema_version: Literal[1] = 1
    purpose: Literal["openmemory_vllm_v4_joint_compatibility"] = (
        "openmemory_vllm_v4_joint_compatibility"
    )
    hypothesis_test_eligible: Literal[False] = False
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    pin_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_raw_sha256: Literal[DATASET_RAW_SHA256] = DATASET_RAW_SHA256
    dataset_canonical_sha256: Literal[DATASET_CANONICAL_SHA256] = (
        DATASET_CANONICAL_SHA256
    )
    preflight_raw_sha256: Literal[PREFLIGHT_RAW_SHA256] = PREFLIGHT_RAW_SHA256
    preflight_canonical_sha256: Literal[PREFLIGHT_CANONICAL_SHA256] = (
        PREFLIGHT_CANONICAL_SHA256
    )
    status: Literal["preflight_failed", "complete"]
    canary: VllmV4CanaryResult
    paired_run: OpenMemoryPairedRun | None
    audits: tuple[VllmDecisionAudit, ...]
    setup_usage: Usage
    headline_usage: Usage
    total_usage: Usage
    passed: bool

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        expected_audits = 1 if self.status == "preflight_failed" else EXPECTED_CALLS
        if len(self.audits) != expected_audits:
            raise ValueError("v4 audit count differs from terminal run status")
        if self.setup_usage != self.canary.usage:
            raise ValueError("setup usage must equal the single canary call")
        if self.total_usage != self.setup_usage.plus(self.headline_usage):
            raise ValueError("total usage differs from setup plus headline")
        if self.status == "preflight_failed":
            if self.paired_run is not None or self.headline_usage != Usage():
                raise ValueError("failed preflight cannot contain scenario results")
            if self.passed:
                raise ValueError("failed preflight cannot pass")
        else:
            if self.paired_run is None:
                raise ValueError("complete v4 run requires the paired matrix")
            raw_complete = all(audit.validation.accepted for audit in self.audits)
            expected_pass = (
                self.canary.semantic_passed
                and self.paired_run.metrics.gate_passed
                and raw_complete
            )
            if self.passed != expected_pass:
                raise ValueError("v4 passed flag differs from frozen gate")
        return self


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _load_frozen_inputs() -> tuple[VllmDecisionRuntimePin, VllmV4Preflight]:
    if _sha256(DATASET_PATH) != DATASET_RAW_SHA256:
        raise ValueError("OpenMemory v4 dataset bytes differ from the frozen pin")
    dataset = load_openmemory_diagnostic(DATASET_PATH)
    if openmemory_diagnostic_sha256(dataset) != DATASET_CANONICAL_SHA256:
        raise ValueError("OpenMemory v4 dataset semantics differ from the frozen pin")
    if _sha256(PREFLIGHT_PATH) != PREFLIGHT_RAW_SHA256:
        raise ValueError("vLLM v4 preflight bytes differ from the frozen pin")
    raw_preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if _canonical_sha256(raw_preflight) != PREFLIGHT_CANONICAL_SHA256:
        raise ValueError("vLLM v4 preflight semantics differ from the frozen pin")
    preflight = VllmV4Preflight.model_validate(raw_preflight)
    pin = VllmDecisionRuntimePin.model_validate_json(
        PIN_PATH.read_text(encoding="utf-8")
    )
    return pin, preflight


def _server_config(pin: VllmDecisionRuntimePin) -> dict[str, object]:
    return {
        "artifact_manifest_sha256": pin.artifact.manifest_sha256,
        "generation_config": "vllm",
        "host": "127.0.0.1",
        "log_deltas": False,
        "log_outputs": False,
        "max_model_len": pin.max_model_len,
        "max_num_seqs": 1,
        "multimodal_mode": "text-only-compat",
        "paged_attention": True,
        "port": 18000,
        "served_model": pin.served_model,
        "speculative_decoding": False,
        "structured_output_backend": "xgrammar",
    }


def _declared_probe(pin: VllmDecisionRuntimePin) -> VllmProbeSnapshot:
    packages = {item.name: item.version for item in pin.runtime_packages}
    return VllmProbeSnapshot(
        health_ok=False,
        base_url=pin.base_url,
        vllm_version=pin.vllm_server_version,
        model_ids=(pin.served_model,),
        model_artifact_manifest_sha256=pin.artifact.manifest_sha256,
        server_config=_server_config(pin),
        runtime_packages=packages,
        structured_output_backend="xgrammar",
        generation_config="vllm",
        max_model_len=pin.max_model_len,
        max_num_seqs=1,
        speculative_decoding=False,
    )


def _verify_source_commit(repo_root: Path, source_commit: str) -> None:
    if len(source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in source_commit
    ):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != source_commit:
        raise ValueError("source_commit differs from HEAD")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("v4 run requires a clean source worktree")


def _sum_usage(usages: list[Usage]) -> Usage:
    return reduce(Usage.plus, usages, Usage())


async def run_openmemory_vllm_v4(
    *,
    artifact_root: Path,
    api_key: str,
    source_commit: str,
) -> OpenMemoryVllmV4Run:
    """Run exactly one canary and, only on success, one frozen 8x2 matrix."""

    _verify_source_commit(REPO_ROOT, source_commit)
    pin, preflight = _load_frozen_inputs()
    if canonical_json_sha256(_server_config(pin)) != pin.server_config_sha256:
        raise ValueError("server configuration differs from the frozen pin")
    verify_vllm_artifact(artifact_root, pin.artifact)
    client = OpenAIExternalVllmClient(
        base_url=pin.base_url,
        api_key=api_key,
        request_timeout_seconds=pin.request_timeout_seconds,
    )
    probe = HttpOperatorVllmProbe(
        declared=_declared_probe(pin),
        api_key=api_key,
        timeout_seconds=pin.request_timeout_seconds,
    )
    model = VllmOpenMemoryDecisionModel(
        pin=pin,
        api_key=api_key,
        artifact_root=artifact_root,
        client=client,
        probe=probe,
    )
    try:
        prompt = build_openmemory_vllm_user_envelope(
            now=preflight.event.at.isoformat(),
            current_event_id=preflight.event.id,
            context_events=[preflight.event],
            decision_history=[],
            memory_view=None,
            retrospective_recall=preflight.retrospective_recall,
        )
        call = await model.decide(DecisionRequest(event=preflight.event, prompt=prompt))
        semantic_passed = not call.decision.actions
        canary = VllmV4CanaryResult(
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            decision=call.decision,
            usage=call.usage,
            latency_ms=call.latency_ms,
            raw_completion=call.raw_completion,
            accepted=not call.parse_error,
            semantic_passed=semantic_passed,
        )
        common = {
            "source_commit": source_commit,
            "pin_sha256": _sha256(PIN_PATH),
            "canary": canary,
            "setup_usage": call.usage,
        }
        if call.parse_error or not semantic_passed:
            return OpenMemoryVllmV4Run(
                **common,
                status="preflight_failed",
                paired_run=None,
                audits=tuple(model.audits),
                headline_usage=Usage(),
                total_usage=call.usage,
                passed=False,
            )
        artifact = load_openmemory_diagnostic(DATASET_PATH)
        paired = await run_openmemory_decision_diagnostic(
            artifact,
            model=model,
            prompt_builder=build_openmemory_vllm_user_envelope,
        )
        headline_usage = _sum_usage([item.usage for item in paired.calls])
        return OpenMemoryVllmV4Run(
            **common,
            status="complete",
            paired_run=paired,
            audits=tuple(model.audits),
            headline_usage=headline_usage,
            total_usage=call.usage.plus(headline_usage),
            passed=(
                paired.metrics.gate_passed
                and all(audit.validation.accepted for audit in model.audits)
            ),
        )
    finally:
        await client.aclose()


def _contained_output(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (REPO_ROOT / "results/runs/local/openmemory_vllm_v4").resolve()
    if not resolved.is_relative_to(allowed) or resolved.suffix != ".json":
        raise ValueError("output must be a JSON file under the frozen v4 run folder")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = _contained_output(args.output)
    result = asyncio.run(
        run_openmemory_vllm_v4(
            artifact_root=args.artifact_root,
            api_key=args.api_key,
            source_commit=args.source_commit,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OpenMemoryVllmV4Run",
    "VllmV4CanaryResult",
    "VllmV4Preflight",
    "main",
    "run_openmemory_vllm_v4",
]
