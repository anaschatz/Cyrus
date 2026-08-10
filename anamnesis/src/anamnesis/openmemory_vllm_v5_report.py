"""Strict report for the two-call aligned-schema compatibility run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from anamnesis.openmemory_vllm import (
    build_openmemory_vllm_aligned_request,
    build_openmemory_vllm_user_envelope,
)
from anamnesis.openmemory_vllm_run import REPO_ROOT, _verify_source_commit
from anamnesis.openmemory_vllm_v5 import (
    FIXTURE_PATH,
    PIN_PATH,
    OpenMemoryVllmV5Run,
    _load_inputs,
    _semantic_passed,
)
from anamnesis.runner import DecisionRequest
from anamnesis.vllm_runtime import canonical_json_sha256

TITLE = "OpenMemory + vLLM v5 schema compatibility — diagnostic only"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_run(path: Path) -> OpenMemoryVllmV5Run:
    result = OpenMemoryVllmV5Run.model_validate_json(path.read_text(encoding="utf-8"))
    _verify_source_commit(REPO_ROOT, result.source_commit)
    pin, fixture, runtime = _load_inputs()
    if result.pin_sha256 != _sha256(PIN_PATH):
        raise ValueError("v5 result pin hash differs")
    if result.fixture_sha256 != pin.fixture_raw_sha256:
        raise ValueError("v5 result fixture hash differs")
    for expected, observed in zip(fixture.cases, result.cases, strict=True):
        if observed.case_id != expected.id:
            raise ValueError("v5 result case order differs")
        prompt = build_openmemory_vllm_user_envelope(
            now=expected.event.at.isoformat(),
            current_event_id=expected.event.id,
            context_events=[expected.event],
            decision_history=[],
            memory_view=None,
            retrospective_recall=expected.retrospective_recall,
        )
        request = build_openmemory_vllm_aligned_request(
            runtime, DecisionRequest(event=expected.event, prompt=prompt)
        )
        if observed.prompt_sha256 != hashlib.sha256(prompt.encode()).hexdigest():
            raise ValueError("v5 prompt hash differs")
        if observed.audit.request_sha256 != canonical_json_sha256(request):
            raise ValueError("v5 raw request hash differs")
        if observed.audit.raw_completion != observed.raw_completion:
            raise ValueError("v5 raw completion differs from audit")
        if observed.audit.usage != observed.usage:
            raise ValueError("v5 usage differs from audit")
        if observed.semantic_passed != _semantic_passed(expected, observed.decision):
            raise ValueError("v5 semantic projection differs")
    return result


def _csv_bytes(result: OpenMemoryVllmV5Run) -> bytes:
    fields = [
        "passed",
        "model_calls",
        "accepted_calls",
        "semantic_passes",
        "input_tokens",
        "output_tokens",
        "provider_api_cost_usd",
    ]
    row = {
        "passed": str(result.passed).lower(),
        "model_calls": len(result.cases),
        "accepted_calls": sum(case.audit.validation.accepted for case in result.cases),
        "semantic_passes": sum(case.semantic_passed for case in result.cases),
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "provider_api_cost_usd": result.usage.cost_usd,
    }
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue().encode()


def _markdown_bytes(result: OpenMemoryVllmV5Run) -> bytes:
    gate = "PASS" if result.passed else "FAIL"
    rows = "\n".join(
        f"| {case.case_id} | {case.expected_mode} | "
        f"{str(case.audit.validation.accepted).lower()} | "
        f"{str(case.semantic_passed).lower()} | "
        f"{case.audit.validation.error_stage or 'none'} |"
        for case in result.cases
    )
    return f"""# {TITLE}

The frozen two-call post-fix compatibility gate **{gate}**. This validates the
additive aligned JSON Schema; it is not a rerun of v4, a recall-quality result,
or a hypothesis test. Provider API cost is `${result.usage.cost_usd}`;
electricity, hardware and human review are unmeasured.

| Case | Expected | Structured accepted | Semantic pass | Error stage |
|---|---|---:|---:|---|
{rows}

Total usage: {result.usage.input_tokens} input and
{result.usage.output_tokens} output tokens. Exactly two calls were authorized,
with no retry, repair, cache, alternate schema, or v4-case reuse.
""".encode()


def _output(path: Path, suffix: str) -> Path:
    resolved = path.resolve()
    results = (REPO_ROOT / "results").resolve()
    if (
        not resolved.is_relative_to(results)
        or resolved.is_relative_to((results / "runs").resolve())
        or resolved.suffix != suffix
    ):
        raise ValueError("v5 report output must be tracked under results/")
    return resolved


def write_report(
    *, run_path: Path, csv_path: Path, markdown_path: Path, provenance_path: Path
) -> bool:
    run_path = run_path.resolve()
    allowed = (
        REPO_ROOT / "results/runs/local/openmemory_vllm_v5_compatibility"
    ).resolve()
    if not run_path.is_relative_to(allowed):
        raise ValueError("v5 run must be under its frozen run folder")
    csv_path = _output(csv_path, ".csv")
    markdown_path = _output(markdown_path, ".md")
    provenance_path = _output(provenance_path, ".json")
    result = validate_run(run_path)
    csv_data = _csv_bytes(result)
    markdown_data = _markdown_bytes(result)
    provenance = {
        "schema_version": 1,
        "title": TITLE,
        "hypothesis_test_eligible": False,
        "source_commit": result.source_commit,
        "inputs": {
            "run": {
                "path": run_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(run_path),
            },
            "fixture": {
                "path": FIXTURE_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(FIXTURE_PATH),
            },
            "pin": {
                "path": PIN_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(PIN_PATH),
            },
        },
        "outputs": {
            "csv": {
                "path": csv_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": hashlib.sha256(csv_data).hexdigest(),
            },
            "markdown": {
                "path": markdown_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": hashlib.sha256(markdown_data).hexdigest(),
            },
        },
    }
    provenance_data = (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(csv_data)
    markdown_path.write_bytes(markdown_data)
    provenance_path.write_bytes(provenance_data)
    return result.passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    return (
        0
        if write_report(
            run_path=args.run,
            csv_path=args.csv,
            markdown_path=args.markdown,
            provenance_path=args.provenance,
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TITLE", "validate_run", "write_report"]
