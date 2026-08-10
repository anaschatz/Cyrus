"""Strict report for the prospective canonicalized-memory diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from anamnesis.action_canonicalizer import canonicalize_immediate_decision
from anamnesis.openmemory_vllm import (
    build_openmemory_vllm_aligned_request,
    build_openmemory_vllm_user_envelope,
)
from anamnesis.openmemory_vllm_run import REPO_ROOT, _verify_source_commit
from anamnesis.openmemory_vllm_v6 import _correct
from anamnesis.openmemory_vllm_v7 import (
    FIXTURE_PATH,
    PIN_PATH,
    V7Run,
    _load_inputs,
)
from anamnesis.runner import DecisionRequest
from anamnesis.vllm_runtime import canonical_json_sha256

TITLE = "OpenMemory canonicalizer v7 prospective diagnostic — not a hypothesis test"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_run(path: Path) -> V7Run:
    result = V7Run.model_validate_json(path.read_text())
    _verify_source_commit(REPO_ROOT, result.source_commit)
    pin, fixture, runtime = _load_inputs()
    if result.pin_sha256 != _sha256(PIN_PATH):
        raise ValueError("v7 pin hash differs")
    for case, observed in zip(fixture.cases, result.cases, strict=True):
        if observed.case_id != case.id:
            raise ValueError("v7 case order differs")
        if observed.retrieved_memory_ids != case.expected_retrieved_memory_ids:
            raise ValueError("v7 retrieval differs")
        expected_contents = tuple(
            next(item.content for item in case.memories if item.id == memory_id)
            for memory_id in case.expected_retrieved_memory_ids
        )
        if observed.retrieved_contents != expected_contents:
            raise ValueError("v7 retrieved content differs")
        for arm, recall in (
            (observed.baseline, None),
            (observed.recall, observed.retrieved_contents),
        ):
            prompt = build_openmemory_vllm_user_envelope(
                now=case.event.at.isoformat(),
                current_event_id=case.event.id,
                context_events=[case.event],
                decision_history=[],
                memory_view=None,
                retrospective_recall=recall,
            )
            request = build_openmemory_vllm_aligned_request(
                runtime, DecisionRequest(event=case.event, prompt=prompt)
            )
            normalized = canonicalize_immediate_decision(
                event=case.event,
                retrospective_recall=recall,
                decision=arm.raw_decision,
            )
            checks = (
                arm.prompt_sha256 == hashlib.sha256(prompt.encode()).hexdigest(),
                arm.audit.request_sha256 == canonical_json_sha256(request),
                arm.canonical_decision == normalized.decision,
                arm.changes == normalized.changes,
                arm.correct == _correct(case.expected, normalized.decision),
                arm.audit.usage == arm.usage,
            )
            if not all(checks):
                raise ValueError("v7 request, normalization, score, or usage differs")
    return result


def _csv(result: V7Run) -> bytes:
    fields = [
        "passed",
        "cases",
        "retrieval_correct",
        "baseline_correct",
        "recall_correct",
        "helpful_gain",
        "safety_regressions",
        "accepted_calls",
        "canonicalizer_changes",
        "input_tokens",
        "output_tokens",
        "provider_api_cost_usd",
    ]
    row = result.metrics.model_dump() | {
        "passed": str(result.passed).lower(),
        "cases": len(result.cases),
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "provider_api_cost_usd": result.usage.cost_usd,
    }
    row.pop("gate_passed")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue().encode()


def _markdown(result: V7Run) -> bytes:
    rows = "\n".join(
        f"| {case.case_id} | {','.join(case.retrieved_memory_ids) or 'none'} | "
        f"{str(case.baseline.correct).lower()} | "
        f"{str(case.recall.correct).lower()} | "
        f"{len(case.baseline.changes)}/{len(case.recall.changes)} |"
        for case in result.cases
    )
    return f"""# {TITLE}

The one authorized fresh paired run **{"PASS" if result.passed else "FAIL"}**.
Both arms used the same frozen source-grounded canonicalizer; raw decisions,
canonical decisions and transformations are retained. This is a development
diagnostic over the local indexed OpenMemory-compatible boundary, not the
upstream Cavira SDK or a hypothesis test.

| Case | Retrieved memory | Baseline correct | Recall correct | Changes B/R |
|---|---|---:|---:|---:|
{rows}

Retrieval {result.metrics.retrieval_correct}/6; baseline
{result.metrics.baseline_correct}/6; recall {result.metrics.recall_correct}/6;
helpful gain {result.metrics.helpful_gain}; safety regressions
{result.metrics.safety_regressions}; accepted calls
{result.metrics.accepted_calls}/12; canonicalizer transformations
{result.metrics.canonicalizer_changes}. Usage: {result.usage.input_tokens} input
and {result.usage.output_tokens} output tokens at `${result.usage.cost_usd}`
provider API cost. Electricity and hardware are unmeasured.
""".encode()


def write_report(
    *, run_path: Path, csv_path: Path, markdown_path: Path, provenance_path: Path
) -> bool:
    run_path = run_path.resolve()
    if not run_path.is_relative_to(
        (REPO_ROOT / "results/runs/local/openmemory_vllm_v7").resolve()
    ):
        raise ValueError("v7 run path differs")
    result = validate_run(run_path)
    csv_data, markdown_data = _csv(result), _markdown(result)
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
                "path": csv_path.resolve().relative_to(REPO_ROOT).as_posix(),
                "sha256": hashlib.sha256(csv_data).hexdigest(),
            },
            "markdown": {
                "path": markdown_path.resolve().relative_to(REPO_ROOT).as_posix(),
                "sha256": hashlib.sha256(markdown_data).hexdigest(),
            },
        },
    }
    csv_path.write_bytes(csv_data)
    markdown_path.write_bytes(markdown_data)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return result.passed


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("run", "csv", "markdown", "provenance"):
        parser.add_argument(f"--{name}", type=Path, required=True)
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
