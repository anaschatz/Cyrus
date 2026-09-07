"""Strict report for the real local indexed-memory paired diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from anamnesis.io import report_output_paths
from anamnesis.openmemory_vllm import (
    VllmOpenMemoryAlignedDecisionModel,
    build_openmemory_vllm_aligned_request,
    build_openmemory_vllm_user_envelope,
)
from anamnesis.openmemory_vllm_run import REPO_ROOT, _verify_source_commit
from anamnesis.openmemory_vllm_v6 import (
    FIXTURE_PATH,
    PIN_PATH,
    RealMemoryRun,
    _correct,
    _load_inputs,
    _metrics,
)
from anamnesis.runner import DecisionRequest
from anamnesis.schema import Decision
from anamnesis.vllm_runtime import canonical_json_sha256

TITLE = "OpenMemory-compatible real indexed-memory diagnostic — not a hypothesis test"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_run(path: Path) -> RealMemoryRun:
    result = RealMemoryRun.model_validate_json(path.read_text())
    _verify_source_commit(REPO_ROOT, result.source_commit)
    pin, fixture, runtime = _load_inputs()
    if (
        result.pin_sha256 != _sha256(PIN_PATH)
        or result.fixture_sha256 != pin.fixture_raw_sha256
    ):
        raise ValueError("v6 result input hashes differ")
    for case, observed in zip(fixture.cases, result.cases, strict=True):
        if (
            observed.case_id != case.id
            or observed.retrieved_memory_ids != case.expected_retrieved_memory_ids
        ):
            raise ValueError("v6 case order or retrieval differs")
        expected_contents = tuple(
            next(item.content for item in case.memories if item.id == memory_id)
            for memory_id in case.expected_retrieved_memory_ids
        )
        if (
            observed.retrieved_contents != expected_contents
            or not observed.retrieval_correct
        ):
            raise ValueError("v6 retrieved content differs from frozen memory")
        for name, arm, recall in (
            ("baseline", observed.baseline, None),
            ("recall", observed.recall, observed.retrieved_contents),
        ):
            decoded = (
                VllmOpenMemoryAlignedDecisionModel.wire_model.model_validate_json(
                    arm.audit.raw_completion
                ).to_domain()
                if arm.audit.validation.accepted
                else Decision()
            )
            if arm.decision != decoded or arm.arm != name:
                raise ValueError("v6 arm or decision differs from raw completion")
            if arm.latency_ms != arm.audit.latency_ms:
                raise ValueError("v6 latency differs from audit")
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
            if arm.prompt_sha256 != hashlib.sha256(
                prompt.encode()
            ).hexdigest() or arm.audit.request_sha256 != canonical_json_sha256(request):
                raise ValueError("v6 prompt or request hash differs")
            if (
                arm.correct != _correct(case.expected, arm.decision)
                or arm.audit.usage != arm.usage
            ):
                raise ValueError("v6 score or usage differs")
    if result.metrics != _metrics(fixture, result.cases):
        raise ValueError("v6 aggregate metrics differ from cases")
    return result


def _csv(result: RealMemoryRun) -> bytes:
    fields = [
        "passed",
        "cases",
        "retrieval_correct",
        "baseline_correct",
        "recall_correct",
        "helpful_gain",
        "safety_regressions",
        "accepted_calls",
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


def _markdown(result: RealMemoryRun) -> bytes:
    rows = "\n".join(
        f"| {case.case_id} | {','.join(case.retrieved_memory_ids) or 'none'} | "
        f"{str(case.baseline.correct).lower()} | "
        f"{str(case.recall.correct).lower()} |"
        for case in result.cases
    )
    summary = (
        f"Retrieval: {result.metrics.retrieval_correct}/8. "
        f"Baseline: {result.metrics.baseline_correct}/8; "
        f"recall: {result.metrics.recall_correct}/8; "
        f"helpful gain: {result.metrics.helpful_gain}; "
        f"safety regressions: {result.metrics.safety_regressions}. "
        f"All {result.metrics.accepted_calls}/16 structured calls were accepted. "
        f"Usage was {result.usage.input_tokens} input and "
        f"{result.usage.output_tokens} output tokens at "
        f"`${result.usage.cost_usd}` provider API cost. "
        "Electricity and hardware are unmeasured."
    )
    return f"""# {TITLE}

The one authorized paired run **{"PASS" if result.passed else "FAIL"}**.
Memory records were added and searched at runtime through the non-authoritative
OpenMemory boundary using the pinned local FastEmbed index; recall text was not
injected from the fixture. This tests the architecture, not the upstream
Cavira SDK.

| Case | Retrieved memory | Baseline correct | Recall correct |
|---|---|---:|---:|
{rows}

{summary}
""".encode()


def write_report(
    *, run_path: Path, csv_path: Path, markdown_path: Path, provenance_path: Path
) -> bool:
    run_path = run_path.resolve()
    if not run_path.is_relative_to(
        (REPO_ROOT / "results/runs/local/openmemory_vllm_v6").resolve()
    ):
        raise ValueError("v6 run path differs")
    csv_path, markdown_path, provenance_path = report_output_paths(
        REPO_ROOT / "results", csv_path, markdown_path, provenance_path
    )
    result = validate_run(run_path)
    csv_data, md_data = _csv(result), _markdown(result)
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
                "sha256": hashlib.sha256(md_data).hexdigest(),
            },
        },
    }
    for path in (csv_path, markdown_path, provenance_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(csv_data)
    markdown_path.write_bytes(md_data)
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
