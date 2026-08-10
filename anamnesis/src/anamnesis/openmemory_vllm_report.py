"""Strict report generator for the single frozen OpenMemory vLLM v4 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

from anamnesis.openmemory_diagnostic import (
    build_openmemory_case_prompts,
    load_openmemory_diagnostic,
    score_openmemory_pair,
)
from anamnesis.openmemory_vllm import (
    build_openmemory_vllm_request,
    build_openmemory_vllm_user_envelope,
)
from anamnesis.openmemory_vllm_run import (
    DATASET_PATH,
    PIN_PATH,
    PREFLIGHT_PATH,
    REPO_ROOT,
    OpenMemoryVllmV4Run,
    _load_frozen_inputs,
)
from anamnesis.runner import DecisionRequest
from anamnesis.vllm_runtime import canonical_json_sha256

TITLE = "OpenMemory + vLLM v4 local diagnostic — not a hypothesis test"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_reporting_checkout(measurement_commit: str) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("v4 report requires a clean worktree")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", measurement_commit, head],
        cwd=REPO_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("measurement commit is not an ancestor of reporter HEAD")
    changed = set(
        subprocess.run(
            ["git", "diff", "--name-only", f"{measurement_commit}..{head}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    allowed = {
        "README.md",
        "results/README.md",
        "results/local_openmemory_vllm_v4_analysis.md",
        "src/anamnesis/openmemory_vllm_report.py",
        "tests/test_openmemory_vllm_run.py",
    }
    if not changed <= allowed:
        raise ValueError("measurement inputs changed after the frozen v4 run")
    return head


def _validate_run(path: Path) -> OpenMemoryVllmV4Run:
    result = OpenMemoryVllmV4Run.model_validate_json(path.read_text(encoding="utf-8"))
    _verify_reporting_checkout(result.source_commit)
    pin, preflight = _load_frozen_inputs()
    if result.pin_sha256 != _sha256(PIN_PATH):
        raise ValueError("run pin hash differs from tracked runtime pin")
    expected_requests: list[dict[str, object]] = []
    canary_prompt = build_openmemory_vllm_user_envelope(
        now=preflight.event.at.isoformat(),
        current_event_id=preflight.event.id,
        context_events=[preflight.event],
        decision_history=[],
        memory_view=None,
        retrospective_recall=preflight.retrospective_recall,
    )
    expected_requests.append(
        build_openmemory_vllm_request(
            pin, DecisionRequest(event=preflight.event, prompt=canary_prompt)
        )
    )
    if result.status == "complete":
        if result.paired_run is None:
            raise ValueError("complete result is missing paired run")
        artifact = load_openmemory_diagnostic(DATASET_PATH)
        baseline = {}
        recall = {}
        for case in artifact.cases:
            prompts = build_openmemory_case_prompts(
                case, prompt_builder=build_openmemory_vllm_user_envelope
            )
            for _arm, prompt in zip(("baseline", "recall"), prompts, strict=True):
                expected_requests.append(
                    build_openmemory_vllm_request(
                        pin, DecisionRequest(event=case.event, prompt=prompt)
                    )
                )
        for call in result.paired_run.calls:
            destination = baseline if call.arm == "baseline" else recall
            if call.case_id in destination:
                raise ValueError("paired run repeats a case arm")
            destination[call.case_id] = call.decision
        recomputed = score_openmemory_pair(artifact, baseline=baseline, recall=recall)
        if recomputed != result.paired_run.metrics:
            raise ValueError("reported v4 metrics differ from recomputed decisions")
    if len(expected_requests) != len(result.audits):
        raise ValueError("request reconstruction count differs from audit count")
    for audit, request in zip(result.audits, expected_requests, strict=True):
        if audit.request_sha256 != canonical_json_sha256(request):
            raise ValueError("raw request differs from frozen reconstruction")
    if result.canary.accepted != result.audits[0].validation.accepted:
        raise ValueError("canary accepted flag differs from first audit")
    if result.audits[0].raw_completion != result.canary.raw_completion:
        raise ValueError("canary raw completion differs from first audit")
    if result.audits[0].usage != result.canary.usage:
        raise ValueError("canary usage differs from first audit")
    if result.paired_run is not None:
        for audit, call in zip(result.audits[1:], result.paired_run.calls, strict=True):
            if audit.raw_completion != call.raw_completion or audit.usage != call.usage:
                raise ValueError("paired call differs from its raw audit")
            if audit.validation.accepted == call.parse_error:
                raise ValueError("paired parse status differs from its raw audit")
    return result


def _row(result: OpenMemoryVllmV4Run) -> dict[str, object]:
    metrics = result.paired_run.metrics if result.paired_run is not None else None
    return {
        "status": result.status,
        "passed": str(result.passed).lower(),
        "case_count": metrics.case_count if metrics else 0,
        "baseline_correct": metrics.baseline_correct if metrics else 0,
        "recall_correct": metrics.recall_correct if metrics else 0,
        "helpful_gain": metrics.helpful_gain if metrics else 0,
        "safety_regressions": metrics.safety_regressions if metrics else 0,
        "no_hit_regressions": metrics.no_hit_regressions if metrics else 0,
        "recall_false_actions": metrics.recall_false_actions if metrics else 0,
        "recall_evidence_contaminations": (
            metrics.recall_evidence_contaminations if metrics else 0
        ),
        "structured_invalid_calls": sum(
            not audit.validation.accepted for audit in result.audits[1:]
        ),
        "setup_input_tokens": result.setup_usage.input_tokens,
        "setup_output_tokens": result.setup_usage.output_tokens,
        "headline_input_tokens": result.headline_usage.input_tokens,
        "headline_output_tokens": result.headline_usage.output_tokens,
        "provider_api_cost_usd": result.total_usage.cost_usd,
    }


def _csv_bytes(row: dict[str, object]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue().encode()


def _markdown_bytes(row: dict[str, object]) -> bytes:
    gate = "PASS" if row["passed"] == "true" else "FAIL"
    setup_tokens = f"{row['setup_input_tokens']} / {row['setup_output_tokens']}"
    headline_tokens = (
        f"{row['headline_input_tokens']} / {row['headline_output_tokens']}"
    )
    text = f"""# {TITLE}

The frozen v4 joint model-artifact + structured-runtime compatibility cell **{gate}**.
This is a development diagnostic, not a causal comparison with Ollama and not a
hypothesis test. Hardware, electricity, model acquisition and human review are
unmeasured; provider API cost is exactly `${row["provider_api_cost_usd"]}`.

| Metric | Value |
|---|---:|
| Status | {row["status"]} |
| Cases | {row["case_count"]} |
| Baseline correct | {row["baseline_correct"]} |
| Recall correct | {row["recall_correct"]} |
| Helpful gain | {row["helpful_gain"]} |
| Safety regressions | {row["safety_regressions"]} |
| No-hit regressions | {row["no_hit_regressions"]} |
| Recall false actions | {row["recall_false_actions"]} |
| Recall evidence contaminations | {row["recall_evidence_contaminations"]} |
| Structured-invalid scenario calls | {row["structured_invalid_calls"]} |
| Setup tokens (input/output, excluded) | {setup_tokens} |
| Headline tokens (input/output) | {headline_tokens} |

The frozen gate requires the canary and every structured call to validate,
one helpful recall gain, and zero safety, no-hit, false-action, or evidence
contamination regressions. No retry, repair, alternate artifact, or selected
duplicate run is permitted on these v4 cases.
"""
    return text.encode()


def _output_path(path: Path, *, suffix: str) -> Path:
    resolved = path.resolve()
    results = (REPO_ROOT / "results").resolve()
    runs = (results / "runs").resolve()
    if (
        not resolved.is_relative_to(results)
        or resolved.is_relative_to(runs)
        or resolved.suffix != suffix
    ):
        raise ValueError("report outputs must be tracked files under results/")
    return resolved


def write_report(
    *, run_path: Path, csv_path: Path, markdown_path: Path, provenance_path: Path
) -> bool:
    run_path = run_path.resolve()
    allowed_runs = (REPO_ROOT / "results/runs/local/openmemory_vllm_v4").resolve()
    if not run_path.is_relative_to(allowed_runs):
        raise ValueError("run artifact must be under the frozen v4 run folder")
    csv_path = _output_path(csv_path, suffix=".csv")
    markdown_path = _output_path(markdown_path, suffix=".md")
    provenance_path = _output_path(provenance_path, suffix=".json")
    targets = {csv_path, markdown_path, provenance_path}
    sources = {
        run_path,
        DATASET_PATH.resolve(),
        PIN_PATH.resolve(),
        PREFLIGHT_PATH.resolve(),
    }
    if len(targets) != 3 or targets & sources:
        raise ValueError("report output collides with an input or another output")
    result = _validate_run(run_path)
    reporter_commit = _verify_reporting_checkout(result.source_commit)
    row = _row(result)
    csv_data = _csv_bytes(row)
    markdown_data = _markdown_bytes(row)
    provenance = {
        "schema_version": 1,
        "title": TITLE,
        "hypothesis_test_eligible": False,
        "measurement_source_commit": result.source_commit,
        "reporter_commit": reporter_commit,
        "inputs": {
            "run": {
                "path": run_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(run_path),
            },
            "dataset": {
                "path": DATASET_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(DATASET_PATH),
            },
            "runtime_pin": {
                "path": PIN_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(PIN_PATH),
            },
            "preflight": {
                "path": PREFLIGHT_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(PREFLIGHT_PATH),
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
    passed = write_report(
        run_path=args.run,
        csv_path=args.csv,
        markdown_path=args.markdown,
        provenance_path=args.provenance,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TITLE", "main", "write_report"]
