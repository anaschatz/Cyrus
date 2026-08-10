"""Integrity checks for the retained OpenMemory transport failure."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_PATH = (
    ROOT / "results" / "local_openmemory_diagnostic_preflight_failure.provenance.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_at_commit(path: Path, commit: str) -> str:
    relative = path.relative_to(ROOT).as_posix()
    value = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
    )
    return hashlib.sha256(value).hexdigest()


def test_openmemory_failure_provenance_hashes_every_retained_artifact() -> None:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))

    assert provenance["hypothesis_test_eligible"] is False
    assert provenance["diagnostic_metrics_defined"] is False
    assert provenance["rerun_same_v1_cases_allowed"] is False
    source_bound = {
        "execution_protocol",
        "task_source",
        "runtime_source",
        "diagnostic_source",
    }
    for name, artifact in provenance["artifacts"].items():
        if artifact.get("published") is False:
            continue
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert path.is_relative_to(ROOT)
        if name in source_bound:
            assert (
                _sha256_at_commit(path, provenance["source_commit"])
                == artifact["sha256"]
            )
        else:
            assert _sha256(path) == artifact["sha256"]


def test_openmemory_failure_summary_is_exact_cancelled_max_tokens_shape() -> None:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    observed = provenance["observed"]
    raw = provenance["artifacts"]["raw_eval_log"]
    summary_path = ROOT / provenance["artifacts"]["sanitized_log_summary"]["path"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert raw["published"] is False
    assert "path" not in raw
    assert summary["source_log_sha256"] == raw["sha256"]
    assert summary["inspect_status"] == observed["inspect_status"] == "cancelled"
    assert summary["sample_count"] == observed["sample_count"] == 1
    assert len(summary["events"]) == observed["model_event_count"] == 2
    assert summary["events"][0] == {
        "index": 0,
        "stop_reason": "max_tokens",
        "input_tokens": 707,
        "output_tokens": 3389,
        "total_tokens": 4096,
        "completion_bytes": 0,
        "total_cost_usd": 0.0,
    }
    assert summary["events"][1]["usage_retained"] is False


def test_failure_report_makes_no_openmemory_metric_claim() -> None:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    report = (ROOT / provenance["artifacts"]["failure_report"]["path"]).read_text(
        encoding="utf-8"
    )

    assert "not evidence for or against OpenMemory recall" in report
    assert "Completed matrix: 0/8 pairs" in report
    assert "No OpenMemory usefulness" in report
    assert "will not be rerun or tuned" in report
