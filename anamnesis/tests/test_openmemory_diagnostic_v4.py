"""Fresh-v4 freeze and prior-version disjointness checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from anamnesis.io import canonical_sha256
from anamnesis.openmemory_diagnostic import (
    build_openmemory_case_prompts,
    load_openmemory_diagnostic,
    openmemory_diagnostic_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
PATHS = [
    ROOT / "eval" / "openmemory" / f"decision_diagnostic.v{version}.json"
    for version in range(1, 5)
]
V4_PATH = PATHS[-1]
MANIFEST_PATH = ROOT / "eval" / "openmemory" / "decision_diagnostic.v4.manifest.json"
RAW_SHA256 = "9f4fb7bdf000858c769b0702acb5585e0ef8e67eb7709bcfa2c8d83c5fbdd0d9"
CANONICAL_SHA256 = "30dc9cbdae1b399e5ca3de58b1efb8fe9c7ae448f6d38e7a2a27b727086ac524"


def _surfaces(path: Path) -> set[str]:
    artifact = load_openmemory_diagnostic(path)
    return {case.event.text.casefold() for case in artifact.cases} | {
        hit.content.casefold() for case in artifact.cases for hit in case.hits
    }


def test_v4_exact_hashes_counts_records_and_planned_cell() -> None:
    artifact = load_openmemory_diagnostic(V4_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert hashlib.sha256(V4_PATH.read_bytes()).hexdigest() == RAW_SHA256
    assert openmemory_diagnostic_sha256(artifact) == CANONICAL_SHA256
    assert manifest["file_sha256"] == RAW_SHA256
    assert manifest["canonical_artifact_sha256"] == CANONICAL_SHA256
    assert manifest["source_commit_before_dataset"] == (
        "45e52f71d221f891ee43edcf4286bf7511d9e7b0"
    )
    assert len(artifact.cases) == manifest["case_count"] == 8
    assert sum(len(case.hits) for case in artifact.cases) == manifest["hit_count"] == 7
    assert manifest["record_sha256"] == {
        case.id: canonical_sha256(case) for case in artifact.cases
    }
    assert manifest["planned_cell"]["not_transport_only"] is True
    assert manifest["review_status"]["independent_human_review"] == "pending"


def test_v4_ids_and_surfaces_are_disjoint_from_v1_v2_v3() -> None:
    prior = [load_openmemory_diagnostic(path) for path in PATHS[:-1]]
    v4 = load_openmemory_diagnostic(V4_PATH)
    assert {case.id for a in prior for case in a.cases}.isdisjoint(
        case.id for case in v4.cases
    )
    assert {case.event.id for a in prior for case in a.cases}.isdisjoint(
        case.event.id for case in v4.cases
    )
    assert {
        hit.fixture_id for a in prior for case in a.cases for hit in case.hits
    }.isdisjoint(hit.fixture_id for case in v4.cases for hit in case.hits)
    assert (
        set()
        .union(*(_surfaces(path) for path in PATHS[:-1]))
        .isdisjoint(_surfaces(V4_PATH))
    )


def test_v4_model_boundary_hides_evaluator_fields() -> None:
    for case in load_openmemory_diagnostic(V4_PATH).cases:
        baseline, recall = build_openmemory_case_prompts(case)
        assert case.event.text in baseline and case.event.text in recall
        assert case.id not in recall and case.family not in recall
        assert '"label":' not in recall and '"expected":' not in recall
        for hit in case.hits:
            assert hit.content not in baseline and hit.content in recall
            assert hit.fixture_id not in recall


def test_prior_canonical_artifacts_remain_unchanged() -> None:
    expected = (
        "b8da030f0e632c5e85523e75ba9ff948c85950435f8d55ca1b0aa3381e830126",
        "7ce91e19d9ca13e6244ea5917c7a3a4a8e499af458b534f90127abedd2bcea61",
        "36ed70646b4f9f0fc78605d148599cc75dac8691d47ef8b531f722f2a73fb146",
    )
    assert (
        tuple(
            openmemory_diagnostic_sha256(load_openmemory_diagnostic(path))
            for path in PATHS[:-1]
        )
        == expected
    )
