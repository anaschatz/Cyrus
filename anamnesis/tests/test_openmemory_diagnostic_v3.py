"""Fresh-v3 freeze, prior-version disjointness and model-boundary checks."""

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
V1_PATH = ROOT / "eval" / "openmemory" / "decision_diagnostic.v1.json"
V2_PATH = ROOT / "eval" / "openmemory" / "decision_diagnostic.v2.json"
V3_PATH = ROOT / "eval" / "openmemory" / "decision_diagnostic.v3.json"
MANIFEST_PATH = ROOT / "eval" / "openmemory" / "decision_diagnostic.v3.manifest.json"
RAW_SHA256 = "9bc0e73e8b7b2299ea83ff630379447c9acfe83424d588cf612fb99de76f2cd9"
CANONICAL_SHA256 = "36ed70646b4f9f0fc78605d148599cc75dac8691d47ef8b531f722f2a73fb146"

V3_ENTITIES = {
    "Northstar Conservation",
    "Juniper Walk",
    "Indigo Hall",
    "Amber Vault",
    "Thermal Echo",
    "rehearsal violin bows",
    "studio lease extension",
    "stage electrician",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _surfaces(path: Path) -> set[str]:
    artifact = load_openmemory_diagnostic(path)
    return {case.event.text.casefold() for case in artifact.cases} | {
        hit.content.casefold() for case in artifact.cases for hit in case.hits
    }


def test_v3_exact_hashes_counts_records_and_freeze_order() -> None:
    artifact = load_openmemory_diagnostic(V3_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert _sha256(V3_PATH) == RAW_SHA256
    assert openmemory_diagnostic_sha256(artifact) == CANONICAL_SHA256
    assert manifest["file_sha256"] == RAW_SHA256
    assert manifest["canonical_artifact_sha256"] == CANONICAL_SHA256
    assert manifest["source_commit_before_dataset"] == (
        "a57c4d129a5f67e343ae37fb4fab763e57616291"
    )
    assert len(artifact.cases) == manifest["case_count"] == 8
    assert sum(len(case.hits) for case in artifact.cases) == manifest["hit_count"] == 7
    assert manifest["record_sha256"] == {
        case.id: canonical_sha256(case) for case in artifact.cases
    }
    assert manifest["family_counts"] == {case.family: 1 for case in artifact.cases}
    assert manifest["freeze_order"]["model_calls_before_freeze"] == 0
    assert manifest["review_status"]["independent_human_review"] == "pending"


def test_v3_is_surface_and_identifier_disjoint_from_v1_v2() -> None:
    prior = [load_openmemory_diagnostic(V1_PATH), load_openmemory_diagnostic(V2_PATH)]
    v3 = load_openmemory_diagnostic(V3_PATH)

    prior_case_ids = {case.id for artifact in prior for case in artifact.cases}
    prior_event_ids = {case.event.id for artifact in prior for case in artifact.cases}
    prior_hit_ids = {
        hit.fixture_id
        for artifact in prior
        for case in artifact.cases
        for hit in case.hits
    }
    assert prior_case_ids.isdisjoint(case.id for case in v3.cases)
    assert prior_event_ids.isdisjoint(case.event.id for case in v3.cases)
    assert prior_hit_ids.isdisjoint(
        hit.fixture_id for case in v3.cases for hit in case.hits
    )
    assert (_surfaces(V1_PATH) | _surfaces(V2_PATH)).isdisjoint(_surfaces(V3_PATH))
    prior_text = "\n".join(_surfaces(V1_PATH) | _surfaces(V2_PATH))
    v3_text = "\n".join(_surfaces(V3_PATH))
    assert all(entity.casefold() not in prior_text for entity in V3_ENTITIES)
    assert all(entity.casefold() in v3_text for entity in V3_ENTITIES)


def test_v3_expected_actions_are_grounded_only_in_current_event_or_helpful_hit() -> (
    None
):
    artifact = load_openmemory_diagnostic(V3_PATH)

    for case in artifact.cases:
        expected = case.expected
        if expected.mode == "no_action":
            continue
        assert expected.action_key == case.event.id
        assert expected.evidence_event_ids == (case.event.id,)
        assert expected.payload is not None
        if case.family == "reference_resolution":
            helpful_text = " ".join(
                hit.content
                for hit in case.hits
                if hit.fixture_id in case.helpful_hit_ids
            )
            assert expected.payload["recipient"] in helpful_text
            assert expected.payload["address"] in helpful_text
        else:
            assert not case.helpful_hit_ids


def test_v3_model_prompts_hide_evaluator_fields() -> None:
    artifact = load_openmemory_diagnostic(V3_PATH)

    for case in artifact.cases:
        baseline, recall = build_openmemory_case_prompts(case)
        assert case.event.text in baseline and case.event.text in recall
        assert "Retrospective recall" not in baseline
        assert "Retrospective recall" in recall
        assert case.id not in recall
        assert case.family not in recall
        assert '"label":' not in recall
        assert '"helpful_hit_ids":' not in recall
        assert '"expected":' not in recall
        for hit in case.hits:
            assert hit.content not in baseline
            assert hit.content in recall
            assert hit.fixture_id not in recall


def test_v1_v2_canonical_semantics_remain_unchanged() -> None:
    assert openmemory_diagnostic_sha256(load_openmemory_diagnostic(V1_PATH)) == (
        "b8da030f0e632c5e85523e75ba9ff948c85950435f8d55ca1b0aa3381e830126"
    )
    assert openmemory_diagnostic_sha256(load_openmemory_diagnostic(V2_PATH)) == (
        "7ce91e19d9ca13e6244ea5917c7a3a4a8e499af458b534f90127abedd2bcea61"
    )
