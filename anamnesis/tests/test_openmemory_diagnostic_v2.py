"""Fresh-v2 freeze, disjointness and prompt-boundary checks."""

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
MANIFEST_PATH = ROOT / "eval" / "openmemory" / "decision_diagnostic.v2.manifest.json"
RAW_SHA256 = "18d69eec94c35c2b750d2ad75f03db8056881405aaeb7a2838fb36d26593de20"
CANONICAL_SHA256 = "7ce91e19d9ca13e6244ea5917c7a3a4a8e499af458b534f90127abedd2bcea61"

V1_ENTITIES = {
    "Cedar Dock",
    "Harbor Annex",
    "Basil Lantern",
    "gallery license",
    "passport renewal",
    "greenhouse electrician",
}
V2_ENTITIES = {
    "Larkspur Registry",
    "Aster Quay",
    "Copper Atrium",
    "Glass Pier",
    "Acoustic Bloom",
    "courtyard telescope",
    "grant renewal",
    "lighting technician",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _surfaces(path: Path) -> set[str]:
    artifact = load_openmemory_diagnostic(path)
    return {case.event.text.casefold() for case in artifact.cases} | {
        hit.content.casefold() for case in artifact.cases for hit in case.hits
    }


def test_v2_exact_bytes_hashes_counts_and_records_are_frozen() -> None:
    artifact = load_openmemory_diagnostic(V2_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert _sha256(V2_PATH) == RAW_SHA256
    assert openmemory_diagnostic_sha256(artifact) == CANONICAL_SHA256
    assert manifest["file_sha256"] == RAW_SHA256
    assert manifest["canonical_artifact_sha256"] == CANONICAL_SHA256
    assert manifest["source_commit_before_dataset"] == (
        "c3ddbfc74da02d05ecc313dd371d63005ca59c86"
    )
    assert len(artifact.cases) == manifest["case_count"] == 8
    assert sum(len(case.hits) for case in artifact.cases) == manifest["hit_count"] == 7
    assert manifest["record_sha256"] == {
        case.id: canonical_sha256(case) for case in artifact.cases
    }
    assert manifest["family_counts"] == {case.family: 1 for case in artifact.cases}
    assert manifest["freeze_order"]["model_calls_before_freeze"] == 0
    assert manifest["review_status"]["independent_human_review"] == "pending"


def test_v2_is_disjoint_from_opened_v1() -> None:
    v1 = load_openmemory_diagnostic(V1_PATH)
    v2 = load_openmemory_diagnostic(V2_PATH)

    assert {case.id for case in v1.cases}.isdisjoint(case.id for case in v2.cases)
    assert {case.event.id for case in v1.cases}.isdisjoint(
        case.event.id for case in v2.cases
    )
    assert {hit.fixture_id for case in v1.cases for hit in case.hits}.isdisjoint(
        hit.fixture_id for case in v2.cases for hit in case.hits
    )
    assert _surfaces(V1_PATH).isdisjoint(_surfaces(V2_PATH))
    assert V1_ENTITIES.isdisjoint(V2_ENTITIES)
    v1_text = "\n".join(_surfaces(V1_PATH))
    v2_text = "\n".join(_surfaces(V2_PATH))
    assert all(entity.casefold() in v1_text for entity in V1_ENTITIES)
    assert all(entity.casefold() in v2_text for entity in V2_ENTITIES)


def test_v2_model_prompts_expose_only_event_and_recall_content() -> None:
    artifact = load_openmemory_diagnostic(V2_PATH)

    for case in artifact.cases:
        baseline, recall = build_openmemory_case_prompts(case)
        assert case.event.text in baseline and case.event.text in recall
        assert "Retrospective recall" not in baseline
        assert "Retrospective recall" in recall
        assert case.id not in recall
        assert case.family not in recall
        assert '"label":' not in recall
        assert '"helpful_hit_ids":' not in recall
        for hit in case.hits:
            assert hit.content not in baseline
            assert hit.content in recall
            assert hit.fixture_id not in recall


def test_v1_frozen_semantics_remain_loadable_after_additive_v2_support() -> None:
    v1 = load_openmemory_diagnostic(V1_PATH)

    assert openmemory_diagnostic_sha256(v1) == (
        "b8da030f0e632c5e85523e75ba9ff948c85950435f8d55ca1b0aa3381e830126"
    )
