from __future__ import annotations

import hashlib
from pathlib import Path

from anamnesis.io import load_scenarios
from anamnesis.local_experiment import LocalExperimentManifest
from anamnesis.oracle import ORACLE_SYSTEM_NAME, load_oracle_artifact

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "eval/local_oracle_manifest.template.json"


def test_local_oracle_manifest_template_binds_the_annotation_artifact() -> None:
    manifest = LocalExperimentManifest.model_validate_json(
        TEMPLATE.read_text(encoding="utf-8")
    )

    assert manifest.status == "draft"
    assert manifest.phase == "oracle_smoke"
    assert manifest.compiler_mode == "oracle"
    assert manifest.systems == [ORACLE_SYSTEM_NAME]
    assert manifest.model.same_model_for_compiler_and_decision is False
    assert manifest.oracle_annotations is not None

    annotation_path = ROOT / manifest.oracle_annotations.path
    assert hashlib.sha256(annotation_path.read_bytes()).hexdigest() == (
        manifest.oracle_annotations.sha256
    )
    artifact = load_oracle_artifact(
        annotation_path,
        load_scenarios(ROOT / manifest.dataset.path),
    )
    assert artifact.hypothesis_test_eligible is False
