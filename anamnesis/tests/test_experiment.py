from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from anamnesis.experiment import ExperimentManifest

TEMPLATE = Path("eval/experiment_manifest.template.json")
HASH = "a" * 64
COMMIT = "b" * 40
EMBEDDING_REVISION = "52398278842ec682c6f32300af41344b1c0b0bb2"
EMBEDDING_ARTIFACT_SHA256 = (
    "d435d05b3411502ad9a280cc9ac0157f7bcd9f176df2fdc8971f788a121a02d7"
)
PRICING_SHA256 = "45558119c159181ff987c37acf6df0225a16a81ef7030bc001ccdc0ad0d87319"


def test_draft_manifest_template_is_valid() -> None:
    manifest = ExperimentManifest.model_validate_json(TEMPLATE.read_text())

    assert manifest.status == "draft"
    assert manifest.phase == "baseline"
    assert manifest.sealed_opened is False
    assert manifest.embedding.top_k == 5
    assert manifest.embedding.revision == EMBEDDING_REVISION
    assert manifest.embedding.artifact_sha256 == EMBEDDING_ARTIFACT_SHA256
    assert manifest.model.provider_args.responses_api is False
    assert manifest.model.pricing.sha256 == PRICING_SHA256
    assert manifest.execution.max_retries == 0
    assert manifest.execution.log_model_api is True


def test_frozen_baseline_requires_every_reproducibility_pin() -> None:
    raw = json.loads(TEMPLATE.read_text())
    raw["status"] = "frozen"

    with pytest.raises(ValidationError, match="model.snapshot"):
        ExperimentManifest.model_validate(raw)


@pytest.mark.parametrize(
    "provider_args",
    [
        {"responses_api": True},
        {"responses_api": False, "service_tier": "flex"},
        {},
    ],
)
def test_model_provider_route_is_exact(provider_args: dict[str, object]) -> None:
    raw = json.loads(TEMPLATE.read_text())
    raw["model"]["provider_args"] = provider_args

    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw)


def test_complete_final_manifest_is_accepted() -> None:
    raw = json.loads(TEMPLATE.read_text())
    raw.update(
        {
            "status": "frozen",
            "phase": "final",
            "scenario_count": 50,
            "sealed_opened": True,
            "systems": [
                "no_memory",
                "full_context",
                "vector_rag",
                "anamnesis",
            ],
            "git_commit": COMMIT,
            "decision_prompt_sha256": HASH,
            "memory_compiler_sha256": HASH,
            "system_config_sha256": {
                "no_memory": HASH,
                "full_context": HASH,
                "vector_rag": HASH,
                "anamnesis": HASH,
            },
        }
    )
    raw["dataset"]["path"] = "eval/scenarios/all.jsonl"
    raw["execution"]["seeds"] = [101, 202, 303]
    raw["execution"]["repetitions"] = 3
    raw["model"]["snapshot"] = "provider/frozen-snapshot"
    raw["embedding"]["revision"] = "c" * 40
    raw["embedding"]["artifact_sha256"] = HASH
    for key in (
        "dataset",
        "dependency_lock",
        "research_contract",
        "architecture_contract",
    ):
        raw[key]["sha256"] = HASH
    raw["model"]["pricing"]["sha256"] = HASH
    raw["model"]["preflight"]["sha256"] = HASH

    manifest = ExperimentManifest.model_validate(raw)

    assert manifest.status == "frozen"
    assert manifest.phase == "final"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scenario_count", 34),
        ("sealed_opened", True),
        ("systems", ["no_memory", "full_context"]),
    ],
)
def test_baseline_matrix_cannot_drift(field: str, value: object) -> None:
    raw = json.loads(TEMPLATE.read_text())
    raw[field] = value

    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw)
