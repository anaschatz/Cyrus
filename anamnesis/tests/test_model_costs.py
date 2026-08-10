from __future__ import annotations

import hashlib
import json
from pathlib import Path

from inspect_ai.model import ModelCost

from anamnesis.experiment import ArtifactPin
from anamnesis.preflight import _configured_model_cost

MODEL = "openai/gpt-4.1-mini-2025-04-14"
PRICING_PATH = Path("eval/model_costs.json")
EXPECTED_COST = ModelCost(
    input=0.40,
    output=1.60,
    input_cache_write=0.40,
    input_cache_read=0.10,
)


def test_tracked_model_pricing_has_one_exact_inspect_entry() -> None:
    raw = json.loads(PRICING_PATH.read_text(encoding="utf-8"))

    assert raw == {MODEL: EXPECTED_COST.model_dump()}
    artifact = ArtifactPin(
        path=str(PRICING_PATH),
        sha256=hashlib.sha256(PRICING_PATH.read_bytes()).hexdigest(),
    )
    assert _configured_model_cost(artifact, MODEL) == EXPECTED_COST
