from __future__ import annotations

import json
from pathlib import Path

import pytest
from inspect_ai.model import ModelCost

from anamnesis.local_experiment import LOCAL_MODEL_ID, validate_zero_api_pricing

PRICING_PATH = Path("eval/local_model_costs.json")
ZERO_COST = ModelCost(
    input=0.0,
    output=0.0,
    input_cache_write=0.0,
    input_cache_read=0.0,
)


def test_local_pricing_has_one_exact_all_zero_inspect_entry() -> None:
    raw = json.loads(PRICING_PATH.read_text(encoding="utf-8"))

    assert raw == {LOCAL_MODEL_ID: ZERO_COST.model_dump()}
    assert validate_zero_api_pricing(PRICING_PATH) == (
        "c185e2fad06d6bd2abaaf0be81a1720fc245555fa2a477c1b1bea558b28c2f74"
    )


def test_nonzero_or_additional_pricing_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "costs.json"
    path.write_text(
        json.dumps(
            {
                LOCAL_MODEL_ID: {
                    "input": 0.01,
                    "output": 0.0,
                    "input_cache_write": 0.0,
                    "input_cache_read": 0.0,
                },
                "another/model": ZERO_COST.model_dump(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one exact all-zero"):
        validate_zero_api_pricing(path)
