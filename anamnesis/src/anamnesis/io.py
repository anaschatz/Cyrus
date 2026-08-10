"""Strict JSONL input/output helpers and reproducibility fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from anamnesis.schema import Scenario, ScenarioRun

ModelT = TypeVar("ModelT", bound=BaseModel)


def canonical_sha256(model: BaseModel) -> str:
    """Hash a model independently of JSON formatting and key order."""

    payload = json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def dataset_sha256(scenarios: list[Scenario]) -> str:
    """Hash an ordered scenario collection from its canonical record hashes."""

    joined = "\n".join(canonical_sha256(scenario) for scenario in scenarios)
    return hashlib.sha256(joined.encode()).hexdigest()


def require_preregistered_final_dataset(
    path: str | Path,
    scenarios: list[Scenario],
) -> None:
    """Fail closed unless the adjacent release manifest clears final use."""

    dataset_path = Path(path)
    release_path = dataset_path.with_suffix(".manifest.json")
    if not release_path.is_file():
        raise ValueError(f"final dataset release manifest is missing: {release_path}")
    try:
        release = json.loads(release_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid final dataset release manifest: {release_path}"
        ) from error
    review = release.get("review_status")
    human_review = (
        review.get("independent_human_review") if isinstance(review, dict) else None
    )
    if human_review != "passed":
        raise ValueError("final dataset requires passed independent human review")
    if release.get("preregistered_final_eligible") is not True:
        raise ValueError("dataset is not eligible for a preregistered final run")
    if release.get("canonical_dataset_sha256") != dataset_sha256(scenarios):
        raise ValueError("final dataset release hash differs from dataset content")


def _load_jsonl(path: str | Path, model_type: type[ModelT]) -> list[ModelT]:
    source = Path(path)
    items: list[ModelT] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                items.append(model_type.model_validate_json(line))
            except (ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
    if not items:
        raise ValueError(f"{source} contains no records")
    return items


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load a scenario dataset and reject duplicate IDs."""

    scenarios = _load_jsonl(path, Scenario)
    ids = [scenario.id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate scenario IDs")
    return scenarios


def load_runs(path: str | Path) -> list[ScenarioRun]:
    """Load raw JSONL runs or extract final runs from an Inspect `.eval` log."""

    source = Path(path)
    if source.suffix == ".eval":
        return _load_inspect_runs(source)
    return _load_jsonl(path, ScenarioRun)


def _load_inspect_runs(path: Path) -> list[ScenarioRun]:
    from inspect_ai.log import read_eval_log

    log = read_eval_log(path)
    if log.status != "success":
        raise ValueError(f"Inspect log did not complete successfully: {path}")
    if not log.samples:
        raise ValueError(f"Inspect log contains no samples: {path}")

    runs: list[ScenarioRun] = []
    for sample in log.samples:
        if sample.output is None:
            raise ValueError(f"Inspect sample {sample.id} has no final output")
        try:
            runs.append(ScenarioRun.model_validate_json(sample.output.completion))
        except ValueError as error:
            raise ValueError(
                f"Inspect sample {sample.id} does not contain a ScenarioRun"
            ) from error
    return runs


def write_runs(path: str | Path, runs: list[ScenarioRun]) -> None:
    """Write raw runs atomically as canonical JSONL."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(run.model_dump_json() + "\n")
    temporary.replace(destination)
