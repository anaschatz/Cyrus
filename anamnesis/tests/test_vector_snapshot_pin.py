from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from anamnesis.cli import _validate_vector_snapshot_task_arg
from anamnesis.experiment import ExperimentManifest


@pytest.fixture(scope="module")
def eval_module() -> dict[str, object]:
    return runpy.run_path("eval/anamnesis_eval.py")


@pytest.mark.parametrize("snapshot_path", [None, "", "   "])
def test_frozen_vector_snapshot_path_must_be_explicit(
    eval_module: dict[str, object],
    snapshot_path: str | None,
) -> None:
    require_snapshot = eval_module["_require_local_embedding_snapshot"]

    with pytest.raises(ValueError, match="explicit embedding_snapshot_path"):
        require_snapshot(snapshot_path)


def test_frozen_vector_snapshot_path_must_be_absolute_and_exist(
    eval_module: dict[str, object],
    tmp_path: Path,
) -> None:
    require_snapshot = eval_module["_require_local_embedding_snapshot"]

    with pytest.raises(ValueError, match="absolute local path"):
        require_snapshot(".")
    with pytest.raises(ValueError, match="existing directory"):
        require_snapshot(str(tmp_path / "missing"))
    assert require_snapshot(str(tmp_path)) == str(tmp_path.resolve())


def test_frozen_vector_task_rejects_missing_path_before_manifest_read(
    eval_module: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = ExperimentManifest.model_validate_json(
        Path("eval/experiment_manifest.template.json").read_text(encoding="utf-8")
    )
    system_task = eval_module["_system_task"]
    monkeypatch.setitem(
        system_task.__globals__,
        "_validated_experiment_manifest",
        lambda *args, **kwargs: manifest,
    )

    with pytest.raises(ValueError, match="explicit embedding_snapshot_path"):
        system_task(
            "vector_rag",
            seed=101,
            repetition=1,
            embedding_revision=manifest.embedding.revision,
            manifest="does-not-exist.json",
        )


@pytest.mark.parametrize(
    "task_args",
    [
        {},
        {"embedding_snapshot_path": None},
        {"embedding_snapshot_path": ""},
        {"embedding_snapshot_path": "   "},
    ],
)
def test_strict_log_requires_explicit_vector_snapshot_arg(
    task_args: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="explicit embedding_snapshot_path"):
        _validate_vector_snapshot_task_arg(task_args)


def test_strict_log_requires_absolute_but_not_machine_specific_snapshot(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="absolute local path"):
        _validate_vector_snapshot_task_arg(
            {"embedding_snapshot_path": "relative/snapshot"}
        )

    machine_specific_path = tmp_path / "may-no-longer-exist-on-report-host"
    _validate_vector_snapshot_task_arg(
        {"embedding_snapshot_path": str(machine_specific_path)}
    )
