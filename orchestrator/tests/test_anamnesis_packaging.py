from __future__ import annotations

import tomllib
from pathlib import Path


def test_orchestrator_declares_the_qualified_anamnesis_distribution() -> None:
    project_root = Path(__file__).resolve().parents[1]
    document = tomllib.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert "anamnesis-memory-eval==0.1.0" in document["project"]["dependencies"]
    assert document["tool"]["uv"]["sources"]["anamnesis-memory-eval"] == {
        "path": "../anamnesis"
    }


def test_frozen_lock_uses_the_non_editable_local_anamnesis_distribution() -> None:
    project_root = Path(__file__).resolve().parents[1]
    lock = tomllib.loads((project_root / "uv.lock").read_text(encoding="utf-8"))
    packages = {package["name"]: package for package in lock["package"]}

    anamnesis = packages["anamnesis-memory-eval"]
    assert anamnesis["version"] == "0.1.0"
    assert anamnesis["source"] == {"directory": "../anamnesis"}
    assert anamnesis["dependencies"] == [{"name": "pydantic"}]

    orchestrator = packages["jarvis-orchestrator"]
    assert {dependency["name"] for dependency in orchestrator["dependencies"]} >= {
        "anamnesis-memory-eval"
    }
