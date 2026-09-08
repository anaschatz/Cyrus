from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


class AnamnesisPackagingTests(unittest.TestCase):
    def test_orchestrator_declares_the_qualified_anamnesis_distribution(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        document = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertIn("anamnesis-memory-eval==0.1.0", document["project"]["dependencies"])
        self.assertEqual(
            document["tool"]["uv"]["sources"]["anamnesis-memory-eval"],
            {"path": "../anamnesis"},
        )

    def test_frozen_lock_uses_the_non_editable_local_anamnesis_distribution(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        lock = tomllib.loads((project_root / "uv.lock").read_text(encoding="utf-8"))
        packages = {package["name"]: package for package in lock["package"]}
        anamnesis = packages["anamnesis-memory-eval"]
        self.assertEqual(anamnesis["version"], "0.1.0")
        self.assertEqual(anamnesis["source"], {"directory": "../anamnesis"})
        self.assertEqual(anamnesis["dependencies"], [{"name": "pydantic"}])
        orchestrator = packages["jarvis-orchestrator"]
        self.assertIn("anamnesis-memory-eval", {dependency["name"] for dependency in orchestrator["dependencies"]})
