from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_SOURCE = ROOT / "orchestrator" / "src" / "jarvis_orchestrator"


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            imports.add(prefix + (node.module or ""))
    return imports


class DependencyBoundaryTests(unittest.TestCase):
    def test_research_test_dependency_is_exact_locked_and_dev_only(self) -> None:
        orchestrator_project = tomllib.loads(
            (ROOT / "orchestrator" / "pyproject.toml").read_text(encoding="utf-8")
        )
        research_project = tomllib.loads(
            (ROOT / "research" / "pyproject.toml").read_text(encoding="utf-8")
        )
        locked = tomllib.loads(
            (ROOT / "orchestrator" / "uv.lock").read_text(encoding="utf-8")
        )

        runtime_dependencies = orchestrator_project["project"]["dependencies"]
        self.assertNotIn("jarvis-research==1.0.0", runtime_dependencies)
        self.assertNotIn("publicsuffixlist==1.0.2.20260726", runtime_dependencies)
        self.assertEqual(
            orchestrator_project["dependency-groups"]["dev"],
            ["jarvis-research==1.0.0"],
        )
        self.assertEqual(
            orchestrator_project["tool"]["uv"]["sources"]["jarvis-research"],
            {"path": "../research"},
        )
        self.assertIn(
            "publicsuffixlist==1.0.2.20260726",
            research_project["project"]["dependencies"],
        )

        packages = {package["name"]: package for package in locked["package"]}
        self.assertEqual(packages["jarvis-research"]["source"], {"directory": "../research"})
        self.assertEqual(
            packages["publicsuffixlist"]["version"], "1.0.2.20260726"
        )
        self.assertEqual(
            packages["jarvis-orchestrator"]["dev-dependencies"]["dev"],
            [{"name": "jarvis-research"}],
        )

    def test_optional_sdk_imports_are_not_module_level_dependencies(self) -> None:
        forbidden = {
            "common_schema.py": {"pydantic_ai"},
            "pydantic_adapter.py": {"pydantic_ai"},
            "model.py": {"httpx"},
            "anamnesis_shadow.py": {"httpx"},
            "cli.py": {
                ".anamnesis_adapter",
                ".anamnesis_shadow",
                ".cloud_provider",
                ".cloud_qualification",
                ".pydantic_adapter",
                ".research_adapter",
            },
        }
        for filename, blocked in forbidden.items():
            with self.subTest(module=filename):
                imports = _top_level_imports(ORCHESTRATOR_SOURCE / filename)
                self.assertTrue(blocked.isdisjoint(imports), sorted(imports & blocked))

    def test_common_schema_import_does_not_require_pydantic_ai(self) -> None:
        script = textwrap.dedent(
            f"""
            import importlib.abc
            import sys

            class BlockPydanticAi(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "pydantic_ai" or fullname.startswith("pydantic_ai."):
                        raise ModuleNotFoundError(fullname)
                    return None

            sys.meta_path.insert(0, BlockPydanticAi())
            sys.path[:0] = [
                {str(ROOT / "orchestrator" / "src")!r},
                {str(ROOT / "src")!r},
                {str(ROOT / "inference" / "src")!r},
                {str(ROOT / "anamnesis" / "src")!r},
                {str(ROOT / "research" / "src")!r},
            ]
            from jarvis_orchestrator.common_schema import route_decision_schema
            assert route_decision_schema()["type"] == "object"
            assert not any(name == "pydantic_ai" or name.startswith("pydantic_ai.") for name in sys.modules)
            """
        )

        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
