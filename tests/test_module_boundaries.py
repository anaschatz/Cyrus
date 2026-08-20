from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERAL_MODULE_LIMIT = 1_000
SOURCE_ROOTS = (
    ROOT / "src" / "jarvis_voice",
    ROOT / "orchestrator" / "src" / "jarvis_orchestrator",
    ROOT / "inference" / "src" / "jarvis_inference",
    ROOT / "anamnesis" / "src" / "anamnesis",
    ROOT / "research" / "src" / "jarvis_research",
)

# Existing monoliths are explicit debt, never an implicit exemption. Each
# ceiling is a one-way ratchet: growth or a new >1,000-line module fails CI;
# when a module is split below 1,000 lines its entry must be removed.
LEGACY_MODULE_CEILINGS = {
    "anamnesis/src/anamnesis/local_report.py": 1_125,
    "anamnesis/src/anamnesis/local_runtime.py": 2_675,
    "anamnesis/src/anamnesis/memory.py": 1_975,
    "anamnesis/src/anamnesis/vllm_runtime.py": 1_175,
    "anamnesis/src/anamnesis/writer_report.py": 1_050,
    "anamnesis/src/anamnesis/writer_report_w2.py": 1_225,
    "anamnesis/src/anamnesis/writer_report_w3.py": 1_350,
    "orchestrator/src/jarvis_orchestrator/anamnesis_shadow.py": 1_200,
    "orchestrator/src/jarvis_orchestrator/cli.py": 1_500,
    "orchestrator/src/jarvis_orchestrator/macos_beta_session.py": 1_900,
    "orchestrator/src/jarvis_orchestrator/memory_crypto.py": 1_200,
    "orchestrator/src/jarvis_orchestrator/memory_key_provider.py": 2_850,
    "orchestrator/src/jarvis_orchestrator/operations.py": 1_200,
    "orchestrator/src/jarvis_orchestrator/readiness.py": 1_150,
    "orchestrator/src/jarvis_orchestrator/readiness_evidence.py": 2_300,
    "orchestrator/src/jarvis_orchestrator/research.py": 1_100,
    "orchestrator/src/jarvis_orchestrator/scheduler_store.py": 2_475,
    "orchestrator/src/jarvis_orchestrator/service.py": 1_975,
    "orchestrator/src/jarvis_orchestrator/store.py": 8_575,
    "src/jarvis_voice/macos/greektts15_pilot.py": 1_150,
    "src/jarvis_voice/macos/greektts15_training.py": 2_475,
}


def production_modules() -> dict[str, int]:
    modules: dict[str, int] = {}
    for source_root in SOURCE_ROOTS:
        for path in source_root.rglob("*.py"):
            relative = path.relative_to(ROOT)
            if not path.stem.isidentifier() or not all(
                part.isidentifier() for part in relative.parent.parts
            ):
                continue
            modules[relative.as_posix()] = len(
                path.read_text(encoding="utf-8").splitlines()
            )
    return modules


class ModuleBoundaryTests(unittest.TestCase):
    def test_production_modules_obey_closed_size_ratchet(self) -> None:
        modules = production_modules()
        oversized = {
            name: lines
            for name, lines in modules.items()
            if lines > GENERAL_MODULE_LIMIT
        }

        self.assertEqual(set(oversized), set(LEGACY_MODULE_CEILINGS))
        for name, lines in sorted(oversized.items()):
            with self.subTest(module=name):
                self.assertLessEqual(lines, LEGACY_MODULE_CEILINGS[name])


if __name__ == "__main__":
    unittest.main()
