from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def project(path: str) -> dict[str, object]:
    return tomllib.loads((ROOT / path / "pyproject.toml").read_text())


class CyrusBrandingContractTests(unittest.TestCase):
    def test_product_entrypoints_have_legacy_compatibility_aliases(self) -> None:
        voice = project("")["project"]
        orchestrator = project("orchestrator")["project"]
        inference = project("inference")["project"]
        research = project("research")["project"]

        pairs = (
            (voice, "cyrus-voice", "jarvis-voice"),
            (orchestrator, "cyrus-orchestrator", "jarvis-orchestrator"),
            (orchestrator, "cyrus-action-host", "jarvis-action-host"),
            (orchestrator, "cyrus-stack", "jarvis-stack"),
            (inference, "cyrus-inference", "jarvis-inference"),
            (
                inference,
                "cyrus-qwen35-sampling-candidate",
                "jarvis-qwen35-sampling-candidate",
            ),
            (research, "cyrus-research-api", "jarvis-research-api"),
            (research, "cyrus-research-qualify", "jarvis-research-qualify"),
            (
                research,
                "cyrus-research-online-qualify",
                "jarvis-research-online-qualify",
            ),
            (research, "cyrus-research-benchmark", "jarvis-research-benchmark"),
            (research, "cyrus-egress-guard", "jarvis-egress-guard"),
            (research, "cyrus-research-fixture", "jarvis-research-fixture"),
            (
                research,
                "cyrus-searx-proxy-bridge",
                "jarvis-searx-proxy-bridge",
            ),
        )
        for metadata, product_name, legacy_name in pairs:
            scripts = metadata["scripts"]
            self.assertEqual(scripts[product_name], scripts[legacy_name])

    def test_native_ui_is_cyrus_but_reads_the_frozen_voice_contract(self) -> None:
        package = (ROOT / "macos" / "CyrusUI" / "Package.swift").read_text()
        runtime = (
            ROOT
            / "macos"
            / "CyrusUI"
            / "Sources"
            / "CyrusUI"
            / "CyrusRuntimeState.swift"
        ).read_text()

        self.assertIn('name: "CyrusUI"', package)
        self.assertNotIn("JarvisUI", package)
        self.assertFalse((ROOT / "macos" / "JarvisUI").exists())
        self.assertIn('version == "jarvis-voice-ui-state/2"', runtime)
        self.assertIn("Application Support/JARVIS Voice/state/ui-state.json", runtime)

    def test_active_docs_publish_cyrus_and_explain_legacy_identifiers(self) -> None:
        readme = (ROOT / "README.md").read_text()
        compatibility = (
            ROOT / "docs" / "branding-and-compatibility.md"
        ).read_text()

        self.assertTrue(readme.startswith("# CYRUS"))
        self.assertIn("macos/CyrusUI", readme)
        self.assertNotIn("macos/JarvisUI", readme)
        self.assertIn("This is a product rename", compatibility)
        self.assertIn("jarvis.voice/1", compatibility)

    def test_runtime_persona_and_operator_labels_are_cyrus(self) -> None:
        llm_context = (
            ROOT
            / "orchestrator"
            / "src"
            / "jarvis_orchestrator"
            / "llm_context.py"
        ).read_text()
        scheduler = (
            ROOT
            / "orchestrator"
            / "src"
            / "jarvis_orchestrator"
            / "scheduler_store.py"
        ).read_text()
        tray = (ROOT / "src" / "jarvis_voice" / "tray_app.py").read_text()

        self.assertIn("You are CYRUS", llm_context)
        self.assertNotIn("You are JARVIS", llm_context)
        self.assertIn('"private_title": "CYRUS"', scheduler)
        self.assertIn("CYRUS Voice —", tray)


if __name__ == "__main__":
    unittest.main()
