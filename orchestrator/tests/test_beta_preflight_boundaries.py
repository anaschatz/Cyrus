from __future__ import annotations

from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
import test_beta_preflight as fixtures
from jarvis_orchestrator import beta_preflight as preflight
from jarvis_orchestrator.stack_config import PrivateStackConfig, StackConfigError


class PreflightBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MacosBetaPreflightTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.stack = PrivateStackConfig.load(self.fixture.stack_path)

    def evaluate(self, **kwargs):
        with patch.object(preflight, "_live_memory_crypto_state", return_value=None):
            return preflight.evaluate_macos_beta_preflight(
                self.stack,
                **{
                    "doctor_report": {"ready": True},
                    "private_state_report": self.fixture.private_state(True),
                    "voice_private_state_report": self.fixture.voice_private_state(
                        True
                    ),
                    **kwargs,
                },
            )

    def test_empty_doctor_report_never_falls_back_to_native_probes(self):
        with patch(
            "jarvis_voice.macos.doctor.macos_doctor_report",
            side_effect=AssertionError("unexpected native probe"),
        ) as native:
            report = self.evaluate(doctor_report={})
        self.assertFalse(report["technical_beta_preflight_ready"])
        self.assertEqual(
            report["voice_environment_blockers"], ["voice.environment_not_ready"]
        )
        native.assert_not_called()

    def test_malformed_blocker_status_stays_content_free(self):
        for value in ([], {}, {"private": str(self.fixture.root)}):
            with self.subTest(status=value):
                report = self.evaluate(
                    doctor_report={
                        "ready": False,
                        "checks": {
                            "audio": {"status": value, "code": str(self.fixture.root)}
                        },
                    }
                )
                self.assertFalse(report["technical_beta_preflight_ready"])
                self.assertEqual(
                    report["voice_environment_blockers"],
                    ["voice.environment_not_ready"],
                )
                self.assertNotIn(str(self.fixture.root), json.dumps(report))

    def test_unenforced_or_non_boolean_private_state_is_not_ready(self):
        for field, factory in (
            ("private_state_report", self.fixture.private_state),
            ("voice_private_state_report", self.fixture.voice_private_state),
        ):
            for changed in ({"platform_enforced": False}, {"ready": 1}):
                with self.subTest(field=field, changed=changed):
                    report = self.evaluate(**{field: replace(factory(True), **changed)})
                    self.assertFalse(report["technical_beta_preflight_ready"])
                    self.assertIs(report["checks"]["owner_only_state"], False)

    def test_changed_qualification_is_revalidated_before_any_probe(self):
        document = json.loads(self.fixture.report.read_text())
        document["status"] = "fail"
        self.fixture.report.write_text(json.dumps(document))
        with self.assertRaises(StackConfigError):
            self.evaluate()

    def test_changed_stack_manifest_cannot_use_cached_component_paths(self):
        copied = self.fixture.root / "other-voice.json"
        copied.write_bytes(self.fixture.voice.read_bytes())
        copied.chmod(0o600)
        document = json.loads(self.fixture.stack_path.read_text())
        document["voice_config_path"] = str(copied)
        self.fixture.stack_path.write_text(json.dumps(document))
        with self.assertRaisesRegex(StackConfigError, "stack_config_changed"):
            self.evaluate()

    def test_changes_during_probe_break_deployment_binding(self):
        def changed(config):
            self.fixture.runner.write_text("changed runner\n")
            return None

        with patch.object(preflight, "_live_memory_crypto_state", side_effect=changed):
            report = preflight.evaluate_macos_beta_preflight(
                self.stack,
                doctor_report={"ready": True},
                private_state_report=self.fixture.private_state(True),
                voice_private_state_report=self.fixture.voice_private_state(True),
            )
        self.assertFalse(report["technical_beta_preflight_ready"])
        self.assertFalse(report["checks"]["deployment_identity_bound"])


if __name__ == "__main__":
    unittest.main()
