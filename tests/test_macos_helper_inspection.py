from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401


class MacOSHelperInspectionTests(unittest.TestCase):
    def _module(self):
        import importlib.util

        path = Path(__file__).parents[1] / "scripts" / "inspect_macos_helper.py"
        spec = importlib.util.spec_from_file_location("inspect_macos_helper", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_rejects_adhoc_signature(self) -> None:
        module = self._module()
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "helper"
            helper.write_bytes(b"mach-o")
            helper.chmod(0o700)
            responses = [
                subprocess.CompletedProcess([], 0, "Mach-O 64-bit executable arm64\n", ""),
                subprocess.CompletedProcess([], 0, b"", b""),
                subprocess.CompletedProcess([], 0, b"Signature=adhoc\nTeamIdentifier=not set\n", b""),
                subprocess.CompletedProcess([], 0, b"designated => true\n", b""),
            ]
            with patch.object(module.subprocess, "run", side_effect=responses):
                with self.assertRaisesRegex(ValueError, "Developer ID"):
                    module.inspect_helper(helper)

    def test_emits_binary_and_code_identity_digests(self) -> None:
        module = self._module()
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "helper"
            helper.write_bytes(b"qualified-mach-o")
            helper.chmod(0o700)
            responses = [
                subprocess.CompletedProcess([], 0, "Mach-O 64-bit executable arm64\n", ""),
                subprocess.CompletedProcess([], 0, b"", b""),
                subprocess.CompletedProcess(
                    [], 0,
                    b"Authority=Developer ID Application: Example (ABCDE12345)\nTeamIdentifier=ABCDE12345\n",
                    b"",
                ),
                subprocess.CompletedProcess([], 0, b"designated => identifier example\n", b""),
            ]
            with patch.object(module.subprocess, "run", side_effect=responses):
                manifest = module.inspect_helper(helper)
            self.assertEqual(manifest["schema_version"], "jarvis-macos-helper-artifact/1")
            self.assertEqual(manifest["size_bytes"], len(b"qualified-mach-o"))
            self.assertEqual(len(manifest["binary_sha256"]), 64)
            self.assertEqual(len(manifest["code_identity_sha256"]), 64)
            from jarvis_voice.scheduler_native import MacOSSchedulerHostAdapter
            adapter = MacOSSchedulerHostAdapter.__new__(MacOSSchedulerHostAdapter)
            adapter.helper_path = helper
            adapter._timeout = 5.0
            requirements = b"Executable=/Applications/JARVIS.app/helper\ndesignated => identifier example\n"

            def run(command, **_kwargs):
                return subprocess.CompletedProcess(command, 0, requirements if "--requirements" in command else b"", b"")

            adapter._validate_helper(manifest["binary_sha256"], manifest["code_identity_sha256"], run)
            requirements = b"Executable=/Applications/JARVIS.app/helper\ndesignated => identifier changed\n"
            with self.assertRaisesRegex(ValueError, "identity invalid"):
                adapter._validate_helper(manifest["binary_sha256"], manifest["code_identity_sha256"], run)


if __name__ == "__main__":
    unittest.main()
