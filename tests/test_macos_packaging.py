from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.config import ArtifactConfig, voice_config_from_mapping
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.package_profile import (
    HELPER_RELATIVE_PATH,
    PROFILE_NAME,
    apply_packaged_macos_profile,
    build_macos_package_profile,
    validate_macos_package_profile,
)


class MacPackageProfileTests(unittest.TestCase):
    def _profile(self, helper: bytes) -> dict[str, object]:
        return {
            "schema_version": "jarvis-macos-runtime-profile/1",
            "product_version": "0.7.0",
            "voice_protocol": 1,
            "minimum_macos": "14.0",
            "architecture": "arm64",
            "python_version": "3.11.14",
            "pyinstaller_version": "6.21.0",
            "helper": {
                "relative_path": HELPER_RELATIVE_PATH,
                "size": len(helper),
                "sha256": hashlib.sha256(helper).hexdigest(),
            },
            "models_embedded": False,
        }

    def test_profile_binds_packaged_helper_and_keeps_model_external(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            helper = root / HELPER_RELATIVE_PATH
            helper.parent.mkdir()
            helper.write_bytes(b"native-arm64-helper")
            (root / PROFILE_NAME).write_text(json.dumps(self._profile(helper.read_bytes())))
            config = voice_config_from_mapping(
                {
                    "macos": {
                        "whisper_cpp": {
                            "model": {"path": "/external/model.bin", "size": 100, "sha256": "a" * 64}
                        }
                    }
                }
            )
            packaged = apply_packaged_macos_profile(config, resources_root=root)
            self.assertEqual(packaged.macos.whisper_cpp.executable.path, str(helper.resolve()))
            self.assertEqual(packaged.macos.whisper_cpp.model, ArtifactConfig("/external/model.bin", 100, "a" * 64))

    def test_tampered_helper_and_profile_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            helper = root / HELPER_RELATIVE_PATH
            helper.parent.mkdir()
            helper.write_bytes(b"expected")
            (root / PROFILE_NAME).write_text(json.dumps(self._profile(b"expected")))
            helper.write_bytes(b"tampered")
            with self.assertRaisesRegex(BackendError, "packaging.macos_helper_integrity"):
                apply_packaged_macos_profile(voice_config_from_mapping({}), resources_root=root)
        value = self._profile(b"x")
        value["models_embedded"] = True
        with self.assertRaisesRegex(BackendError, "packaging.macos_profile_contract"):
            validate_macos_package_profile(value)

    def test_build_profile_requires_exact_native_toolchain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            helper = Path(temporary) / "whisper-server"
            helper.write_bytes(b"helper")
            with mock.patch("jarvis_voice.macos.package_profile.sys.platform", "darwin"), mock.patch(
                "jarvis_voice.macos.package_profile.platform.machine", return_value="arm64"
            ), mock.patch("jarvis_voice.macos.package_profile.sys.version_info", (3, 11, 14)):
                profile = build_macos_package_profile(helper, pyinstaller_version="6.21.0")
            self.assertFalse(profile["models_embedded"])
            self.assertNotIn(str(helper), str(profile))

    def test_spec_declares_bundle_privacy_and_external_models(self) -> None:
        root = Path(__file__).resolve().parents[1]
        spec = (root / "packaging" / "jarvis_voice_macos.spec").read_text()
        entitlements = (root / "packaging" / "macos-entitlements.plist").read_text()
        script = (root / "scripts" / "build_macos.sh").read_text()
        self.assertIn('bundle_identifier="com.jarvis.voice"', spec)
        self.assertIn("NSMicrophoneUsageDescription", spec)
        self.assertIn("com.apple.security.device.audio-input", entitlements)
        self.assertIn("notarytool submit", script)
        self.assertNotIn("model.bin", spec)


if __name__ == "__main__":
    unittest.main()
