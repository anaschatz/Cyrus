from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import _bootstrap  # noqa: F401

from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.inference import InferenceIdentityEndpointUnavailable
from jarvis_orchestrator.stack_cli import _beta_session, _status, _voice_auth_ready
from jarvis_orchestrator.stack_config import (
    STACK_SCHEMA_VERSION,
    PrivateStackConfig,
    StackConfigError,
    write_private_stack_config,
)
from jarvis_voice.voice_auth import create_voice_auth_key_file
from jarvis_voice.voice_auth_transport import AUTH_PROBE_SUCCESS_FRAME


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class FakeHttpResponse:
    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _maximum: int | None = None) -> bytes:
        return self.body


class PrivateStackConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.voice_auth_key = self.root / "voice-auth.json"
        create_voice_auth_key_file(self.voice_auth_key, key=b"k" * 32)
        self.voice = self.private_file(
            "voice.json",
            {
                "orchestrator_url": "ws://127.0.0.1:8765",
                "voice_auth_key_file": str(self.voice_auth_key),
            },
        )
        self.orchestrator = self.private_file(
            "orchestrator.json",
            {
                "database_path": str(self.root / "state.sqlite3"),
                "control_bearer_token": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8",
                "inference_bearer_token": "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
                "voice_auth_key_file": str(self.voice_auth_key),
            },
        )
        self.runner = self.root / "runner.py"
        self.runner.write_text("pass\n", encoding="utf-8")
        self.runner.chmod(0o700)
        self.manifest_document = {
            "alias": "jarvis-local",
            "manifest_schema_version": 1,
        }
        self.manifest = self.private_file("manifest.json", self.manifest_document)
        self.report = self.private_file(
            "report.json",
            {
                "status": "pass",
                "identity": {
                    "manifest_digest": hashlib.sha256(
                        canonical(self.manifest_document)
                    ).hexdigest()
                },
            },
        )
        self.stack_path = self.root / "jarvis.private.json"

    def private_file(self, name: str, value: object) -> Path:
        path = self.root / name
        path.write_bytes(canonical(value) + b"\n")
        path.chmod(0o600)
        return path

    def document(self) -> dict[str, object]:
        return {
            "schema_version": STACK_SCHEMA_VERSION,
            "voice_config_path": str(self.voice),
            "orchestrator_config_path": str(self.orchestrator),
            "inference": {
                "runner_path": str(self.runner),
                "manifest_path": str(self.manifest),
                "qualification_report_path": str(self.report),
            },
        }

    def test_write_load_and_hash_are_deterministic_and_private(self) -> None:
        write_private_stack_config(self.stack_path, self.document())
        first = PrivateStackConfig.load(self.stack_path)
        second = PrivateStackConfig.load(self.stack_path)
        self.assertEqual(first.sha256, second.sha256)
        identity = first.deployment_identity()
        self.assertEqual(identity["deployment_sha256"], first.sha256)
        self.assertEqual(
            set(identity),
            {
                "auth_binding",
                "auth_key_version",
                "components",
                "deployment_sha256",
                "schema_version",
            },
        )
        self.assertNotIn(
            "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8",
            json.dumps(identity),
        )
        self.assertNotIn(
            "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
            json.dumps(identity),
        )
        self.assertEqual(first.voice_config_path, self.voice.resolve())
        self.assertEqual(first.orchestrator_config_path, self.orchestrator.resolve())
        self.assertEqual(first.inference.manifest_path, self.manifest.resolve())
        if os.name != "nt":
            self.assertEqual(self.stack_path.stat().st_mode & 0o777, 0o600)

    def test_deployment_identity_changes_when_component_content_changes(self) -> None:
        write_private_stack_config(self.stack_path, self.document())
        before = PrivateStackConfig.load(self.stack_path).sha256
        voice = json.loads(self.voice.read_text(encoding="utf-8"))
        voice["ptt_key"] = "CONTROL_OPTION"
        self.voice.write_bytes(canonical(voice) + b"\n")
        self.voice.chmod(0o600)

        after = PrivateStackConfig.load(self.stack_path).sha256

        self.assertNotEqual(before, after)

    def test_component_symlink_is_rejected_instead_of_resolved(self) -> None:
        linked_runner = self.root / "runner-link.py"
        linked_runner.symlink_to(self.runner)
        document = self.document()
        document["inference"] = {
            **document["inference"],  # type: ignore[arg-type]
            "runner_path": str(linked_runner),
        }
        self.private_file("jarvis.private.json", document)

        with self.assertRaisesRegex(
            StackConfigError, "inference_runner_unavailable"
        ):
            PrivateStackConfig.load(self.stack_path)

    def test_status_requires_authenticated_voice_probe(self) -> None:
        write_private_stack_config(self.stack_path, self.document())
        stack = PrivateStackConfig.load(self.stack_path)
        capability = SimpleNamespace(text=True, stream=True)
        expected_manifest, expected_runtime = (
            stack.expected_inference_runtime_identity()
        )
        live_identity = SimpleNamespace(
            manifest_digest=expected_manifest,
            runtime_identity_sha256=expected_runtime,
        )
        with (
            patch(
                "jarvis_orchestrator.stack_cli.InferenceControlClient"
            ) as inference,
            patch("jarvis_orchestrator.stack_cli._readyz", return_value=True),
            patch(
                "jarvis_orchestrator.stack_cli._beta_session",
                return_value={
                    "core_ready": True,
                    "effective_capabilities": {
                        "cloud": False,
                        "memory_retrospective": False,
                        "memory_shadow": False,
                        "research": False,
                        "tools": False,
                    },
                    "schema_version": "jarvis-orchestrator-beta-session/2",
                    "voice_client_connected": False,
                    "voice_connection_generation": 1,
                    "voice_connection_tracker_id": "00000000-0000-4000-8000-000000000001",
                },
            ),
            patch(
                "jarvis_orchestrator.stack_cli._voice_auth_ready",
                new=AsyncMock(return_value=False),
            ),
        ):
            inference.return_value.capabilities.return_value = capability
            inference.return_value.runtime_identity.return_value = live_identity
            report = _status(stack)

        self.assertFalse(report["voice_auth_ready"])
        self.assertFalse(report["ready"])
        self.assertTrue(report["inference_identity_ready"])
        self.assertFalse(report["services_ready"])
        self.assertFalse(report["voice_client_connected"])
        self.assertFalse(report["foreground_session_ready"])
        self.assertEqual(report["inference_identity_source"], "runtime_endpoint")
        self.assertEqual(report["version"], "jarvis-stack-status/5")

    def test_beta_session_parser_requires_exact_v2_continuity_fields(self) -> None:
        config = OrchestratorConfig.load(self.orchestrator)
        valid = {
            "core_ready": True,
            "effective_capabilities": {
                "cloud": False,
                "memory_retrospective": False,
                "memory_shadow": False,
                "research": False,
                "tools": False,
            },
            "schema_version": "jarvis-orchestrator-beta-session/2",
            "voice_client_connected": True,
            "voice_connection_generation": 3,
            "voice_connection_tracker_id": (
                "00000000-0000-4000-8000-000000000001"
            ),
        }
        with patch(
            "jarvis_orchestrator.stack_cli.urllib.request.urlopen",
            return_value=FakeHttpResponse(canonical(valid)),
        ):
            self.assertEqual(_beta_session(config), valid)

        invalid_documents = (
            {**valid, "schema_version": "jarvis-orchestrator-beta-session/1"},
            {
                key: value
                for key, value in valid.items()
                if key != "voice_connection_generation"
            },
            {**valid, "voice_connection_generation": True},
            {**valid, "voice_connection_tracker_id": "not-a-uuid"},
        )
        for document in invalid_documents:
            with self.subTest(document=document):
                with patch(
                    "jarvis_orchestrator.stack_cli.urllib.request.urlopen",
                    return_value=FakeHttpResponse(canonical(document)),
                ):
                    self.assertIsNone(_beta_session(config))

    def test_status_rejects_a_different_live_inference_identity(self) -> None:
        write_private_stack_config(self.stack_path, self.document())
        stack = PrivateStackConfig.load(self.stack_path)
        capability = SimpleNamespace(text=True, stream=True)
        live_identity = SimpleNamespace(
            manifest_digest="0" * 64,
            runtime_identity_sha256="1" * 64,
        )
        with (
            patch(
                "jarvis_orchestrator.stack_cli.InferenceControlClient"
            ) as inference,
            patch("jarvis_orchestrator.stack_cli._readyz", return_value=True),
            patch(
                "jarvis_orchestrator.stack_cli._beta_session",
                return_value={
                    "core_ready": True,
                    "effective_capabilities": {
                        "cloud": False,
                        "memory_retrospective": False,
                        "memory_shadow": False,
                        "research": False,
                        "tools": False,
                    },
                    "schema_version": "jarvis-orchestrator-beta-session/2",
                    "voice_client_connected": True,
                    "voice_connection_generation": 1,
                    "voice_connection_tracker_id": "00000000-0000-4000-8000-000000000001",
                },
            ),
            patch(
                "jarvis_orchestrator.stack_cli._voice_auth_ready",
                new=AsyncMock(return_value=True),
            ),
        ):
            inference.return_value.capabilities.return_value = capability
            inference.return_value.runtime_identity.return_value = live_identity
            report = _status(stack)

        self.assertTrue(report["inference_ready"])
        self.assertFalse(report["inference_identity_ready"])
        self.assertFalse(report["ready"])

    def test_status_is_ready_only_with_a_connected_real_voice_client(self) -> None:
        write_private_stack_config(self.stack_path, self.document())
        stack = PrivateStackConfig.load(self.stack_path)
        expected_manifest, expected_runtime = (
            stack.expected_inference_runtime_identity()
        )
        with (
            patch(
                "jarvis_orchestrator.stack_cli.InferenceControlClient"
            ) as inference,
            patch("jarvis_orchestrator.stack_cli._readyz", return_value=True),
            patch(
                "jarvis_orchestrator.stack_cli._voice_auth_ready",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "jarvis_orchestrator.stack_cli._beta_session",
                return_value={
                    "core_ready": True,
                    "effective_capabilities": {
                        "cloud": False,
                        "memory_retrospective": False,
                        "memory_shadow": False,
                        "research": False,
                        "tools": False,
                    },
                    "schema_version": "jarvis-orchestrator-beta-session/2",
                    "voice_client_connected": True,
                    "voice_connection_generation": 1,
                    "voice_connection_tracker_id": "00000000-0000-4000-8000-000000000001",
                },
            ),
        ):
            inference.return_value.capabilities.return_value = SimpleNamespace(
                text=True, stream=True
            )
            inference.return_value.runtime_identity.return_value = SimpleNamespace(
                manifest_digest=expected_manifest,
                runtime_identity_sha256=expected_runtime,
            )
            report = _status(stack)

        self.assertTrue(report["services_ready"])
        self.assertTrue(report["voice_client_connected"])
        self.assertTrue(report["ready"])

    def test_status_accepts_legacy_identity_only_with_a_verified_live_lease(self) -> None:
        write_private_stack_config(self.stack_path, self.document())
        stack = PrivateStackConfig.load(self.stack_path)
        with (
            patch("jarvis_orchestrator.stack_cli.InferenceControlClient") as inference,
            patch("jarvis_orchestrator.stack_cli._readyz", return_value=True),
            patch(
                "jarvis_orchestrator.stack_cli._voice_auth_ready",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "jarvis_orchestrator.stack_cli._beta_session",
                return_value={
                    "core_ready": True,
                    "effective_capabilities": {
                        "cloud": False,
                        "memory_retrospective": False,
                        "memory_shadow": False,
                        "research": False,
                        "tools": False,
                    },
                    "schema_version": "jarvis-orchestrator-beta-session/2",
                    "voice_client_connected": True,
                    "voice_connection_generation": 1,
                    "voice_connection_tracker_id": "00000000-0000-4000-8000-000000000001",
                },
            ),
            patch(
                "jarvis_orchestrator.stack_cli.validate_live_session_lease",
                return_value=True,
            ) as lease,
        ):
            inference.return_value.capabilities.return_value = SimpleNamespace(
                text=True, stream=True
            )
            inference.return_value.runtime_identity.side_effect = (
                InferenceIdentityEndpointUnavailable()
            )
            report = _status(stack)

        lease.assert_called_once_with(stack)
        self.assertTrue(report["foreground_session_ready"])
        self.assertTrue(report["inference_identity_ready"])
        self.assertEqual(report["inference_identity_source"], "foreground_lease")
        self.assertTrue(report["ready"])

    def test_status_never_uses_lease_to_override_an_identity_mismatch(self) -> None:
        write_private_stack_config(self.stack_path, self.document())
        stack = PrivateStackConfig.load(self.stack_path)
        with (
            patch("jarvis_orchestrator.stack_cli.InferenceControlClient") as inference,
            patch("jarvis_orchestrator.stack_cli._readyz", return_value=True),
            patch(
                "jarvis_orchestrator.stack_cli._voice_auth_ready",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "jarvis_orchestrator.stack_cli._beta_session",
                return_value={
                    "core_ready": True,
                    "effective_capabilities": {
                        "cloud": False,
                        "memory_retrospective": False,
                        "memory_shadow": False,
                        "research": False,
                        "tools": False,
                    },
                    "schema_version": "jarvis-orchestrator-beta-session/2",
                    "voice_client_connected": True,
                    "voice_connection_generation": 1,
                    "voice_connection_tracker_id": "00000000-0000-4000-8000-000000000001",
                },
            ),
            patch(
                "jarvis_orchestrator.stack_cli.validate_live_session_lease",
                return_value=True,
            ) as lease,
        ):
            inference.return_value.capabilities.return_value = SimpleNamespace(
                text=True, stream=True
            )
            inference.return_value.runtime_identity.return_value = SimpleNamespace(
                manifest_digest="0" * 64,
                runtime_identity_sha256="1" * 64,
            )
            report = _status(stack)

        lease.assert_not_called()
        self.assertFalse(report["foreground_session_ready"])
        self.assertFalse(report["inference_identity_ready"])
        self.assertEqual(report["inference_identity_source"], "none")
        self.assertFalse(report["ready"])

    def test_voice_auth_probe_requires_explicit_semantic_success_ack(self) -> None:
        class Socket:
            def __init__(self, reply: str) -> None:
                self.reply = reply

            async def recv(self) -> str:
                return self.reply

        class Connection:
            def __init__(self, reply: str) -> None:
                self.socket = Socket(reply)

            async def __aenter__(self) -> Socket:
                return self.socket

            async def __aexit__(self, *_args: object) -> None:
                return None

        config = SimpleNamespace(voice_host="127.0.0.1", voice_port=8765)
        authenticate = AsyncMock(return_value=None)
        with (
            patch(
                "websockets.asyncio.client.connect",
                side_effect=lambda *_args, **_kwargs: Connection(
                    AUTH_PROBE_SUCCESS_FRAME
                ),
            ),
            patch(
                "jarvis_voice.voice_auth_transport.authenticate_async_client",
                new=authenticate,
            ),
            patch(
                "jarvis_orchestrator.stack_cli._voice_auth_key",
                return_value=object(),
            ),
        ):
            self.assertTrue(asyncio.run(_voice_auth_ready(config)))

        with (
            patch(
                "websockets.asyncio.client.connect",
                side_effect=lambda *_args, **_kwargs: Connection(
                    '{"protocol":"jarvis.voice-auth-probe/1","status":"invalid"}'
                ),
            ),
            patch(
                "jarvis_voice.voice_auth_transport.authenticate_async_client",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "jarvis_orchestrator.stack_cli._voice_auth_key",
                return_value=object(),
            ),
        ):
            self.assertFalse(asyncio.run(_voice_auth_ready(config)))

    def test_endpoint_mismatch_fails_closed(self) -> None:
        self.voice.write_bytes(
            canonical(
                {
                    "orchestrator_url": "ws://127.0.0.1:9999",
                    "voice_auth_key_file": str(self.voice_auth_key),
                }
            )
            + b"\n"
        )
        self.voice.chmod(0o600)
        self.private_file("jarvis.private.json", self.document())
        with self.assertRaisesRegex(StackConfigError, "voice_orchestrator_mismatch"):
            PrivateStackConfig.load(self.stack_path)

    def test_voice_auth_key_mismatch_fails_closed(self) -> None:
        second_key = self.root / "voice-auth-second.json"
        create_voice_auth_key_file(second_key, key=b"z" * 32)
        self.voice.write_bytes(
            canonical(
                {
                    "orchestrator_url": "ws://127.0.0.1:8765",
                    "voice_auth_key_file": str(second_key),
                }
            )
            + b"\n"
        )
        self.voice.chmod(0o600)
        self.private_file("jarvis.private.json", self.document())
        with self.assertRaisesRegex(StackConfigError, "voice_auth_key_mismatch"):
            PrivateStackConfig.load(self.stack_path)

    def test_configured_voice_auth_key_symlink_fails_closed(self) -> None:
        linked_key = self.root / "voice-auth-link.json"
        linked_key.symlink_to(self.voice_auth_key)
        self.voice.write_bytes(
            canonical(
                {
                    "orchestrator_url": "ws://127.0.0.1:8765",
                    "voice_auth_key_file": str(linked_key),
                }
            )
            + b"\n"
        )
        self.voice.chmod(0o600)
        self.orchestrator.write_bytes(
            canonical(
                {
                    "database_path": str(self.root / "state.sqlite3"),
                    "control_bearer_token": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8",
                    "inference_bearer_token": "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
                    "voice_auth_key_file": str(linked_key),
                }
            )
            + b"\n"
        )
        self.orchestrator.chmod(0o600)
        self.private_file("jarvis.private.json", self.document())

        with self.assertRaisesRegex(StackConfigError, "voice_auth_key_invalid"):
            PrivateStackConfig.load(self.stack_path)

    def test_qualification_digest_mismatch_fails_closed(self) -> None:
        self.report.write_bytes(
            canonical(
                {
                    "status": "pass",
                    "identity": {"manifest_digest": "0" * 64},
                }
            )
            + b"\n"
        )
        self.report.chmod(0o600)
        self.private_file("jarvis.private.json", self.document())
        with self.assertRaisesRegex(
            StackConfigError, "inference_qualification_mismatch"
        ):
            PrivateStackConfig.load(self.stack_path)

    def test_duplicate_json_key_is_rejected(self) -> None:
        self.stack_path.write_text(
            '{"schema_version":"jarvis-private-stack/2",'
            '"schema_version":"jarvis-private-stack/2"}',
            encoding="utf-8",
        )
        self.stack_path.chmod(0o600)
        with self.assertRaisesRegex(StackConfigError, "stack_json_invalid"):
            PrivateStackConfig.load(self.stack_path)


if __name__ == "__main__":
    unittest.main()
