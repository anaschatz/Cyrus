from __future__ import annotations

import unittest
import urllib.error
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_orchestrator.domain import OrchestratorError
from jarvis_orchestrator.inference import (
    InferenceControlClient,
    InferenceIdentityEndpointUnavailable,
)


class InferenceControlIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = InferenceControlClient(
            "http://127.0.0.1:8766/v1", "x" * 32
        )

    def test_exact_live_identity_is_accepted(self) -> None:
        with patch.object(
            self.client,
            "_get",
            return_value={
                "manifest_digest": "a" * 64,
                "runtime_identity_sha256": "b" * 64,
                "schema_version": "jarvis-inference-live-identity/1",
            },
        ):
            identity = self.client.runtime_identity()

        self.assertEqual(identity.manifest_digest, "a" * 64)
        self.assertEqual(identity.runtime_identity_sha256, "b" * 64)

    def test_malformed_or_extended_identity_fails_closed(self) -> None:
        invalid = (
            {},
            {
                "manifest_digest": "a" * 64,
                "runtime_identity_sha256": "b" * 64,
                "schema_version": "jarvis-inference-live-identity/1",
                "extra": True,
            },
            {
                "manifest_digest": "not-a-digest",
                "runtime_identity_sha256": "b" * 64,
                "schema_version": "jarvis-inference-live-identity/1",
            },
        )
        for value in invalid:
            with self.subTest(value=value), patch.object(
                self.client, "_get", return_value=value
            ), self.assertRaises(OrchestratorError):
                self.client.runtime_identity()

    def test_only_explicit_missing_identity_route_is_distinguished(self) -> None:
        missing = urllib.error.HTTPError(
            "http://127.0.0.1:8766/jarvis/v1/identity",
            404,
            "not found",
            {},
            None,
        )
        with patch(
            "urllib.request.urlopen", side_effect=missing
        ), self.assertRaises(InferenceIdentityEndpointUnavailable):
            self.client.runtime_identity()

        unavailable = urllib.error.URLError("connection refused")
        with patch(
            "urllib.request.urlopen", side_effect=unavailable
        ), self.assertRaises(OrchestratorError):
            self.client.runtime_identity()


if __name__ == "__main__":
    unittest.main()
