from __future__ import annotations

import copy
import unittest
import uuid

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import OrchestratorError, canonical_sha256
from jarvis_orchestrator.memory_release import (
    MEMORY_RELEASE_POLICY,
    MEMORY_RELEASE_POLICY_SHA256,
    MEMORY_RELEASE_POLICY_VERSION,
    ShadowHistoryReaderFacade,
    shadow_safety_fields,
    validate_shadow_audit_payload,
)

EXPECTED_POLICY_SHA256 = (
    "0f846b77a38da9c4f71a9e7b650aa41d663ca151b1c275fd7486f3ac160486b6"
)


def _proposal_payload() -> dict[str, object]:
    proposal = {
        "operation": "upsert",
        "memory_key": "profile.response_style",
        "supersedes_source_event_ids": [],
        "summary": "Prefers concise responses",
        "writer_contract_sha256": "a" * 64,
    }
    proposal_sha256 = canonical_sha256(proposal)
    fingerprint = "b" * 64
    return {
        **shadow_safety_fields(),
        "status": "proposed",
        "source_event_id": "source-1",
        "text_sha256": "c" * 64,
        "proposal": proposal,
        "proposal_sha256": proposal_sha256,
        "authority_state_before_sha256": fingerprint,
        "authority_state_after_sha256": fingerprint,
        "provenance": {
            "current_source_event_id": "source-1",
            "source_text_sha256": "c" * 64,
            "superseded_source_event_ids": [],
            "writer_contract_sha256": "a" * 64,
            "proposal_sha256": proposal_sha256,
        },
    }


class MemoryReleasePolicyTests(StoreCase, unittest.TestCase):
    def test_release_policy_is_frozen_and_rejects_config_drift(self) -> None:
        self.assertEqual(MEMORY_RELEASE_POLICY_SHA256, EXPECTED_POLICY_SHA256)
        self.assertEqual(
            MEMORY_RELEASE_POLICY["version"], MEMORY_RELEASE_POLICY_VERSION
        )
        self.assertFalse(
            MEMORY_RELEASE_POLICY["production_v1"]["automatic_side_effects"]
        )
        self.assertEqual(
            MEMORY_RELEASE_POLICY["production_v1"]["generative_compilers"],
            "not_qualified",
        )
        with self.assertRaises(TypeError):
            MEMORY_RELEASE_POLICY["version"] = "mutated"  # type: ignore[index]
        with self.assertRaises(TypeError):
            MEMORY_RELEASE_POLICY["production_v1"][  # type: ignore[index]
                "automatic_side_effects"
            ] = True
        with self.assertRaisesRegex(ValueError, "memory_release_policy"):
            OrchestratorConfig.from_test_mapping(
                {
                    "database_path": "/tmp/jarvis-memory-release-test.sqlite3",
                    "control_bearer_token": "control-private",
                    "inference_bearer_token": "inference-private",
                    "memory_release_policy": "jarvis.memory-release-policy.v2",
                }
            )

    def test_shadow_payload_binds_policy_provenance_and_authority_state(self) -> None:
        payload = _proposal_payload()
        validate_shadow_audit_payload("memory.shadow.proposed", payload)
        for key, value in (
            ("automatic_side_effect_requested", True),
            ("action_evidence_emitted", True),
            ("authoritative_state_changed", True),
            ("authority_state_after_sha256", "d" * 64),
        ):
            with self.subTest(key=key):
                tampered = copy.deepcopy(payload)
                tampered[key] = value
                with self.assertRaises(ValueError):
                    validate_shadow_audit_payload("memory.shadow.proposed", tampered)
        tampered = copy.deepcopy(payload)
        tampered["provenance"]["proposal_sha256"] = "e" * 64  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "provenance binding"):
            validate_shadow_audit_payload("memory.shadow.proposed", tampered)

    def test_shadow_history_facade_is_defensive_and_has_no_write_capabilities(
        self,
    ) -> None:
        source = [{"payload": {"proposal": {"operation": "ignore"}}}]
        facade = ShadowHistoryReaderFacade(lambda conversation_id: source)
        first = facade.memory_shadow_history("conversation-1")
        first[0]["payload"]["proposal"]["operation"] = "upsert"  # type: ignore[index]
        self.assertEqual(
            facade.memory_shadow_history("conversation-1")[0]["payload"]["proposal"][
                "operation"
            ],
            "ignore",
        )
        for forbidden in (
            "add_memory_record",
            "delete_memory_record",
            "create_approval",
            "reserve_capability",
        ):
            self.assertFalse(hasattr(facade, forbidden))

    def test_tool_authority_is_off_by_default_and_opt_in_only(self) -> None:
        store = self.make_store()
        task = self.executing(
            store,
            self.admitted(
                store,
                event_id=str(uuid.uuid4()),
                turn_id=str(uuid.uuid4()),
            ),
        )
        with self.assertRaisesRegex(OrchestratorError, "forbidden"):
            store.create_approval(
                task.task_id,
                task.epoch,
                kind="tool",
                name="lamp",
                arguments={},
                now=NOW,
            )
        with self.assertRaisesRegex(OrchestratorError, "forbidden"):
            store.reserve_capability(
                task.task_id,
                task.epoch,
                kind="tool",
                request_key="call-1",
                arguments={},
                now=NOW,
            )
        with store._lock:
            self.assertEqual(
                store._connection.execute("SELECT COUNT(*) FROM approvals").fetchone()[
                    0
                ],
                0,
            )
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM capability_reservations"
                ).fetchone()[0],
                0,
            )

        enabled = self.make_store(tool_authority_enabled=True)
        enabled_task = self.executing(
            enabled,
            self.admitted(
                enabled,
                event_id=str(uuid.uuid4()),
                turn_id=str(uuid.uuid4()),
            ),
        )
        approval_id = enabled.create_approval(
            enabled_task.task_id,
            enabled_task.epoch,
            kind="tool",
            name="test-only",
            arguments={},
            now=NOW,
        )
        self.assertTrue(approval_id)


if __name__ == "__main__":
    unittest.main()
