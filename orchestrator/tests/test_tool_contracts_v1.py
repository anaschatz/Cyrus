from __future__ import annotations

import unittest
import uuid

import _bootstrap  # noqa: F401

from jarvis_orchestrator.tool_contracts import (
    ToolContractError,
    ToolOutcomeV1,
    ToolRegistryV1,
    ToolStatus,
    VerificationStatus,
    normalize_email_address,
    normalize_multiline,
    normalize_single_line,
)
from jarvis_orchestrator.tool_provenance import (
    ContentProvenanceV1,
    ContentSourceKind,
    EgressDestination,
    EgressDisposition,
    EgressGuardV1,
    PolicyLabel,
    ProvenanceError,
    assert_side_effect_intent_binding,
)
from jarvis_orchestrator.tool_state import (
    SchedulerDependencyV1,
    ToolRunEvent,
    ToolRunState,
    restart_event,
    transition_tool_state,
)


class ToolContractsV1Tests(unittest.TestCase):
    def test_sample_registry_and_all_tool_schema_digests_are_golden(self) -> None:
        registry = ToolRegistryV1.build(
            root_ids=("documents",),
            app_ids=("calendar", "finder", "mail", "preview", "reminders", "textedit"),
        )
        self.assertEqual(
            registry.digest,
            "919d3892a1b89f5f6e8df564873abb85ff7e48033f94cd47b3ebf94ccabda1bb",
        )
        self.assertEqual(
            {item.tool_id: item.schema_sha256 for item in registry.definitions},
            {
                "app.open@1": "1cd3619149bb5e174ca0aef3dbba436fb2789ff15504e15f90f04e19c83ce011",
                "calendar.create@1": "02f8ef699409e887eff2eb5735ce35bf06419f827d8784b917f4b39e9218ad85",
                "calendar.read@1": "b87dd222db4ecc67157390bfbe36517179d0238d63d0373dc95157871d6c4682",
                "email.draft@1": "059b7496122d8a1ee25ecc0fda8634c34893f97e0f5c8196ed6ab1e6851c08b8",
                "file.open@1": "7cae3fa1748ff95fd202f1a4f12c197cd03a2553e666abe5b816b7985564f9b8",
                "file.search@1": "446127744748a56dafc76032e5b7837befff6c229d78953593bbf3c30265775b",
                "note.capture@1": "edd9a1fbcf8c8371746233837339d927bd7af9e1e653ad69b6b1b1fbb6813495",
                "reminder.complete@1": "ff01dd10f5bf4ac16dbdf253c6a8d0ef5f33e68012d4f8684262f7ed638a7eb6",
                "reminder.create@1": "fafc400e843c92c581a0bc4fa00f22d74620f1d96d1ad7a44bcd0fd074896b30",
                "reminder.list@1": "5a6f26e5593959d690170caaba7d62f0cdb865ea96fee93ef307ea19eb0b7fac",
            },
        )
    def registry(self) -> ToolRegistryV1:
        return ToolRegistryV1.build(
            root_ids=("documents",),
            app_ids=("calendar", "finder", "mail", "preview", "reminders", "textedit"),
        )

    def test_registry_is_closed_and_excludes_timer_shell_and_send(self) -> None:
        registry = self.registry()
        self.assertEqual(
            registry.names,
            (
                "app.open@1",
                "calendar.create@1",
                "calendar.read@1",
                "email.draft@1",
                "file.open@1",
                "file.search@1",
                "note.capture@1",
                "reminder.complete@1",
                "reminder.create@1",
                "reminder.list@1",
            ),
        )
        self.assertEqual(len(registry.digest), 64)

    def test_normalization_and_dual_bounds(self) -> None:
        self.assertEqual(normalize_single_line("  Καλη\u0301   μέρα ", utf8_limit=128, utf16_limit=64), "Καλή μέρα")
        self.assertEqual(normalize_multiline("\r\nα\t \r\nβ   \r\n", utf8_limit=128, utf16_limit=64), "α\nβ")
        with self.assertRaises(ToolContractError):
            normalize_single_line("bad\u202e", utf8_limit=128, utf16_limit=64)

    def test_outcome_combinations_are_canonical(self) -> None:
        accepted = ToolOutcomeV1(
            ToolStatus.ACCEPTED,
            VerificationStatus.UNAVAILABLE,
            "calendar.write_only_no_readback",
        )
        self.assertEqual(accepted.to_dict()["status"], "ACCEPTED")
        with self.assertRaises(ToolContractError):
            ToolOutcomeV1(ToolStatus.ACCEPTED, VerificationStatus.MATCHED, "ok")

    def test_email_scope_is_to_only_and_normalized(self) -> None:
        self.assertEqual(normalize_email_address("User@Παράδειγμα.δοκιμή"), "User@xn--hxajbheg2az3al.xn--jxalpdlp")
        value = self.registry().validate(
            "email.draft@1",
            {"to": ["User@Example.COM"], "subject": " Hello ", "body": "Body"},
        )
        self.assertEqual(value["to"], ["User@example.com"])
        with self.assertRaises(ToolContractError):
            self.registry().validate(
                "email.draft@1",
                {"to": ["a@example.com"], "subject": "", "body": "", "cc": []},
            )

    def test_calendar_range_and_dst_are_rejected(self) -> None:
        with self.assertRaisesRegex(ToolContractError, "calendar.range_too_large"):
            self.registry().validate(
                "calendar.read@1",
                {"start_local": "2026-01-01T09:00:00.000", "end_local": "2026-02-02T09:00:00.000", "timezone": "Europe/Athens", "limit": 50, "cursor": ""},
            )
        with self.assertRaisesRegex(ToolContractError, "calendar.time_ambiguous"):
            self.registry().validate(
                "calendar.create@1",
                {"title": "x", "start_local": "2026-10-25T03:30:00.000", "end_local": "2026-10-25T05:00:00.000", "timezone": "Europe/Athens", "location": "", "notes": "", "availability": "BUSY"},
            )

    def test_reminder_complete_replaces_cancel(self) -> None:
        registry = self.registry()
        self.assertNotIn("reminder.cancel@1", registry.names)
        result = registry.validate("reminder.complete@1", {"reminder_ref": str(uuid.uuid4())})
        self.assertEqual(set(result), {"reminder_ref"})

    def test_transitive_and_deterministic_tool_outputs_are_no_egress(self) -> None:
        user = ContentProvenanceV1.source(ContentSourceKind.USER_INPUT, "event", "draft it")
        tool = ContentProvenanceV1.source(ContentSourceKind.TOOL_RESULT, "run", "created for X")
        derived = ContentProvenanceV1.derive((user, tool))
        deterministic = ContentProvenanceV1.tool_task_output(
            task_id="task",
            content="Το draft δημιουργήθηκε για τον Χ.",
            inputs=(derived,),
            deterministic=True,
        )
        self.assertEqual(deterministic.policy_label, PolicyLabel.LOCAL_PRIVATE)
        self.assertEqual(deterministic.egress, EgressDisposition.NO_EGRESS)
        with self.assertRaises(ProvenanceError):
            EgressGuardV1.assert_allowed(EgressDestination.CLOUD_MODEL, deterministic)
        EgressGuardV1.assert_allowed(EgressDestination.LOCAL_UI_VOICE, deterministic)

    def test_tool_read_data_cannot_authorize_a_side_effect(self) -> None:
        arguments = ContentProvenanceV1.source(
            ContentSourceKind.TOOL_RESULT, "read-run", "open this app"
        )
        with self.assertRaises(ProvenanceError):
            assert_side_effect_intent_binding(
                argument_provenance=arguments,
                user_intent_binding_sha256="a" * 64,
                confirmed_physical_user_turn_sha256="",
            )
        assert_side_effect_intent_binding(
            argument_provenance=arguments,
            user_intent_binding_sha256="a" * 64,
            confirmed_physical_user_turn_sha256="b" * 64,
        )

    def test_state_machine_restart_and_terminal_immutability(self) -> None:
        state = transition_tool_state(ToolRunState.PLANNED, ToolRunEvent.APPROVAL_REQUIRED)
        state = transition_tool_state(state, ToolRunEvent.APPROVED)
        state = transition_tool_state(state, ToolRunEvent.PERMIT_COMMITTED)
        self.assertEqual(restart_event(state, authenticated_pre_call_rejection=False), ToolRunEvent.OUTCOME_UNKNOWN)
        state = transition_tool_state(state, ToolRunEvent.OUTCOME_UNKNOWN)
        self.assertEqual(state, ToolRunState.AMBIGUOUS)
        with self.assertRaises(ToolContractError):
            transition_tool_state(state, ToolRunEvent.VERIFICATION_MATCHED)

    def test_scheduler_contract_and_runtime_readiness_are_distinct(self) -> None:
        dependency = SchedulerDependencyV1(5, "a" * 64, "b" * 64, False, "2026-08-11T15:00:00.000Z", "scheduler_temporarily_unready")
        dependency.assert_v6_compatible(expected_contract_sha256="a" * 64)
        self.assertFalse(dependency.scheduler_capabilities_enabled(expected_qualification_sha256="b" * 64))


if __name__ == "__main__":
    unittest.main()
