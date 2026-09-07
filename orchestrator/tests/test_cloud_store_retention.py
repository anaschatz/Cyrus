from __future__ import annotations

import sqlite3
import unittest
import uuid
from dataclasses import replace
from datetime import timedelta

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase
from jarvis_orchestrator.cloud_contract import (
    DEEPSEEK_MODEL, MONTHLY_HARD_CAP_MICROUSD, CloudApprovalBindingV1,
    build_dispatch_body, reserve_cost_microusd,
)
from jarvis_orchestrator.cloud_routing import DeterministicModelRouterV1
from jarvis_orchestrator.domain import BudgetExhaustedError, ReasonCode, TaskState, utc_text
from jarvis_orchestrator.jcs import domain_sha256
from jarvis_orchestrator.store import OrchestratorStore
from test_cloud_release_qualification import endpoint
from test_m1c_cloud_contract import facts, profile


def cloud_approval(store):
    task = store.admit_transcript(
        event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4()),
        payload={"text": "analyse", "language": {"code": "en"}}, now=NOW,
    ).task
    store.transition(task.task_id, task.epoch, TaskState.ADMITTED, TaskState.ROUTING, now=NOW)
    selected_profile = profile()
    selected_facts = replace(facts(task.task_id, selected_profile.sha256), epoch=task.epoch)
    plan = DeterministicModelRouterV1.decide(selected_profile, selected_facts)
    store.persist_route_artifacts(task.task_id, task.epoch, selected_profile, selected_facts, plan, now=NOW)
    store.transition(task.task_id, task.epoch, TaskState.ROUTING, TaskState.LOADING_CONTEXT, now=NOW)
    task = store.transition(task.task_id, task.epoch, TaskState.LOADING_CONTEXT, TaskState.EXECUTING, now=NOW)
    selected_endpoint = endpoint(DEEPSEEK_MODEL)
    store.register_endpoint_qualification(selected_endpoint, enabled=True, now=NOW)
    body = build_dispatch_body(selected_endpoint, messages=({"role": "user", "content": "analyse"},), output_token_limit=1024)
    reserved = reserve_cost_microusd(
        prompt_cap=1000, completion_cap=1024,
        prompt_rate_per_m=selected_endpoint.prompt_rate_per_m,
        completion_rate_per_m=selected_endpoint.completion_rate_per_m,
    )
    binding = CloudApprovalBindingV1(
        conversation_id=task.conversation_id, task_id=task.task_id, epoch=task.epoch,
        expires_at=utc_text(NOW + timedelta(minutes=10)),
        egress_sha256=domain_sha256("cloud-egress/1", {"messages": body["messages"]}),
        route_plan_sha256=plan.sha256, dispatch_body_sha256=domain_sha256("cloud-dispatch-body/1", body),
        model_canonical_id=selected_endpoint.model_canonical_id,
        provider_endpoint_tag=selected_endpoint.provider_endpoint_tag,
        quantization=selected_endpoint.quantization,
        parameter_profile_sha256=selected_endpoint.parameter_profile_sha256,
        qualification_version=selected_endpoint.qualification_version,
        qualification_sha256=selected_endpoint.qualification_sha256,
        stable_catalog_sha256=selected_endpoint.stable_catalog_sha256,
        privacy_policy_sha256="c" * 64, billable_prompt_tokens=1000,
        input_token_limit=128_000, output_token_limit=1024,
        max_reserved_microusd=reserved, included_turn_ids=(),
    )
    session_id = str(uuid.uuid4())
    created = store.create_cloud_approval(task.task_id, task.epoch, binding, body, owner_session_id=session_id, now=NOW)
    nonce = store._connection.execute(
        "SELECT decision_nonce FROM cloud_presentations WHERE presentation_id=?", (created["presentation_id"],)
    ).fetchone()[0]
    decision = dict(
        decision_event_id=str(uuid.uuid4()), semantic_sha256="d" * 64,
        approval_id=created["approval_id"], presentation_id=created["presentation_id"],
        requested_event_id=created["requested_event_id"], approval_binding_sha256=binding.sha256,
        decision_nonce=nonce, decision="approve", session_id=session_id, capture_idle=True,
    )
    permit_request = dict(
        approval_id=created["approval_id"], approval_binding_sha256=binding.sha256,
        stable_catalog_sha256=selected_endpoint.stable_catalog_sha256,
        qualification_sha256=selected_endpoint.qualification_sha256,
        reserved_microusd=reserved, key_remaining_microusd=MONTHLY_HARD_CAP_MICROUSD,
        utc_month=NOW.strftime("%Y-%m"), now=NOW,
    )
    return task, binding, decision, permit_request


class CloudStoreRetentionTests(StoreCase, unittest.TestCase):
    def test_deleted_cost_remains_in_monthly_budget_after_restart_and_blocks_dispatch(self):
        store = self.make_store()
        task, binding, decision, request = cloud_approval(store)
        store.decide_cloud_approval(**decision, now=NOW)
        store.resume_cloud_decision(decision["approval_id"], now=NOW)
        permit = store.issue_cloud_dispatch_permit(task.task_id, task.epoch, **request)
        store.consume_cloud_dispatch_permit(permit["permit_id"], task.task_id, task.epoch, binding.dispatch_body_sha256, now=NOW)
        store.reconcile_cloud_dispatch(permit["reservation_id"], actual_microusd=MONTHLY_HARD_CAP_MICROUSD, now=NOW)
        cancelled = store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED, now=NOW)
        store.delete_conversation(task.conversation_id, cancelled["epoch"], now=NOW)
        self.assertEqual(store.cloud_monthly_remaining("2026-08"), 0)
        self.assertEqual(store.cloud_monthly_remaining("2026-09"), MONTHLY_HARD_CAP_MICROUSD)
        for table in ("tasks", "conversations", "cloud_cost_reservations", "cloud_approval_bindings", "cloud_provider_receipts"):
            self.assertEqual(store._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)
        reopened = OrchestratorStore(store.path)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.cloud_monthly_remaining("2026-08"), 0)
        next_task, _, next_decision, next_request = cloud_approval(reopened)
        reopened.decide_cloud_approval(**next_decision, now=NOW)
        reopened.resume_cloud_decision(next_decision["approval_id"], now=NOW)
        with self.assertRaises(BudgetExhaustedError):
            reopened.issue_cloud_dispatch_permit(next_task.task_id, next_task.epoch, **next_request)

    def test_ambiguous_cost_is_retained_and_failed_delete_does_not_double_count(self):
        store = self.make_store()
        task, binding, decision, request = cloud_approval(store)
        store.decide_cloud_approval(**decision, now=NOW)
        store.resume_cloud_decision(decision["approval_id"], now=NOW)
        permit = store.issue_cloud_dispatch_permit(task.task_id, task.epoch, **request)
        store.consume_cloud_dispatch_permit(permit["permit_id"], task.task_id, task.epoch, binding.dispatch_body_sha256, now=NOW)
        cancelled = store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED, now=NOW)
        expected = MONTHLY_HARD_CAP_MICROUSD - request["reserved_microusd"]
        store._connection.execute("CREATE TRIGGER review_fail_delete BEFORE DELETE ON conversations BEGIN SELECT RAISE(ABORT,'synthetic'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            store.delete_conversation(task.conversation_id, cancelled["epoch"], now=NOW)
        self.assertEqual(store.cloud_monthly_remaining("2026-08"), expected)
        store._connection.execute("DROP TRIGGER review_fail_delete")
        store.delete_conversation(task.conversation_id, cancelled["epoch"], now=NOW)
        self.assertEqual(store.cloud_monthly_remaining("2026-08"), expected)

    def test_multiple_deletions_accumulate_charges_but_not_unused_reservations(self):
        store = self.make_store()
        charged = 0
        for dispatch in (True, False, True):
            task, binding, decision, request = cloud_approval(store)
            store.decide_cloud_approval(**decision, now=NOW)
            store.resume_cloud_decision(decision["approval_id"], now=NOW)
            permit = store.issue_cloud_dispatch_permit(task.task_id, task.epoch, **request)
            if dispatch:
                store.consume_cloud_dispatch_permit(permit["permit_id"], task.task_id, task.epoch, binding.dispatch_body_sha256, now=NOW)
                store.reconcile_cloud_dispatch(permit["reservation_id"], actual_microusd=100, now=NOW)
                charged += 100
            cancelled = store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED, now=NOW)
            store.delete_conversation(task.conversation_id, cancelled["epoch"], now=NOW)
            self.assertEqual(store.cloud_monthly_remaining("2026-08"), MONTHLY_HARD_CAP_MICROUSD - charged)
        retained = store._connection.execute(
            "SELECT key,value FROM schema_meta WHERE key LIKE 'cloud_retired_microusd:%'"
        ).fetchall()
        self.assertEqual([tuple(row) for row in retained], [("cloud_retired_microusd:2026-08", "200")])

    def test_expired_decision_ack_and_rejection_are_deliverable(self):
        store = self.make_store()
        task, _, decision, _ = cloud_approval(store)
        selected = NOW + timedelta(minutes=10)
        outcome = store.decide_cloud_approval(**decision, now=selected)
        self.assertEqual(outcome["result"]["reason_code"], "approval_expired")
        events = store.claim_cloud_extension_events(task.conversation_id, now=selected)
        self.assertEqual(
            {event["event_id"] for event in events},
            {outcome["ack_event_id"], outcome["result_event_id"]},
        )
        self.assertEqual(store.get_task(task.task_id).state, TaskState.AWAITING_APPROVAL)


if __name__ == "__main__":
    unittest.main()
