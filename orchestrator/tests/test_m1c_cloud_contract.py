from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401

from jarvis_orchestrator.cloud_contract import (
    DEEPSEEK_MODEL,
    CloudApprovalBindingV1,
    QualifiedEndpointV1,
    build_dispatch_body,
    reserve_cost_microusd,
    TERRA_MODEL,
)
from jarvis_orchestrator.cloud_provider import (
    CloudModelResultV1,
    OpenRouterCloudAdapter,
    OpenRouterRevalidationV1,
)
from jarvis_orchestrator.cloud_privacy import JarvisPrivacyClassifierV1, PrivacyClass
from jarvis_orchestrator.cloud_routing import (
    CandidateFactsV1,
    ContextScope,
    DeterministicModelRouterV1,
    OutputContract,
    RouteReason,
    RouteTarget,
    RoutingFactsV1,
    TaskKind,
    TaskProfileV1,
    complexity_score,
)
from jarvis_orchestrator.domain import TaskState, utc_text
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.cloud_contract import EndpointHealthV1
from jarvis_orchestrator.jcs import CanonicalJsonError, domain_sha256, jcs_text
from jarvis_orchestrator.store import OrchestratorStore
from jarvis_voice.cloud_approval import (
    CLOUD_APPROVAL_PROTOCOL,
    CloudApprovalController,
    CloudApprovalEnvelope,
    CloudApprovalEventFactory,
    CloudApprovalJournal,
)


NOW = datetime(2026, 8, 11, 9, 0, tzinfo=UTC)
DIGEST = "a" * 64


def candidate(alias: str, *, cost: int = 10_000) -> CandidateFactsV1:
    return CandidateFactsV1(
        alias=alias,
        qualified=True,
        ready=True,
        circuit_open=False,
        capabilities=("text", "structured_output", "tools"),
        context_window=128_000 if alias != "local" else 4096,
        sample_count=20,
        p95_ttft_ms=100,
        p95_total_ms=500,
        max_reserved_microusd=cost,
        stable_catalog_sha256=DIGEST,
        qualification_sha256=DIGEST,
    )


def profile() -> TaskProfileV1:
    return TaskProfileV1(
        version="task-profile/1",
        task_kind=TaskKind.ANALYSIS,
        reasoning_depth=1,
        constraint_count=3,
        domains=("general", "software"),
        context_scope=ContextScope.CONVERSATION,
        output_contract=OutputContract.PLAIN_TEXT,
        response_language="el",
    )


def facts(task_id: str, profile_sha256: str, *, approval: str = "absent") -> RoutingFactsV1:
    return RoutingFactsV1(
        version="routing-facts/1",
        task_id=task_id,
        epoch=1,
        profile_sha256=profile_sha256,
        estimated_input_tokens=1000,
        reserved_output_tokens=1024,
        remaining_active_ms=100_000,
        remaining_absolute_ms=800_000,
        durable_output_started=False,
        privacy_eligible=True,
        contains_memory=False,
        contains_research=False,
        contains_tool_result=False,
        extension_ready=True,
        connection_ready=True,
        approval_status=approval,
        cloud_dispatch_count=0,
        monthly_remaining_microusd=5_000_000,
        flow_kind="local",
        local_failure="",
        local=candidate("local", cost=0),
        deepseek=candidate("deepseek"),
        terra=candidate("terra", cost=198_000),
    )


class RoutingContractTests(unittest.TestCase):
    def test_complexity_thresholds_are_frozen_at_39_40_69_70(self) -> None:
        fixtures = (
            (
                39,
                TaskProfileV1(
                    version="task-profile/1",
                    task_kind=TaskKind.CLASSIFICATION,
                    reasoning_depth=1,
                    constraint_count=4,
                    domains=("general", "software"),
                    context_scope=ContextScope.CONVERSATION,
                    output_contract=OutputContract.TOOL_CHOICE,
                    response_language="en",
                ),
            ),
            (
                40,
                TaskProfileV1(
                    version="task-profile/1",
                    task_kind=TaskKind.SUMMARY,
                    reasoning_depth=1,
                    constraint_count=3,
                    domains=("general", "software"),
                    context_scope=ContextScope.CONVERSATION,
                    output_contract=OutputContract.TOOL_CHOICE,
                    response_language="en",
                ),
            ),
            (
                69,
                TaskProfileV1(
                    version="task-profile/1",
                    task_kind=TaskKind.ANALYSIS,
                    reasoning_depth=2,
                    constraint_count=4,
                    domains=("general", "software"),
                    context_scope=ContextScope.CONVERSATION,
                    output_contract=OutputContract.TOOL_CHOICE,
                    response_language="en",
                ),
            ),
            (
                70,
                TaskProfileV1(
                    version="task-profile/1",
                    task_kind=TaskKind.PLANNING,
                    reasoning_depth=2,
                    constraint_count=3,
                    domains=("general", "software"),
                    context_scope=ContextScope.CONVERSATION,
                    output_contract=OutputContract.TOOL_CHOICE,
                    response_language="en",
                ),
            ),
        )
        self.assertEqual([complexity_score(item) for _, item in fixtures], [score for score, _ in fixtures])
        targets = [
            DeterministicModelRouterV1.decide(item, facts(str(uuid.uuid4()), item.sha256)).target
            for _, item in fixtures
        ]
        self.assertEqual(
            targets,
            [RouteTarget.LOCAL, RouteTarget.DEEPSEEK, RouteTarget.DEEPSEEK, RouteTarget.TERRA],
        )

    def test_complexity_and_preapproval_route_are_deterministic(self) -> None:
        selected = profile()
        self.assertEqual(complexity_score(selected), 52)
        evidence = facts(str(uuid.uuid4()), selected.sha256)
        first = DeterministicModelRouterV1.decide(selected, evidence)
        second = DeterministicModelRouterV1.decide(selected, evidence)
        self.assertEqual(first.target, RouteTarget.DEEPSEEK)
        self.assertEqual(first.reason, RouteReason.CLOUD_APPROVAL_REQUIRED)
        self.assertEqual(first.canonical_json(), second.canonical_json())
        self.assertEqual(first.sha256, second.sha256)

    def test_memory_and_durable_output_never_escalate(self) -> None:
        selected = profile()
        raw = facts(str(uuid.uuid4()), selected.sha256)
        memory = replace(raw, flow_kind="memory")
        self.assertEqual(DeterministicModelRouterV1.decide(selected, memory).target, RouteTarget.LOCAL)

    def test_project_owned_profile_and_facts_are_closed_schemas(self) -> None:
        selected = profile()
        evidence = facts(str(uuid.uuid4()), selected.sha256)
        self.assertEqual(TaskProfileV1.from_mapping(selected.canonical()), selected)
        self.assertEqual(RoutingFactsV1.from_mapping(evidence.canonical()), evidence)
        extra = {**evidence.canonical(), "provider_override": "attacker"}
        with self.assertRaises(ValueError):
            RoutingFactsV1.from_mapping(extra)

    def test_jcs_rejects_floats_and_binding_changes_with_expiry(self) -> None:
        with self.assertRaises(CanonicalJsonError):
            jcs_text({"price": 0.2})
        self.assertEqual(jcs_text({"β": 1}) if False else jcs_text({"a": "é"}), '{"a":"é"}')
        common = dict(
            conversation_id=str(uuid.uuid4()),
            task_id=str(uuid.uuid4()),
            epoch=1,
            egress_sha256=DIGEST,
            route_plan_sha256=DIGEST,
            dispatch_body_sha256=DIGEST,
            model_canonical_id=DEEPSEEK_MODEL,
            provider_endpoint_tag="provider/endpoint",
            quantization="fp8",
            parameter_profile_sha256=DIGEST,
            qualification_version="qualification/1",
            qualification_sha256=DIGEST,
            stable_catalog_sha256=DIGEST,
            privacy_policy_sha256=DIGEST,
            billable_prompt_tokens=1000,
            input_token_limit=128_000,
            output_token_limit=2048,
            max_reserved_microusd=20_000,
            included_turn_ids=(),
        )
        first = CloudApprovalBindingV1(expires_at="2026-08-11T09:10:00.000Z", **common)
        second = CloudApprovalBindingV1(expires_at="2026-08-11T09:10:01.000Z", **common)
        self.assertNotEqual(first.sha256, second.sha256)

    def test_terra_policy_fixture_is_exact(self) -> None:
        self.assertEqual(
            reserve_cost_microusd(
                prompt_cap=60_000,
                completion_cap=2_000,
                prompt_rate_per_m="2.50",
                completion_rate_per_m="15",
            ),
            198_000,
        )

    def test_exact_deepseek_and_terra_request_profiles(self) -> None:
        common = dict(
            provider_endpoint_tag="provider/full",
            quantization="fp8",
            supported_parameters=("max_tokens", "reasoning", "temperature", "top_p"),
            qualification_version="q/1",
            qualification_sha256="b" * 64,
            context_limit=128_000,
            output_limit=2048,
            prompt_rate_per_m="0.09",
            completion_rate_per_m="0.18",
            request_rate="0",
            runtime_provider_name="Provider Inc",
        )
        deepseek = QualifiedEndpointV1(model_canonical_id=DEEPSEEK_MODEL, **common)
        deepseek_body = build_dispatch_body(
            deepseek,
            messages=({"role": "user", "content": "hello"},),
            output_token_limit=2048,
        )
        self.assertEqual(deepseek_body["reasoning"], {"enabled": False})
        self.assertEqual(deepseek_body["max_tokens"], 2048)
        self.assertNotIn("max_completion_tokens", deepseek_body)
        self.assertFalse(deepseek_body["provider"]["allow_fallbacks"])
        self.assertEqual(deepseek_body["provider"]["max_price"]["request"], "0")
        self.assertEqual(
            deepseek_body,
            json.loads(
                (ROOT / "orchestrator" / "tests" / "snapshots" / "openrouter_deepseek.json").read_text(
                    encoding="utf-8"
                )
            ),
        )

        terra = QualifiedEndpointV1(
            **{
                **common,
                "model_canonical_id": TERRA_MODEL,
                "supported_parameters": ("max_completion_tokens", "reasoning"),
                "context_limit": 60_000,
                "output_limit": 2000,
                "prompt_rate_per_m": "2.50",
                "completion_rate_per_m": "15",
            }
        )
        terra_body = build_dispatch_body(
            terra,
            messages=({"role": "user", "content": "hard problem"},),
            output_token_limit=2000,
        )
        self.assertEqual(terra_body["reasoning"], {"effort": "medium"})
        self.assertEqual(terra_body["max_completion_tokens"], 2000)
        self.assertNotIn("temperature", terra_body)
        self.assertNotIn("max_tokens", terra_body)
        self.assertEqual(
            terra_body,
            json.loads(
                (ROOT / "orchestrator" / "tests" / "snapshots" / "openrouter_terra.json").read_text(
                    encoding="utf-8"
                )
            ),
        )

    def test_combined_privacy_is_fail_closed(self) -> None:
        classifier = JarvisPrivacyClassifierV1()
        self.assertEqual(classifier.classify("Public product documentation").classification, PrivacyClass.PUBLIC)
        self.assertEqual(classifier.classify("My API key is sk-secretvalue").classification, PrivacyClass.SECRET_CREDENTIAL)
        self.assertFalse(classifier.classify("", memory_derived=True).cloud_eligible)


class ExtensionProtocolTests(unittest.TestCase):
    def requested(self, factory: CloudApprovalEventFactory) -> CloudApprovalEnvelope:
        return factory.create(
            "cloud.approval.requested",
            {
                "approval_binding_sha256": DIGEST,
                "approval_id": str(uuid.uuid4()),
                "approve_key": "F10",
                "billable_prompt_tokens": 1000,
                "decision_nonce": str(uuid.uuid4()),
                "deny_key": "F11",
                "egress_sha256": DIGEST,
                "epoch": 1,
                "expires_at": "2099-08-11T09:10:00.000Z",
                "included_turn_ids": [],
                "input_token_limit": 128000,
                "max_reserved_microusd": 10000,
                "model_canonical_id": DEEPSEEK_MODEL,
                "output_token_limit": 2048,
                "presentation_id": str(uuid.uuid4()),
                "provider_endpoint_tag": "provider/endpoint",
                "qualification_sha256": DIGEST,
                "quantization": "fp8",
                "route_plan_sha256": DIGEST,
                "stable_catalog_sha256": DIGEST,
                "task_id": str(uuid.uuid4()),
            },
            sent_at=utc_text(NOW),
        )

    def test_exact_envelope_and_conflicting_journal_duplicate(self) -> None:
        factory = CloudApprovalEventFactory(str(uuid.uuid4()))
        event = self.requested(factory)
        self.assertEqual(CloudApprovalEnvelope.from_json(event.to_json()), event)
        with self.assertRaises(Exception):
            CloudApprovalEnvelope.from_json(event.to_json()[:-1] + ',"extra":1}')
        with self.assertRaises(Exception):
            CloudApprovalEnvelope.from_json(event.to_json().replace('"seq":1', '"seq":1,"seq":1'))
        oversized = event.to_json().replace("provider/endpoint", "x" * (16 * 1024))
        with self.assertRaises(Exception):
            CloudApprovalEnvelope.from_json(oversized)
        with tempfile.TemporaryDirectory() as directory:
            journal = CloudApprovalJournal(Path(directory) / "cloud.sqlite3")
            self.assertFalse(journal.observe_server_event(event, expires_at=event.payload["expires_at"]))
            conflicting = CloudApprovalEnvelope(
                protocol=event.protocol,
                type=event.type,
                event_id=event.event_id,
                session_id=event.session_id,
                seq=event.seq + 1,
                sent_at=event.sent_at,
                payload={**event.payload, "max_reserved_microusd": 10_001},
            )
            with self.assertRaises(Exception):
                journal.observe_server_event(conflicting, expires_at=event.payload["expires_at"])
            journal.close()

    def test_f10_is_persisted_before_dispatch_and_restart_retransmits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.sqlite3"
            journal = CloudApprovalJournal(path)
            controller = CloudApprovalController(session_id=str(uuid.uuid4()), journal=journal)
            server_factory = CloudApprovalEventFactory(controller.factory.session_id)
            requested = self.requested(server_factory)
            acks = controller.handle_server_event(requested, key_counter=7, both_keys_up=True)
            self.assertEqual(acks[0].type, "cloud.approval.event_ack")
            self.assertIsNone(controller.key_down("F10", key_counter=7, capture_idle=True, autorepeat=False))
            self.assertIsNone(controller.key_down("F10", key_counter=8, capture_idle=False, autorepeat=False))
            self.assertIsNone(controller.key_down("F10", key_counter=8, capture_idle=True, autorepeat=True))
            decision = controller.key_down("F10", key_counter=8, capture_idle=True, autorepeat=False)
            self.assertIsNotNone(decision)
            self.assertEqual(len(journal.pending_decisions()), 1)
            decision_ack = server_factory.create(
                "cloud.approval.decision_ack",
                {"decision_event_id": decision.event_id, "status": "accepted"},
            )
            self.assertEqual(
                controller.handle_server_event(decision_ack, key_counter=8, both_keys_up=True),
                [],
                "a decision ACK must not create an ACK-of-ACK frame",
            )
            self.assertEqual(journal.pending_decisions(), [])
            journal.close()
            restarted_journal = CloudApprovalJournal(path)
            restarted = CloudApprovalController(session_id=controller.factory.session_id, journal=restarted_journal)
            replay = restarted.due_decision_retransmissions(now=0.0)
            self.assertEqual(replay, [])
            restarted_journal.close()

    def test_unacknowledged_decision_survives_voice_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.sqlite3"
            session_id = str(uuid.uuid4())
            journal = CloudApprovalJournal(path)
            controller = CloudApprovalController(session_id=session_id, journal=journal)
            requested = self.requested(CloudApprovalEventFactory(session_id))
            controller.handle_server_event(requested, key_counter=1, both_keys_up=True)
            decision = controller.key_down("F10", key_counter=2, capture_idle=True, autorepeat=False)
            assert decision is not None
            journal.close()
            restarted_journal = CloudApprovalJournal(path)
            restarted = CloudApprovalController(session_id=session_id, journal=restarted_journal)
            replay = restarted.due_decision_retransmissions(now=0.0)
            self.assertEqual([item.event_id for item in replay], [decision.event_id])
            restarted_journal.close()


class StoreCloudTransactionTests(unittest.TestCase):
    def test_approval_dispatch_and_cost_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = OrchestratorStore(Path(directory) / "state.sqlite3")
            self.addCleanup(store.close)
            admitted = store.admit_transcript(
                event_id=str(uuid.uuid4()),
                turn_id=str(uuid.uuid4()),
                payload={"text": "analyse", "language": {"code": "en"}},
                now=NOW,
            ).task
            assert admitted is not None
            routing = store.transition(admitted.task_id, admitted.epoch, TaskState.ADMITTED, TaskState.ROUTING, now=NOW)
            selected_profile = profile()
            selected_facts = facts(routing.task_id, selected_profile.sha256)
            plan = DeterministicModelRouterV1.decide(selected_profile, selected_facts)
            store.persist_route_artifacts(routing.task_id, routing.epoch, selected_profile, selected_facts, plan, now=NOW)
            store.transition(routing.task_id, routing.epoch, TaskState.ROUTING, TaskState.LOADING_CONTEXT, now=NOW)
            executing = store.transition(routing.task_id, routing.epoch, TaskState.LOADING_CONTEXT, TaskState.EXECUTING, now=NOW)

            endpoint = QualifiedEndpointV1(
                model_canonical_id=DEEPSEEK_MODEL,
                provider_endpoint_tag="provider/full-endpoint",
                quantization="fp8",
                supported_parameters=("max_tokens", "reasoning", "temperature", "top_p"),
                qualification_version="qualification/1",
                qualification_sha256="b" * 64,
                context_limit=128_000,
                output_limit=2048,
                prompt_rate_per_m="0.09",
                completion_rate_per_m="0.18",
                request_rate="0",
                runtime_provider_name="Provider Inc",
            )
            store.register_endpoint_qualification(endpoint, enabled=True, now=NOW)
            body = build_dispatch_body(endpoint, messages=({"role": "user", "content": "analyse"},), output_token_limit=1024)
            reserved = reserve_cost_microusd(
                prompt_cap=1000,
                completion_cap=1024,
                prompt_rate_per_m=endpoint.prompt_rate_per_m,
                completion_rate_per_m=endpoint.completion_rate_per_m,
            )
            binding = CloudApprovalBindingV1(
                conversation_id=executing.conversation_id,
                task_id=executing.task_id,
                epoch=executing.epoch,
                expires_at=utc_text(NOW + timedelta(minutes=10)),
                egress_sha256=domain_sha256("cloud-egress/1", {"messages": body["messages"]}),
                route_plan_sha256=plan.sha256,
                dispatch_body_sha256=domain_sha256("cloud-dispatch-body/1", body),
                model_canonical_id=endpoint.model_canonical_id,
                provider_endpoint_tag=endpoint.provider_endpoint_tag,
                quantization=endpoint.quantization,
                parameter_profile_sha256=endpoint.parameter_profile_sha256,
                qualification_version=endpoint.qualification_version,
                qualification_sha256=endpoint.qualification_sha256,
                stable_catalog_sha256=endpoint.stable_catalog_sha256,
                privacy_policy_sha256="c" * 64,
                billable_prompt_tokens=1000,
                input_token_limit=128_000,
                output_token_limit=1024,
                max_reserved_microusd=reserved,
                included_turn_ids=(),
            )
            session_id = str(uuid.uuid4())
            mutated_body = {**body, "seed": 7}
            mutated_binding = replace(
                binding,
                dispatch_body_sha256=domain_sha256("cloud-dispatch-body/1", mutated_body),
            )
            with self.assertRaises(Exception):
                store.create_cloud_approval(
                    executing.task_id,
                    executing.epoch,
                    mutated_binding,
                    mutated_body,
                    owner_session_id=session_id,
                    now=NOW,
                )
            created = store.create_cloud_approval(executing.task_id, executing.epoch, binding, body, owner_session_id=session_id, now=NOW)
            self.assertEqual(store.get_task(executing.task_id).state, TaskState.AWAITING_APPROVAL)
            old_nonce = store._connection.execute(
                "SELECT decision_nonce FROM cloud_presentations WHERE presentation_id=?",
                (created["presentation_id"],),
            ).fetchone()[0]
            new_session_id = str(uuid.uuid4())
            ownership = store.claim_cloud_presentation_owner(
                new_session_id,
                now=NOW + timedelta(milliseconds=100),
            )
            assert ownership is not None
            self.assertEqual(ownership["old_session_id"], session_id)
            stale = store.decide_cloud_approval(
                decision_event_id=str(uuid.uuid4()),
                semantic_sha256="f" * 64,
                approval_id=created["approval_id"],
                presentation_id=created["presentation_id"],
                requested_event_id=created["requested_event_id"],
                approval_binding_sha256=binding.sha256,
                decision_nonce=old_nonce,
                decision="approve",
                session_id=session_id,
                capture_idle=True,
                now=NOW + timedelta(milliseconds=200),
            )
            self.assertEqual(stale["result"]["reason_code"], "presentation_superseded")
            self.assertEqual(stale["result"]["result"], "rejected")
            session_id = new_session_id
            created = {
                "approval_id": created["approval_id"],
                "presentation_id": ownership["presentation_id"],
                "requested_event_id": ownership["requested_event_id"],
            }
            decision_factory = CloudApprovalEventFactory(session_id)
            decision = decision_factory.create(
                "cloud.approval.decision",
                {
                    "approval_binding_sha256": binding.sha256,
                    "approval_id": created["approval_id"],
                    "decision": "approve",
                    "decision_nonce": store._connection.execute("SELECT decision_nonce FROM cloud_presentations WHERE presentation_id=?", (created["presentation_id"],)).fetchone()[0],
                    "presentation_id": created["presentation_id"],
                    "requested_event_id": created["requested_event_id"],
                },
                sent_at=utc_text(NOW + timedelta(seconds=1)),
            )
            outcome = store.decide_cloud_approval(
                decision_event_id=decision.event_id,
                semantic_sha256=decision.semantic_sha256,
                approval_id=created["approval_id"],
                presentation_id=created["presentation_id"],
                requested_event_id=created["requested_event_id"],
                approval_binding_sha256=binding.sha256,
                decision_nonce=decision.payload["decision_nonce"],
                decision="approve",
                session_id=session_id,
                capture_idle=True,
                now=NOW + timedelta(seconds=1),
            )
            self.assertEqual(outcome["result"]["result"], "applied")
            store.resume_cloud_decision(created["approval_id"], now=NOW + timedelta(seconds=1))
            with self.assertRaises(Exception):
                store.issue_cloud_dispatch_permit(
                    executing.task_id,
                    executing.epoch,
                    approval_id=created["approval_id"],
                    approval_binding_sha256=binding.sha256,
                    stable_catalog_sha256=endpoint.stable_catalog_sha256,
                    qualification_sha256=endpoint.qualification_sha256,
                    reserved_microusd=reserved,
                    key_remaining_microusd=5_000_000,
                    utc_month="2026-07",
                    now=NOW + timedelta(seconds=2),
                )
            permit = store.issue_cloud_dispatch_permit(
                executing.task_id,
                executing.epoch,
                approval_id=created["approval_id"],
                approval_binding_sha256=binding.sha256,
                stable_catalog_sha256=endpoint.stable_catalog_sha256,
                qualification_sha256=endpoint.qualification_sha256,
                reserved_microusd=reserved,
                key_remaining_microusd=5_000_000,
                utc_month="2026-08",
                now=NOW + timedelta(seconds=2),
            )
            store.consume_cloud_dispatch_permit(
                permit["permit_id"],
                executing.task_id,
                executing.epoch,
                binding.dispatch_body_sha256,
                now=NOW + timedelta(seconds=2),
            )
            self.assertEqual(store.reconcile_cloud_dispatch(permit["reservation_id"], actual_microusd=None, ambiguous=True), reserved)
            self.assertEqual(store._connection.execute("SELECT cloud_dispatch_count FROM tasks WHERE task_id=?", (executing.task_id,)).fetchone()[0], 1)


class _UnusedLocalModel:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request):
        self.calls += 1
        if False:
            yield request


class _Profiler:
    async def profile_task_with_correction(self, request, *, context_scope, output_contract, response_language):
        del request
        return TaskProfileV1(
            version="task-profile/1",
            task_kind=TaskKind.ANALYSIS,
            reasoning_depth=1,
            constraint_count=3,
            domains=("general", "software"),
            context_scope=context_scope,
            output_contract=output_contract,
            response_language=response_language,
        )


class _CloudFake:
    def __init__(self, store: OrchestratorStore, endpoint: QualifiedEndpointV1) -> None:
        self.store = store
        self.endpoint = endpoint
        self.dispatches = 0

    async def revalidate(self, endpoint):
        self.assert_endpoint(endpoint)
        health = EndpointHealthV1(
            model_canonical_id=endpoint.model_canonical_id,
            provider_endpoint_tag=endpoint.provider_endpoint_tag,
            status=0,
            uptime_last_30m_basis_points=10_000,
            p90_latency_ms=100,
            observed_at=utc_text(NOW),
            age_ms=0,
        )
        return OpenRouterRevalidationV1(endpoint, health, 5_000_000, 0, "monthly")

    def assert_endpoint(self, endpoint):
        if endpoint.qualification_sha256 != self.endpoint.qualification_sha256:
            raise AssertionError("wrong endpoint")

    async def dispatch(self, **kwargs):
        self.dispatches += 1
        self.store.consume_cloud_dispatch_permit(
            kwargs["permit_id"],
            kwargs["task_id"],
            kwargs["epoch"],
            kwargs["dispatch_body_sha256"],
            now=NOW,
        )
        self.store.reconcile_cloud_dispatch(kwargs["reservation_id"], actual_microusd=100, now=NOW)
        return CloudModelResultV1(
            text="Cloud answer.",
            generation_id="gen-qualified",
            input_tokens=100,
            output_tokens=3,
            reasoning_tokens=0,
            actual_microusd=100,
            receipt={"provider": "qualified"},
        )


class CloudLifecycleE2ETests(unittest.IsolatedAsyncioTestCase):
    async def test_medium_route_requires_f10_then_dispatches_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            live_now = datetime.now(UTC)
            store = OrchestratorStore(Path(directory) / "state.sqlite3")
            self.addCleanup(store.close)
            endpoint = QualifiedEndpointV1(
                model_canonical_id=DEEPSEEK_MODEL,
                provider_endpoint_tag="provider/full-endpoint",
                quantization="fp8",
                supported_parameters=("max_tokens", "reasoning", "temperature", "top_p"),
                qualification_version="qualification/1",
                qualification_sha256="d" * 64,
                context_limit=128_000,
                output_limit=2048,
                prompt_rate_per_m="0.09",
                completion_rate_per_m="0.18",
                request_rate="0",
                runtime_provider_name="Provider Inc",
            )
            store.register_endpoint_qualification(endpoint, enabled=True, now=NOW)
            local = _UnusedLocalModel()
            cloud = _CloudFake(store, endpoint)
            config = OrchestratorConfig(
                database_path=Path(directory) / "state.sqlite3",
                cloud_enabled=True,
                final_response_output_tokens=256,
            )
            service = OrchestratorService(
                config,
                store,
                local,
                task_profiler=_Profiler(),
                cloud_model=cloud,
                cloud_endpoints=(endpoint,),
            )
            session_id = str(uuid.uuid4())
            service.attach_client("connection", cloud_session_id=session_id)
            admitted = store.admit_transcript(
                event_id=str(uuid.uuid4()),
                turn_id=str(uuid.uuid4()),
                payload={"text": "Analyse these constraints", "language": {"code": "en"}},
                now=live_now,
            ).task
            assert admitted is not None
            waiting = await service.execute_task(admitted.task_id)
            self.assertEqual(waiting.state, TaskState.AWAITING_APPROVAL)
            run = store.cloud_run(admitted.task_id)
            assert run is not None
            presentation = store._connection.execute(
                "SELECT * FROM cloud_presentations WHERE approval_id=? AND state='PENDING'",
                (run["approval_id"],),
            ).fetchone()
            decision_event_id = str(uuid.uuid4())
            decided = store.decide_cloud_approval(
                decision_event_id=decision_event_id,
                semantic_sha256=DIGEST,
                approval_id=run["approval_id"],
                presentation_id=presentation["presentation_id"],
                requested_event_id=presentation["requested_event_id"],
                approval_binding_sha256=run["binding_sha256"],
                decision_nonce=presentation["decision_nonce"],
                decision="approve",
                session_id=session_id,
                capture_idle=True,
                now=live_now + timedelta(seconds=1),
            )
            self.assertEqual(decided["result"]["reason_code"], "approved")
            store.resume_cloud_decision(run["approval_id"], now=live_now + timedelta(seconds=1))
            completed = await service.execute_task(admitted.task_id)
            self.assertEqual(completed.state, TaskState.COMPLETED)
            self.assertEqual(cloud.dispatches, 1)
            self.assertEqual(local.calls, 0)


class OpenRouterBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_revalidation_is_exact_and_unguarded_post_is_rejected(self) -> None:
        try:
            import httpx
        except ImportError:
            self.skipTest("httpx environment required")
        with tempfile.TemporaryDirectory() as directory:
            store = OrchestratorStore(Path(directory) / "state.sqlite3")
            self.addCleanup(store.close)
            endpoint = QualifiedEndpointV1(
                model_canonical_id=DEEPSEEK_MODEL,
                provider_endpoint_tag="provider/full-endpoint",
                quantization="fp8",
                supported_parameters=("max_tokens", "reasoning", "temperature", "top_p"),
                qualification_version="qualification/1",
                qualification_sha256="e" * 64,
                context_limit=128_000,
                output_limit=2048,
                prompt_rate_per_m="0.09",
                completion_rate_per_m="0.18",
                request_rate="0",
                runtime_provider_name="Provider Inc",
            )
            store.register_endpoint_qualification(endpoint, enabled=True)

            async def control(request):
                if request.url.path == "/api/v1/endpoints/zdr":
                    return httpx.Response(
                        200,
                        json={
                            "data": [
                                {
                                    "context_length": 128000,
                                    "latency_last_30m": {"p90": 0.1},
                                    "max_completion_tokens": 2048,
                                    "model_id": DEEPSEEK_MODEL,
                                    "pricing": {"prompt": "0.00000009", "completion": "0.00000018", "request": "0"},
                                    "quantization": "fp8",
                                    "status": 0,
                                    "supported_parameters": ["top_p", "temperature", "reasoning", "max_tokens"],
                                    "tag": "provider/full-endpoint",
                                    "uptime_last_30m": 99.9,
                                }
                            ]
                        },
                        request=request,
                    )
                if request.url.path == "/api/v1/key":
                    return httpx.Response(
                        200,
                        json={"data": {"limit": 10, "limit_remaining": 9, "usage_monthly": 1, "limit_reset": "monthly"}},
                        request=request,
                    )
                raise AssertionError(request.url.path)

            adapter = OpenRouterCloudAdapter(
                store,
                bearer_token="private-key",
                transport=httpx.MockTransport(lambda request: httpx.Response(500, request=request)),
                control_transport=httpx.MockTransport(control),
            )
            self.addAsyncCleanup(adapter.aclose)
            result = await adapter.revalidate(endpoint)
            self.assertEqual(result.key_remaining_microusd, 9_000_000)
            with self.assertRaisesRegex(Exception, "cloud_dispatch_without_permit"):
                await adapter._dispatch_client.post("/chat/completions", json={})


if __name__ == "__main__":
    unittest.main()
