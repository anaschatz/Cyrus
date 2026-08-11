from __future__ import annotations

import asyncio
import tempfile
import unittest
import uuid
import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase

from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.control import ControlApplication
from jarvis_orchestrator.domain import (
    CanonicalEffectV1,
    ContinuationArtifactV1,
    ModelChunk,
    ModelRequest,
    ModelUsage,
    OrchestratorError,
    ReasonCode,
    TaskState,
    canonical_sha256,
    utc_now,
    utc_text,
)
from jarvis_orchestrator.model import BudgetedModelPort
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.store import OrchestratorStore
from jarvis_orchestrator.pydantic_adapter import build_local_pydantic_port
from jarvis_voice.protocol import EventFactory


class FakeModel:
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        yield ModelChunk("Γει")
        yield ModelChunk("ά σου. Δεύτερη πρόταση.", ModelUsage(output_tokens=8, input_tokens=4))


def config(path: Path) -> OrchestratorConfig:
    return OrchestratorConfig.from_mapping(
        {
            "database_path": str(path),
            "control_bearer_token": "control-private",
            "inference_bearer_token": "inference-private",
        }
    )


class ServiceAndControlTests(StoreCase, unittest.TestCase):
    def test_m1a_guarded_pydantic_admission_commit_and_reconnect(self) -> None:
        try:
            import httpx
        except ImportError:
            self.skipTest("pinned Pydantic AI integration environment required")
        captured: list[dict[str, object]] = []

        async def handler(request):
            captured.append(json.loads(request.content))
            chunks = [
                {"id": "chatcmpl-e2e", "object": "chat.completion.chunk", "created": 1, "model": "jarvis-local", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None, "finish_reason": None}], "usage": None},
                {"id": "chatcmpl-e2e", "object": "chat.completion.chunk", "created": 1, "model": "jarvis-local", "choices": [{"index": 0, "delta": {"content": "Έτοιμο."}, "logprobs": None, "finish_reason": None}], "usage": None},
                {"id": "chatcmpl-e2e", "object": "chat.completion.chunk", "created": 1, "model": "jarvis-local", "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}], "usage": None},
                {"id": "chatcmpl-e2e", "object": "chat.completion.chunk", "created": 1, "model": "jarvis-local", "choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}},
            ]
            body = "".join(f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

        store = self.make_store()
        port = build_local_pydantic_port(
            store,
            base_url="http://127.0.0.1:9/v1",
            bearer_token="private",
            timeout_seconds=2,
            http_transport=httpx.MockTransport(handler),
        )
        service = OrchestratorService(config(store.path), store, port)
        envelope = EventFactory().create(
            "input.transcript.final",
            {
                "text": "Απάντησε.",
                "language": {"code": "el", "score": 0.99},
                "asr_score": {"value": 0.8, "method": "exp_weighted_avg_logprob", "raw_avg_logprob": -0.2, "calibrated": False},
            },
            turn_id=str(uuid.uuid4()),
        )
        admission = service.admit_envelope(envelope)
        completed = asyncio.run(service.execute_task(admission.task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        first = store.claim_outbox(completed.conversation_id, "connection-1")
        for event in first:
            store.finish_outbox_attempt(event.event_id, "connection-1", complete=True)
        replay = store.claim_outbox(completed.conversation_id, "connection-2")
        self.assertEqual([item.event_id for item in first], [item.event_id for item in replay])
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["max_completion_tokens"], 256)

    def test_local_end_to_end_commits_text_and_early_speech(self) -> None:
        store = self.make_store()
        service = OrchestratorService(config(store.path), store, BudgetedModelPort(store, FakeModel()))
        task = self.admitted(
            store,
            event_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            now=utc_now(),
        )
        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        history = store.canonical_conversation(task.conversation_id)
        self.assertEqual(history.turns[0].assistant_text, "Γειά σου. Δεύτερη πρόταση.")
        events = store.claim_outbox(task.conversation_id, "connection")
        speech = [event for event in events if event.event_type == "response.speech.commit"]
        self.assertEqual([event.payload["unit_index"] for event in speech], [1, 2])
        self.assertEqual([event.payload["text"] for event in speech], ["Γειά σου.", "Δεύτερη πρόταση."])

    def test_control_mutations_replay_exact_result_and_reject_key_reuse(self) -> None:
        store = self.make_store()
        service = OrchestratorService(config(store.path), store, BudgetedModelPort(store, FakeModel()))
        app = ControlApplication(service.config, service)
        task = self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4()))
        key = str(uuid.uuid4())
        first = app.mutate(
            "POST",
            f"/v1/tasks/{task.task_id}/cancel",
            {"reason_code": "user_cancelled"},
            key,
            task.epoch,
        )
        second = app.mutate(
            "POST",
            f"/v1/tasks/{task.task_id}/cancel",
            {"reason_code": "user_cancelled"},
            key,
            task.epoch,
        )
        self.assertEqual(first, second)
        with self.assertRaisesRegex(OrchestratorError, "idempotency_conflict"):
            app.mutate("POST", "/v1/conversations", {}, key, 0)

    def test_control_distinguishes_unknown_path_from_wrong_method(self) -> None:
        store = self.make_store()
        service = OrchestratorService(config(store.path), store, BudgetedModelPort(store, FakeModel()))
        app = ControlApplication(service.config, service)
        with self.assertRaisesRegex(OrchestratorError, "method_not_allowed"):
            app.get("/v1/conversations")
        status, body = app.mutate("POST", "/readyz", {}, str(uuid.uuid4()), 0)
        self.assertEqual(
            (status, body),
            (405, {"error": {"code": "method_not_allowed", "retryable": False}}),
        )
        with self.assertRaisesRegex(OrchestratorError, "not_found"):
            app.get("/v1/does-not-exist")

    def test_busy_delete_then_logical_delete(self) -> None:
        store = self.make_store()
        service = OrchestratorService(config(store.path), store, BudgetedModelPort(store, FakeModel()))
        app = ControlApplication(service.config, service)
        task = self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4()))
        busy = app.mutate("DELETE", f"/v1/conversations/{task.conversation_id}", None, str(uuid.uuid4()), task.epoch)
        self.assertEqual(busy, (409, {"error": {"code": "conversation_busy", "retryable": True}}))
        cancellation = service.cancel_task(task.task_id, task.epoch)
        status, body = app.mutate(
            "DELETE",
            f"/v1/conversations/{task.conversation_id}",
            None,
            str(uuid.uuid4()),
            cancellation["epoch"],
        )
        self.assertEqual((status, body), (204, None))
        self.assertIsNone(store.active_conversation())


class ContinuationAndRecoveryTests(StoreCase, unittest.TestCase):
    def test_continuation_is_versioned_and_fails_closed_on_digest_change(self) -> None:
        store = self.make_store(tool_authority_enabled=True)
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        arguments = {"target": "lamp"}
        store.create_approval(task.task_id, task.epoch, kind="tool", name="switch", arguments=arguments, now=NOW)
        canonical = store.canonical_conversation(task.conversation_id)
        artifact = ContinuationArtifactV1(
            artifact_id=str(uuid.uuid4()),
            artifact_version=1,
            task_id=task.task_id,
            conversation_id=task.conversation_id,
            epoch=task.epoch,
            framework_version="pydantic-ai/2.27.0",
            canonical_revision=canonical.revision,
            canonical_sha256=canonical_sha256(asdict(canonical)),
            model_digest="model-v1",
            profile_digest="profile-v1",
            tool_schema_digest="tools-v1",
            pydantic_conversation_id=str(uuid.uuid4()),
            paused_run_id=str(uuid.uuid4()),
            messages_json="[]",
            deferred_requests_json="[]",
            arguments_sha256=canonical_sha256(arguments),
            principal_sha256="principal-v1",
            policy_version="jarvis-permissions/v1",
            usage_json="{}",
            reservations_json="[]",
            created_at=utc_text(NOW),
            expires_at=utc_text(NOW + timedelta(minutes=10)),
        )
        store.save_continuation(artifact)
        loaded = store.load_continuation(
            artifact.artifact_id,
            task_id=task.task_id,
            epoch=task.epoch,
            model_digest="model-v1",
            profile_digest="profile-v1",
            tool_schema_digest="tools-v1",
            principal_sha256="principal-v1",
            policy_version="jarvis-permissions/v1",
            arguments_sha256=canonical_sha256(arguments),
            now=NOW,
        )
        self.assertEqual(loaded, artifact)
        with self.assertRaisesRegex(OrchestratorError, "continuation_incompatible"):
            store.load_continuation(
                artifact.artifact_id,
                task_id=task.task_id,
                epoch=task.epoch,
                model_digest="changed",
                profile_digest="profile-v1",
                tool_schema_digest="tools-v1",
                principal_sha256="principal-v1",
                policy_version="jarvis-permissions/v1",
                arguments_sha256=canonical_sha256(arguments),
                now=NOW,
            )

    def test_capability_result_requires_second_epoch_cas(self) -> None:
        store = self.make_store(tool_authority_enabled=True)
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        reservation = store.reserve_capability(
            task.task_id, task.epoch, kind="tool", request_key="call-1", arguments={"x": 1}, now=NOW
        )
        store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED, now=NOW)
        with self.assertRaises(OrchestratorError):
            store.reconcile_capability(
                reservation,
                task.task_id,
                task.epoch,
                effect=CanonicalEffectV1("tool", "fake", "COMPLETED", canonical_sha256({"x": 1}), "result"),
                now=NOW,
            )

    def test_restart_preserves_suspended_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            first = OrchestratorStore(path)
            suspended = first.admit_transcript(event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4()), payload={"text": "next", "language": {"code": "en"}}, now=NOW).task
            suspended = self.executing(first, suspended)
            first.create_approval(suspended.task_id, suspended.epoch, kind="research", name="search", arguments={}, now=NOW)
            first.close()
            recovered = OrchestratorStore(path)
            self.addCleanup(recovered.close)
            self.assertEqual(recovered.recover_after_restart(now=NOW), 0)
            self.assertEqual(recovered.get_task(suspended.task_id).state, TaskState.AWAITING_APPROVAL)

    def test_restart_interrupts_active_model_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            first = OrchestratorStore(path)
            active = self.executing(first, self.admitted(first, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
            first.close()
            recovered = OrchestratorStore(path)
            self.addCleanup(recovered.close)
            self.assertEqual(recovered.recover_after_restart(now=NOW), 1)
            self.assertEqual(recovered.get_task(active.task_id).state, TaskState.INTERRUPTED)


if __name__ == "__main__":
    unittest.main()
