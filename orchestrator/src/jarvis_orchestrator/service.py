"""The only component that owns the complete CYRUS task lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import re
import threading
import uuid
from collections.abc import Callable, Mapping
from datetime import timedelta
from decimal import Decimal
from typing import Any

from .config import OrchestratorConfig
from .context_assembler import (
    CURRENT_TRANSCRIPT_MAX_BYTES,
    HISTORY_MAX_PAIRS,
    ContextAssemblerV1,
    ContextAssemblyError,
    ContextAssemblyV1,
)
from .domain import (
    BudgetExhaustedError,
    CanonicalEffectV1,
    ModelRequest,
    OrchestratorError,
    ReasonCode,
    StaleEpochError,
    TaskSnapshot,
    TaskState,
    RouteDecision,
    canonical_sha256,
    new_uuid4,
    nfc,
    parse_utc,
    utc_now,
    utc_text,
)
from .cloud_contract import (
    DEEPSEEK_MODEL,
    TERRA_INPUT_FLOOR_PER_M,
    TERRA_MODEL,
    TERRA_OUTPUT_FLOOR_PER_M,
    CloudApprovalBindingV1,
    CloudModelResultV1,
    QualifiedEndpointV1,
    build_dispatch_body,
    reserve_cost_microusd,
)
from .cloud_privacy import JarvisPrivacyClassifierV1
from .cloud_routing import (
    CandidateFactsV1,
    ContextScope,
    DeterministicModelRouterV1,
    OutputContract,
    RoutePlanV1,
    RouteTarget,
    RoutingFactsV1,
    TaskProfileV1,
    complexity_score,
    is_advanced_reasoning,
    local_eligible,
)
from .jcs import domain_sha256
from .llm_context import (
    assemble_final_local_context,
    build_system_prompt,
    validate_buffered_model_text,
    validate_stream_fragment,
)
from .inference import InferenceControlClient
from .memory import (
    DisabledShadowMemoryService,
    MemoryObservation,
    MemoryRecall,
    MemoryRecallRequest,
    MemoryProposal,
    ShadowMemoryServicePort,
    parse_explicit_memory_command,
)
from .memory_release import shadow_safety_fields
from .ports import (
    DisabledMemoryPort,
    DisabledResearchPort,
    MemoryPort,
    ModelPort,
    ResearchPlannerPort,
    ResearchPort,
    CloudModelPort,
    TaskProfilerPort,
    ToolExecutorPort,
    ToolPlannerPort,
)
from .research import (
    EvidenceBundleV1,
    FacetResultV1,
    FacetState,
    ResearchCorpusV1,
    ResearchRequestV1,
    ResearchResultV1,
    ResearchStatus,
    bundle_from_json,
    build_research_result,
    plan_from_json,
    render_bundle,
    speech_units,
)
from .speech import SpeechSegmenterV1, resolve_locale
from .store import OrchestratorStore
from .scheduler_store import SchedulerStore
from .scheduler_contract import (
    MAXIMUM_SNOOZE_GENERATIONS,
    MINIMUM_INTERVAL_MS,
    MINIMUM_LEAD_MS,
    PROPOSAL_INPUT_SCHEMA,
    ProposalOutcome,
)
from .schedule_admission import SCHEDULE_ADMISSION_CONTRACT, classify_schedule_request
from .research_security import reject_sensitive_text
from .runtime_context import (
    resolve_temporal_query,
    select_relevant_history,
)
from .tool_contracts import ToolOutcomeV1
from .tool_store import ToolStoreV1
from .tool_state import ToolRunState
from .translation import (
    DisabledTranslationPort,
    TranslationPort,
)
BETA_SESSION_SCHEMA_VERSION = "jarvis-orchestrator-beta-session/2"
EXPLICIT_MEMORY_COMMAND_MAX_BYTES = 8_192


def valid_voice_connection_tracker_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return False
    return parsed.version == 4 and str(parsed) == value


class OrchestratorService:
    def __init__(
        self,
        config: OrchestratorConfig,
        store: OrchestratorStore,
        model: ModelPort,
        inference_control: InferenceControlClient | None = None,
        memory: ShadowMemoryServicePort | None = None,
        active_memory: MemoryPort | None = None,
        research_planner: ResearchPlannerPort | None = None,
        research: ResearchPort | None = None,
        task_profiler: TaskProfilerPort | None = None,
        cloud_model: CloudModelPort | None = None,
        cloud_endpoints: tuple[QualifiedEndpointV1, ...] = (),
        cloud_release_verified: bool = False,
        scheduler: SchedulerStore | None = None,
        schedule_compiler: Any | None = None,
        tool_store: ToolStoreV1 | None = None,
        tool_planner: ToolPlannerPort | None = None,
        tool_executor: ToolExecutorPort | None = None,
        input_translator: TranslationPort | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.context_assembler = ContextAssemblerV1()
        self.scheduler = scheduler
        self.schedule_compiler = schedule_compiler
        if config.tools_enabled and (
            tool_store is None or tool_planner is None or tool_executor is None
        ):
            raise RuntimeError(
                "tools_enabled requires a qualified registry, store, local planner and executor"
            )
        self.tool_store = tool_store
        self.tool_planner = tool_planner
        self.tool_executor = tool_executor
        self.model = model
        if config.input_translation_mode != "disabled" and input_translator is None:
            raise RuntimeError(
                "input_translation_mode requires a qualified TranslationPort"
            )
        self.input_translator = input_translator or DisabledTranslationPort()
        self.inference_control = inference_control
        self.memory = memory or DisabledShadowMemoryService()
        if config.memory_enabled and active_memory is None:
            raise RuntimeError("memory_enabled requires a qualified MemoryPort")
        self.active_memory = active_memory or DisabledMemoryPort()
        if config.research_enabled and (research_planner is None or research is None):
            raise RuntimeError("research_enabled requires qualified planner and ResearchPort")
        self.research_planner = research_planner
        self.research = research or DisabledResearchPort()
        if config.cloud_enabled and (
            not cloud_release_verified
            or task_profiler is None
            or cloud_model is None
            or not cloud_endpoints
        ):
            raise RuntimeError(
                "cloud_enabled requires verified release evidence, a qualified profiler, CloudModelPort and endpoint registry"
            )
        self.task_profiler = task_profiler
        self.cloud_model = cloud_model
        self.cloud_endpoints = tuple(cloud_endpoints)
        self.privacy_classifier = JarvisPrivacyClassifierV1()
        self._connection_lock = threading.RLock()
        self._connected_sessions: set[str] = set()
        self._voice_connection_tracker_id = new_uuid4()
        self._voice_connection_generation = 0
        self._cloud_sessions: dict[str, str] = {}
        self._scheduler_sessions: dict[str, str] = {}
        self._scheduler_action_sessions: dict[str, str] = {}
        self._execution_scheduler: Callable[[str], None] | None = None

    def set_execution_scheduler(self, scheduler: Callable[[str], None] | None) -> None:
        self._execution_scheduler = scheduler

    def _schedule(self, task_id: str) -> None:
        if self._execution_scheduler is not None:
            self._execution_scheduler(task_id)

    def _response_language_code(self, input_language: str) -> str:
        if self.config.response_language == "en":
            return "en"
        return input_language if input_language in {"el", "en"} else "default"

    def _response_locale(self, input_language: str) -> str:
        if self.config.response_language == "en":
            return "en-US"
        return resolve_locale(input_language, self.config.default_speech_locale)

    def _scheduler_response_locale(self, input_language: str) -> str:
        if self.config.response_language == "en":
            return "en-US"
        return "en-US" if input_language == "en" else self.config.scheduler_default_locale

    def _validate_response_language(self, text: str) -> None:
        if self.config.response_language == "en" and re.search(
            r"[\u0370-\u03ff\u1f00-\u1fff]",
            text,
        ):
            raise OrchestratorError("response_language_mismatch", status=502)

    def startup_recovery(self) -> int:
        recovered = self.store.recover_after_restart()
        if self.tool_store is not None:
            recovered += self.tool_store.recover_uncertain()
        return recovered

    def ready(self) -> bool:
        return self.readiness_snapshot()["core_ready"]

    def readiness_snapshot(self) -> dict[str, bool | str]:
        """Capture storage, memory, and inference readiness once per request."""

        try:
            self.store.active_conversation()
            storage_ready = True
        except Exception:  # noqa: BLE001 - readiness is fail-closed.
            storage_ready = False
        memory_retrospective_ready = self.memory_retrospective_ready()
        memory_ready = memory_retrospective_ready
        memory_requirement_satisfied = (
            memory_ready if self.config.memory_enabled else True
        )
        try:
            inference_ready = (
                True
                if self.inference_control is None
                else self.inference_control.ready()
            )
        except Exception:  # noqa: BLE001 - readiness is fail-closed.
            inference_ready = False
        return {
            "core_ready": (
                storage_ready
                and memory_requirement_satisfied
                and inference_ready
            ),
            "inference_ready": inference_ready,
            "memory_ready": memory_ready,
            "memory_requirement_satisfied": memory_requirement_satisfied,
            "memory_retrospective_ready": memory_retrospective_ready,
            "memory_state": (
                "READY"
                if memory_ready
                else "BLOCKED"
                if self.config.memory_enabled
                else "DISABLED"
            ),
            "storage_ready": storage_ready,
        }

    def memory_storage_status(self) -> dict[str, Any]:
        """Expose only the store's cached, content-free memory status."""

        return self.store.memory_storage_status()

    def memory_retrospective_ready(self) -> bool:
        if not self.config.memory_enabled:
            return False
        try:
            status = self.memory_storage_status()
            return (
                status.get("storage_state") == "ENCRYPTED"
                and status.get("ready") is True
                and status.get("pending_operation_count") == 0
            )
        except Exception:  # noqa: BLE001 - readiness is fail-closed.
            return False

    def has_client(self) -> bool:
        with self._connection_lock:
            return bool(self._connected_sessions)

    def voice_connection_state(self) -> tuple[bool, str, int]:
        """Return content-free state owned by this service instance."""

        with self._connection_lock:
            return (
                bool(self._connected_sessions),
                self._voice_connection_tracker_id,
                self._voice_connection_generation,
            )

    def attach_client(
        self,
        connection_id: str,
        *,
        cloud_session_id: str | None = None,
        scheduler_session_id: str | None = None,
        scheduler_action_session_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self._connection_lock:
            is_new_connection = connection_id not in self._connected_sessions
            self._connected_sessions.add(connection_id)
            if is_new_connection:
                self._voice_connection_generation += 1
            if cloud_session_id is not None:
                self._cloud_sessions[connection_id] = cloud_session_id
            if scheduler_session_id is not None:
                self._scheduler_sessions[connection_id] = scheduler_session_id
            if scheduler_action_session_id is not None:
                self._scheduler_action_sessions[connection_id] = scheduler_action_session_id
        conversation = self.store.active_conversation()
        if conversation and conversation["active_task_id"]:
            try:
                task = self.store.set_reconnected(conversation["active_task_id"], conversation["epoch"])
                if conversation["active_task_state"] == TaskState.DECISION_WAITING_RECONNECT.value:
                    self._schedule(task.task_id)
            except OrchestratorError:
                pass
        return conversation

    def detach_client(self, connection_id: str) -> None:
        with self._connection_lock:
            was_connected = connection_id in self._connected_sessions
            self._connected_sessions.discard(connection_id)
            if was_connected:
                self._voice_connection_generation += 1
            self._cloud_sessions.pop(connection_id, None)
            self._scheduler_sessions.pop(connection_id, None)
            self._scheduler_action_sessions.pop(connection_id, None)
            if self._connected_sessions:
                return
        conversation = self.store.active_conversation()
        if conversation and conversation["active_task_id"]:
            try:
                self.store.set_disconnected(conversation["active_task_id"], conversation["epoch"])
            except OrchestratorError:
                pass

    def has_cloud_client(self) -> bool:
        with self._connection_lock:
            return bool(self._cloud_sessions)

    def cloud_approval_available(self) -> bool:
        """Return whether the server can actually service the cloud extension."""

        return bool(
            self.config.cloud_enabled
            and self.task_profiler is not None
            and self.cloud_model is not None
            and self.cloud_endpoints
        )

    def cloud_owner_session_id(self) -> str | None:
        with self._connection_lock:
            if not self._cloud_sessions:
                return None
            return self._cloud_sessions[next(reversed(self._cloud_sessions))]

    def has_scheduler_client(self) -> bool:
        with self._connection_lock:
            return bool(self._scheduler_sessions)

    def has_scheduler_action_client(self) -> bool:
        with self._connection_lock:
            return bool(self._scheduler_action_sessions)

    def _validate_command_or_model_transcript(
        self, value: Any
    ) -> tuple[str, str] | None:
        """Apply one closed outer bound before command-aware validation."""

        if not isinstance(value, str) or not value:
            raise ContextAssemblyError("invalid_request", status=400)
        # A Python code point occupies at least one UTF-8 byte. This first test
        # prevents an unbounded temporary allocation for obviously huge input.
        if len(value) > EXPLICIT_MEMORY_COMMAND_MAX_BYTES:
            raise ContextAssemblyError("request_too_large", status=413)
        try:
            size = len(value.encode("utf-8", errors="strict"))
        except UnicodeEncodeError:
            raise ContextAssemblyError("invalid_request", status=400) from None
        if size > EXPLICIT_MEMORY_COMMAND_MAX_BYTES:
            raise ContextAssemblyError("request_too_large", status=413)

        # Explicit commands must be recognized before the smaller model-bound
        # transcript cap is applied. The shared parser is the only command gate.
        command = parse_explicit_memory_command(value)
        self.context_assembler.validate_transcript(
            value,
            max_bytes=(
                EXPLICIT_MEMORY_COMMAND_MAX_BYTES
                if command is not None
                else CURRENT_TRANSCRIPT_MAX_BYTES
            ),
        )
        return command

    def admit_envelope(self, envelope: Any) -> Any:
        if (
            envelope.type != "input.transcript.final"
            or envelope.turn_id is None
            or envelope.response_id is not None
        ):
            raise OrchestratorError("invalid_request", status=400)
        if not isinstance(envelope.payload, Mapping):
            raise OrchestratorError("invalid_request", status=400)
        self._validate_command_or_model_transcript(envelope.payload.get("text"))
        return self.store.admit_transcript(
            event_id=envelope.event_id,
            turn_id=envelope.turn_id,
            payload=envelope.payload,
            absolute_lifetime=timedelta(seconds=self.config.absolute_lifetime_seconds),
        )

    async def _final_local_context(
        self,
        task: TaskSnapshot,
        current_transcript: str,
        recall: MemoryRecall,
        tool_projection: Mapping[str, Any] | None,
    ) -> ContextAssemblyV1:
        return await assemble_final_local_context(
            config=self.config,
            store=self.store,
            input_translator=self.input_translator,
            context_assembler=self.context_assembler,
            task=task,
            current_transcript=current_transcript,
            recall=recall,
            tool_projection=tool_projection,
        )

    async def _recall_memory(
        self, task: TaskSnapshot, input_record: Mapping[str, str]
    ) -> MemoryRecall:
        if not self.config.memory_enabled:
            return MemoryRecall()
        query = input_record["text"]
        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        started = self.store.begin_memory_recall(
            task.task_id, task.epoch, query_sha256=query_sha256
        )
        if not started:
            return MemoryRecall()
        try:
            request = MemoryRecallRequest(
                task_id=task.task_id,
                conversation_id=task.conversation_id,
                epoch=task.epoch,
                query=query,
                limit=self.config.memory_recall_limit,
            )
            async with asyncio.timeout(self.config.memory_timeout_seconds):
                recall = await self.active_memory.search(request)
            if not isinstance(recall, MemoryRecall):
                raise TypeError("memory provider returned an invalid recall")
            result_document = [
                {"text": item.text, "score": item.score} for item in recall.items
            ]
            self.store.finish_memory_recall(
                task.task_id,
                task.epoch,
                status="completed" if recall.items else "empty",
                query_sha256=query_sha256,
                result_sha256=canonical_sha256(result_document),
                result_count=len(recall.items),
            )
            return recall
        except StaleEpochError:
            raise
        except TimeoutError:
            self.store.finish_memory_recall(
                task.task_id,
                task.epoch,
                status="failed",
                query_sha256=query_sha256,
                error_code="timeout",
            )
            return MemoryRecall()
        except Exception:
            self.store.finish_memory_recall(
                task.task_id,
                task.epoch,
                status="failed",
                query_sha256=query_sha256,
                error_code="service_error",
            )
            return MemoryRecall()

    async def _remember_completed_turn(
        self, task: TaskSnapshot, input_record: Mapping[str, str]
    ) -> None:
        if not self.config.memory_enabled:
            return
        observation = MemoryObservation(
            task_id=task.task_id,
            conversation_id=task.conversation_id,
            turn_id=task.turn_id,
            source_event_id=input_record["event_id"],
            text=input_record["text"],
            language=input_record["language"],
            epoch=task.epoch,
        )
        try:
            async with asyncio.timeout(self.config.memory_timeout_seconds):
                deletion = await self.active_memory.forget(observation)
                if not deletion.requested:
                    await self.active_memory.remember(observation)
        except Exception:
            # The canonical response is already committed. Memory is optional
            # and never changes or invalidates that completed result.
            return

    async def _observe_memory_shadow(
        self, task: TaskSnapshot, input_record: Mapping[str, str]
    ) -> None:
        if not self.config.memory_shadow_enabled:
            return
        text = input_record["text"]
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            started = self.store.begin_memory_shadow_audit(
                task.task_id,
                task.epoch,
                source_event_id=input_record["event_id"],
                text_sha256=text_sha256,
            )
        except Exception:
            return
        if not started:
            return
        try:
            authority_before = self.store.shadow_authority_fingerprint(task.task_id)
        except Exception:
            return
        try:
            observation = MemoryObservation(
                task_id=task.task_id,
                conversation_id=task.conversation_id,
                turn_id=task.turn_id,
                source_event_id=input_record["event_id"],
                text=text,
                language=input_record["language"],
                epoch=task.epoch,
            )
            async with asyncio.timeout(self.config.memory_shadow_timeout_seconds):
                proposal = await self.memory.observe(observation)
            if not isinstance(proposal, MemoryProposal):
                raise TypeError("memory service returned an invalid proposal")
            proposal_payload = {
                "operation": proposal.operation.value,
                "memory_key": proposal.memory_key,
                "supersedes_source_event_ids": list(
                    proposal.supersedes_source_event_ids
                ),
                "summary": proposal.summary,
                "writer_contract_sha256": proposal.writer_contract_sha256,
            }
            proposal_sha256 = canonical_sha256(proposal_payload)
            authority_after = self.store.shadow_authority_fingerprint(task.task_id)
            payload = {
                **shadow_safety_fields(),
                "status": "proposed",
                "source_event_id": observation.source_event_id,
                "text_sha256": text_sha256,
                "proposal": proposal_payload,
                "proposal_sha256": proposal_sha256,
                "authority_state_before_sha256": authority_before,
                "authority_state_after_sha256": authority_after,
                "provenance": {
                    "current_source_event_id": observation.source_event_id,
                    "source_text_sha256": text_sha256,
                    "superseded_source_event_ids": list(
                        proposal.supersedes_source_event_ids
                    ),
                    "writer_contract_sha256": proposal.writer_contract_sha256,
                    "proposal_sha256": proposal_sha256,
                },
            }
            self.store.finish_memory_shadow_audit(
                task.task_id,
                task.epoch,
                event_type="memory.shadow.proposed",
                payload=payload,
            )
        except TimeoutError:
            try:
                authority_after = self.store.shadow_authority_fingerprint(task.task_id)
                self.store.finish_memory_shadow_audit(
                    task.task_id,
                    task.epoch,
                    event_type="memory.shadow.failed",
                    payload={
                        **shadow_safety_fields(),
                        "status": "failed",
                        "source_event_id": input_record["event_id"],
                        "text_sha256": text_sha256,
                        "error_code": "timeout",
                        "authority_state_before_sha256": authority_before,
                        "authority_state_after_sha256": authority_after,
                    },
                )
            except Exception:
                pass
        except Exception:
            try:
                authority_after = self.store.shadow_authority_fingerprint(task.task_id)
                self.store.finish_memory_shadow_audit(
                    task.task_id,
                    task.epoch,
                    event_type="memory.shadow.failed",
                    payload={
                        **shadow_safety_fields(),
                        "status": "failed",
                        "source_event_id": input_record["event_id"],
                        "text_sha256": text_sha256,
                        "error_code": "service_error",
                        "authority_state_before_sha256": authority_before,
                        "authority_state_after_sha256": authority_after,
                    },
                )
            except Exception:
                pass

    def _publish_deterministic_response(
        self,
        task: TaskSnapshot,
        *,
        text: str,
        speech: tuple[tuple[str, str], ...],
        default_locale: str,
    ) -> TaskSnapshot:
        self.store.start_response(task.task_id, task.epoch, default_locale)
        self.store.append_response_event(
            task.task_id,
            task.epoch,
            "response.text.delta",
            {"chunk_index": 0, "text": nfc(text)},
        )
        trailing = [
            (index, locale, nfc(unit))
            for index, (locale, unit) in enumerate(speech, start=1)
            if unit
        ]
        return self.store.complete_response(
            task.task_id,
            task.epoch,
            nfc(text),
            chunk_count=1,
            trailing_speech=trailing,
            last_unit_index=len(trailing),
        )

    async def _plan_tool_capability(
        self, task: TaskSnapshot, input_record: Mapping[str, str]
    ) -> TaskSnapshot | None:
        """Select and freeze one local tool call, then enter approval."""

        if (
            not self.config.tools_enabled
            or self.tool_store is None
            or self.tool_planner is None
        ):
            return None
        remaining = min(
            task.remaining_active_seconds(limit=self.config.active_budget_seconds),
            task.remaining_absolute_seconds(),
            self.config.model_timeout_seconds,
        )
        budget = self.store.remaining_model_budget(task.task_id, task.epoch)
        if remaining <= 0 or budget["requests"] < 1 or budget["output_tokens"] < 1:
            raise BudgetExhaustedError()
        route_request = ModelRequest(
            task_id=task.task_id,
            epoch=task.epoch,
            purpose="route",
            system_prompt=(
                "Choose the exact route for the current request. Select tool_local only "
                "when one registered tool is necessary; otherwise select local. /no_think"
            ),
            messages=({"role": "user", "content": input_record["text"] + "\n/no_think"},),
            output_allowance=min(256, budget["output_tokens"]),
            timeout_seconds=remaining,
            stream=False,
        )
        try:
            decision = await self.tool_planner.plan_with_correction(route_request)
        except (TimeoutError, asyncio.TimeoutError, OrchestratorError, ValueError):
            return None
        if not isinstance(decision, RouteDecision) or decision.route != "tool_local":
            return None
        definition = self.tool_store.registry.resolve(decision.tool_name)
        self.store.persist_planner_route(
            task.task_id,
            task.epoch,
            {
                "memory_query": decision.memory_query,
                "research_query": decision.research_query,
                "response_language": decision.response_language,
                "route": decision.route,
                "tool_name": decision.tool_name,
            },
        )
        task = self.store.transition(
            task.task_id,
            task.epoch,
            TaskState.ROUTING,
            TaskState.LOADING_CONTEXT,
            payload={"route": "tool_local", "tool_name": decision.tool_name},
        )
        task = self.store.transition(
            task.task_id,
            task.epoch,
            TaskState.LOADING_CONTEXT,
            TaskState.EXECUTING,
        )
        remaining_budget = self.store.remaining_model_budget(task.task_id, task.epoch)
        argument_request = ModelRequest(
            task_id=task.task_id,
            epoch=task.epoch,
            purpose="tool_arguments",
            system_prompt=(
                "Extract only the exact arguments for the selected local tool. "
                "Do not invent identifiers. /no_think"
            ),
            messages=({"role": "user", "content": input_record["text"] + "\n/no_think"},),
            output_allowance=min(512, remaining_budget["output_tokens"]),
            timeout_seconds=min(
                remaining,
                task.remaining_active_seconds(limit=self.config.active_budget_seconds),
                task.remaining_absolute_seconds(),
            ),
            stream=False,
        )
        arguments = await self.tool_planner.plan_tool_arguments(
            argument_request, definition=definition
        )
        canonical_arguments = self.tool_store.registry.validate(
            decision.tool_name, arguments
        )
        tool_call_id = self._deterministic_uuid(
            "tool-call", task.task_id, str(task.epoch), decision.tool_name
        )
        idempotency_key = self._deterministic_uuid(
            "tool-idempotency", task.task_id, str(task.epoch), decision.tool_name
        )
        self.store.reserve_capability(
            task.task_id,
            task.epoch,
            kind="tool",
            request_key=tool_call_id,
            arguments=canonical_arguments,
        )
        self.tool_store.create_tool_run(
            task_id=task.task_id,
            epoch=task.epoch,
            tool_call_id=tool_call_id,
            tool_id=decision.tool_name,
            arguments=canonical_arguments,
            preview={"arguments": canonical_arguments, "tool_id": decision.tool_name},
            idempotency_key=idempotency_key,
            policy_sha256=self.config.tool_policy_sha256,
            scheduler_contract_sha256=self.config.scheduler_contract_sha256,
            user_intent_binding_sha256=domain_sha256(
                "jarvis.tool-user-intent/1",
                {
                    "arguments": canonical_arguments,
                    "task_id": task.task_id,
                    "text": input_record["text"],
                    "tool_id": decision.tool_name,
                },
            ),
        )
        return self.store.get_task(task.task_id)

    @staticmethod
    def _deterministic_uuid(namespace: str, *parts: str) -> str:
        digest = bytearray(
            hashlib.sha256(
                ("jarvis:" + namespace + ":" + ":".join(parts)).encode("utf-8")
            ).digest()[:16]
        )
        digest[6] = (digest[6] & 0x0F) | 0x40
        digest[8] = (digest[8] & 0x3F) | 0x80
        return str(uuid.UUID(bytes=bytes(digest)))

    async def _execute_tool_resume(
        self, task: TaskSnapshot, input_record: Mapping[str, str]
    ) -> tuple[TaskSnapshot | None, Mapping[str, Any] | None]:
        if self.tool_store is None:
            return None, None
        run = self.tool_store.task_run(task.task_id, task.epoch)
        if run is None:
            return None, None
        reservation = self.store.capability_reservation(
            task.task_id,
            task.epoch,
            kind="tool",
            request_key=run["tool_call_id"],
        )
        if reservation is None:
            raise OrchestratorError("tool.binding_mismatch", status=409)
        if run["status"] == ToolRunState.DENIED.value:
            if reservation["status"] == "RESERVED":
                self.store.reconcile_capability(
                    reservation["reservation_id"],
                    task.task_id,
                    task.epoch,
                    effect=CanonicalEffectV1(
                        kind="ToolDenied",
                        name=run["tool_id"],
                        status="DENIED",
                        arguments_sha256=run["arguments_sha256"],
                    ),
                )
            locale = self._response_locale(input_record["language"])
            text = (
                "Η ενέργεια απορρίφθηκε και δεν εκτελέστηκε."
                if locale == "el-GR"
                else "The action was denied and was not executed."
            )
            completed = self._publish_deterministic_response(
                task,
                text=text,
                speech=((locale, text),),
                default_locale=locale,
            )
            await self._remember_completed_turn(completed, input_record)
            await self._observe_memory_shadow(completed, input_record)
            return completed, None
        terminal_success = {
            ToolRunState.VERIFIED.value,
            ToolRunState.ACCEPTED.value,
        }
        if run["status"] in terminal_success:
            if self.tool_executor is not None:
                try:
                    await self.tool_executor.acknowledge(
                        run["execution_id"], run["idempotency_key"]
                    )
                except Exception:
                    # A durable result is authoritative. Host ACK recovery is
                    # best-effort and must never cause a second native action.
                    pass
            return None, self.tool_store.result_projection(run["execution_id"])
        if run["status"] != ToolRunState.APPROVED.value or self.tool_executor is None:
            raise OrchestratorError("tool.invalid_state", status=409)
        permit = self.tool_store.issue_execution_permit(
            run["execution_id"],
            task_id=task.task_id,
            epoch=task.epoch,
            policy_sha256=self.config.tool_policy_sha256,
            scheduler_contract_sha256=self.config.scheduler_contract_sha256,
        )
        deadline_seconds = min(
            float(permit["definition"].timeout_seconds),
            task.remaining_active_seconds(limit=self.config.active_budget_seconds),
            task.remaining_absolute_seconds(),
        )
        if deadline_seconds <= 0:
            raise BudgetExhaustedError()
        try:
            async with asyncio.timeout(deadline_seconds):
                outcome, output = await self.tool_executor.execute_typed(
                    execution_id=run["execution_id"],
                    permit_id=permit["permit_id"],
                    tool_call_id=permit["tool_call_id"],
                    definition=permit["definition"],
                    arguments=permit["arguments"],
                    arguments_sha256=permit["arguments_sha256"],
                    idempotency_key=permit["idempotency_key"],
                    registry_sha256=permit["registry_sha256"],
                    policy_sha256=permit["policy_sha256"],
                    scheduler_contract_sha256=permit["scheduler_contract_sha256"],
                    user_intent_binding_sha256=permit[
                        "user_intent_binding_sha256"
                    ],
                    deadline_utc=utc_text(utc_now() + timedelta(seconds=deadline_seconds)),
                )
            if not isinstance(outcome, ToolOutcomeV1):
                raise OrchestratorError("tool.output_invalid", status=502)
            projection = self.tool_store.complete_execution(
                run["execution_id"],
                task_id=task.task_id,
                epoch=task.epoch,
                outcome=outcome,
                result=output,
            )
            try:
                await self.tool_executor.acknowledge(
                    run["execution_id"], run["idempotency_key"]
                )
            except Exception:
                # The result is already durable. The host retains the same
                # idempotent result for a later query/ACK; never repeat the OS call.
                pass
        except StaleEpochError:
            raise
        except BaseException:
            self.tool_store.mark_dispatch_ambiguous(run["execution_id"])
            raise
        if reservation["status"] == "RESERVED":
            self.store.reconcile_capability(
                reservation["reservation_id"],
                task.task_id,
                task.epoch,
                effect=CanonicalEffectV1(
                    kind="tool",
                    name=run["tool_id"],
                    status=outcome.status.value,
                    arguments_sha256=run["arguments_sha256"],
                    result_sha256=self.tool_store.task_run(task.task_id, task.epoch)[
                        "result_sha256"
                    ],
                ),
            )
        return None, projection

    async def _execute_schedule_admission(
        self, task: TaskSnapshot, input_record: Mapping[str, str]
    ) -> TaskSnapshot:
        """Compile one proposal and suspend the owning task for F10/F11."""

        task = self.store.transition(
            task.task_id,
            task.epoch,
            TaskState.ROUTING,
            TaskState.LOADING_CONTEXT,
            payload={"route": "scheduler_proposal"},
        )
        task = self.store.transition(
            task.task_id, task.epoch, TaskState.LOADING_CONTEXT, TaskState.EXECUTING
        )
        locale = self._scheduler_response_locale(input_record["language"])
        result_kind = "UNAVAILABLE"
        clarification: str | None = None
        if self.scheduler is not None and self.schedule_compiler is not None:
            model_input = {
                "schema_version": PROPOSAL_INPUT_SCHEMA,
                "request_id": task.task_id,
                "request_text": input_record["text"],
                "request_received_at_utc": task.admitted_at,
                "default_timezone": self.config.scheduler_default_timezone,
                "default_locale": locale,
                "selected_target": {
                    "job_id": None,
                    "job_revision": None,
                    "occurrence_id": None,
                    "snooze_generation": None,
                },
                "policy": {
                    "minimum_lead_ms": MINIMUM_LEAD_MS,
                    "minimum_interval_ms": MINIMUM_INTERVAL_MS,
                    "maximum_snooze_generations": MAXIMUM_SNOOZE_GENERATIONS,
                },
            }
            semantic = domain_sha256(
                SCHEDULE_ADMISSION_CONTRACT,
                {
                    "request_received_at_utc": task.admitted_at,
                    "request_text": input_record["text"],
                    "timezone": self.config.scheduler_default_timezone,
                    "locale": locale,
                },
            )
            remaining = min(
                task.remaining_active_seconds(limit=self.config.active_budget_seconds),
                task.remaining_absolute_seconds(),
                self.config.model_timeout_seconds,
            )
            if remaining <= 0:
                raise BudgetExhaustedError()
            try:
                async with asyncio.timeout(remaining):
                    compiled = await asyncio.to_thread(
                        self.schedule_compiler.compile, model_input
                    )
                result_kind = str(
                    compiled.proposal.get(
                        "outcome",
                        "PROPOSAL" if compiled.first_due_at_utc is not None else "PROPOSAL_INVALID",
                    )
                )
                clarification = compiled.proposal.get("clarification_question")
                if compiled.first_due_at_utc is not None:
                    self.scheduler.create_proposal(
                        compiled,
                        task_id=task.task_id,
                        epoch=task.epoch,
                        api_event_id=task.task_id,
                        api_semantic_sha256=semantic,
                    )
                    return self.store.get_task(task.task_id)
            except TimeoutError:
                raise
            except Exception:
                # Explicit reminder intent must never fall through to a general
                # model that could falsely claim the side effect occurred.
                result_kind = "UNAVAILABLE"

        greek = locale == "el-GR"
        if result_kind == ProposalOutcome.NEEDS_CLARIFICATION.value:
            text = clarification or (
                "Χρειάζομαι ακριβέστερη ημερομηνία ή ώρα πριν δημιουργήσω την υπενθύμιση."
                if greek
                else "I need a more precise date or time before creating the reminder."
            )
        elif result_kind == ProposalOutcome.UNSUPPORTED.value:
            text = (
                "Αυτό το είδος προγραμματισμού δεν υποστηρίζεται ακόμη."
                if greek
                else "That kind of schedule is not supported yet."
            )
        else:
            text = (
                "Ο ασφαλής προγραμματισμός υπενθυμίσεων δεν είναι διαθέσιμος αυτή τη στιγμή."
                if greek
                else "Safe reminder scheduling is currently unavailable."
            )
        return self._publish_deterministic_response(
            task,
            text=text,
            speech=((locale, text),),
            default_locale=locale,
        )

    async def _execute_scheduler_resume(
        self, task: TaskSnapshot, input_record: Mapping[str, str]
    ) -> TaskSnapshot | None:
        if self.scheduler is None:
            return None
        run = self.scheduler.task_run(task.task_id, task.epoch)
        if run is None:
            return None
        locale = self._scheduler_response_locale(input_record["language"])
        greek = locale == "el-GR"
        if run["state"] == "PENDING":
            raise OrchestratorError("invalid_state", status=409)
        if run["state"] == "APPROVED":
            operation = run["operation"]
            if operation == "CREATE_JOB":
                text = "Η υπενθύμιση δημιουργήθηκε." if greek else "The reminder was created."
            elif operation == "CANCEL_JOB":
                text = "Η υπενθύμιση ακυρώθηκε." if greek else "The reminder was cancelled."
            else:
                text = "Η υπενθύμιση αναβλήθηκε." if greek else "The reminder was snoozed."
        else:
            text = (
                "Δεν έγινε καμία αλλαγή στην υπενθύμιση."
                if greek
                else "No reminder change was made."
            )
        completed = self._publish_deterministic_response(
            task,
            text=text,
            speech=((locale, text),),
            default_locale=locale,
        )
        await self._remember_completed_turn(completed, input_record)
        await self._observe_memory_shadow(completed, input_record)
        return completed

    async def _execute_research(self, task: TaskSnapshot, input_record: Mapping[str, str]) -> TaskSnapshot:
        run = self.store.research_run(task.task_id)
        if run is None:
            raise OrchestratorError("research_run_missing", status=409)
        plan = plan_from_json(run["plan_json"])
        default_locale = "el-GR" if plan.language == "el" else "en-US"
        if run["approval_status"] == "DENIED":
            return self._publish_deterministic_response(
                task,
                text=(
                    "Η έρευνα δεν εκτελέστηκε, επειδή δεν εγκρίθηκε η αποστολή του ερωτήματος "
                    "σε δημόσιες υπηρεσίες αναζήτησης."
                    if plan.language == "el"
                    else "Research was not run because sending the query to public search services was not approved."
                ),
                speech=((default_locale, "Η έρευνα δεν εκτελέστηκε." if plan.language == "el" else "Research was not run."),),
                default_locale=default_locale,
            )
        if run["approval_status"] != "APPROVED":
            raise OrchestratorError("research_not_approved", status=409)
        if run["stage"] == "COMPLETED":
            if not run["bundle_json"]:
                raise OrchestratorError("research_bundle_missing", status=409)
            bundle = bundle_from_json(run["bundle_json"])
        else:
            reservation_id = self.store.begin_research_run(
                task.task_id,
                task.epoch,
                request_key=plan.plan_hash,
            )
            request = ResearchRequestV1(
                task_id=task.task_id,
                conversation_id=task.conversation_id,
                epoch=task.epoch,
                plan=plan,
            )
            remaining = min(
                task.remaining_active_seconds(limit=self.config.active_budget_seconds),
                task.remaining_absolute_seconds(),
                self.config.research_timeout_seconds,
            )
            if remaining <= 0:
                raise BudgetExhaustedError()
            try:
                async with asyncio.timeout(remaining):
                    corpus = await self.research.research(
                        request, idempotency_key=reservation_id
                    )
                    if not isinstance(corpus, ResearchCorpusV1):
                        raise TypeError("research provider returned an invalid corpus")
                    if corpus.plan_hash != plan.plan_hash:
                        raise ValueError("research corpus plan mismatch")
                    if any(
                        source.authority_registry_version
                        != self.config.research_authority_registry_version
                        or source.language_detector_version
                        != self.config.research_language_detector_version
                        for source in corpus.sources
                    ):
                        raise ValueError("research qualification identity mismatch")
                    selector_remaining = min(
                        task.remaining_active_seconds(limit=self.config.active_budget_seconds),
                        task.remaining_absolute_seconds(),
                        remaining,
                    )
                    if selector_remaining <= 0:
                        raise BudgetExhaustedError()
                    assert self.research_planner is not None
                    selections = await self.research_planner.select_research_spans(
                        task_id=task.task_id,
                        epoch=task.epoch,
                        plan=plan,
                        passages=corpus.passages,
                        documents=corpus.documents,
                        output_allowance=768,
                        timeout_seconds=selector_remaining,
                    )
                    result = build_research_result(plan, corpus, selections)
            except StaleEpochError:
                raise
            except BudgetExhaustedError:
                raise
            except (TimeoutError, asyncio.TimeoutError):
                result = ResearchResultV1(
                    EvidenceBundleV1(
                        plan.plan_hash,
                        ResearchStatus.FAILED,
                        tuple(
                            FacetResultV1(facet.facet_id, FacetState.TECHNICAL_FAILURE, (), "deadline")
                            for facet in plan.facets
                        ),
                        (),
                        (),
                    ),
                    {},
                )
            except Exception:
                result = ResearchResultV1(
                    EvidenceBundleV1(
                        plan.plan_hash,
                        ResearchStatus.FAILED,
                        tuple(
                            FacetResultV1(facet.facet_id, FacetState.TECHNICAL_FAILURE, (), "service_error")
                            for facet in plan.facets
                        ),
                        (),
                        (),
                    ),
                    {},
                )
            self.store.commit_research_result(
                reservation_id,
                task.task_id,
                task.epoch,
                bundle=result.bundle,
                cited_documents=result.cited_documents,
            )
            bundle = result.bundle
        text = render_bundle(plan, bundle)
        completed = self._publish_deterministic_response(
            task,
            text=text,
            speech=speech_units(plan, bundle),
            default_locale=default_locale,
        )
        await self._remember_completed_turn(completed, input_record)
        await self._observe_memory_shadow(completed, input_record)
        return completed

    @staticmethod
    def _estimate_billable_tokens(messages: tuple[Mapping[str, str], ...]) -> int:
        # Frozen conservative local estimator: UTF-8 bytes/3 rounded up, plus
        # explicit per-message and request overhead. Provider token counting is
        # never invoked before approval.
        total = 32
        for message in messages:
            content = str(message["content"])
            total += 12 + (len(content.encode("utf-8")) + 2) // 3
        return total

    @staticmethod
    def _candidate(
        alias: str,
        *,
        endpoint: QualifiedEndpointV1 | None,
        ready: bool,
        p95_total_ms: int = 0,
        max_reserved_microusd: int = 0,
        local: bool = False,
    ) -> CandidateFactsV1:
        return CandidateFactsV1(
            alias=alias,
            qualified=local or endpoint is not None,
            ready=ready,
            circuit_open=False,
            capabilities=("text", "structured_output", "tools"),
            context_window=4096 if local else endpoint.context_limit if endpoint is not None else 0,
            sample_count=0 if local else 20,
            p95_ttft_ms=0,
            p95_total_ms=p95_total_ms,
            max_reserved_microusd=max_reserved_microusd,
            stable_catalog_sha256="" if endpoint is None else endpoint.stable_catalog_sha256,
            qualification_sha256="" if endpoint is None else endpoint.qualification_sha256,
        )

    def _endpoint_for_model(self, model_canonical_id: str) -> QualifiedEndpointV1 | None:
        matches = [item for item in self.cloud_endpoints if item.model_canonical_id == model_canonical_id]
        return matches[0] if matches else None

    async def _prepare_cloud_approval(
        self,
        task: TaskSnapshot,
        input_record: Mapping[str, str],
    ) -> bool:
        """Persist a frozen cloud request and F10 presentation, or stay local."""

        if not self.config.cloud_enabled or self.task_profiler is None or self.cloud_model is None:
            return False
        profile_budget = self.store.remaining_model_budget(task.task_id, task.epoch)
        if profile_budget["requests"] < 1 or profile_budget["output_tokens"] < 1:
            return False
        history = select_relevant_history(
            input_record["text"],
            self.store.cloud_safe_history(
                task.conversation_id,
                limit_pairs=HISTORY_MAX_PAIRS,
            ),
        )
        context_scope = ContextScope.CONVERSATION if history else ContextScope.LATEST_TURN
        profile_request = ModelRequest(
            task_id=task.task_id,
            epoch=task.epoch,
            purpose="task_profile",
            system_prompt="Classify the current task only. Return the exact structured profile. /no_think",
            messages=({"role": "user", "content": input_record["text"] + "\n/no_think"},),
            output_allowance=min(256, profile_budget["output_tokens"]),
            timeout_seconds=min(
                10.0,
                task.remaining_active_seconds(limit=self.config.active_budget_seconds),
                task.remaining_absolute_seconds(),
            ),
            stream=False,
        )
        try:
            profile = await self.task_profiler.profile_task_with_correction(
                profile_request,
                context_scope=context_scope,
                output_contract=OutputContract.PLAIN_TEXT,
                response_language=self._response_language_code(input_record["language"]),
            )
        except (TimeoutError, asyncio.TimeoutError, OrchestratorError, ValueError, RuntimeError):
            # Invalid/unavailable profiler is local-only, never cloud-by-default.
            return False

        local_ready = True
        if self.inference_control is not None:
            try:
                local_ready = bool(self.inference_control.ready())
            except Exception:
                local_ready = False
        score = complexity_score(profile)
        desired_model = (
            TERRA_MODEL
            if score >= 70 and is_advanced_reasoning(profile)
            else DEEPSEEK_MODEL if score >= 40 or not local_ready else ""
        )
        endpoint = self._endpoint_for_model(desired_model) if desired_model else None
        owner_session_id = self.cloud_owner_session_id()

        selected_history = list(history)
        def messages_for_history() -> tuple[Mapping[str, str], ...]:
            messages: list[Mapping[str, str]] = [
                {"role": "system", "content": build_system_prompt(self.config, task)}
            ]
            for turn in selected_history:
                messages.append({"role": "user", "content": turn.user_text})
                messages.append({"role": "assistant", "content": turn.assistant_text})
            messages.append({"role": "user", "content": input_record["text"] + "\n/no_think"})
            return tuple(messages)

        messages = messages_for_history()
        privacy = await self.privacy_classifier.classify_bounded(
            (str(item["content"]) for item in messages),
            memory_derived=False,
            research_evidence=False,
        )
        revalidation = None
        if endpoint is not None and owner_session_id is not None and privacy.cloud_eligible:
            try:
                revalidation = await self.cloud_model.revalidate(endpoint)
            except (OrchestratorError, TimeoutError, asyncio.TimeoutError):
                revalidation = None

        remaining_output = self.store.remaining_model_budget(task.task_id, task.epoch)["output_tokens"]
        model_output_limit = 0
        input_limit = 0
        max_per_request = 0
        reserved_microusd = 0
        dispatch_body: Mapping[str, Any] | None = None
        if endpoint is not None:
            model_output_limit = min(endpoint.output_limit, 2000 if desired_model == TERRA_MODEL else 2048, remaining_output)
            input_limit = min(endpoint.context_limit, 60_000 if desired_model == TERRA_MODEL else 128_000)
            max_per_request = 200_000 if desired_model == TERRA_MODEL else 20_000
            prompt_rate = Decimal(endpoint.prompt_rate_per_m)
            completion_rate = Decimal(endpoint.completion_rate_per_m)
            if desired_model == TERRA_MODEL:
                prompt_rate = max(prompt_rate, TERRA_INPUT_FLOOR_PER_M)
                completion_rate = max(completion_rate, TERRA_OUTPUT_FLOOR_PER_M)
            while model_output_limit >= 256:
                messages = messages_for_history()
                billable = self._estimate_billable_tokens(messages)
                if billable > input_limit and selected_history:
                    selected_history.pop(0)
                    continue
                if billable > input_limit:
                    model_output_limit = 0
                    break
                reserved_microusd = reserve_cost_microusd(
                    prompt_cap=billable,
                    completion_cap=model_output_limit,
                    prompt_rate_per_m=prompt_rate,
                    completion_rate_per_m=completion_rate,
                    request_fee_usd=endpoint.request_rate,
                )
                if reserved_microusd <= max_per_request:
                    dispatch_body = build_dispatch_body(
                        endpoint,
                        messages=messages,
                        output_token_limit=model_output_limit,
                    )
                    break
                if selected_history:
                    selected_history.pop(0)
                else:
                    model_output_limit -= 128

        billable_tokens = self._estimate_billable_tokens(messages)
        cloud_ready = revalidation is not None and dispatch_body is not None
        local_candidate = self._candidate("local", endpoint=None, ready=local_ready, local=True)
        deepseek_endpoint = self._endpoint_for_model(DEEPSEEK_MODEL)
        terra_endpoint = self._endpoint_for_model(TERRA_MODEL)
        deepseek = self._candidate(
            "deepseek",
            endpoint=deepseek_endpoint,
            ready=cloud_ready and desired_model == DEEPSEEK_MODEL,
            p95_total_ms=revalidation.health.p90_latency_ms if cloud_ready and desired_model == DEEPSEEK_MODEL else 0,
            max_reserved_microusd=reserved_microusd if desired_model == DEEPSEEK_MODEL else 0,
        )
        terra = self._candidate(
            "terra",
            endpoint=terra_endpoint,
            ready=cloud_ready and desired_model == TERRA_MODEL,
            p95_total_ms=revalidation.health.p90_latency_ms if cloud_ready and desired_model == TERRA_MODEL else 0,
            max_reserved_microusd=reserved_microusd if desired_model == TERRA_MODEL else 0,
        )
        selected_now = utc_now()
        facts = RoutingFactsV1(
            version="routing-facts/1",
            task_id=task.task_id,
            epoch=task.epoch,
            profile_sha256=profile.sha256,
            estimated_input_tokens=billable_tokens,
            reserved_output_tokens=max(0, model_output_limit),
            remaining_active_ms=max(0, int(task.remaining_active_seconds(now=selected_now, limit=self.config.active_budget_seconds) * 1000)),
            remaining_absolute_ms=max(0, int(task.remaining_absolute_seconds(now=selected_now) * 1000)),
            durable_output_started=task.partial_output,
            privacy_eligible=privacy.cloud_eligible,
            contains_memory=False,
            contains_research=False,
            contains_tool_result=False,
            extension_ready=owner_session_id is not None,
            connection_ready=owner_session_id is not None,
            approval_status="absent",
            cloud_dispatch_count=0,
            monthly_remaining_microusd=self.store.cloud_monthly_remaining(selected_now.strftime("%Y-%m")),
            flow_kind="local",
            local_failure="backend" if not local_ready else "",
            local=local_candidate,
            deepseek=deepseek,
            terra=terra,
        )
        plan = DeterministicModelRouterV1.decide(profile, facts)
        self.store.persist_route_artifacts(task.task_id, task.epoch, profile, facts, plan)
        if (
            plan.target not in {RouteTarget.DEEPSEEK, RouteTarget.TERRA}
            or endpoint is None
            or dispatch_body is None
            or revalidation is None
            or owner_session_id is None
            or not privacy.cloud_eligible
        ):
            return False

        task = self.store.transition(task.task_id, task.epoch, TaskState.ROUTING, TaskState.LOADING_CONTEXT, payload={"route": plan.target.value})
        task = self.store.transition(task.task_id, task.epoch, TaskState.LOADING_CONTEXT, TaskState.EXECUTING)
        expires = min(
            utc_now() + timedelta(seconds=self.config.approval_ttl_seconds),
            parse_utc(task.absolute_expires_at),
        )
        dispatch_body_sha256 = domain_sha256("cloud-dispatch-body/1", dispatch_body)
        binding = CloudApprovalBindingV1(
            conversation_id=task.conversation_id,
            task_id=task.task_id,
            epoch=task.epoch,
            expires_at=utc_text(expires),
            egress_sha256=domain_sha256("cloud-egress/1", {"messages": list(messages)}),
            route_plan_sha256=plan.sha256,
            dispatch_body_sha256=dispatch_body_sha256,
            model_canonical_id=endpoint.model_canonical_id,
            provider_endpoint_tag=endpoint.provider_endpoint_tag,
            quantization=endpoint.quantization,
            parameter_profile_sha256=endpoint.parameter_profile_sha256,
            qualification_version=endpoint.qualification_version,
            qualification_sha256=endpoint.qualification_sha256,
            stable_catalog_sha256=endpoint.stable_catalog_sha256,
            privacy_policy_sha256=privacy.policy_sha256,
            billable_prompt_tokens=billable_tokens,
            input_token_limit=input_limit,
            output_token_limit=model_output_limit,
            max_reserved_microusd=reserved_microusd,
            included_turn_ids=tuple(turn.turn_id for turn in selected_history),
        )
        self.store.create_cloud_approval(
            task.task_id,
            task.epoch,
            binding,
            dispatch_body,
            owner_session_id=owner_session_id,
        )
        return True

    async def _execute_cloud(
        self,
        task: TaskSnapshot,
        input_record: Mapping[str, str],
        run: Mapping[str, Any],
    ) -> TaskSnapshot | None:
        if run["approval_status"] == "DENIED":
            artifacts = self.store.route_artifacts(task.task_id, task.epoch)
            profile = TaskProfileV1.from_mapping(artifacts["profile"])
            facts = RoutingFactsV1.from_mapping(artifacts["facts"])
            local_ok, _reason = local_eligible(profile, facts)
            if (
                not local_ok
                or task.remaining_active_seconds(limit=self.config.active_budget_seconds) <= 0
                or task.remaining_absolute_seconds() <= 0
            ):
                raise OrchestratorError("capability_mismatch", status=409)
            return None
        if run["approval_status"] != "APPROVED" or self.cloud_model is None:
            raise OrchestratorError("cloud_not_approved", status=409)
        binding = run["binding"]
        endpoint = next(
            (
                item
                for item in self.cloud_endpoints
                if item.qualification_sha256 == binding["qualification_sha256"]
                and item.model_canonical_id == binding["model_canonical_id"]
                and item.provider_endpoint_tag == binding["provider_endpoint_tag"]
            ),
            None,
        )
        if endpoint is None:
            raise OrchestratorError("cloud_unqualified", status=409)
        revalidation = await self.cloud_model.revalidate(endpoint)
        if endpoint.stable_catalog_sha256 != binding["stable_catalog_sha256"]:
            raise OrchestratorError("binding_changed", status=409)
        month = utc_now().strftime("%Y-%m")
        permit = self.store.issue_cloud_dispatch_permit(
            task.task_id,
            task.epoch,
            approval_id=run["approval_id"],
            approval_binding_sha256=run["binding_sha256"],
            stable_catalog_sha256=binding["stable_catalog_sha256"],
            qualification_sha256=binding["qualification_sha256"],
            reserved_microusd=int(binding["max_reserved_microusd"]),
            key_remaining_microusd=revalidation.key_remaining_microusd,
            utc_month=month,
        )
        remaining = min(
            task.remaining_active_seconds(limit=self.config.active_budget_seconds),
            task.remaining_absolute_seconds(),
            self.config.cloud_timeout_seconds,
        )
        if remaining <= 0:
            raise BudgetExhaustedError()
        async with asyncio.timeout(remaining):
            result = await self.cloud_model.dispatch(
                task_id=task.task_id,
                epoch=task.epoch,
                endpoint=endpoint,
                permit_id=permit["permit_id"],
                reservation_id=permit["reservation_id"],
                dispatch_body=run["dispatch_body"],
                dispatch_body_sha256=binding["dispatch_body_sha256"],
                input_token_limit=int(binding["input_token_limit"]),
                output_token_limit=int(binding["output_token_limit"]),
                max_reserved_microusd=int(binding["max_reserved_microusd"]),
            )
        if not isinstance(result, CloudModelResultV1):
            raise OrchestratorError("provider_response_invalid", status=502)
        text = validate_buffered_model_text(result.text)
        self._validate_response_language(text)
        locale = self._response_locale(input_record["language"])
        self.store.start_response(task.task_id, task.epoch, locale)
        self.store.append_response_event(
            task.task_id,
            task.epoch,
            "response.text.delta",
            {"chunk_index": 0, "text": text},
        )
        segmenter = SpeechSegmenterV1()
        units = segmenter.feed(text) + segmenter.finish()
        completed = self.store.complete_response(
            task.task_id,
            task.epoch,
            text,
            chunk_count=1,
            trailing_speech=[(index, locale, unit) for index, unit in enumerate(units, start=1)],
            last_unit_index=len(units),
        )
        await self._remember_completed_turn(completed, input_record)
        await self._observe_memory_shadow(completed, input_record)
        return completed

    async def execute_task(self, task_id: str) -> TaskSnapshot:
        task = self.store.get_task(task_id)
        try:
            explicit_memory_command = self._validate_command_or_model_transcript(
                self.store.task_input(task_id)["text"]
            )
            if task.state == TaskState.ADMITTED:
                task = self.store.transition(task_id, task.epoch, TaskState.ADMITTED, TaskState.ROUTING)
            recall = MemoryRecall()
            tool_projection: Mapping[str, Any] | None = None
            if task.state == TaskState.ROUTING:
                if explicit_memory_command is not None:
                    task = self.store.transition(
                        task_id,
                        task.epoch,
                        TaskState.ROUTING,
                        TaskState.LOADING_CONTEXT,
                        payload={"route": "memory_command"},
                    )
            if task.state == TaskState.ROUTING:
                routing_input = self.store.task_input(task_id)
                temporal_answer = resolve_temporal_query(
                    routing_input["text"],
                    input_language=routing_input["language"],
                    response_language=self.config.response_language,
                    observed_at=parse_utc(task.admitted_at),
                    timezone_name=self.config.scheduler_default_timezone,
                )
                if temporal_answer is not None:
                    task = self.store.transition(
                        task_id,
                        task.epoch,
                        TaskState.ROUTING,
                        TaskState.LOADING_CONTEXT,
                        payload={"route": "runtime_temporal"},
                    )
            if task.state == TaskState.ROUTING:
                routing_input = self.store.task_input(task_id)
                if classify_schedule_request(routing_input["text"]).admitted:
                    return await self._execute_schedule_admission(task, routing_input)
                tool_waiting = await self._plan_tool_capability(task, routing_input)
                if tool_waiting is not None:
                    return tool_waiting
                research_plan = None
                if self.config.research_enabled:
                    assert self.research_planner is not None
                    research_plan = await self.research_planner.plan_research(
                        task_id=task.task_id,
                        epoch=task.epoch,
                        query=routing_input["text"],
                        language=routing_input["language"],
                        timezone=self.config.research_timezone,
                        browser_fallback_allowed=self.config.research_browser_fallback,
                        output_allowance=384,
                    )
                if research_plan is None:
                    recall = await self._recall_memory(task, routing_input)
                    if not recall.items and await self._prepare_cloud_approval(task, routing_input):
                        return self.store.get_task(task_id)
                task = self.store.transition(
                    task_id,
                    task.epoch,
                    TaskState.ROUTING,
                    TaskState.LOADING_CONTEXT,
                    payload={
                        "route": (
                            "research_local"
                            if research_plan is not None
                            else "memory_local" if recall.items else "local"
                        )
                    },
                )
                if research_plan is not None:
                    reject_sensitive_text(research_plan.canonical_query)
                    for subquery in research_plan.subqueries:
                        reject_sensitive_text(subquery)
                    task = self.store.transition(
                        task_id,
                        task.epoch,
                        TaskState.LOADING_CONTEXT,
                        TaskState.EXECUTING,
                    )
                    self.store.create_research_approval(
                        task.task_id, task.epoch, research_plan
                    )
                    return self.store.get_task(task_id)
            if task.state == TaskState.LOADING_CONTEXT:
                task = self.store.transition(task_id, task.epoch, TaskState.LOADING_CONTEXT, TaskState.EXECUTING)
            if task.state != TaskState.EXECUTING:
                raise OrchestratorError("invalid_state", status=409)
            input_record = self.store.task_input(task_id)
            if explicit_memory_command is not None:
                locale = self._response_locale(input_record["language"])
                completed, _ = self.store.complete_explicit_memory_command(
                    task.task_id,
                    task.epoch,
                    enabled=self.config.memory_enabled,
                    locale=locale,
                )
                await self._observe_memory_shadow(completed, input_record)
                return completed
            temporal_answer = resolve_temporal_query(
                input_record["text"],
                input_language=input_record["language"],
                response_language=self.config.response_language,
                observed_at=parse_utc(task.admitted_at),
                timezone_name=self.config.scheduler_default_timezone,
            )
            if temporal_answer is not None:
                locale = self._response_locale(input_record["language"])
                self._validate_response_language(temporal_answer.text)
                completed = self._publish_deterministic_response(
                    task,
                    text=temporal_answer.text,
                    speech=((locale, temporal_answer.text),),
                    default_locale=locale,
                )
                await self._remember_completed_turn(completed, input_record)
                await self._observe_memory_shadow(completed, input_record)
                return completed
            scheduler_completed = await self._execute_scheduler_resume(
                task, input_record
            )
            if scheduler_completed is not None:
                return scheduler_completed
            tool_completed, tool_projection = await self._execute_tool_resume(
                task, input_record
            )
            if tool_completed is not None:
                return tool_completed
            if self.config.research_enabled and self.store.research_run(task_id) is not None:
                return await self._execute_research(task, input_record)
            cloud_run = self.store.cloud_run(task_id) if self.config.cloud_enabled else None
            if cloud_run is not None:
                cloud_completed = await self._execute_cloud(task, input_record, cloud_run)
                if cloud_completed is not None:
                    return cloud_completed
            remaining_seconds = min(
                task.remaining_active_seconds(limit=self.config.active_budget_seconds),
                task.remaining_absolute_seconds(),
                self.config.model_timeout_seconds,
            )
            if remaining_seconds <= 0:
                raise BudgetExhaustedError()
            budget = self.store.remaining_model_budget(task_id, task.epoch)
            allowance = min(
                self.config.final_response_output_tokens,
                budget["output_tokens"],
            )
            if budget["requests"] < 1 or allowance < 1:
                raise BudgetExhaustedError()
            locale = self._response_locale(input_record["language"])
            context = await self._final_local_context(
                task,
                input_record["text"],
                recall,
                tool_projection,
            )
            remaining_seconds = min(
                task.remaining_active_seconds(limit=self.config.active_budget_seconds),
                task.remaining_absolute_seconds(),
                self.config.model_timeout_seconds,
            )
            if remaining_seconds <= 0:
                raise BudgetExhaustedError()
            request = ModelRequest(
                task_id=task_id,
                epoch=task.epoch,
                purpose="final",
                system_prompt=context.system_prompt,
                messages=context.messages,
                output_allowance=allowance,
                timeout_seconds=remaining_seconds,
                stream=True,
            )
            segmenter = SpeechSegmenterV1()
            full_text: list[str] = []
            chunk_count = 0
            unit_index = 0
            response_started = False
            model_stream = self.model.stream(request)
            try:
                async with asyncio.timeout(remaining_seconds):
                    async for chunk in model_stream:
                        text = nfc(chunk.text)
                        if not text:
                            continue
                        validate_stream_fragment(text)
                        self._validate_response_language(text)
                        if not response_started and not text.strip():
                            continue
                        if not response_started:
                            self.store.start_response(task_id, task.epoch, locale)
                            response_started = True
                        self.store.append_response_event(
                            task_id,
                            task.epoch,
                            "response.text.delta",
                            {"chunk_index": chunk_count, "text": text},
                        )
                        chunk_count += 1
                        full_text.append(text)
                        for unit in segmenter.feed(text):
                            unit_index += 1
                            self.store.append_response_event(
                                task_id,
                                task.epoch,
                                "response.speech.commit",
                                {"unit_index": unit_index, "locale": locale, "text": unit},
                            )
            finally:
                close_stream = getattr(model_stream, "aclose", None)
                if close_stream is not None:
                    await close_stream()
            if not response_started:
                raise OrchestratorError("local_structural_empty", status=502)
            assistant_text = validate_buffered_model_text(nfc("".join(full_text)))
            trailing: list[tuple[int, str, str]] = []
            for unit in segmenter.finish():
                unit_index += 1
                trailing.append((unit_index, locale, unit))
            completed = self.store.complete_response(
                task_id,
                task.epoch,
                assistant_text,
                chunk_count=chunk_count,
                trailing_speech=trailing,
                last_unit_index=unit_index,
            )
            await self._remember_completed_turn(completed, input_record)
            await self._observe_memory_shadow(completed, input_record)
            return completed
        except StaleEpochError:
            return self.store.get_task(task_id)
        except ContextAssemblyError as exc:
            current = self.store.get_task(task_id)
            if current.state not in {
                TaskState.CANCELLED,
                TaskState.INTERRUPTED,
                TaskState.BUDGET_EXHAUSTED,
                TaskState.FAILED,
            }:
                try:
                    self.store.terminate_failure(
                        task_id,
                        current.epoch,
                        state=TaskState.FAILED,
                        reason=ReasonCode.BACKEND_FAILURE,
                        error_code=exc.code,
                    )
                except OrchestratorError:
                    pass
            return self.store.get_task(task_id)
        except BudgetExhaustedError as exc:
            current = self.store.get_task(task_id)
            if current.state in {TaskState.CANCELLED, TaskState.INTERRUPTED, TaskState.BUDGET_EXHAUSTED, TaskState.FAILED}:
                return current
            try:
                self.store.terminate_failure(
                    task_id,
                    current.epoch,
                    state=TaskState.BUDGET_EXHAUSTED,
                    reason=(
                        ReasonCode.ABSOLUTE_LIFETIME_EXHAUSTED
                        if exc.code == "absolute_lifetime_exhausted"
                        else ReasonCode.ACTIVE_BUDGET_EXHAUSTED
                    ),
                    error_code=exc.code,
                )
            except OrchestratorError:
                pass
            return self.store.get_task(task_id)
        except (TimeoutError, asyncio.TimeoutError):
            current = self.store.get_task(task_id)
            if current.state in {TaskState.CANCELLED, TaskState.INTERRUPTED, TaskState.BUDGET_EXHAUSTED, TaskState.FAILED}:
                return current
            absolute = current.remaining_absolute_seconds() <= 0
            active = current.remaining_active_seconds(limit=self.config.active_budget_seconds) <= 0
            try:
                if absolute or active:
                    self.store.terminate_failure(
                        task_id,
                        current.epoch,
                        state=TaskState.BUDGET_EXHAUSTED,
                        reason=(ReasonCode.ABSOLUTE_LIFETIME_EXHAUSTED if absolute else ReasonCode.ACTIVE_BUDGET_EXHAUSTED),
                        error_code=("absolute_lifetime_exhausted" if absolute else "active_budget_exhausted"),
                    )
                else:
                    self.store.terminate_failure(
                        task_id,
                        current.epoch,
                        state=TaskState.FAILED,
                        reason=ReasonCode.BACKEND_FAILURE,
                        error_code="backend_failure",
                    )
            except OrchestratorError:
                pass
            return self.store.get_task(task_id)
        except OrchestratorError:
            current = self.store.get_task(task_id)
            if current.state not in {TaskState.CANCELLED, TaskState.INTERRUPTED, TaskState.BUDGET_EXHAUSTED, TaskState.FAILED}:
                try:
                    self.store.terminate_failure(
                        task_id,
                        current.epoch,
                        state=TaskState.FAILED,
                        reason=ReasonCode.BACKEND_FAILURE,
                        error_code="backend_failure",
                    )
                except OrchestratorError:
                    pass
            return self.store.get_task(task_id)
        except Exception as exc:
            current = self.store.get_task(task_id)
            if current.state not in {TaskState.CANCELLED, TaskState.INTERRUPTED, TaskState.BUDGET_EXHAUSTED, TaskState.FAILED}:
                try:
                    if isinstance(exc, AttributeError):
                        diagnostic_code = "attribute_error"
                    elif isinstance(exc, KeyError):
                        diagnostic_code = "key_error"
                    elif isinstance(exc, LookupError):
                        diagnostic_code = "lookup_error"
                    elif isinstance(exc, OSError):
                        diagnostic_code = "os_error"
                    elif isinstance(exc, TypeError):
                        diagnostic_code = "type_error"
                    elif isinstance(exc, ValueError):
                        diagnostic_code = "value_error"
                    elif isinstance(exc, RuntimeError):
                        diagnostic_code = "runtime_error"
                    else:
                        diagnostic_code = "other_error"
                    self.store.record_internal_diagnostic(
                        task_id, current.epoch, diagnostic_code
                    )
                    self.store.terminate_failure(
                        task_id,
                        current.epoch,
                        state=TaskState.FAILED,
                        reason=ReasonCode.INTERNAL_FAILURE,
                        error_code="internal_failure",
                    )
                except OrchestratorError:
                    pass
            return self.store.get_task(task_id)

    def cancel_task(self, task_id: str, expected_epoch: int) -> dict[str, Any]:
        return self.store.cancel_task(task_id, expected_epoch, ReasonCode.USER_CANCELLED)

    def handle_playback_terminal(
        self,
        *,
        event_id: str,
        semantic_sha256: str,
        response_id: str,
        terminal_type: str,
        client_sent_at: str,
    ) -> str:
        return self.store.record_voice_playback_terminal(
            event_id=event_id,
            semantic_sha256=semantic_sha256,
            response_id=response_id,
            terminal_type=terminal_type,
            client_sent_at=client_sent_at,
        )

    def create_approval(self, task_id: str, epoch: int, *, kind: str, name: str, arguments: Mapping[str, Any]) -> str:
        return self.store.create_approval(task_id, epoch, kind=kind, name=name, arguments=arguments)

    def decide_approval(self, approval_id: str, epoch: int, decision: str) -> dict[str, Any]:
        kind = self.store.approval_kind(approval_id)
        if kind == "cloud_model":
            raise OrchestratorError("forbidden", status=403)
        if kind == "tool":
            if self.tool_store is None:
                raise OrchestratorError("tool.disabled", status=503)
            result = self.tool_store.decide_action_approval(
                approval_id=approval_id,
                epoch=epoch,
                decision=decision,
            )
        else:
            result = self.store.decide_approval(approval_id, epoch, decision, client_connected=self.has_client())
        if result["task_state"] == TaskState.EXECUTING.value:
            self._schedule(result["task_id"])
        return result

    def sweep(self) -> int:
        return self.store.expire_deadlines()
