"""Diagnostic-only Anamnesis strategy augmented by OpenMemory recall.

The evaluated path is deliberately search-only.  A caller supplies a factory
that opens a fresh, pre-populated and independently pinned recall snapshot for
each scenario.  Search results are rendered as untrusted text; they never enter
the deterministic store, MemoryView, evidence ledger, compiler state, or event
history.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from time import perf_counter

from anamnesis.baselines import (
    AnamnesisMemoryStrategy,
    ContextSelection,
    RetrievalUsage,
    StrategyWork,
)
from anamnesis.memory import InMemoryAnamnesis, MemoryCompiler
from anamnesis.openmemory_recall import RecallIndex
from anamnesis.prompts import build_decision_prompt
from anamnesis.runtime_contract import anamnesis_runtime_contract
from anamnesis.schema import Decision, ObservableEvent
from anamnesis.wire import DecisionWire

OPENMEMORY_RECALL_STRATEGY_VERSION = "openmemory-recall.v0.1"
OPENMEMORY_RECALL_PROMPT_VERSION = "openmemory-recall-prompt.v0.1"


class AnamnesisOpenMemoryRecallStrategy:
    """Add bounded retrospective snippets without delegating memory authority."""

    name = "anamnesis_openmemory_recall"

    def __init__(
        self,
        *,
        compiler: MemoryCompiler,
        recall_factory: Callable[[], RecallIndex],
        memory: InMemoryAnamnesis | None = None,
        top_k: int = 5,
    ) -> None:
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be in [1, 100]")
        self._core = AnamnesisMemoryStrategy(compiler=compiler, memory=memory)
        self._recall_factory = recall_factory
        self.top_k = top_k
        self._recall: RecallIndex | None = None
        self._recall_by_event: dict[str, tuple[str, ...]] = {}

    @property
    def memory(self) -> InMemoryAnamnesis:
        """Expose only the authoritative deterministic store for auditing."""

        return self._core.memory

    def reset(self) -> None:
        """Reset core state and open a fresh caller-isolated recall snapshot."""

        self._core.reset()
        recall = self._recall_factory()
        for attribute in (
            "authoritative",
            "supports_action_evidence",
            "mutates_anamnesis",
        ):
            if getattr(recall, attribute, None) is not False:
                raise ValueError(
                    "recall index must explicitly declare a non-authoritative, "
                    f"evidence-free boundary: {attribute}=False"
                )
        self._recall = recall
        self._recall_by_event = {}

    async def ingest(self, event: ObservableEvent) -> StrategyWork:
        """Compile the event normally, then query the read-only recall snapshot."""

        core_work = await self._core.ingest(event)
        recall = self._require_recall()
        query = json.dumps(
            {
                "at": event.at.isoformat(),
                "kind": event.kind,
                "text": event.text,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        started = perf_counter()
        result = await recall.search(query, limit=self.top_k)
        latency_ms = (perf_counter() - started) * 1000
        if result.authoritative or result.evidence_event_ids:
            raise ValueError("recall boundary returned authoritative material")
        self._recall_by_event[event.id] = tuple(
            match.content for match in result.matches
        )
        return core_work.plus(
            StrategyWork(
                local_usage=RetrievalUsage(latency_ms=latency_ms),
                # OpenMemory does not expose provider-neutral token/embedding
                # accounting.  Diagnostic runs therefore remain incomplete.
                usage_complete=False,
                cost_complete=False,
            )
        )

    def select(self, current: ObservableEvent) -> ContextSelection:
        """Return core selection plus text-only, non-authoritative recall."""

        core = self._core.select(current)
        if current.id not in self._recall_by_event:
            raise RuntimeError("select requires successful recall ingest")
        return ContextSelection(
            events=core.events,
            decisions=core.decisions,
            memory_view=core.memory_view,
            retrospective_recall=self._recall_by_event[current.id],
            state_sha256=core.state_sha256,
            due_candidate_ids=core.due_candidate_ids,
            usage=core.usage,
        )

    def commit(self, current: ObservableEvent, decision: Decision) -> StrategyWork:
        """Commit only to Anamnesis; OpenMemory never receives decisions."""

        return self._core.commit(current, decision)

    def strategy_contract(self) -> dict[str, object]:
        """Return the policy bytes bound into fallback system identity."""

        return {
            "authoritative": False,
            "deterministic_memory": anamnesis_runtime_contract(),
            "evidence_ids_allowed": False,
            "online_writes": False,
            "provider_usage_complete": False,
            "top_k": self.top_k,
            "version": OPENMEMORY_RECALL_STRATEGY_VERSION,
        }

    def _require_recall(self) -> RecallIndex:
        if self._recall is None:
            raise RuntimeError("strategy must be reset before ingest")
        return self._recall


def openmemory_recall_prompt_contract() -> str:
    """Fingerprint the additive prompt section used by this diagnostic arm."""

    sentinel = ObservableEvent.model_validate(
        {
            "id": "<event-id>",
            "at": "2000-01-01T00:00:00+00:00",
            "kind": "user_message",
            "text": "<event-text>",
        }
    )
    rendered = build_decision_prompt(
        now="<current-time>",
        current_event_id="<current-event-id>",
        context_events=[sentinel],
        decision_history=[],
        memory_view=None,
        retrospective_recall=("<untrusted-recall>",),
    )
    schema = json.dumps(
        DecisionWire.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{OPENMEMORY_RECALL_PROMPT_VERSION}\n{rendered}\n{schema}"


def openmemory_recall_prompt_sha256() -> str:
    return hashlib.sha256(openmemory_recall_prompt_contract().encode()).hexdigest()


__all__ = [
    "AnamnesisOpenMemoryRecallStrategy",
    "OPENMEMORY_RECALL_PROMPT_VERSION",
    "OPENMEMORY_RECALL_STRATEGY_VERSION",
    "openmemory_recall_prompt_contract",
    "openmemory_recall_prompt_sha256",
]
