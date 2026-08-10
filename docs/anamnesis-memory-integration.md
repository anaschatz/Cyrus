# Anamnesis memory integration qualification

## Decision

JARVIS qualifies the real Anamnesis `OpenMemoryRecallIndex` boundary for
local, non-authoritative retrospective recall. It does **not** promote the
experimental `InMemoryAnamnesis` temporal/reminder reducer to production
durable storage.

The accepted runtime contract SHA-256 is:

`832de106ffcfa543d8544bb7b29084a141666d6a3f6e2d40e90ff03ade2b1352`

Startup fails closed if the imported Anamnesis runtime contract changes.

## Inspected Anamnesis surface

| Concern | Qualified finding |
|---|---|
| Location | `anamnesis/` inside the JARVIS workspace |
| Startup | No production daemon or HTTP API exists. Anamnesis is a source-run evaluation package with validation/report CLIs. |
| Python interface | `anamnesis.runtime_contract.anamnesis_runtime_contract()` and `anamnesis.openmemory_recall.OpenMemoryRecallIndex` |
| Input | Bounded text, namespace, user partition, scalar metadata and search limit |
| Output | Closed Pydantic `RecallSearchResult`/`RecallMatch` objects; recalled text is explicitly non-authoritative and cannot carry action evidence |
| Persistence | The Anamnesis temporal store is explicitly pure in-memory. JARVIS therefore persists accepted recall records in Orchestrator SQLite. |
| Deletion | Anamnesis exposes record deletion on its recall boundary. JARVIS additionally deletes all conversation memory inside the same logical conversation-deletion transaction. |
| Authentication | No network service is opened, so there is no bearer-auth surface. The adapter requires positive local-only attestation and performs no network request. |
| Timeout/cancellation | Owned by the Orchestrator active timer with a configurable bounded timeout and post-call epoch CAS. |
| Idempotency | `UNIQUE(conversation_id, source_event_id)`; same content/metadata returns the existing record, conflicting reuse fails closed. |

The additive SQLite schema is migration version 2. Existing version-1
Orchestrator databases are upgraded inside `BEGIN IMMEDIATE`; unknown future or
older versions fail closed.

## Runtime behavior

- Memory is off by default.
- A completed turn is eligible for storage only when it explicitly begins with
  a phrase such as `Να θυμάσαι ότι…`, `Θυμήσου…`, `Remember that…` or
  `Please remember…`.
- Partial, cancelled, interrupted, budget-exhausted and failed turns cannot
  write normal memory.
- Recall runs during `ROUTING`. A non-empty qualified result selects
  `memory_local`; empty, timed-out or unavailable recall falls back to `local`.
- Recalled strings enter the model request only as labelled untrusted JSON.
  They never enter canonical history and cannot supply action evidence.
- Recall audit rows contain hashes, counts and stable error codes—not raw memory
  or provider exception text.
- Records are partitioned by conversation. A new conversation cannot retrieve
  a deleted conversation's memory.

## Enable for source-run

Set these fields in the private Orchestrator configuration:

```json
{
  "memory_enabled": true,
  "memory_timeout_seconds": 0.25,
  "memory_recall_limit": 5
}
```

Then include all source roots:

```sh
cd orchestrator
PYTHONPATH=src:../src:../anamnesis/src \
  uv run --frozen jarvis-orchestrator --config orchestrator.local.json serve
```

No Anamnesis evaluation dependency, model or external memory server is started
by this integration. The existing local inference appliance remains the only
model backend.

## Deliberate non-goals

The qualified path does not enable prospective reminders, triggers,
OpenMemory/Cavira SDK installation, embeddings, hosted inference, automatic
fact extraction from every conversation, or cross-conversation identity.
Those require separate policy, privacy and model-quality qualification.
