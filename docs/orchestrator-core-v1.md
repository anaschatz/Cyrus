# Orchestrator Core v1.0 contract map

This document maps the frozen implementation contract to the source tree. The executable contract is the test suite under `orchestrator/tests`.

| Concern | Implementation |
|---|---|
| Lifecycle and terminal immutability | `domain.py`, `reducer.py`, `service.py` |
| SQLite WAL, migrations, inbox, canonical history, epoch CAS, budgets, approvals, continuation, outbox | `store.py` |
| Guarded physical model dispatch | `model.py`, `pydantic_adapter.py` |
| CommonSchema and hybrid route decision | `common_schema.py`, `pydantic_adapter.py` |
| Deterministic authorization | `policy.py` |
| Voice WebSocket publication fence | `gateway.py` |
| Control API and CLI | `control.py`, `cli.py` |
| Persisted voice replay dedup | `src/jarvis_voice/delivery_journal.py` |
| Qualified Anamnesis retrospective recall | `anamnesis_adapter.py`, `memory.py`, `store.py` migration v2 |

## Fixed clocks

- Active execution: 120 seconds, paused only while awaiting approval or reconnect after a decision.
- Approval: 10 wall-clock minutes, capped by the absolute task lifetime.
- Absolute task lifetime: 15 minutes, never paused.
- Active disconnect, decision reconnect and terminal replay: 30 seconds each.
- WebSocket write and cancellation lock acquisition: one second each; barge-in target is at most two seconds.

## Terminal wire mapping

`CANCELLED` and `INTERRUPTED` emit one `response.cancelled`. A budget/backend/internal failure before visible output emits `server.error` followed by `response.cancelled`; after visible output, prior output is tombstoned and only `response.cancelled` remains replayable. `response.cancelled` and `response.speech.done` are mutually exclusive.

## Control endpoints

```text
GET    /readyz
GET    /v1/approvals
POST   /v1/approvals/{approval_id}/decision
POST   /v1/tasks/{task_id}/cancel
GET    /v1/conversations/active
POST   /v1/conversations
DELETE /v1/conversations/{conversation_id}
```

Except for `/readyz`, every endpoint requires the configured bearer token. Every mutation additionally requires UUID-v4 `Idempotency-Key` and a non-negative `If-Match-Epoch`. All endpoints enforce IPv4 loopback peer, strict numeric Host, forwarded/origin rejection and bounded headers/bodies.

## Qualified and disabled integrations

The production composition always uses the local inference appliance. Explicit
retrospective Anamnesis recall is qualified but remains off by default. Its
records are conversation-scoped Orchestrator state and its results pass a
post-call epoch CAS. Prospective/automatic memory, research, cloud and
side-effect tool ports remain disabled so later integrations cannot acquire
lifecycle authority.
