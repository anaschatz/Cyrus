# JARVIS Orchestrator Core v1.0

The Orchestrator is the sole owner of task lifecycle, one-active-conversation state, routing, budgets, approvals, epochs, cancellation, canonical history and the durable server-event outbox. The voice package remains a transport/TTS client and the inference package remains a frozen local appliance.

## Install and run

Python 3.11 is required. Dependencies are resolved and hashed in `uv.lock`.

```sh
cd orchestrator
uv sync --frozen --python 3.11
cp config.example.json orchestrator.local.json
# Replace the example database path and both private bearer tokens.
PYTHONPATH=src:../src:../anamnesis/src uv run --frozen jarvis-orchestrator --config orchestrator.local.json serve
```

The server performs bounded capability discovery before opening the voice gateway. It never reads provider configuration from the environment. The model alias is fixed to `jarvis-local`; SDK retries and redirects are disabled.

Control examples:

```sh
PYTHONPATH=src uv run --frozen jarvis-orchestrator --config orchestrator.local.json ready
PYTHONPATH=src uv run --frozen jarvis-orchestrator --config orchestrator.local.json approvals list
PYTHONPATH=src uv run --frozen jarvis-orchestrator --config orchestrator.local.json conversations active
```

Conversation deletion always prompts for the full conversation ID. Logical deletion does not claim forensic erasure of SQLite WAL or free pages.

## Durable boundaries

- Every semantic mutation runs inside `BEGIN IMMEDIATE` and is fenced by task ID, expected epoch/state, tombstone state and the conversation epoch.
- Admission, pending canonical user turn, task creation and accepted ACK are one transaction. Duplicate ACK identities are stable under retransmission.
- Cancellation increments the conversation epoch, tombstones the old task/response, invalidates pending output, approvals and reservations, then commits before in-process cancellation signals.
- The outbox uses lease claims and rechecks epoch/replayability. Delivery is bounded at-least-once with immutable logical event IDs and a 30-second terminal window.
- Completed canonical history uses project-owned versioned dataclasses. Pydantic messages are adapters; serialized Pydantic state is allowed only in a validated `ContinuationArtifactV1`.
- Every physical Chat Completions POST consumes one persisted reservation permit. The reserved output allowance becomes the exact `max_completion_tokens` value.

## Verification

```sh
cd orchestrator
PYTHONPATH=src:../src:../inference/src:../anamnesis/src uv run --frozen python -m unittest discover -s tests -v
PYTHONPATH=src:../src uv run --frozen python -m compileall -q src ../src/jarvis_voice
```

The suite includes captured Pydantic AI 2.27.0 HTTP snapshots for text streaming, native structured routing and synthetic output-tool routing. The latter two are also passed through the inference appliance's own request/schema qualification code.

Production v1 is local-only. Explicit retrospective memory has a qualified,
off-by-default adapter; prospective memory, research, cloud and side-effect
tools remain disabled until separately qualified and activated by policy.

## Qualified Anamnesis recall

An additive, off-by-default Anamnesis adapter is available for local
retrospective recall. It validates the pinned Anamnesis runtime fingerprint and
uses its real closed `OpenMemoryRecallIndex` contract. Durable records remain
under Orchestrator SQLite ownership so restart, epoch fencing, idempotency and
conversation deletion remain enforceable.

Set `memory_enabled:true` only with `../anamnesis/src` on `PYTHONPATH`. The
adapter stores only explicit completed statements such as `Να θυμάσαι ότι…`
or `Remember that…`. Non-empty recall selects `memory_local`; timeout, provider
failure or no match safely falls back to `local`. See
[`docs/anamnesis-memory-integration.md`](../docs/anamnesis-memory-integration.md)
for the qualification inventory and limitations.

## Memory shadow mode

The Orchestrator now exposes an audit-only `ShadowMemoryServicePort`. It runs
after a successful response has already been committed, never adds recalled
memory to the model prompt, never changes authoritative memory, and cannot emit
action evidence. Its bounded result is stored idempotently in `task_events` as
`memory.shadow.proposed` or `memory.shadow.failed`; the raw user text is not
duplicated in the audit payload.

Shadow mode is off by default. Setting `memory_shadow_enabled` only activates
the boundary; a separately qualified memory adapter must still be injected.
Timeouts and provider failures are recorded with stable error codes and never
change the completed JARVIS response. Automatic approval remains disabled.

`AnamnesisShadowAdapter` is the first concrete lifecycle implementation of
that boundary. Its writer may choose only `ignore`, `create`, `replace` or
`cancel`, plus an index into the current active-memory list for the last two.
JARVIS derives opaque memory keys and exact supersession IDs itself, rebuilds
active shadow state from hash-checked audit events and canonical source turns,
and never lets shadow output mutate canonical history or enter the answer
prompt. This remains an evaluation seam; a separately qualified structured
writer is required before enabling it in a production composition.

The source-run composition now provides that local structured writer when
shadow mode is enabled. It calls only the configured loopback `/v1/chat/completions`
endpoint, requires the appliance's `json_schema` capability, uses a closed
two-field decision schema, deterministic sampling, no streaming, no tools and
no retry. User text and active-memory text remain canonical JSON in a single
user message; lifecycle rules remain in the system message. The example keeps
the feature disabled and allocates a 30-second bound for an explicitly enabled
local shadow call.
