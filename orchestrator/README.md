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

## Planner → approval → execution

Schema v7 connects the previously separate capability components to the owning
task lifecycle. Explicit schedule requests run only through the qualified local
schedule compiler, create a scheduler proposal linked to the task, pause at
`AWAITING_APPROVAL`, and resume the same task after F10/F11. Approval activates
or rejects the scheduler operation atomically before the deterministic final
voice response is committed.

When tools are enabled, the local route planner selects one closed-registry
tool, a second strict structured call extracts its arguments, and the
Orchestrator persists the route, capability reservation, encrypted arguments,
preview, and approval binding. F10 permits exactly one epoch-checked dispatch;
F11 produces a durable denial and no adapter call. A qualified `ToolExecutorPort`
returns a typed outcome which is validated, encrypted, reconciled into canonical
effects, and supplied only to the local follow-up model. The production executor
uses the authenticated Action Host v2 client; it ACKs a native result only after
the Orchestrator commits it durably and can query/ACK that same idempotent result
after restart without repeating the action. Cancellation invalidates pending
scheduler/tool approvals, while a dispatched ambiguous action can never publish
stale output. Production startup fails closed if tools are enabled without the
configured Action Host socket, token and qualified executor.

## Verification

```sh
cd orchestrator
PYTHONPATH=src:../src:../inference/src:../anamnesis/src uv run --frozen python -m unittest discover -s tests -v
PYTHONPATH=src:../src uv run --frozen python -m compileall -q src ../src/jarvis_voice
```

The suite includes captured Pydantic AI 2.27.0 HTTP snapshots for text streaming, native structured routing and synthetic output-tool routing. The latter two are also passed through the inference appliance's own request/schema qualification code.

Production starts local-only. Explicit retrospective memory, research and the
M1C cloud router are independent, off-by-default adapters. Side-effect tools
remain disabled until separately qualified and activated by policy.

Operational background startup, content-free diagnostics, verified SQLite
backup/restore, the 1,000-turn synthetic lifecycle gate, 100-turn live gate,
restart gate and 24-hour bounded soak are defined in
[`docs/operational-ready-gate-v1.md`](docs/operational-ready-gate-v1.md). Cloud,
memory and research readiness never substitutes for this whole-system gate.

## M1C cloud router (off by default)

M1C adds a negotiated `jarvis.cloud-approval/1` transport without adding any
event to the closed `jarvis.voice/1` enum. A local structured profiler feeds a
pure deterministic router. Medium reasoning is eligible only for the qualified
DeepSeek registry alias; exact advanced reasoning is eligible only for the
qualified Terra alias. Anamnesis recalls, research evidence and tool results
are never included in cloud egress.

Every request is frozen before presentation and bound to the endpoint tag,
quantization, exact dispatch body, stable catalog/pricing digest, privacy
policy, token caps, maximum micro-USD and expiry. F10 approves that one binding;
F11 denies it. The transport persists the decision before sending it and the
server persists a one-shot dispatch permit before the OpenRouter POST. Output
remains buffered until the generation receipt verifies the model/provider and
usage. Timeout after a physical POST is never retried and consumes the full
reservation once.

Activation requires all of the following:

- a real per-endpoint qualification registry based on
  [`cloud-qualification.example.json`](cloud-qualification.example.json);
- an owner-only OpenRouter key file outside the repository (never an
  environment-derived credential);
- `cloud_enabled:true` and the two absolute file paths in local config;
- the fixed `$5` UTC-month Orchestrator ledger. The external approximately
  `€10` account limit remains a separate billing envelope.
- a hash-bound `cloud-release-qualification/1` bundle proving real F10/F11,
  Greek/English/mixed privacy cases, exact endpoint receipts, cost
  reconciliation, fault injection, no fallback and at least 30 days of
  shadow operation. See
  [`docs/cloud-operational-qualification-v1.md`](docs/cloud-operational-qualification-v1.md).

At startup all persisted endpoint qualifications are disabled first. Only the
exact DeepSeek tuple that passes both the release bundle and live key/ZDR probe
is re-enabled. Terra additionally requires `cloud_terra_enabled:true` and its
own passing tuple; otherwise the Terra route is absent. `cloud_enabled:true`
alone never grants cloud authority.

The ZDR catalog and current-key endpoint are rechecked immediately before the
POST. They are health/budget oracles, not qualification authorities. Endpoint
failure never makes local `/readyz` unavailable and never promotes research to
a cloud fallback.

## Research System v1 contract

The off-by-default Research milestone now has closed Python contracts for
immutable search plans, full source metadata, normalized cited snapshots,
absolute quote locators, extractive claims and exact facet/overall statuses.
SQLite schema v3 binds each plan hash to approval, uses one research capability
reservation for the entire run, stores only cited normalized documents, and
checks the live epoch/tombstone again before committing the bundle and canonical
effect. Conversation deletion removes both research tables transactionally.

A research turn has exactly two generative operations: the qualified local
structured route/search-plan request (384 output tokens) and the sidecar's
span-selection request (768 output tokens). Publication invokes no model. It is
rendered from code-owned Markdown and TTS templates, and denial or insufficient
evidence never falls back to model knowledge. The production composition
requires structured-output capability, an authenticated loopback Research API,
a pinned offline language-detector record, and matching qualification digests.

The sibling [`research/`](../research/README.md) deployment freezes the network
topology: all workload networks are internal, SearXNG and Crawl4AI have no host
ports, and only the authenticated egress guard is dual-homed. Browser support is
a separate qualification profile. Keep `research_enabled:false` until the
router, egress, direct-fetch and provider gates all pass with immutable image
digests; enable browser fallback only after its additional security gate.

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

Explicit retrospective deletion is equally closed: `Forget that…` or
`Ξέχασε ότι…` deletes only when the normalized command target equals exactly
one stored record in the active conversation. Zero or multiple matches are
content-free audited outcomes and leave all records unchanged; no model selects
the record to delete.

## Memory shadow mode

The Orchestrator now exposes an audit-only `ShadowMemoryServicePort`. It runs
after a successful response has already been committed, never adds recalled
memory to the model prompt, never changes authoritative memory, and cannot emit
action evidence. Its bounded result is stored idempotently in `task_events` as
`memory.shadow.proposed` or `memory.shadow.failed`; the raw user text is not
duplicated in the audit payload.

The immutable `jarvis.memory-release-policy.v1` boundary limits production v1
to explicit retrospective memory commands and explicit deterministic
reminders. Generative prospective compilers remain unqualified and shadow-only:
they cannot request automatic side effects, emit action evidence, or receive
tool authority. Each audit result binds the source-text hash, selected and
superseded source IDs, writer contract, proposal hash, release-policy hash and
identical before/after fingerprints of the authoritative memory/approval/tool
state. The production composition gives the shadow adapter only a defensive
read-only history facade, never the `OrchestratorStore`.

Shadow mode is off by default. Setting `memory_shadow_enabled` only activates
the boundary; a separately qualified memory adapter must still be injected.
`memory_shadow_mode: "structured_fallback"` retains the schema-constrained
local fallback. For local inference appliances without JSON Schema support,
the conservative trial setting `memory_shadow_mode: "deterministic_only"`
uses gate v3 for clear commands and resolves every ambiguous or hedged command
to `ignore`; it never makes a lifecycle model call.
Final voice responses default to a closed 256-token ceiling through
`final_response_output_tokens` (valid range 32–1024), preventing a local
model that misses its stop condition from consuming the former 1,024-token
allowance.

The first isolated live probes used the real loopback inference appliance for
a three-turn create/replace/cancel conversation while authoritative memory was
disabled. They first exposed a stale chat renderer that missed the model stop.
After restarting with the qualified renderer, an exact `STOP_OK` canary ended
naturally in three tokens and the final three-turn probe produced 12, 34 and 46
tokens in 1.167, 1.570 and 2.067 seconds. Lifecycle state was exact, audit
coverage was 100%, and no authoritative record was written. This engineering
probe is not the required 20-conversation/100-turn real-session trial.
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

The source-run composition now uses a deterministic-first lifecycle writer.
Explicit bilingual `remember`, `correction` and `forget` commands are resolved
without a model when the target is unambiguous. Correction and cancellation
select automatically only when exactly one shadow memory is active; multiple
possible targets and incomplete lifecycle commands fall back to the local
structured writer. Ordinary messages resolve to `ignore` without a lifecycle
model call.

The fallback calls only the configured loopback `/v1/chat/completions`
endpoint, requires the appliance's `json_schema` capability, uses a closed
single-decision enum schema, deterministic sampling, no streaming, no tools
and no retry. User text and active-memory text remain canonical JSON in a
single user message; lifecycle rules remain in the system message. Shadow
memory remains disabled by default and non-authoritative.

The first frozen natural-language shadow diagnostic contains 48 English and
Greek cases. Gate v1 scored 37/48: all 10 ordinary, quoted or negated messages
were safely ignored and all 6 ambiguity cases routed to fallback, but 11 clear
memory commands using unsupported paraphrases were missed as `ignore`. Gate v1
is therefore rejected for a JARVIS pilot. The dataset must remain frozen; a
future gate revision requires a new evaluation set rather than tuning and
claiming success on these same cases.

Gate v2 was implemented as an additive contract and evaluated once on a new,
balanced 64-case corpus. It improved to 59/64, including 32/32 English cases,
16/16 safe ignores and 33/36 explicit memory commands. It still failed the
preregistered fallback-routing requirement at 6/8 and scored 27/32 in Greek,
so it was not wired into the source-run composition. All five failures point
to incomplete normalization of Greek rule literals and markers. Any fix is a
new gate revision and must use another evaluation corpus.

Gate v3 changed only normalization: input, prefixes, markers and hedge words
now share one accent-insensitive boundary. Its first frozen corpus passed
48/48, split evenly between English and Greek, including 28/28 explicit
commands, 12/12 safe ignores and 8/8 fallback routes. Gate v3 is therefore the
deterministic gate used by the source-run shadow composition. This promotion
applies only to the off-by-default, non-authoritative shadow pilot; it does not
enable durable production memory.

The first frozen service-level pilot runs gate v3 through the complete
OrchestratorService boundary across 8 independent multi-turn conversations and
30 English, Greek and mixed-language turns. It passed 8/8 conversations and
30/30 turns, including create/replace/cancel chains, safe ignores, zero-active
commands, two simultaneous shadow memories and two offline-observed fallback
routes. Every canonical response completed, every shadow proposal remained
content-free and non-authoritative, and the authoritative memory-record count
remained exactly zero. The pilot used a static offline assistant and an
ignore-only fallback stub, made no lifecycle model or provider API calls, and
cost $0. It promotes gate v3 only to a larger real-service shadow trial; it does
not qualify authoritative writes or durable production memory.

For the next real-session stage, stop the local service and generate a
content-free report from its SQLite database:

    PYTHONPATH=orchestrator/src:inference/src orchestrator/.venv/bin/python \
      orchestrator/scripts/report_lifecycle_shadow_sessions.py \
      --database /absolute/path/to/orchestrator.sqlite3

The reporter opens SQLite in read-only mode and emits no user text, event IDs,
conversation IDs or database path. It fails the integrity gate on pending or
failed audits, invalid reconstructed histories, release-policy drift, any
side-effect/action-evidence flag, or unequal authoritative-state fingerprints.
Explicit-command retrospective records are reported separately and do not
count as shadow authority. A minimum of 20 real conversations and 100 audited
turns is required before human correctness review. Even a clean report cannot
automatically enable authoritative memory or prospective actions.
