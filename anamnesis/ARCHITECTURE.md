# Anamnesis v0 architecture

Status: frozen implementation contract for the research prototype. This
document describes the treatment being evaluated; it is not a claim that the
hypothesis is supported.

## Research boundary

Anamnesis changes the memory supplied to an otherwise shared action-decision
policy. It does not receive a different action engine, extra gold annotations,
or access to future events.

Every evaluated system receives the same sequence of sanitized observable
events and makes one final structured decision at every checkpoint. Anamnesis
may make one additional memory-compiler call for each user message or external
observation. Those calls use the same frozen model snapshot as the final
decision and all of their tokens, cost, errors, and latency count toward the
Anamnesis result.

```text
ObservableEvent
  -> MemoryCompiler
  -> validated MemoryDelta
  -> deterministic temporal store and trigger engine
  -> compact MemoryView
  -> shared decision model
  -> occurrence and execution ledger
```

Clock ticks do not invoke the memory compiler. The shared decision model is
still invoked after every clock tick and after every other authored event,
including checkpoints where Anamnesis has no due candidate.

## Observable boundary

The runtime event contract contains only `id`, `at`, `kind`, and `text`.
Authored `supersedes` links, expected and forbidden actions, scenario tags,
descriptions, hidden evidence sets, and future events never cross this
boundary. The full scenario remains available only to the deterministic scorer
and for the canonical dataset hash.

The three simple baselines and Anamnesis implement one lifecycle:

```python
class MemoryStrategy(Protocol):
    name: str

    def reset(self) -> None: ...
    async def ingest(self, event: ObservableEvent) -> StrategyWork: ...
    def select(self, current: ObservableEvent) -> ContextSelection: ...
    def commit(self, current: ObservableEvent, decision: Decision) -> StrategyWork: ...
```

`ingest` applies the current event before context selection. `commit` records
the decision before the next checkpoint. Full-context and vector RAG preserve
assistant decisions through this method; Anamnesis records typed executions
instead of parsing a synthetic conversation event.

## Memory records

The v0 store is reset for every scenario and lives entirely in Python memory.
It contains four versioned or append-only record families:

- `ObservableEvent`: immutable audit source. The full audit log is never
  rendered into Anamnesis decision context.
- `FactRevision`: opaque normalized entity/attribute key, typed value and
  optional unit, half-open validity interval, prior revision, and raw source
  event. Exactly one revision per fact key is current.
- `IntentRevision`: immutable full revision with a stable root `action_key`,
  trigger, conditions, action template, validity interval, prior revision, and
  field-level provenance. Updates do not create a new root action key.
- `Occurrence` and `ExecutionRecord`: concrete one-shot or recurring instance,
  its terminal state, emission checkpoint, and payload hash. Recurring
  occurrences terminate independently.

The compiler emits a strict `MemoryDelta` containing fact assertions, intent
creates, intent patches, and cancellations. The runner supplies the source
event ID; the model cannot invent provenance. Applying a delta is atomic. A
schema error, unknown action key, conflicting mutation, or invalid reference
leaves the store unchanged and receives no retry in a measured run.

A new fact closes the previous revision at the event time, including
same-value reaffirmations. An intent patch materializes a complete new revision
and carries unchanged field values and provenance forward. Cancellation closes
the active revision and cancels pending occurrences. Task completion is a fact
used as a blocker rather than an irreversible intent mutation, allowing a later
correction before the deadline.

## Trigger language

The closed v0 trigger language supports only:

- one absolute aware datetime;
- a local time on selected weekdays within inclusive start/end dates;
- a condition transition within an active interval;
- conjunction across required conditions;
- any matching blocker;
- exact `eq`, `gte`, and `lte` comparisons over typed fact values; and
- `{date}` and `{weekday}` occurrence variables in fact and action templates.

Condition evaluation is three-valued. Every required condition must be true;
an unknown required condition prevents eligibility. A true blocker suppresses
the occurrence; an unknown blocker does not. A condition trigger is eligible
only on a post-creation transition from false or unknown to true. General
Boolean expressions, OR, cron, arbitrary RRULEs, and general temporal logic are
outside v0.

An absolute or recurring occurrence becomes due at the first checkpoint that
reaches its scheduled time. A due occurrence that is not emitted at that
checkpoint expires instead of firing late. A condition occurrence is anchored
to the event that caused the qualifying transition. Terminal occurrence keys
cannot fire twice.

## Context and provenance

Anamnesis renders only due candidates, facts actually consulted for those
candidates, resolved payload fields, relevant prior executions, and raw source
event IDs. The current observable event is always shown separately. Memory
record IDs are never valid action evidence.

Gold provenance follows a minimal-causal rule. An acceptable evidence set
contains the creating request, sources of the active trigger and action fields,
sources of facts actually consulted for required or blocker evaluation, and
the current trigger checkpoint. Scenario gold may list multiple sets when more
than one minimal explanation is defensible.

## Reproducibility and accounting

Each run fingerprints the shared decision contract and its runtime system
configuration: the memory-writer prompt and wire schema where applicable,
explicit reducer, trigger-engine and renderer versions, a derived memory-schema
digest, vector-RAG parameters and embedding revision, model snapshot,
generation settings, and pricing configuration. The clean Git revision and
frozen manifest additionally bind the full implementation and experiment
policy; both task construction and strict reporting verify them.

Headline Anamnesis usage is the sum of memory-compiler and final-decision
usage. Per-purpose usage and latency remain available diagnostically. Missing
token usage, missing pricing configuration for a cost-bearing hosted model, or
an incomplete scenario/system/repetition matrix invalidates a measured result.
Local embedding work is reported in inputs, characters, and wall-clock latency;
its API cost is zero and hardware/energy cost is explicitly outside v0.

Before the first measured scenario in each Inspect task, the hosted model makes
one strict, synthetic, non-dataset warmup call. Its raw output, usage, cost and
latency are copied into every run as one task-level attestation; its latency is
charged once to setup latency, while its tokens and cost are deliberately
excluded from headline scenario usage. Strict reporting reconciles the Inspect
log total to headline usage plus exactly one warmup. Every run also records the
exact byte SHA-256 of the frozen manifest, so replacing a manifest at the same
path invalidates the report.

Per-checkpoint traces retain the accepted or rejected delta, state hash, due
candidates, rendered-context hash, model output, parse status, usage, and
latency. They contain no hidden gold.

The deterministic fake compiler exists only for unit/end-to-end harness tests.
An oracle or manually annotated compiler may be published as a diagnostic
ceiling, but it is never the headline Anamnesis system. Every measured
Anamnesis run uses the same frozen hosted model as the shared decision step.

## Experimental gates

The 35 development scenarios may be used for implementation and declared
ablations. A valid 15-scenario sealed split is excluded from development
commands and opened only after the Anamnesis prompt, schema, and configuration
are frozen. The checked-in holdout-shaped candidate was already inspected
during implementation, so it is explicitly ineligible and must be replaced
post-freeze. AutoResearch-style automation, if added later, is restricted to
the development split and cannot replace the preregistered final runs.

The hypothesis is supported only when all three preregistered repetitions meet
all three conditions in `RESEARCH.md`. A clean rejection remains a successful
research outcome.

## Explicit non-goals

There is no database, vector index inside Anamnesis, knowledge graph, agent
framework, background scheduler service, UI, model training, PyTorch,
multimodal input, multi-user state, confidence/decay model, or production
persistence in v0.
