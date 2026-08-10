# Oracle compiler smoke diagnostic

`smoke_memory_deltas.v1.json` is a gold-assisted, diagnostic-only compiler
ceiling for the 10 open smoke scenarios. It is not the Anamnesis system, is not
hypothesis-test eligible, and must never enter a headline baseline, success
gate, or sealed-set result.

The artifact contains 53 explicit records: exactly one for every sanitized
non-clock `ObservableEvent` in the smoke dataset, including explicit empty
deltas. Each record is bound to the SHA-256 of only `id`, `at`, `kind`, and
`text`. Records contain no expected or forbidden actions, authored
`supersedes` links, tags, future-event content, gold evidence sets, or
provenance IDs. At runtime, the oracle compiler releases records strictly in
observable-event order and verifies the current event hash; later records are
not included in the compiler response or active memory state.

The annotations are intentionally gold-assisted so that this diagnostic can
separate compiler failures from failures in the deterministic store, trigger
engine, renderer, and shared decision policy. Each mutation is nevertheless
placed no earlier than the observable event that supports it. The offline
acceptance replay uses only sanitized runtime events and reaches 8 TP, 0 FP,
0 FN, and exact provenance for all 8 expected actions.

## Temporal annotation convention

- A condition-transition request without an explicit end remains active from
  the current observable event until exactly seven days later.
- A request explicitly scoped to Friday is active only during that Friday in
  the scenario timezone.
- Explicit deadlines and recurring ranges are resolved directly from the
  current observable event's timestamp and wording.

## Accounting boundary

The oracle performs no model compiler calls. Its compiler token usage and
provider API cost are zero. Any reported oracle token count is therefore a
decision-only lower bound, not an Anamnesis efficiency result. Human annotation
time, annotation cost, review effort, electricity, and hardware amortization
are unmeasured and excluded.
