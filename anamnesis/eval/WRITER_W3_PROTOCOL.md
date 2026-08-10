# Local writer W3 preregistration protocol

W3 is a prompt-only, bundled compiler repair for the pinned
`ollama/qwen3:4b-instruct` model. It is development-only diagnostic evidence,
is not hypothesis-test eligible, and must not be described as a single-factor
ablation. Its fixed intervention label is `bundled-repair`.

The v4 diagnostic dataset was frozen before the W3 prompt at Git commit
`d8b9972bd2dd69f4336d784a5d8f9442aa808119`. Prompt authors remain blind to
all v4 scenario and oracle contents. This protocol contains no exact preflight
case wording, event records, compact memory states, or illustrative wire/domain
objects.

## Intervention boundary

W3 adds `local.v0.4` as the exact W2 instructions followed by one semantic
validation addendum. The addendum fixes a model-side procedure for normalized
identifiers and non-empty units, calendar resolution, trigger selection,
stable update targets, complete sparse payloads, full compound updates, and
fail-closed ambiguity.

The W1 and W2 prompt builders and contracts remain executable and byte-for-byte
unchanged. W3 deliberately reuses the W2 local wire schema. It does not tighten
the permissive transport schema or change the decision prompt/schema,
provider-neutral domain models, reducer, trigger engine, renderer, or scoring.
Consequently, strict wire-to-domain conversion and the zero-invalid gate remain
responsible for exposing any model-side normalization failure.

| Contract | Frozen value |
| --- | --- |
| W3 compiler version | `local.v0.4` |
| W3 addendum SHA-256 | `84897bc8493dc4c89272aacd9ec6aaf869de92e63b1e225b954d97af84877793` |
| W3 sentinel prompt SHA-256 | `412a63d6b42ea6b5e294401cabbcbacf5a6b7facddbd8fe04ca7b91914c141e5` |
| W3 local-wire contract SHA-256 | `b90298df967f81c91cd6aed6289190768b1f4fe28af4743fb118920d11f8ec51` |
| Unchanged local wire-model schema SHA-256 | `f0e0ab9c3aef10f9b99ca5055d1ee1f2e6d7f091be666ee95035040e564302ec` |
| Unchanged Inspect compiler response schema SHA-256 | `8871ff344eb3a2e88a53b964ef2f24f089a72507c69073ec323cf26a428c3030` |
| Unchanged decision prompt SHA-256 | `871fe15e3160e66abe7480cbde15dfb943dec2d0ff89bb01a03849ad35defd8d` |
| Unchanged decision schema SHA-256 | `1b7c38d3f4bf150523ecc1e468ad3fb1f94753611f190d70f93abbf5ec582426` |
| Frozen W1 local-wire contract SHA-256 | `1ac94e36a5db89ef03798b091424494b9cf50f52ac8e7aaa70e8cfcfc3b0ebd8` |
| Frozen W2 local-wire contract SHA-256 | `cb46570bfb1a101bff51008315ba121e07cea38a93de38fe6c79693d746f72c9` |
| Machine-readable protocol SHA-256 | `7f63c156a8af74ced2d5e5530b3e8083da95c7f46e14e1afafaaf864b3ce1915` |

The machine-readable protocol is
[`preflight/local_writer_w3.protocol.v1.json`](preflight/local_writer_w3.protocol.v1.json).
Any byte change requires a newly named protocol version.

## Post-prompt semantic preflight

The category order and acceptance projections are frozen now. Only after this
prompt/protocol commit may a separate, v4-blind author create neutral exact
case material. A dataset custodian must verify, as a pass/fail audit without
revealing v4 content to the prompt or case author, that the material uses
different entities, actions, dates, and identifiers from v4. It must be
content-addressed before any model call and may not cause a prompt, schema,
category, order, or acceptance change.

The future fixture contains eight compiler cases and one decision case in this
exact order:

| Order | Category | Frozen semantic acceptance projection |
| --- | --- | --- |
| C1 | `normalization_fact` | Exactly one fact assertion; exact normalized entity/attribute, typed value, and explicit non-empty unit or null when absent; no other mutation; strict domain conversion. |
| C2 | `bare_weekday_at` | Exactly one create with an `at` trigger at the exact first strictly-future local occurrence of the named weekday/time, empty conditions/blockers, and exact domain payload. |
| C3 | `condition_transition_and` | Exactly one create with a condition-transition trigger, exact active window, every explicit AND conjunct, exact normalized condition keys/values/units, and no synthetic negative. |
| C4 | `recurrence_iana_range` | Exactly one create with an exact recurring local time, weekdays, start/end dates, IANA timezone, and domain payload. |
| C5 | `stable_id_trigger_update` | Exactly one update whose ID is copied from the unique active intent, with the full changed trigger and all unchanged top-level fields omitted or null. |
| C6 | `full_action_template_update` | Exactly one update with the copied active ID and full current action template; preserve every unchanged sourced leaf, change only event-licensed leaves, and omit other top-level fields. |
| C7 | `complete_sparse_payload_including_zero` | Exactly one create with the exact complete domain payload, every explicit action argument in its matching slot, explicit numeric zero preserved, and no unsourced or filler values. |
| C8 | `ambiguous_empty` | All four mutation arrays and the domain delta are empty; no guessed intent target. |
| D1 | `no_action` | Decision wire mode is `no_action`, with empty wire and domain action lists. |

The model will receive only each future case's runtime input. It must never
receive category names, semantic acceptance data, or authoring material. Create
case intent IDs and summaries may be structural-only where their exact spelling
is semantically underdetermined; update IDs are exact.

The gate is exactly C1 through C8 followed by D1: eight compiler calls and one
decision call. Passing requires exact order, zero parse/domain invalid calls,
zero semantic invalid calls, complete usage and cost for every call, exact
local model identity/residency, and provider API cost `$0.00`.

## Frozen execution and stopping rule

All W3 calls use seed `101`, temperature `0`, one connection, the same model for
compiler and decision, response cache disabled, zero retries, and zero
structured-output repair calls. Raw model calls must be retained.

There is exactly one standalone preflight attempt. Failure preserves the log,
rejects W3 at preflight, and forbids scenario execution. If it passes, there is
exactly one measured-task attempt; that task repeats the same ordered nine-call
gate once as contemporaneous setup before any scenario call. Setup usage and
latency are reported separately from the scenario headline.

The scenario writer gate is fixed as:

- compiler calls: `39`;
- parse/domain invalid: `0`;
- semantic/store invalid: `0`;
- accepted deltas: `39`;
- due-candidate false positives: `0`; and
- due-candidate false negatives: `0`.

The measured and reference due-candidate multisets must match exactly. Decision
actions remain diagnostic-only and are excluded from the writer gate.

A valid failed gate is the final W3 result. No output repair, prompt correction,
acceptance correction, or second W3 run is allowed on v4. W4 may not use v4;
any later prompt revision requires a newly authored and frozen dataset. An
integrity failure is not an experimental result and does not authorize another
model attempt.

This commit intentionally preregisters only the W3 prompt and protocol. Runtime,
preflight task, manifest, and reporter implementation must be added later
without changing any frozen contract above and before exact case material or a
model call is permitted.
