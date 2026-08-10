# Frozen writer diagnostic v2

Writer diagnostic v2 is a fresh, locally authored set for the next online
memory-compiler intervention. It was frozen before any W2 prompt, model call,
or W2 result existed. It is development-only diagnostic evidence and is not
evidence for or against the Anamnesis research hypothesis.

## Frozen artifacts

- Dataset: `scenarios/writer_diagnostic.v2.jsonl`
- Dataset manifest: `scenarios/writer_diagnostic.v2.manifest.json`
- Gold-assisted reference: `oracle/writer_diagnostic_memory_deltas.v2.json`

The dataset contains exactly 10 seven-day scenarios, 69 authored checkpoints,
46 non-clock compiler events, 8 expected actions, 18 forbidden actions, and 2
no-action scenarios. There is one case for each frozen writer family:
`basic_deadline`, `cancellation`, `conjunctive_trigger`, `deadline_update`,
`entity_grounding`, `fact_update`, `negative_control`,
`recurring_intention`, `reversible_completion`, and `threshold_trigger`.

All cases use new dates, entities, event IDs, titles, descriptions, and event
wording. They were authored without inspecting the contents of the v1 writer
dataset, its oracle, or the smoke datasets. The automated freeze test checks
that v2 scenario IDs, event IDs, canonical record hashes, and exact authored
text surfaces are disjoint from the existing datasets. This is an exact
anti-overlap attestation, not a claim that common English vocabulary or family
semantics never recur.

The manifest pins the exact dataset bytes, canonical dataset hash, every
canonical scenario hash, counts, family assignment, exact oracle bytes,
canonical oracle hash, origin, review status, and anti-overlap method. Any
content correction requires a new dataset version and new hashes. A W2 result
must never be used to edit v2.

## Oracle isolation and ceiling

The reference contains one explicit `MemoryDelta` for each of the 46
non-clock sanitized observable events. Each record is bound to the canonical
hash of only `ObservableEvent(id, at, kind, text)`. It contains no tags,
`supersedes`, scenario descriptions, expected actions, forbidden actions,
gold evidence sets, or future-event field.

Offline replay uses the real `InMemoryAnamnesis` reducer, trigger engine,
occurrence lifecycle, execution ledger, and scorer. The frozen ceiling is 8
true positives, 0 false positives, 0 false negatives, 0 obsolete-memory
errors, and exact provenance for all 8 expected actions. This only validates
the annotations and deterministic memory path. The oracle is gold-assisted,
uses zero model tokens, and is never an evaluated writer.

## One-cell W2 boundary

This freeze defines no W2 prompt and changes no compiler, decision, runtime,
reporter, or task code. A later source commit may preregister one W2 cell with
one frozen compiler prompt/schema, one separately pinned local model artifact,
one seed, and these scenarios in their frozen order. The evaluated writer may
receive only the current sanitized event and active compact state.

Report the complete W2 cell even if it fails. Do not repair model output,
silently retry a failed semantic preflight, or tune a later prompt on these
same records. Independent human review remains pending, so all v2 results must
remain `hypothesis_test_eligible=false`.
