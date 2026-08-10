# Frozen writer diagnostic v3

Writer diagnostic v3 is the corrected pre-prompt release for the W2 online
memory-compiler intervention. It is derived from the immutable v2 release in
commit `0a8b6714dea57eb9a0c9dded8535569c380be9b1`. V2 was rejected before any W2
prompt existed and before any W2 model call because an independent audit found
three source-to-gold defects. V2 remains preserved and must not be edited or
used for a W2 run.

## Closed correction set

V3 rekeys every scenario, event, action, and evidence ID from `wd2_` to `wd3_`
and makes exactly three causal/gold corrections:

1. `wd3_01` preserves the observable source casing `Orion Dome` in the
   optional `project` payload.
2. `wd3_05` preserves the observable source casing `Cedar incubator` in the
   optional `item` payload.
3. `wd3_09` minimally changes the creating event to explicitly say “send the
   letter to the theater sponsor,” supporting the existing `recipient` payload.

There are no other causal, temporal, action, provenance, or gold changes. An
automated lineage test reverses the namespace change and these three
corrections and requires exact equality with v2. A second independent audit GO
is required before committing v3 or writing the W2 prompt; this document does
not itself assert that GO.

## Frozen artifacts and scope

- Dataset: `scenarios/writer_diagnostic.v3.jsonl`
- Dataset manifest: `scenarios/writer_diagnostic.v3.manifest.json`
- Gold-assisted reference: `oracle/writer_diagnostic_memory_deltas.v3.json`

The release retains 10 seven-day scenarios, 69 authored checkpoints, 46
non-clock compiler events, 8 expected actions, 18 forbidden actions, 2
no-action scenarios, and one case for every frozen writer family. It remains
development-only diagnostic evidence, outside the development 35, sealed set,
and final hypothesis test. Independent human review remains pending.

The oracle contains one hash-bound `MemoryDelta` for every non-clock sanitized
observable event. Offline replay through the real store, trigger engine,
occurrence lifecycle, ledger, and scorer must produce 8 true positives, 0
false positives, 0 false negatives, 0 obsolete-memory errors, and exact
provenance for all 8 actions. The oracle is gold-assisted, never visible to the
evaluated writer, and has zero model-token accounting.

## Public W2 candidate-matching protocol

The W2 writer gate compares deterministic due candidates with this canonical
key, in this exact field order:

`(checkpoint, action_key, due_at, kind, canonical_payload, sorted_evidence)`

`canonical_payload` is the compact JSON serialization of the complete payload
with keys sorted and no insignificant whitespace. `sorted_evidence` is the
lexically sorted tuple of candidate evidence event IDs. Candidate comparison
is multiset-based.

`summary` is noncanonical UX text. No canonical summary-generation policy
exists, so `summary` is explicitly excluded from the W2 gate, along with local
`intent_id` and `occurrence_id`. There must be no hidden exact-summary gate.

For continuity, tests also compute the legacy exact diagnostic key:

`(checkpoint, action_key, due_at, kind, canonical_payload, summary, sorted_evidence)`

That legacy key may diagnose changes in oracle-rendered UX text, but it cannot
change W2 acceptance, candidate TP/FP/FN, or stopping decisions. Oracle records
may retain summaries because the deterministic memory view and final decision
model still need human-readable text.

## W2 boundary

This release changes no compiler prompt, decision prompt, runtime, reporter, or
task implementation. After v3 receives audit GO and is committed, a separate
commit may preregister exactly one W2 prompt/schema, pinned local model, seed,
and this frozen scenario order. The writer receives only the current sanitized
event and active compact state. A failed semantic preflight ends that model
attempt without repair, retry, scenario access, or post-result dataset edits.
