# Frozen writer diagnostic v1

This is a fresh, locally authored diagnostic for the online memory compiler.
It contains 10 simulated seven-day scenarios and is frozen before a W1 writer
prompt, task, model call, or result exists. It is development-only evidence
about one compiler intervention, not evidence for or against the Anamnesis
research hypothesis.

## Frozen artifacts

- Dataset: `scenarios/writer_diagnostic.v1.jsonl`
- Dataset manifest: `scenarios/writer_diagnostic.v1.manifest.json`
- Gold-assisted reference: `oracle/writer_diagnostic_memory_deltas.v1.json`

The manifest pins the exact dataset bytes, canonical dataset hash, every
canonical scenario hash, exact oracle bytes, canonical oracle hash, counts,
families, origin, and review status. Any content correction creates a new
dataset version and new hashes. W1 output must never be used to edit v1.

The records are not members of `smoke.jsonl`, the 35-scenario development set,
the holdout-shaped harness set, or the combined 50-scenario candidate. No
LongMemEval or TriggerBench content is included. The cases cover generic writer
failure families with new wording and entities; they are not transformations
of the original smoke records.

## Oracle isolation

The oracle artifact contains one explicit `MemoryDelta` for every non-clock
observable event and binds each record to the hash of the sanitized
`ObservableEvent(id, at, kind, text)`. Its annotation schema has no expected
actions, forbidden actions, tags, `supersedes`, or author-only scenario fields.
The loader uses the ordered sanitized `RuntimeScenario` only to verify complete
coverage; the oracle compiler then consumes exactly one bound record for each
current request. Neither the artifact nor future records are passed to the
evaluated W1 writer.

Offline replay through the real in-memory reducer, trigger engine, occurrence
state, renderer input, and execution ledger must produce all 8 expected actions
with 0 false positives, 0 false negatives, and exact provenance for all 8. This
is an annotation/store ceiling only. The annotations are human-authored,
gold-assisted, and unmeasured; their zero model-token accounting is not an
Anamnesis efficiency result.

The evaluated W1 writer must never receive the oracle file, expected or
forbidden actions, provenance sets, tags, `supersedes`, scenario titles or
descriptions, or future events. It receives only the current sanitized event
and the active compact state under the existing incremental runtime boundary.

## One-cell W1 boundary

This freeze does not define or modify a prompt. A later, separate source commit
may preregister exactly one W1 writer cell: one frozen compiler prompt/schema,
one pinned local model artifact, one declared seed, and these 10 scenarios in
their frozen order. The decision prompt, scorer, memory engine, dataset, and
oracle reference must remain unchanged. No W1 model call may occur before that
configuration is committed and its artifact hashes are recorded.

Report the complete cell even if it fails. Do not tune W2 on these same 10
records and do not use a W1 result to authorize the 35-scenario development run
or any sealed run. A subsequent writer intervention requires newly authored
diagnostic cases and a new preregistration.

Independent human review is pending. Until it passes, all results remain local
diagnostics and `hypothesis_test_eligible=false`.
