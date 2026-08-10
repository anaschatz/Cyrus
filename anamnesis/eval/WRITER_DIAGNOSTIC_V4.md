# Writer diagnostic v4 freeze

Writer diagnostic v4 is a brand-new, locally authored, diagnostic-only dataset. It was authored and frozen on 2026-08-08 before any W3 compiler prompt existed and before any W3 model call. It is not derived from writer diagnostics v1, v2, or v3, is not part of the development 35 or sealed set, is not hypothesis evidence, and is not eligible for a preregistered final run.

The dataset contains exactly ten seven-day `Europe/Athens` scenarios, one for each writer family: basic deadline, cancellation, conjunctive trigger, deadline update, entity grounding, fact/action update (`fact_update` in the manifest), negative control, recurrence, reversible completion, and threshold transition (`threshold_trigger` in the manifest).

## Frozen counts

| Item | Count |
| --- | ---: |
| Scenarios | 10 |
| Families | 10, one scenario each |
| Observable events | 62 |
| Non-clock events | 39 |
| Reporter-only oracle event records | 39 |
| Nonempty oracle delta events | 21 |
| Oracle mutations | 21 (9 facts, 9 creates, 2 updates, 1 cancellation) |
| Expected actions | 8 |
| Forbidden actions | 13 |
| No-action scenarios | 2 |
| Scenarios with an obsolete-action trap | 3 |
| Optional payload slots | 34 |

Every optional payload value is explicitly present, with identical source casing, in observable text that grounds that expected or forbidden action. This includes both the current and stale alternatives in update and entity-confusion traps. The Quill update inherits the unchanged `item="quill samples"` leaf from the prior active intent while changing only the recipient. The `subject` field is frozen as an article-free, trimmed lowercase verb-plus-object value: `submit permit`, `reserve kiln`, `launch transect`, `inspect crate`, `photograph facade`, `courier samples`, `polish map`, `test alarm`, `dispatch prospectus`, or `close valve`.

## Frozen hashes

| Artifact or semantic object | SHA-256 |
| --- | --- |
| `eval/scenarios/writer_diagnostic.v4.jsonl` bytes | `6b2530cb9f3426c792500f07e854d7f31ad84081ac77104cb8032737234ff91c` |
| Ordered canonical dataset | `ee80a55874ac6d6cfd5ee32484d91113bff78d829d66c9ff46bcb646456eb598` |
| `eval/scenarios/writer_diagnostic.v4.manifest.json` bytes | `9cb287cc2271ff136c59618d6d3a6c07255a65bc5576c4c7a9af8f5de8a63f16` |
| `eval/oracle/writer_diagnostic_memory_deltas.v4.json` bytes | `72308bb34bda758cc72dc651e3f0fd2fd2bd1bff820479e2cf0774ee8d66cf5c` |
| Canonical oracle artifact | `b877bcd6fe15767d9f1bb42a5840a799d2ef5a4a3691eb6a59ae2f9f7d40813b` |

The adjacent manifest additionally freezes the ten canonical per-record hashes and all count, family, origin, grounding, isolation, and replay attestations.

## Public W3 candidate key

W3 candidate matching uses `w3.candidate-key.v1`, the same six-field structure frozen for v3:

1. `checkpoint`
2. `action_key`
3. `due_at`
4. `kind`
5. `canonical_payload`
6. `sorted_evidence`

`canonical_payload` is UTF-8 JSON with keys sorted and separators `,` and `:` without extra whitespace. `sorted_evidence` is the lexical ordering of the candidate evidence event IDs. `summary`, `intent_id`, and `occurrence_id` are explicitly excluded. Summary is noncanonical UX text, and there is no hidden exact-summary gate. A seven-field key that adds summary may be reported only as a legacy diagnostic.

## Oracle and replay boundary

The oracle artifact contains exactly one explicit memory delta, including empty deltas, for each non-clock sanitized observable event. It is bound to the canonical dataset and to the hash of each event's provider-neutral `id`, `at`, `kind`, and `text` fields. It contains no scenario gold fields and is reserved for reporter-only offline replay; it is never writer input and cannot be used by an evaluated writer.

Real `InMemoryAnamnesis` replay with due-candidate copying reaches the frozen ceiling: 8 TP, 0 FP, 0 FN, exact provenance for all 8 matches, 0 obsolete errors, and 0 invalid outputs.

Each nonempty mutation is derivable from the current observable event plus prior active state. Every named weekday/ISO-date pair is checked against the calendar, condition-transition end times are explicit, and the recurrence states its start date, end date, local time, weekday, and `Europe/Athens` IANA timezone in observable text. Update and cancellation wording identifies the sole active intent. The future-scoped wording in `wd4_09_e1` creates a blocker but does not assert a current `dispatched=false` fact; an explicit regression test freezes that boundary. Internal intent IDs and summaries are reporter representations rather than hidden matching targets; the public candidate key excludes both `intent_id` and `summary`.

## Isolation and audit status

The automated test compares v4 against the core scenario sets and writer diagnostics v1, v2, and v3 without printing their contents. It recomputes disjoint scenario IDs, event IDs, canonical record hashes, exact authored surfaces, case-folded authored surfaces, optional payload values, and calendar dates. Two independent agent audits passed on 2026-08-08 after their findings were corrected and a final clean post-fix audit was completed. No human has reviewed this dataset; independent human review remains pending. W3 prompt authoring may begin only from the final hashes below.
