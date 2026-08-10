# OpenMemory recall diagnostic v3 freeze

This artifact is a fresh, development-only paired diagnostic for immediate action interpretation with optional retrospective recall. It was frozen before any v3 decision contract, task, or model call.

The v2 result exposed a contract mismatch: the evaluated prompt described a temporal reminder firing component, while four positive cases were explicit immediate commands. V3 does not repair that mismatch inside the data. It freezes fresh cases first so a later, separately committed immediate-action decision contract can be evaluated without changing the expected outputs after seeing model behavior.

## Frozen boundary

- Eight cases, one per existing recall-safety/usefulness family.
- Four explicit immediate-action expectations and four no-action expectations.
- One helpful reference-resolution opportunity, six forbidden-influence cases, and one empty-recall control.
- IDs, event/hit text, named entities and dates are disjoint from v1 and v2.
- Only the observable event and frozen recall content may enter model prompts. Case family, labels, policies and expected decisions remain evaluator-only.
- Recall is untrusted and non-authoritative. It may resolve an argument of an explicit current action, but it cannot create an action, override current context, prove completion or execution, or provide evidence IDs.

The exact raw and canonical hashes, record hashes, counts and lineage are pinned in `eval/openmemory/decision_diagnostic.v3.manifest.json`. Automated review is complete; independent human review remains pending.

## Future treatment and stopping rule

After this freeze is committed, one dedicated immediate-action contract may be authored and committed. It must keep the same local model, no-thinking transport, schema, seed, temperature, call order, and paired metric. Exactly one 8×2 run is allowed. A valid pass or fail is preserved; no prompt repair or duplicate selection is allowed on v3. Any subsequent intervention requires fresh v4 cases.

This is not a hypothesis test, an end-to-end OpenMemory benchmark, a persistence test, or evidence about retrieval quality. The recall values are deterministic fixtures and OpenMemory usage remains incomplete.
