# Local OpenMemory immediate-action diagnostic v3 execution

The fresh v3 artifact was frozen in commit `d9752079762a1aa00d6fd2b9feadaae97fdb763c` before this decision contract and task were authored.

## Treatment

V2 used a temporal-reminder firing prompt for immediate commands. V3 replaces only that mismatched decision contract with a dedicated immediate-action contract. It explicitly permits a current command to act now, while retaining a narrow recall boundary: recall may fill an optional argument of that explicit action, but cannot create, suppress, cancel, schedule or prove an action, override current text, or supply evidence.

The response schema, byte-pinned 9B model, no-thinking transport, context limit, seed 101, temperature 0, cache/retry/repair policy, call order and paired scorer remain unchanged. The exact paired prompt contract SHA-256 is `1505cfc4df8be3812d6e7f0ef53a1245d2ec82e3865b0e709476f9001d21754e`.

## Frozen execution

- Eight cases, baseline then recall, exactly 16 calls.
- One attempt only; no selected duplicate, retry, repair or prompt modification.
- No live OpenMemory writes; recall hits are a frozen search-only fixture.
- Exact raw completions, parse status, tokens, latency and zero provider API cost retained.
- Retrieval usage is incomplete; electricity and hardware cost are unmeasured.

The paired gate requires all helpful opportunities to improve, zero forbidden-influence and no-hit regressions, zero recall-induced false actions, and zero evidence contamination. Preserve a valid pass or fail. Once any v3 model call begins, no second v3 run is permitted; any further intervention requires fresh v4 cases.

This remains a development diagnostic, not a hypothesis test, temporal-memory benchmark, persistence test, or general OpenMemory evaluation.
