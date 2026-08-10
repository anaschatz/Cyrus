# W3-M2-T1 transport-only preflight — failed

This development diagnostic changed only the effective OpenAI-compatible
Ollama transport. The byte-pinned W3-M2 model, W3 prompt and response schemas,
C1-C8/D1 fixture, context and output budgets, seed 101, temperature 0, and
no-cache/no-retry/no-repair policy were unchanged. Every raw request carried
`extra_body={"reasoning_effort":"none"}`, which the Ollama compatibility
endpoint receives as top-level `reasoning_effort: "none"`. The cell ran exactly
once from clean source commit `ef13b6d222a44435c323db3a2c4b8361964c7111`.

The transport intervention fixed the immediate W3-M2 truncation failure. All
eight compiler calls returned non-empty final content without a `max_tokens`
stop, instead of consuming the entire budget in hidden reasoning. The total
setup time fell from about 41.15 minutes in W3-M2 to about 8.14 minutes here.

The semantic gate nevertheless failed. All eight compiler outputs were JSON
text but none matched the frozen `LocalMemoryDeltaWire`: early cases invented
alternate names such as `entity_id`, `attribute_id`, `id`, and `payload`, while
later update cases copied internal stored revisions, provenance, and status
fields rather than emitting a partial mutation. C7 also emitted unsupported
action-template structure and filler values. The D1 decision call returned the
exact no-action decision and passed.

| Case | Role | Parse valid | Semantic valid | Input tokens | Output tokens | Latency ms |
|---|---|---:|---:|---:|---:|---:|
| C1 | compiler | no | no | 1,950 | 96 | 28,261.0 |
| C2 | compiler | no | no | 1,959 | 188 | 16,807.8 |
| C3 | compiler | no | no | 2,016 | 263 | 23,497.3 |
| C4 | compiler | no | no | 1,992 | 232 | 38,307.5 |
| C5 | compiler | no | no | 2,233 | 447 | 117,386.0 |
| C6 | compiler | no | no | 2,289 | 588 | 97,241.9 |
| C7 | compiler | no | no | 1,992 | 174 | 35,511.7 |
| C8 | compiler | no | no | 2,507 | 609 | 121,532.4 |
| D1 | decision | yes | yes | 715 | 19 | 9,724.9 |

Aggregate usage was 17,653 input and 2,616 output tokens. Total setup latency,
including the residency probe, was 488,280.5 ms (about 8.14 minutes). Ollama
0.31.1 served the exact Qwen35 9.7B Q4_K_M manifest on `127.0.0.1`, with cloud
access disabled and context 4096. Provider API cost was exactly `$0.00`;
electricity, hardware amortization, and human time were not measured.

There were exactly nine raw ModelEvents and no retries, cache hits, repair
calls, or scenario calls. W3-M2-T1 is rejected at preflight. Per the frozen
stopping rule, no scenario dataset was created or evaluated and no second T1
attempt is permitted. The result isolates a second incompatibility: disabling
thinking solves output exhaustion, but this exact Ollama/Qwen3.5 transport does
not enforce or elicit the frozen compiler schema. Any subsequent schema or
transport treatment must be a new preregistered cell; it cannot be presented as
a repair or rerun of T1.
