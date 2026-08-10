# W3-M2 local model-only preflight — failed

This development diagnostic changed only the local model: the frozen W3
prompt, response schemas, C1-C8/D1 fixture, seed 101, temperature 0, context
4096, and no-cache/no-retry/no-repair policy were unchanged. The byte-pinned
`ollama/qwen3.5:9b-q4_K_M` artifact (Qwen35 9.7B, Q4_K_M) was run exactly once
from clean source commit `a3f4338a5f463e8481659304246985d014d43a8a`.

The local runtime and residency checks passed: Ollama 0.31.1 served the exact
manifest digest `6488c96f…3ea7` on `127.0.0.1`, cloud access was disabled, the
model used Metal with context 4096, and the resident allocation reported
5,649,538,743 bytes. Provider API cost was exactly `$0.00`; electricity,
hardware amortization, and human time were not measured.

The semantic gate failed. All eight compiler calls consumed the available
context and returned `stop_reason=max_tokens` with no usable completion, so all
eight were parse-invalid. The final D1 decision call returned the exact
`{"mode":"no_action","actions":[]}` output and passed.

| Case | Role | Parse valid | Semantic valid | Input tokens | Output tokens | Latency ms |
|---|---|---:|---:|---:|---:|---:|
| C1 | compiler | no | no | 1,948 | 2,148 | 244,970.4 |
| C2 | compiler | no | no | 1,957 | 2,139 | 300,789.1 |
| C3 | compiler | no | no | 2,014 | 2,082 | 303,982.6 |
| C4 | compiler | no | no | 1,990 | 2,106 | 312,452.4 |
| C5 | compiler | no | no | 2,231 | 1,865 | 332,880.6 |
| C6 | compiler | no | no | 2,287 | 1,809 | 259,907.3 |
| C7 | compiler | no | no | 1,990 | 2,106 | 270,257.5 |
| C8 | compiler | no | no | 2,505 | 1,591 | 224,248.9 |
| D1 | decision | yes | yes | 1,943 | 10 | 219,456.4 |

Aggregate usage was 18,865 input and 15,856 output tokens. Total setup latency,
including the residency probe, was 2,469,096.5 ms (about 41.15 minutes). There
were exactly nine raw ModelEvents and no retries, cache hits, repair calls, or
scenario calls.

W3-M2 is rejected at preflight. Per the frozen stopping rule, no scenario
dataset was created or evaluated for this cell, and this result does not
authorize a prompt, schema, context-window, or decoding-policy adjustment on
the same fixture. It is not a hypothesis test and says only that this exact
model/runtime/transport combination is incompatible with the frozen W3 gate.
