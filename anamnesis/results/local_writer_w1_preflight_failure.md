# Local W1 writer preflight failure — no scenario run

The W1 memory-compiler prompt was tested once with the pinned local
`ollama/qwen3:4b-instruct` candidate. The synthetic two-call Inspect job had
technical status `success`, but the semantic preflight result was
`passed=false`. Therefore no writer-diagnostic scenario was executed, no
experiment manifest was frozen, and no writer result table was produced.

This is a development-only diagnostic. It is not evidence for or against the
Anamnesis research hypothesis.

## Frozen identity

| Item | Value |
|---|---|
| Source commit | `8d99f68ea185614f2f5da1b16a011be91961a5c7` |
| Blind dataset-freeze commit | `df2d5579528efe4f328550170f4edc1034e334da` |
| Blind W1 prompt commit | `65ab0657a90d958229ecda0a89932aef24f82fb8` |
| Model | `ollama/qwen3:4b-instruct` |
| Ollama model digest | `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0` |
| Compiler prompt SHA-256 | `4a1f6ece3a1a72e98b54f91433039b6d41ff78e766969852a5498916909d1f60` |
| Compiler schema SHA-256 | `8871ff344eb3a2e88a53b964ef2f24f089a72507c69073ec323cf26a428c3030` |
| Decision prompt SHA-256 | `871fe15e3160e66abe7480cbde15dfb943dec2d0ff89bb01a03849ad35defd8d` |
| Decision schema SHA-256 | `1b7c38d3f4bf150523ecc1e468ad3fb1f94753611f190d70f93abbf5ec582426` |
| Seed / temperature | `101` / `0` |
| Provider API cost | `$0.00` |

The exact ignored preflight log is
`results/runs/local/writer_diagnostic/preflight/2026-08-07T22-18-25-00-00_local-model-preflight_HXz7rGqJHvw53spggddpjS.eval`, with SHA-256
`1626afdc93170383390bb4627e61ffa4ced2ea5e8df065c674cd4364781623ce`.
It records clean revision `8d99f68`, Ollama `0.31.1`, context length 4096,
Q4_K_M quantization, and the pinned loopback route.

## Outcome

| Component | Parse | Semantics | Input tokens | Output tokens | Latency |
|---|---:|---:|---:|---:|---:|
| W1 compiler | FAIL | FAIL | 981 | 216 | 11,632.1 ms |
| D1 decision | PASS | PASS | 679 | 12 | 2,623.9 ms |

All usage and cost fields were complete. The residency probe took 3.2 ms. The
semantic artifact validator independently rejected the log with
`local semantic preflight result did not pass`.

## Exact failure

The compiler returned schema-shaped JSON and correctly created a reminder at
`2026-01-05T17:00:00+00:00` with subject `perform compatibility check`.
However, it populated every unused optional action-payload slot with an empty
string (and `quantity` with zero) instead of omitting those slots. In
particular, it returned `date: ""`. The wire object parsed, but conversion to
the provider-neutral `ActionTemplate` failed because `date` must be an ISO
`YYYY-MM-DD` value whenever present. The entire delta was therefore rejected.

The decision preflight independently returned the required no-action output
and passed. This isolates the failure to the local compiler's structured
payload generation, not the decision schema or the deterministic memory
engine.

## Stopping decision

The preregistered writer protocol says that a failed semantic preflight ends
the Qwen 4B candidate attempt and does not authorize scenario calls or output
repair. That rule was followed:

- writer scenarios evaluated: **0 / 10**;
- writer compiler calls on scenario data: **0 / 45**;
- W1 result manifest: **not frozen**;
- W1 candidate status: **rejected at preflight**.

The frozen writer dataset and oracle reference were not used for model
evaluation. A scientifically valid next attempt would keep the frozen W1
prompt/schema and test a separately pinned, stronger local model through a new
synthetic preflight. It must not reinterpret this failed preflight as a
scenario result.
