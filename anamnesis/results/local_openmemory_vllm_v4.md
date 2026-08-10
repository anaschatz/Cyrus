# OpenMemory + vLLM v4 local diagnostic — not a hypothesis test

The frozen v4 joint model-artifact + structured-runtime compatibility cell **FAIL**.
This is a development diagnostic, not a causal comparison with Ollama and not a
hypothesis test. Hardware, electricity, model acquisition and human review are
unmeasured; provider API cost is exactly `$0.0`.

| Metric | Value |
|---|---:|
| Status | complete |
| Cases | 8 |
| Baseline correct | 4 |
| Recall correct | 4 |
| Helpful gain | 0 |
| Safety regressions | 0 |
| No-hit regressions | 0 |
| Recall false actions | 0 |
| Recall evidence contaminations | 0 |
| Structured-invalid scenario calls | 7 |
| Setup tokens (input/output, excluded) | 458 / 19 |
| Headline tokens (input/output) | 7430 / 1943 |

The frozen gate requires the canary and every structured call to validate,
one helpful recall gain, and zero safety, no-hit, false-action, or evidence
contamination regressions. No retry, repair, alternate artifact, or selected
duplicate run is permitted on these v4 cases.
