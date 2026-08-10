# Local writer W2 diagnostic — not a hypothesis test

This development-only diagnostic evaluates the W2 LLM memory writer. It is not hypothesis-test evidence. The gate is computed only from deterministically replayed due-candidate multisets. Summary text and runtime-local intent/occurrence IDs are excluded; final decision actions are diagnostic-only.

| Calls | Parse invalid | Semantic invalid | Accepted | Candidate TP | FP | FN | Precision | Recall | F1 | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 46 | 4 | 1 | 41 | 0 | 3 | 8 | 0.0% | 0.0% | 0.0% | FAIL |

| Compiler tokens in/out | Decision tokens in/out | Total tokens in/out | Compiler ms | Decision ms | Local ms | Total ms | Setup ms | API cost USD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60675/3634 | 47577/920 | 108252/4554 | 228776.0 | 119176.4 | 35.1 | 347987.5 | 13937.4 | 0.000000 |

Final-action diagnostic (excluded from the gate): TP=0, FP=1, FN=8, F1=0.0%.

Gate: zero parse/semantic invalid deltas, 46/46 accepted deltas, zero candidate false positives, and zero candidate false negatives. Provider API cost is exactly zero; electricity and hardware cost are unmeasured.
