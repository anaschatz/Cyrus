# Local smoke diagnostic — not a hypothesis test

| System | TP | FP | FN | Precision | Recall | F1 | False reminders | FAR | Obsolete | Provenance | Input tokens | Reduction vs full | Provider API cost USD | Latency p50/p95 ms | Setup ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anamnesis | 0 | 7 | 8 | 0.0% | 0.0% | 0.0% | 7 | 8.6% | 0 | N/A | 95282 | -19.2% | 0.000000 | 19157.8/33991.9 | 18648.1 |
| full_context | 0 | 0 | 8 | 0.0% | 0.0% | 0.0% | 0 | 0.0% | 0 | N/A | 79922 | 0.0% | 0.000000 | 6661.1/13589.7 | 18856.4 |
| no_memory | 0 | 0 | 8 | 0.0% | 0.0% | 0.0% | 0 | 0.0% | 0 | N/A | 53010 | 33.7% | 0.000000 | 1333.6/4200.9 | 6247.1 |
| vector_rag | 0 | 0 | 8 | 0.0% | 0.0% | 0.0% | 0 | 0.0% | 0 | N/A | 67610 | 15.4% | 0.000000 | 4926.9/6523.9 | 29402.4 |

Provider API cost is zero. Electricity and hardware cost are unmeasured. Setup latency is reported separately.
