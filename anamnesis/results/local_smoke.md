# Local smoke diagnostic — not a hypothesis test

| System | TP | FP | FN | Precision | Recall | F1 | False reminders | FAR | Obsolete | Provenance | Input tokens | Reduction vs full | Provider API cost USD | Latency p50/p95 ms | Setup ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anamnesis | 0 | 7 | 8 | 0.0% | 0.0% | 0.0% | 7 | 8.6% | 0 | N/A | 81617 | -23.5% | 0.000000 | 18012.8/31653.0 | 36797.2 |
| full_context | 0 | 1 | 8 | 0.0% | 0.0% | 0.0% | 1 | 1.4% | 0 | N/A | 66078 | 0.0% | 0.000000 | 6623.0/18459.7 | 11007.9 |
| no_memory | 0 | 1 | 8 | 0.0% | 0.0% | 0.0% | 1 | 1.4% | 0 | N/A | 39048 | 40.9% | 0.000000 | 2025.5/3656.9 | 8113.8 |
| vector_rag | 0 | 1 | 8 | 0.0% | 0.0% | 0.0% | 1 | 1.4% | 0 | N/A | 53721 | 18.7% | 0.000000 | 11403.3/15189.3 | 65864.1 |

Provider API cost is zero. Electricity and hardware cost are unmeasured. Setup latency is reported separately.
