# Local oracle-compiler ceiling — diagnostic only

This is a gold-assisted diagnostic ceiling for the frozen oracle compiler. It is not a headline Anamnesis result, is not hypothesis-test eligible, and has no success gate or baseline comparison.

| System | TP | FP | FN | Precision | Recall | F1 | False reminders | FAR | Obsolete | Provenance | Decision input tokens | Oracle compiler tokens in/out | Provider API cost USD | Latency p50/p95 ms | Setup ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anamnesis_oracle_compiler | 7 | 1 | 1 | 87.5% | 87.5% | 87.5% | 1 | 1.4% | 0 | 0.0% | 40986 | 0/0 | 0.000000 | 1502.7/5410.8 | 6297.0 |

Scenario compiler tokens and provider API cost are exactly zero because the compiler replays frozen oracle annotations locally. Human annotation effort is unmeasured, so the reported token scope is a decision-only lower bound. Electricity and hardware cost are also unmeasured. The two setup preflight model calls are excluded from headline usage; setup latency is reported separately.
