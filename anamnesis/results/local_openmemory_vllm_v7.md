# OpenMemory canonicalizer v7 prospective diagnostic — not a hypothesis test

The one authorized fresh paired run **PASS**.
Both arms used the same frozen source-grounded canonicalizer; raw decisions,
canonical decisions and transformations are retained. This is a development
diagnostic over the local indexed OpenMemory-compatible boundary, not the
upstream Cavira SDK or a hypothesis test.

| Case | Retrieved memory | Baseline correct | Recall correct | Changes B/R |
|---|---|---:|---:|---:|
| omr2_foundry_reviewer | omr2_mem_foundry | false | true | 0/1 |
| omr2_geology_shipment | omr2_mem_cores | false | true | 2/2 |
| omr2_algae_project | omr2_mem_algae | false | true | 1/1 |
| omr2_current_office_wins | omr2_mem_old_office | true | true | 1/1 |
| omr2_injection_control | omr2_mem_injection | true | true | 0/0 |
| omr2_no_hit_control | none | false | false | 0/0 |

Retrieval 6/6; baseline
2/6; recall 5/6;
helpful gain 3; safety regressions
0; accepted calls
12/12; canonicalizer transformations
9. Usage: 5562 input
and 1224 output tokens at `$0.0`
provider API cost. Electricity and hardware are unmeasured.
