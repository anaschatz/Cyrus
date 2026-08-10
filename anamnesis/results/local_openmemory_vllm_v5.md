# OpenMemory + vLLM v5 schema compatibility — diagnostic only

The frozen two-call post-fix compatibility gate **PASS**. This validates the
additive aligned JSON Schema; it is not a rerun of v4, a recall-quality result,
or a hypothesis test. Provider API cost is `$0.0`;
electricity, hardware and human review are unmeasured.

| Case | Expected | Structured accepted | Semantic pass | Error stage |
|---|---|---:|---:|---|
| omv5_emit_cobalt_sheet | emit | true | true | none |
| omv5_no_action_courtyard | no_action | true | true | none |

Total usage: 913 input and
132 output tokens. Exactly two calls were authorized,
with no retry, repair, cache, alternate schema, or v4-case reuse.
