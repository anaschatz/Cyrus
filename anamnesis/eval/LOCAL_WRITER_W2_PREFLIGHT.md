# Local writer W2 preflight fixture

`eval/preflight/local_writer_w2.v1.json` is a diagnostic-only, post-prompt
preflight fixture for the frozen local W2 writer contract. It is not a result,
does not contain model output, and is not eligible for hypothesis testing. Its
only purpose is to check that a live model can express three elementary
compiler boundaries and one elementary decision boundary through the strict
local wire schemas before any separate evaluation is run.

The four categories and their order were fixed before the neutral event texts,
dates, IDs, and reminder subjects were authored. Version 1 contains exactly
three compiler cases followed by one decision case:

| Order | Category | Frozen acceptance projection |
| --- | --- | --- |
| C1 | Trivial explicit same-day `at` intent | One create with the exact trigger and subject-only domain payload; unused optional wire keys may be omitted or null. |
| C2 | Trivial explicit next-day `at` intent | One create with exactly one explicitly sourced, proper-cased `address` optional payload key. |
| C3 | Irrelevant observation | All four compiler mutation arrays are empty and the domain `MemoryDelta` is empty. |
| D1 | Empty structured memory plus an irrelevant raw event | Strict local decision mode is `no_action`, with an empty action list in both wire and domain JSON. |

The `valid_wire_example` and `valid_domain_example` objects are illustrative
valid serializations, not exact-output gold strings. Acceptance is determined
only by each case's explicit `acceptance` projection after strict wire parsing
and conversion to the domain model. For C1 and C2, both omission and JSON null
pass for every unused optional wire slot because both reduce to the same exact
domain payload. A non-null value in an unsourced optional slot fails.

For the two create cases, the mutation type, trigger, domain payload, empty
conditions and blockers, and reminder kind are exact. The generated
`intent_id` and summary are structural-only: any values accepted by the frozen
wire/domain schemas pass, and they are not compared with the illustrative
examples. C2 treats its next-day wording only as trigger resolution; `address`
is the sole optional action-payload value sourced by that event.

## Frozen contracts

The hashes bind the static sentinel prompt and the exact Inspect response
schema serialized by the runtime. The fixture does not modify or substitute
any runtime component.

| Contract | Version | Prompt SHA-256 | Schema SHA-256 |
| --- | --- | --- | --- |
| Compiler | `local.v0.3` | `024641f8d0ec16168eb9b7d8dbee67f92b7049fe6d35b604495aba273319d9dd` | `8871ff344eb3a2e88a53b964ef2f24f089a72507c69073ec323cf26a428c3030` |
| Decision | `ollama.decision.v0.2` | `871fe15e3160e66abe7480cbde15dfb943dec2d0ff89bb01a03849ad35defd8d` | `1b7c38d3f4bf150523ecc1e468ad3fb1f94753611f190d70f93abbf5ec582426` |

Each compiler input stores the exact `ObservableEvent` plus the exact canonical
active-state string supplied to the prompt builder. The decision input stores
all prompt-builder arguments, including an explicit structured memory view
whose `blocks` array is empty. Valid wire examples are validated through
`LocalMemoryDeltaWire` or `LocalDecisionWire`. The test suite first verifies
that each valid wire example reduces to its paired valid domain example, then
scores candidate variants against the separate semantic acceptance projection.
It explicitly proves that omitted and all-unused-null serializations and
alternative schema-valid intent IDs and summaries pass, while non-null filler
or extra payload values fail.

## Byte freeze and change policy

The SHA-256 of `eval/preflight/local_writer_w2.v1.json` is:

`3b82128bab1d801d073118488aa4f0a0a662603b98325f5c9d7dad497f026057`

`tests/test_local_w2_preflight_fixture.py` freezes that hash, the exact case
count and order, the prompt/schema hashes, the semantic acceptance projections,
and the example wire-to-domain reductions. Any change to an input, acceptance
projection, example, category, or ordering requires a newly named fixture
version and a new documented hash; version 1 remains immutable.
