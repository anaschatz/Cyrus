# Local writer W3 semantic preflight fixture

`eval/preflight/local_writer_w3.v1.json` is the diagnostic-only, post-prompt
semantic preflight fixture for the frozen W3 bundled compiler repair. It was
authored after prompt commit
`a9fb1602158c4545f6791c296dfc05d8decc7d90` by a separate author who did not
inspect any writer-diagnostic scenario or oracle content. The fixture is not a
result, contains no model output, and is not hypothesis-test eligible.

The fixture implements the order and acceptance projections frozen earlier in
`eval/preflight/local_writer_w3.protocol.v1.json`. Its event texts, dates,
entities, actions, and IDs were authored from that protocol only. All dates are
in 2034. The proper nouns and complete event texts are disjoint from the public
W1/W2 synthetic preflights and the frozen W3 instruction text. The completed
dataset-custodian audit also verifies, without disclosing v4 material, that full
inputs, IDs, dates, and actions are disjoint from the hash-pinned v4 dataset and
oracle.

## Ordered cases

The model gate is exactly eight compiler calls followed by one decision call:

| Order | Exact neutral boundary | Semantic acceptance |
| --- | --- | --- |
| C1 `normalization_fact` | A multiword Sapphire Observatory battery fact with typed numeric value `73` and no unit. | Exactly one fact at `sapphire_observatory.backup_battery_level`; unit omission and null are equivalent; no other mutation. |
| C2 `bare_weekday_at` | A Monday event requests Monday `09:10` after that day's time has passed and explicitly names item `Aurora Spectrometer`. | Exactly one `at` create for the first strictly future Monday, `2034-08-28T09:10:00+10:00`, with subject `calibrate equipment` and the exact item slot. |
| C3 `condition_transition_and` | A bounded lunar-archive request has two explicit AND conditions. | Exactly one condition-transition create with the exact active window and both normalized conditions as a canonical multiset; no synthetic negative. |
| C4 `recurrence_iana_range` | A Tuesday/Saturday schedule gives an exact range, `America/Toronto`, and item `Zephyr Antenna`. | Exactly one recurring create with exact local time, weekday set, dates, timezone, subject, and item slot. |
| C5 `stable_id_trigger_update` | One active Polaris-lens intent is moved to a new instant. | Exactly one trigger-only update using active ID `align_polaris_lens` character-for-character. |
| C6 `full_action_template_update` | One active Meridian-capsule intent changes only its recipient. | Exactly one action-template-only update using active ID `dispatch_meridian_capsule`, preserving subject, project, room, and summary. |
| C7 `complete_sparse_payload_including_zero` | A supply handoff sources item, project, quantity `0`, recipient, and room. | Exactly one create with every sourced value, original casing, numeric zero, and no trigger-only date or filler. |
| C8 `ambiguous_empty` | Two active prism-catalog intents are equally plausible cancellation targets. | All four wire mutation arrays and the domain delta are empty; no guessed ID. |
| D1 `no_action` | Empty structured memory accompanies an irrelevant basalt-ridge observation. | Wire mode is `no_action`; wire and domain action lists are empty. |

Every case contains an exact `input`, one schema-valid `valid_wire_example`, its
strictly converted `valid_domain_example`, the frozen protocol
`acceptance_projection`, and a concrete `acceptance` object. The examples are
not exact-output gold strings. Acceptance is evaluated after strict wire parsing
and domain conversion.

For create cases, `intent_id` and summary remain structural-only because their
exact surface form is not semantically determined. Any schema-valid alternative
passes when all frozen semantics match. Update IDs are exact and must be copied
from active state. Unused optional payload slots and unchanged update fields may
be omitted or null because both forms reduce to the same domain object. Sourced
payload leaves, triggers, conditions, typed values, and update targets are exact.
C3 conditions are compared as a canonical multiset, and C4 weekdays are
compared as a set; reversing either valid serialization does not change its
semantic acceptance.

The C5, C6, and C8 active-state strings are canonical JSON emitted in the same
shape as the memory store. They contain no facts and only the one or two active
intents needed to resolve—or deliberately fail to resolve—the current request.
All other compiler cases use the canonical empty state.

## Completed custodian audit

The fixture persists the custodian result as `status=passed` with raw and
canonical SHA-256 pins for both v4 artifacts. The test suite reloads those exact
artifacts, verifies all four hashes, and performs set-disjointness checks over
canonical full inputs, identifier values, calendar dates, and canonical action
payload/template fingerprints. The comparison reports only pass/fail by
dimension and never places v4 material in an assertion message, fixture field,
or model prompt.

## Model boundary and validation

Only each case's `input` reaches a prompt builder. Compiler inputs contain one
exact `ObservableEvent` and one exact canonical active-state string. The
decision input contains exactly the five decision-builder arguments. Case IDs,
roles, categories, acceptance objects, examples, and projections never enter
either prompt. Tests mutate every case field outside `input`—for compiler and
decision cases—and prove that the rendered prompt is unchanged.

The test suite validates all example wire-to-domain reductions and explicitly
accepts order-only reversals of the C3 condition multiset and C4 weekday set. It
rejects:

- non-null filler and trigger-only payload dates;
- missing sourced leaves or changed numeric zero;
- wrong normalized keys, value types, units, update IDs, triggers, dates,
  recurrence fields, or IANA timezone;
- a missing AND conjunct or a synthetic negative;
- an incomplete compound update or an extra top-level update field;
- any extra independent compiler mutation;
- a guessed target in C8; and
- an emitted action in D1.

It also proves the fixture path did not exist in the prompt commit and that the
prompt commit is an ancestor of the fixture work, freezing the required
post-prompt authorship order.

## Frozen hashes

| Artifact or contract | SHA-256 |
| --- | --- |
| W3 fixture bytes | `5628c3c1d7f8e1a5da43d6e567d55ac8e4fbabd8b9c4054325de6f4def1da30c` |
| W3 protocol bytes | `7f63c156a8af74ced2d5e5530b3e8083da95c7f46e14e1afafaaf864b3ce1915` |
| v4 dataset raw bytes | `6b2530cb9f3426c792500f07e854d7f31ad84081ac77104cb8032737234ff91c` |
| v4 dataset canonical semantics | `ee80a55874ac6d6cfd5ee32484d91113bff78d829d66c9ff46bcb646456eb598` |
| v4 oracle raw bytes | `72308bb34bda758cc72dc651e3f0fd2fd2bd1bff820479e2cf0774ee8d66cf5c` |
| v4 oracle canonical semantics | `b877bcd6fe15767d9f1bb42a5840a799d2ef5a4a3691eb6a59ae2f9f7d40813b` |
| W3 addendum | `84897bc8493dc4c89272aacd9ec6aaf869de92e63b1e225b954d97af84877793` |
| W3 sentinel prompt | `412a63d6b42ea6b5e294401cabbcbacf5a6b7facddbd8fe04ca7b91914c141e5` |
| W3 local-wire contract | `b90298df967f81c91cd6aed6289190768b1f4fe28af4743fb118920d11f8ec51` |
| Unchanged local wire-model schema | `f0e0ab9c3aef10f9b99ca5055d1ee1f2e6d7f091be666ee95035040e564302ec` |
| Unchanged Inspect compiler response schema | `8871ff344eb3a2e88a53b964ef2f24f089a72507c69073ec323cf26a428c3030` |
| Unchanged decision prompt | `871fe15e3160e66abe7480cbde15dfb943dec2d0ff89bb01a03849ad35defd8d` |
| Unchanged decision schema | `1b7c38d3f4bf150523ecc1e468ad3fb1f94753611f190d70f93abbf5ec582426` |

Any byte change to the fixture requires a newly named fixture version and a new
documented hash. The protocol still permits exactly one standalone attempt and,
after a pass, one contemporaneous replay inside the single measured task. This
fixture adds no runtime, task, manifest, reporter, retry, repair, or model-call
implementation.
