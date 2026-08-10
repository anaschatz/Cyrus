# Local W2 writer diagnostic analysis

This analysis refers only to the strict one-cell result in
[`local_writer_w2.md`](local_writer_w2.md). W2 is a development-only compiler
diagnostic. It is not a hypothesis test, does not authorize the 35-scenario
development benchmark, and cannot be repaired or rerun on these same 10 cases.

## Decision

Reject W2. Its narrow sparse-payload intervention passed the fresh semantic
preflight and eliminated filler values, but the measured writer still failed
all four preregistered gates:

| Gate | Result |
|---|---|
| Zero parse/semantic invalid deltas | **Fail** — 4 parse, 1 semantic |
| All 46 compiler calls accepted | **Fail** — 41/46 |
| Zero due-candidate false positives | **Fail** — 3 |
| Zero due-candidate false negatives | **Fail** — 8 |

The strict replay produced candidate TP=0, FP=3, FN=8, so precision, recall,
and F1 were all 0%. “Accepted” means only that a delta was valid for the
deterministic store; it does not mean that its interpretation was correct.

## Frozen identity and single-run boundary

The corrected v3 diagnostic dataset and gold-assisted reference were frozen
before the W2 prompt. W2 then added only a sparse serialization invariant to
the preserved W1 compiler rules: unused optional payload slots must be omitted
or `null`; empty strings, false values, empty collections, and placeholder
zeroes are forbidden. The compiler schema and shared D1 decision contract did
not change.

| Item | Value |
|---|---|
| Source commit used for measurement | `9388f58cc1366f09c6134086fb36759272b50dbd` |
| Model | `ollama/qwen3:4b-instruct` |
| Model digest | `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0` |
| Compiler version | `local.v0.3` |
| Compiler prompt SHA-256 | `024641f8d0ec16168eb9b7d8dbee67f92b7049fe6d35b604495aba273319d9dd` |
| Compiler schema SHA-256 | `8871ff344eb3a2e88a53b964ef2f24f089a72507c69073ec323cf26a428c3030` |
| Decision prompt SHA-256 | `871fe15e3160e66abe7480cbde15dfb943dec2d0ff89bb01a03849ad35defd8d` |
| Decision schema SHA-256 | `1b7c38d3f4bf150523ecc1e468ad3fb1f94753611f190d70f93abbf5ec582426` |
| Dataset raw / canonical SHA-256 | `34e2e875...af4f` / `37a62b64...115` |
| Reference raw / canonical SHA-256 | `7adb64ed...857` / `9e46d95b...01f3` |
| Seed / temperature | `101` / `0` |
| Provider API cost | `$0.00` |

The standalone C1/C2/C3/D1 preflight passed all four semantic projections:
subject-only payload, one sourced optional address, irrelevant-event empty
delta, and decision no-action. The measured task then repeated that exact
four-call setup gate and evaluated the 10 scenarios once. There were no model
retries, output repairs, or cached calls in the retained artifacts. Exactly one
authorized measured `.eval` is present.

## Sparse-payload intervention

W2 fixed the exact W1 transport failure:

- C1 emitted only `subject`;
- C2 emitted only `subject` and the sourced `address`;
- C3 emitted an empty delta;
- D1 returned `no_action`; and
- none of the 10 scenario action payloads contained `null`, an empty string,
  `false`, an empty collection, or zero filler.

The earlier invalid `date: ""` did not recur. This is evidence that the narrow
serialization instruction was understood. It did not, however, improve the
model's temporal resolution, trigger selection, identifier normalization, or
intent-reference stability. In several cases it also omitted legitimate
source-specific payload fields, so sparse output alone was insufficient.

## Scenario-level forensic replay

| Scenario | Family | Accepted / parse / semantic | Candidate result | Primary failure |
|---|---|---:|---:|---|
| `wd3_01` | Basic deadline | 4 / 0 / 0 | FP1, FN1 | Resolved Thursday as Apr 5 instead of Apr 8 and truncated the payload. |
| `wd3_02` | Cancellation | 4 / 0 / 0 | None | Used May 7 instead of May 8; cancellation occurred before either date and masked the timing error at candidate level. |
| `wd3_03` | Conjunction | 6 / 0 / 0 | FN1 | Used a past/current `at` trigger instead of `condition_transition` and omitted required facts. |
| `wd3_04` | Deadline update | 4 / 0 / 1 | FP1, FN1 | Created an immediate trigger; later update targeted a nonexistent intent ID. |
| `wd3_05` | Entity grounding | 4 / 1 / 0 | FN1 | Initial intent failed domain conversion because the entity key contained spaces. |
| `wd3_06` | Fact/action update | 3 / 1 / 0 | FP1, FN1 | Initial due date was wrong; the update was atomically rejected because `unit` was empty. |
| `wd3_07` | Negative control | 3 / 0 / 0 | None | Correctly created no intent. |
| `wd3_08` | Recurrence | 3 / 1 / 0 | FN1 | Invalid blocker entity and incorrect one-shot/condition representation replaced recurrence. |
| `wd3_09` | Reversible completion | 4 / 1 / 0 | FN1 | Invalid condition entity, incorrect deadline/polarity, and missed completion revisions. |
| `wd3_10` | Threshold transition | 6 / 0 / 0 | FN1 | Used `at(current)+condition`, creating a suppressed one-shot occurrence instead of a transition trigger. |

Across all 46 calls, 24 accepted deltas were empty and 17 accepted deltas were
non-empty.

## Invalid deltas

All four “parse invalid” outputs were valid JSON and passed the local transport
wire. They failed only during conversion to the stricter provider-neutral
domain model:

| Event | Domain failure |
|---|---|
| `wd3_05_e01` | `FactKey.entity="cedar incubator"` violated the normalized identifier pattern. |
| `wd3_06_e04` | `SetFact.unit=""` violated the non-empty unit constraint. |
| `wd3_08_e01` | `FactKey.entity="lagoon reef salinity"` contained spaces. |
| `wd3_09_e01` | `FactKey.entity="theater sponsorship letter"` contained spaces. |

The one semantic/store-invalid delta was `wd3_04_e03`. It tried to update
`rem_1100_fri`, while the actual active intent ID was `rem_1100_wed`; the
deterministic reducer correctly rejected the whole delta as atomic.

This exposes a contract-ergonomics problem: the transport schema permits some
strings that the domain schema rejects. The run remains valid because the
frozen evaluator explicitly counts those conversions as writer-invalid output.
Changing that schema now would be a new intervention.

## Candidate and decision replay

Independent fresh-store replay found 3 measured due candidates and 8 reference
candidates, with an empty multiset intersection:

| Checkpoint | Measured due candidate | Why it was false |
|---|---|---|
| `wd3_01_e02` | Apr 5 at 16:30, payload `submit narration` | Required Apr 8 and a complete payload. |
| `wd3_04_e01` | Jul 12 at 09:35 | Immediate creation-time trigger instead of the Jul 16 updated deadline. |
| `wd3_06_e02` | Oct 4 at 10:00 | Required Oct 9 with the updated address. |

The eight false negatives were the required occurrences in `wd3_01`, `wd3_03`,
`wd3_04`, `wd3_05`, `wd3_06`, `wd3_08`, `wd3_09`, and `wd3_10`.

Decision output is excluded from the writer gate. For completeness, the shared
decision model made 69 parse-valid calls, returned 68 no-action decisions, and
emitted only the false `wd3_04_e01` candidate. Its diagnostic action score was
TP=0, FP=1, FN=8, F1=0%.

## Usage and latency accounting

| Component | Input tokens | Output tokens | Latency |
|---|---:|---:|---:|
| Compiler, 46 scenario calls | 60,675 | 3,634 | 228,775.977 ms |
| Decision, 69 scenario calls | 47,577 | 920 | 119,176.418 ms |
| Local deterministic work | — | — | 35.117 ms |
| Headline scenario total | 108,252 | 4,554 | 347,987.512 ms |
| Four embedded measured-task setup calls, excluded | 4,020 | 356 | 13,937.426 ms |

Raw usage in the measured `.eval` is therefore exactly 112,272 input and 4,910
output tokens. The separate standalone preflight used the same 4,020/356 token
counts and took 25,438.072 ms; it is a distinct artifact and is not included in
either measured headline or embedded setup latency. All usage and latency fields
reconcile with their raw calls. Provider API cost was exactly `$0.00`;
electricity, hardware amortization, and human annotation effort were not
measured.

Latency remains descriptive. This one local-model cell was not designed as a
cross-system latency benchmark.

## Stopping rule and next valid experiment

W2 must remain frozen as a failed diagnostic. No output repair, prompt edit, or
second W2 run is permitted on these v3 cases. In particular, the observed
normalization, timing, and intent-ID failures cannot now be converted into a
W2 patch and retested on the same data.

A valid next compiler experiment requires a newly authored and frozen dataset,
new reference artifact, separately named intervention, fresh post-prompt
preflight, clean source commit, and one declared run. Alternatively, the
preserved W1 prompt could be tested with a separately pinned stronger local
model as a model ablation. Neither path opens the 35 development scenarios or
the final holdout.

## Reproducibility

The [provenance sidecar](local_writer_w2.provenance.json) binds the source
commit, dataset/reference, prompt/schema contracts, model/pricing pins, frozen
manifest, standalone preflight, measured `.eval`, CSV, and generated Markdown
by SHA-256. Unlike earlier local diagnostic publications, the exact W2 manifest
and both raw logs are committed under
`results/runs/local/writer_diagnostic_w2/`, so their exact evidence bytes are
available from Git. Strict reporter replay must use the clean source commit
`9388f58cc1366f09c6134086fb36759272b50dbd` with these three artifacts restored
from the later publication commit; the reporter correctly rejects the later
documentation-only commit as a different runtime source tree. The raw Inspect
artifacts retain local absolute filesystem paths but contain no API keys,
credentials, or bearer tokens.
