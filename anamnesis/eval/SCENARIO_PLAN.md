# Scenario plan and development-candidate record for v0

The 10 records in `eval/scenarios/smoke.jsonl` remain development fixtures for
the harness. They are not evidence for or against the hypothesis on their own.
The corrected v0.1.2 candidate contains 50 seven-day scenarios: 35 visible
development scenarios (including the smoke set) and 15 holdout-shaped
scenarios used to test the split and final-run guards.

The current 15-record holdout candidate is **not eligible for the
preregistered hypothesis test**. It was inspected while the payload schema and
memory compiler contract were still changing. It must be replaced by a fresh,
externally held and independently reviewed 15-scenario set after the system is
frozen. Keeping the candidate in the repository is useful for deterministic
harness testing, but calling it a valid sealed result set would be research
leakage.

## Candidate files

| File | Role | Scenarios | Events | Expected | Forbidden | SHA-256 |
|---|---|---:|---:|---:|---:|---|
| `eval/scenarios/dev.jsonl` | prompt/configuration development | 35 | 219 | 24 | 58 | `790a09e0377cac3589eec2f70d0d034c67eaa82c7c860262ea7059ac0dbf1ce8` |
| `eval/scenarios/sealed.jsonl` | holdout-shaped harness candidate; replace before final | 15 | 77 | 8 | 23 | `d9241ce5fdf453bf4fbd4bb6036d37a7747fd568c1a8106129c858cc8ac843fc` |
| `eval/scenarios/all.jsonl` | candidate combined ordering (`dev` then holdout candidate) | 50 | 296 | 32 | 81 | `79965f548ff7382dc45bae9a3f078d628a32953ec301ded0e02d8fd6d737d3f1` |

The adjacent `*.manifest.json` files are authoritative for canonical hashes,
per-record hashes, family allocation, counts, origin, and review state. The
candidate has passed automated integrity checks. Independent human review of
gold actions, forbidden traps, and provenance remains explicitly pending. The
combined and holdout manifests also mark this candidate as ineligible for a
preregistered final run; neither claim may be changed without a genuinely new
post-freeze holdout and a recorded review.

## Target composition

| Family | Total | Development | Sealed | Required coverage |
|---|---:|---:|---:|---|
| Basic deadlines | 8 | 6 | 2 | unmet and completed before deadline |
| Deadline/time updates | 8 | 5 | 3 | extension, shortening, multiple updates |
| Fact/action-parameter updates | 8 | 5 | 3 | address, recipient, entity, quantity |
| Conditional event triggers | 8 | 6 | 2 | conjunction, threshold, exact entity |
| Completion and cancellation | 6 | 4 | 2 | before and exactly at the boundary |
| Recurring intentions | 6 | 4 | 2 | mixed action/no-action occurrences |
| Negative controls | 6 | 5 | 1 | brainstorming, explicit non-request, distractors |
| **Total** | **50** | **35** | **15** | |

The candidate contains 20 scenarios with an explicit obsolete-memory
trap and 22 scenarios requiring no action. Surface wording, entity names,
weekdays, and event order vary across the development and sealed portions
without changing the preregistered family distribution.

## Authoring contract

- Every event has a unique timestamp and is an observable decision checkpoint.
  The model is called once after that event is applied. A completion exactly at
  a deadline is represented by the completion event at that timestamp, without
  an additional same-time clock event.
- Timestamps include a UTC offset and events are ordered chronologically.
- `action_key` is the ID of the event that created the intention; later updates
  do not create a new action key. `supersedes` means that an explicit revision
  or cancellation makes the referenced intention, schedule, payload field, or
  fact value no longer current. Completing a condition does not by itself
  supersede the conditional intention; it supplies a current blocker fact.
- Expected actions define an exact structured payload, inclusive execution
  window, and one or more exact acceptable evidence-ID sets.
- Payloads use a canonical closed grammar. `subject` is required and is a
  lowercase imperative verb phrase containing a verb and direct object. The
  only optional keys are `address`, `build`, `date`, `flight`, `greenhouse`,
  `item`, `project`, `quantity`, `recipient`, `room`, `shipment`, `tank`, and
  `trip`. Every recurring occurrence has an ISO `YYYY-MM-DD` `date`; weekday
  names are not payload fields.
- Evidence follows the minimal-causal policy: each set contains the creating
  event, the exact decision checkpoint, and only the intervening facts or
  revisions needed to justify the payload and trigger. No accepted set may be a
  strict superset of another accepted set. If the action depends on an explicit
  current fact, the event that asserted that fact is required. An absent or
  unknown blocker contributes no evidence source. When independent observations
  form genuinely alternative minimal traces, all alternatives are retained.
- Forbidden actions use a disjoint taxonomy: `obsolete` is reserved for a
  superseded or cancelled intention, schedule, or payload; completion and other
  true blockers are `condition_satisfied`; early or unsatisfied triggers are
  `premature`; re-emission after an eligible action is `duplicate`; and actions
  without an active matching intention are `unrequested`. A scenario cannot
  contain duplicate forbidden signatures for the same action, payload, and
  window.
- Gold and `supersedes` annotations are hidden from model context.
- An acceptable evidence set cannot contain an event after the action window.
- Scenario writers must not inspect final-system behavior while editing sealed
  gold labels.

## Data provenance

All 50 candidate scenarios are locally authored and retain an author/reviewer
trail in the split manifests. No LongMemEval content was imported because a
repository revision and license record were not already pinned locally. No
TriggerBench content is used. A future externally derived case must record its
upstream item ID, repository revision, license, and exact transformation before
entering a development split.

The v0.1.2 changes are automated semantic corrections, not human review.
Before the final run, a second reviewer must independently inspect every gold
action, forbidden trap, and provenance set in a fresh post-freeze holdout. Any
correction creates a new dataset version and new hashes; it may not silently
mutate a frozen dataset. The current holdout-shaped split remains excluded from
development task defaults and cannot pass the final-run eligibility guard.
