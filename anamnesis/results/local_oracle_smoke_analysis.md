# Local oracle-compiler ceiling analysis

This analysis refers only to the gold-assisted 10-scenario diagnostic in
[`local_oracle_smoke.md`](local_oracle_smoke.md). It is not an Anamnesis
headline result, is not hypothesis-test eligible, and does not justify opening
the 35-scenario development set or any sealed data.

## Decision

The deterministic memory path is capable of representing and triggering every
required action in the smoke set when compilation is correct. The current Qwen
4B decision contract is the remaining bottleneck: it converted 7 of 8 exact due
candidates, created one false alarm, and failed exact provenance on all 7
matched actions.

The next experiment should therefore be a new, frozen decision-prompt version
on the same 10 open smoke scenarios. It should not change the decision schema,
memory engine, scorer, dataset, or thresholds.

## Strict result

- 10 scenarios, 8 expected actions.
- 7 true positives, 1 false positive, and 1 false negative.
- Precision 87.5%, recall 87.5%, and F1 87.5%.
- One false-alarm checkpoint, zero obsolete-memory errors, and zero invalid
  outputs.
- Exact provenance: 0 of 7 matched actions.
- 40,986 decision input tokens and 1,652 output tokens.
- Scenario compiler model calls, compiler tokens, and compiler provider API
  cost: exactly zero. The runtime performed 53 deterministic local annotation
  replays, one for each non-clock observable event.
- Total provider API cost: exactly USD 0. Human annotation effort,
  electricity, and hardware cost were not measured.

The token count is a decision-only lower bound. It cannot be used as evidence
that Anamnesis is token-efficient because the oracle replaces the measured LLM
compiler with frozen human-authored deltas.

## What the oracle isolated

The strict runtime trace contained exactly 8 due candidates, one at each gold
checkpoint, with the correct time, stable root `action_key`, canonical payload,
and active causal state. It contained no due candidate at any negative
checkpoint. The compiler deltas were all accepted, and the reducer and trigger
engine produced no obsolete-memory error.

Compared with the original local Anamnesis smoke run, which scored F1 0 with 7
false reminders and 2 invalid compiler outputs, this ceiling shows that writer
quality was the dominant failure. It also exposes a smaller independent
decision-policy problem.

## Residual failures

### False alarm: `s04-e04`

At the tax-letter user message, the structured memory view explicitly said that
no candidate was due. The model nevertheless invented and emitted a reminder to
mail the tax-office letter. This was correctly scored as a false alarm; it was
not produced by the oracle compiler, store, trigger engine, or scorer.

### Missed recurring occurrence: `s05-e10`

At Thursday 18:00, the memory view contained the exact due occurrence for
uploading Thursday's lab notes, including the correct date and an
`uploaded=false` fact. The model returned `no_action`. The same view included a
prior Tuesday execution with the same root `action_key` but a different
`occurrence_id` and date. The existing prompt says not to repeat an executed
action but does not state clearly that recurring occurrences are independent.

## Why provenance was 0/7

The scorer requires exact equality with an accepted minimal causal evidence
set. It is operating as preregistered and was not changed after seeing the
result.

For scheduled occurrences, the memory view exposed the intent and update
sources but not the current clock checkpoint inside the candidate evidence
list. The current event remained separately visible, yet the decision prompt
only said evidence IDs *may* be used; it did not require appending that
checkpoint or copying all candidate sources. The model also dropped causal
source IDs in several event-triggered cases.

The offline oracle acceptance test reached exact provenance 8/8 because its
deterministic copier explicitly appended the current event ID. That test proved
that the stored state can realize the gold actions, but it did not test the live
LLM provenance handoff. The live result identifies this as a decision-contract
coverage gap, not a reason to relax the scorer or rewrite the gold evidence
post hoc.

## Frozen D1 decision ablation

The smallest justified change is a prompt-only rule for structured memory:

1. Treat a provided structured memory view as authoritative.
2. With no `DUE_CANDIDATE`, return `no_action`, regardless of wording in the
   current raw event.
3. For each `DUE_CANDIDATE`, emit exactly one action and copy its kind,
   `action_key`, payload, and summary value-for-value.
4. Copy the candidate evidence IDs in order and append the current decision
   event if absent.
5. A prior execution suppresses only the same `occurrence_id`; a different
   occurrence of a recurring intent remains independently actionable.

This D1 version must receive a new prompt hash, clean source commit, local
preflight, frozen manifest, and strict smoke report. Because the decision policy
is shared, a fair four-system rerun is required before comparing it with the
original local smoke matrix. No 35-scenario run should occur until that smoke
ablation is complete.

That D1 matrix is now complete and failed its frozen promotion gate. The regular
Anamnesis writer produced the same erroneous compiler trace as D0, so the
decision prompt had only one gold-timed candidate to act on, with a noncanonical
payload. See [`local_smoke_d1_analysis.md`](local_smoke_d1_analysis.md). No D2
will be tuned on these same smoke cases.

## Latency limitation

Latency remains operationally diagnostic only. The oracle artifact load is a
small local setup step outside the recorded latency fields, and the prior
four-system local run required a server restart. Accuracy, raw-call, token,
cost, prompt, schema, artifact, and source-commit accounting are all validated,
but these data do not support a cross-system latency claim.
