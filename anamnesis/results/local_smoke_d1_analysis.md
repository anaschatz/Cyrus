# Local D1 decision-prompt ablation analysis

This analysis refers only to the strict 10-scenario local diagnostic in
[`local_smoke_d1.md`](local_smoke_d1.md). D1 was designed after inspecting the
D0 smoke and oracle failures. It is not a hypothesis test, does not demonstrate
generalization, and does not authorize opening the 35-scenario development set
or any sealed data.

## Decision

Reject D1 as a local smoke candidate. Do not tune a D2 prompt on these same 10
cases.

The new shared decision instructions improved conservatism in the three simple
baselines, but they did not recover any required action. The regular Anamnesis
writer remained the dominant bottleneck and reproduced its D0 compiler-call
outputs, deltas, accounting, and due IDs exactly, including two invalid outputs
and a set of mostly incorrect due candidates.

## Strict D1 result

| System | TP | FP | FN | F1 | False-alarm checkpoints | Input tokens |
|---|---:|---:|---:|---:|---:|---:|
| No memory | 0 | 0 | 8 | 0.0% | 0 | 53,010 |
| Full context | 0 | 0 | 8 | 0.0% | 0 | 79,922 |
| Vector RAG | 0 | 0 | 8 | 0.0% | 0 | 67,610 |
| Anamnesis | 0 | 7 | 8 | 0.0% | 6 | 95,282 |

All systems had precision, recall, and F1 0. Anamnesis emitted 7 false actions
across 6 negative checkpoints; its payload mismatch at the one positive
checkpoint is the seventh FP but not a false-alarm checkpoint. There were no
obsolete-memory errors. Provider API cost was exactly USD 0; electricity,
hardware, and human time were not measured.

Anamnesis input comprised 55,585 decision tokens and 39,697 compiler tokens.
Its compact decision context used 30.5% fewer tokens than D1 full-context, but
the compiler overhead made total Anamnesis input 19.2% higher. The research
token criterion applies to total input, so it was not met.

## Frozen promotion gate

D1 was required to satisfy all four conditions declared before the run:

| Condition | Result |
|---|---|
| Anamnesis recall greater than zero | **Fail** — recall 0 |
| Zero Anamnesis compiler invalid outputs | **Fail** — 2 invalid outputs |
| No simple baseline exceeds its D0 false-alarm count of one | Pass — all three had 0 |
| Positive exact provenance on matched Anamnesis actions | **Fail / N/A** — no matched actions |

Because the gate failed, the complete result is reported without repair,
selection, or rerun, and the planned stopping rule applies.

## What changed from D0

Only the shared local decision prompt and its version/hash changed. The decision
schema, compiler prompt/schema, memory engine, renderer, scorer, dataset, RAG,
model, seed, and runtime policy were frozen.

The D0 and D1 Anamnesis compiler-call artifacts are byte-for-byte identical:
all 53 calls have the same raw outputs, parsed deltas, acceptance, usage, parse
flags, and due IDs, while the 25 clock checkpoints remain compiler-free. State
hashes legitimately diverge after changed decisions update the execution
ledger. Compiler input/output remained exactly 39,697/6,896 tokens. The same two
semantic parse errors occurred in threshold scenario `s08`: non-normalized
entity keys `Aegean flight 482` and `Olympic Air flight 482` were rejected
atomically.

The longer D1 prompt added roughly 13.7–14.0k input tokens to each system. The
descriptive D0→D1 changes were:

- no memory: false reminders 1 → 0, recall unchanged at 0;
- full context: false reminders 1 → 0, recall unchanged at 0;
- vector RAG: false reminders 1 → 0, recall unchanged at 0; and
- Anamnesis: false reminders 7 → 7, invalid outputs 2 → 2, recall unchanged at
  0.

These cross-version deltas are diagnostic, not a formal generalization claim.

## Why D1 could not recover recall

The D1 block was present in every one of the 78 decision prompts. Anamnesis
rendered 66 empty views and 12 due views. The model obeyed `empty => no_action`
at all 66 empty checkpoints. It emitted at 7 of 12 due checkpoints and copied
the candidate's kind, root key, payload, and summary exactly in all 7 emissions;
it followed the complete evidence rule in only 5 of 7.

Only one of the 8 gold checkpoints had a candidate: `s01-e07`. Even there, the
candidate payload said `send the statistics assignment`, while the canonical
gold subject is `send statistics assignment`. The other 7 gold checkpoints had
empty views and therefore no candidate for the decision model to copy.

The other 11 due-candidate checkpoints were not gold action times. They came
from writer errors such as immediate triggers, incorrect rescheduling,
misinterpreted brainstorming, and incomplete recurring/completion state. D1
emitted actions at 7 of the 12 candidate checkpoints and emitted nothing when
no candidate existed, but every emitted action still failed the scorer because
the upstream candidate was wrong or mistimed.

Compared with D0, D1 stopped following the current tax-letter wording in
`s04-e04` and skipped one incorrect recurring candidate. It also followed two
other incorrect compiled candidates in the negative brainstorming and cancelled
reminder scenarios. The total remained 7 false reminders. Full compliance with
the five due candidates that the model ignored would have added five more
negative false positives, not recovered a true positive.

## Interpretation

The oracle ceiling already showed that the reducer, trigger engine, renderer,
and decision model can reach F1 87.5% when compilation is correct. D1 now shows
that a stronger downstream copying rule cannot repair missing or malformed
upstream intentions. It may improve candidate discipline, but the current Qwen
4B writer is not a viable smoke candidate.

The next defensible experiment is not another prompt tuned on these 10 cases.
It requires a newly authored diagnostic set and a separately frozen writer
intervention, such as a stronger local model or a more constrained compiler
representation. The current 35 development scenarios were not run or evaluated
in this local track.

## Latency limitation

Latency is operationally diagnostic only. The four tasks ran sequentially in a
single Ollama server process, whose prompt cache grew to roughly 7 GiB and
caused macOS swap pressure during Anamnesis. Accuracy, raw-call, token, cost,
prompt/schema, artifact, and source-commit accounting passed the strict
reporter, but these data do not support cross-system latency comparisons.

## Distribution limitation

The provenance sidecar binds the exact frozen manifest, new preflight, four raw
`.eval` logs, dataset, CSV, and Markdown by SHA-256. The six run artifacts remain
under the ignored `results/runs/` tree and are available locally, not in Git.
The committed table is therefore tamper-evident, but a Git-only consumer cannot
independently replay the strict validation until those exact ignored bytes are
published as a separate archive.
