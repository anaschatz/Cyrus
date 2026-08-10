# Local smoke diagnostic analysis

This analysis refers only to the 10-scenario local diagnostic in
[`local_smoke.md`](local_smoke.md). It is not a hypothesis test and it does not
justify opening the 35-scenario development set or a sealed set.

## Decision

Do not expand the current `ollama/qwen3:4b-instruct` compiler/prompt
configuration. All four systems missed all 8 expected actions. Anamnesis also
increased input tokens by 23.5% relative to full-context and produced more
false reminders.

## Headline findings

- All systems: 0 true positives, 8 false negatives, F1 0.
- No memory, full-context, and vector RAG: 1 false reminder each.
- Anamnesis: 7 false reminders across 6 checkpoints and 2 invalid compiler
  outputs.
- Anamnesis input: 81,617 tokens, split into 41,920 decision tokens and 39,697
  compiler tokens.
- Full-context input: 66,078 tokens. The Anamnesis reduction target was missed
  by 53.5 percentage points: the required reduction was at least 30%, while the
  observed value was -23.5%.
- Provider API cost was exactly USD 0 for every system. Electricity and hardware
  cost were not measured.

## Observed failure modes

The strict scorer requires the correct root `action_key`, canonical payload,
time window, and an accepted minimal causal evidence set.

- Full-context recognized the assignment reminder in `s01`, but emitted a
  revised-event ID as the action key rather than the stable creating-event ID.
- Vector RAG recognized the flight-delay reminder in `s08`, but paraphrased the
  canonical subject and flight payload, so it did not satisfy the action
  contract.
- No memory emitted an unrelated tax-letter reminder in the address-update
  scenario `s04`.
- Anamnesis kept the correct root key for the assignment reminder but
  paraphrased its subject and omitted required causal evidence.
- Anamnesis fired an immediate passport reminder at creation time in `s04`, then
  emitted an unrelated tax-letter action with the wrong root key at `s04-e04`,
  before the visa reminder's due checkpoint.
- In recurring scenario `s05`, Anamnesis emitted four reminders rather than the
  two required occurrences and omitted the occurrence date from the payload.
- The threshold scenario `s08` contained both invalid Anamnesis compiler
  outputs; no correct action was emitted.

These failures show two different bottlenecks: the 4B writer does not reliably
compile canonical memory records, and the shared 4B decision model does not
reliably preserve the benchmark action contract even when relevant context is
available.

## Subsequent experiments and stopping decision

1. The deterministic/oracle compiler ceiling is now complete. It produced all
   8 correct due candidates; the shared decision model yielded 7 TP, 1 FP, and
   1 FN. See [`local_oracle_smoke_analysis.md`](local_oracle_smoke_analysis.md).
2. The shared D1 decision-contract ablation is now complete for all four
   systems. It reduced simple-baseline false alarms but left every system at F1
   0 and failed its promotion gate. See
   [`local_smoke_d1_analysis.md`](local_smoke_d1_analysis.md).
3. Stop tuning prompts on these 10 smoke cases. A writer intervention or
   stronger local model now requires newly authored diagnostic cases, exact
   artifact pins, a fresh preflight, and a new frozen manifest.
4. Proceed to the 35 development scenarios only after a fresh smoke candidate
   produces non-zero recall and passes its declared gate without repair calls.

## Latency limitation

The Ollama server accumulated an 8+ GiB warm prompt cache during the sequential
baseline tasks and was restarted before Anamnesis to avoid macOS swap and disk
failure. Consequently, the recorded latency remains useful for operational
diagnosis within each run but is not a fair cross-system latency comparison.
Token, cost, accuracy, prompt/schema, and raw-call accounting remain fully
validated.
