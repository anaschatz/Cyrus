# Anamnesis

## Research question
Can an explicit temporal and provenance-aware memory improve an
LLM agent's ability to remember what is currently true and execute the
right action at the right time, with fewer tokens and fewer errors than
simple memory systems?

## Hypothesis
Separating facts, events, future intentions, and their provenance will
improve execution accuracy and reduce obsolete-memory errors while using
fewer input tokens.

The initial v0 uses three preregistered repetitions with seeds 101, 202, and
303. The hypothesis is supported only if, in each repetition, Anamnesis:

- improves F1 by at least 5 percentage points over the best simple baseline,
- reduces input tokens by at least 30% compared with full-context, and
- does not increase false alarms.

The false-alarm comparison uses the same simple baseline that wins F1 in that
repetition, after applying the preregistered tie-breaks below. The token and
cost totals for Anamnesis include both memory-compilation and final-decision
model calls.

## Initial scope
- Text only
- One simulated user
- Seven simulated days
- 50 scenarios
- No UI
- No model training initially

## Example behavior
A user says: "If I have not submitted the assignment by Friday, remind me."
After unrelated events, the agent must retain the intention, monitor whether
its condition was satisfied, act only when appropriate, adapt to deadline
changes, and explain which source information caused the action.

## Systems under comparison
1. No persistent memory
2. Full conversation history
3. Basic vector RAG
4. Anamnesis

## Metrics
- Precision
- Recall
- F1
- False reminders and false-alarm rate
- Obsolete-memory errors
- Input tokens
- Cost
- Latency

## Evaluation protocol
- The 10 smoke scenarios are for harness development, not hypothesis testing.
- A valid final dataset contains 35 development and 15 sealed scenarios. The
  sealed scenarios are excluded from development commands and are opened only
  after the Anamnesis schema, prompts, and configuration are frozen. All 50
  are then run for the final result.
- All systems use the same model snapshot, decision checkpoints, output schema,
  temperature, seed set, and concurrency.
- Anamnesis uses that same model snapshot for its strict memory compiler. The
  compiler runs only on user messages and external observations; the shared
  final decision still runs at every checkpoint for all systems.
- Inspect response caching is disabled for measured runs. Provider cache tokens
  are included in logical input-token totals.
- A measured run is invalid when provider token usage, the frozen pricing
  configuration, or any scenario/system/repetition record is missing.
- Precision, recall, and F1 are action-level micro metrics over all 50 scenarios.
- The best simple baseline is the one with highest F1 in that repetition. Ties
  are broken by fewer false-alarm checkpoints, then fewer input tokens.
- The false-alarm constraint compares checkpoint counts, not rounded rates.
- Extra diagnostic runs may be published but cannot replace a preregistered run.

The 15 holdout-shaped records currently checked into the repository are test
fixtures for the split and release guards, not a valid sealed hypothesis set:
they were inspected while the system contract was changing. They must be
replaced after the freeze, and the final runner rejects them in their current
release state.

## Non-goals
- AGI or a general personal assistant
- A foundation model
- A production SaaS or large UI
- A multimodal or multi-user system

## Definition of done for v0
Run all four systems on the same 50 seven-day scenarios for exactly three
repetitions and produce one reproducible results table. The prototype is a
successful research outcome even if the hypothesis is rejected, provided
that the experiment is valid and reproducible.
