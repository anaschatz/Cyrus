# Anamnesis

An evaluation-first research prototype for temporal and prospective memory in
LLM agents. It tests whether explicit, provenance-aware state helps an agent
remember what is true now and execute an intended action at the right moment
with fewer errors and fewer input tokens than simple memory baselines.

The research question and pass/fail thresholds are frozen in
[`RESEARCH.md`](RESEARCH.md). The treatment and fairness boundary are specified
in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Current status

The v0 harness and in-memory treatment are implemented. The repository now
contains:

- 35 visible development scenarios and 15 holdout-shaped harness scenarios;
- no-memory, full-context, exact top-k vector-RAG, and Anamnesis systems;
- an online LLM memory compiler plus a deterministic versioned store, closed
  trigger DSL, occurrence state, compact memory view, and execution ledger;
- deterministic action/provenance scoring and strict baseline/final reports;
- component-level token, cost, latency, parse-error, and audit accounting;
- frozen-manifest, dataset-release, model-pricing, prompt, git, and embedding
  artifact guards; and
- a synthetic hosted-model compatibility preflight.

No preregistered hosted benchmark result is claimed yet. The first hosted candidate,
[`openai/gpt-5.4-mini-2026-03-17`](https://developers.openai.com/api/docs/models/gpt-5.4-mini),
was rejected before any API or scenario call because the installed Inspect
Responses path could not preserve both seed and temperature-zero semantics. The
preregistered next candidate is the dated
[`openai/gpt-4.1-mini-2025-04-14`](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
snapshot, forced through Chat Completions with `responses_api=false` on the
standard OpenAI endpoint. Its pricing is tracked in `eval/model_costs.json`, but
it is not frozen or accepted until the live compatibility preflight passes. The
FastEmbed artifact is pinned to repository
`qdrant/bge-small-en-v1.5-onnx-q`, revision
`52398278842ec682c6f32300af41344b1c0b0bb2`, and canonical tree SHA-256
`d435d05b3411502ad9a280cc9ac0157f7bcd9f176df2fdc8971f788a121a02d7`.
No scenario result was viewed during selection; an API key and the live
preflight remain operational blockers, so the required 35-scenario development
baseline has not run. The existing 15-record holdout
was inspected during development and is deliberately marked ineligible for
hypothesis testing. A fresh, externally held and independently reviewed
replacement is required after the system freeze. See
[`eval/SCENARIO_PLAN.md`](eval/SCENARIO_PLAN.md).

A separate, zero-provider-cost local diagnostic track is also available. It
uses the already installed `ollama/qwen3:4b-instruct` model through the pinned
`127.0.0.1` Ollama endpoint and never qualifies as the preregistered final
hypothesis test. Its clean two-call compatibility probe and strict four-system
matrix have now run on the 10 smoke scenarios. The result is deliberately
negative: all four systems scored F1 0; Anamnesis produced 7 false reminders,
used 81,617 input tokens versus 66,078 for full-context (23.5% more), and had
two invalid compiler outputs. Provider API cost was exactly `$0.00` throughout.
This rejects the current Qwen 4B compiler/prompt configuration as a candidate
for expansion to the 35 development scenarios; it does not accept or reject the
research hypothesis. See the strict [result table](results/local_smoke.md),
[CSV](results/local_smoke.csv), [failure analysis](results/local_smoke_analysis.md),
and [SHA-256 provenance sidecar](results/local_smoke.provenance.json). Exact
model/blob hashes, server settings, commands, and claim limits are documented in
[`eval/LOCAL_EXECUTION.md`](eval/LOCAL_EXECUTION.md).

A subsequent gold-assisted oracle-compiler ceiling isolated the failure. The
deterministic memory path produced all 8 correct due candidates, and the shared
decision model converted 7 of them into scored true positives: 7 TP, 1 FP, 1 FN,
F1 87.5%, and zero obsolete-memory errors. The remaining false alarm and miss
were decision-policy errors, while exact provenance remained 0/7 because the
current decision contract did not require copying the full causal evidence plus
the current checkpoint. This diagnostic uses frozen human annotations, reports
only a decision-token lower bound, and is not an Anamnesis result or hypothesis
test.
See the strict [oracle table](results/local_oracle_smoke.md),
[CSV](results/local_oracle_smoke.csv),
[analysis](results/local_oracle_smoke_analysis.md), and
[provenance sidecar](results/local_oracle_smoke.provenance.json).

The preregistered D1 shared decision-prompt ablation has also completed on the
same four-system smoke matrix. It reduced each simple baseline's false alarms
from one to zero, but every system still had recall and F1 0. Anamnesis remained
at 7 false reminders and 2 invalid compiler outputs, used 95,282 total input
tokens versus 79,922 for D1 full-context, and therefore failed the frozen D1
promotion gate. All 53 compiler-call outputs, deltas, accounting records, and
due-candidate IDs were unchanged from D0: only one of eight gold checkpoints had
a due candidate, and that candidate had the wrong canonical payload. D1 is
rejected, and no D2 will be tuned on these same smoke cases. See the
[D1 table](results/local_smoke_d1.md),
[CSV](results/local_smoke_d1.csv),
[analysis](results/local_smoke_d1_analysis.md), and
[provenance sidecar](results/local_smoke_d1.provenance.json).

A fresh 10-case writer diagnostic and gold-assisted reference were then frozen
before authoring the W1 compiler prompt. The new Qwen 4B attempt stopped at its
mandatory synthetic preflight: the compiler filled unused optional payload
slots with empty values, including an invalid `date: ""`, so domain validation
rejected the delta. The decision preflight passed, but the combined semantic
gate did not. In accordance with the frozen stopping rule, no writer scenario
was run and no output repair or retry was attempted. See the
[W1 preflight failure record](results/local_writer_w1_preflight_failure.md) and
[SHA-256 provenance sidecar](results/local_writer_w1_preflight_failure.provenance.json).

W2 then tested one narrowly preregistered change on a separately frozen and
corrected 10-case writer set: unused optional payload slots must be omitted or
`null`, never filled with empty strings, false values, empty collections, or
placeholder zeroes. Its fresh four-call semantic preflight passed, and the
scenario cell ran exactly once. The W1 filler failure did not recur—no optional
action payload contained filler—but the writer gate failed: 4 domain-conversion
invalid deltas, 1 store-invalid update, 41/46 accepted deltas, and replayed
candidate TP=0, FP=3, FN=8. Headline scenario usage was 108,252 input and 4,554
output tokens; provider API cost was exactly `$0.00`. W2 is rejected and will
not be repaired or rerun on these cases. See the strict
[W2 table](results/local_writer_w2.md), [CSV](results/local_writer_w2.csv),
[forensic analysis](results/local_writer_w2_analysis.md), and
[provenance sidecar](results/local_writer_w2.provenance.json).

W3 bundled the remaining compiler repairs into a new prompt, froze a fourth
blind 10-case writer set before that prompt, and froze a neutral nine-call
semantic fixture afterward. Its one authorized preflight failed, so the
scenario set was never evaluated. C5 (stable-ID trigger update), C7 (complete
sparse payload including an explicit zero), and D1 (no action) passed; C2/C4
failed domain conversion, while C1/C3/C6/C8 were schema-valid but semantically
wrong. Aggregate usage was 17,326 input and 1,044 output tokens at exactly
`$0.00` provider API cost. W3 is rejected at preflight, with no retry, repair,
manifest freeze, or measured scenario log. Per the frozen stopping rule, no W4
will be tuned on the v4 cases. See the
[W3 failure record](results/local_writer_w3_preflight_failure.md) and
[provenance sidecar](results/local_writer_w3_preflight_failure.provenance.json).

A separately preregistered W3-M2 model-only cell then kept the full W3
prompt/schema/fixture contract unchanged and replaced only the local model with
the byte-pinned `ollama/qwen3.5:9b-q4_K_M` (Qwen35 9.7B, Q4_K_M). The artifact
fit and ran locally on the Apple M3, but all eight compiler calls exhausted the
4096-token context and produced no parseable completion. D1 no-action passed.
The single nine-call run used 18,865 input and 15,856 output tokens at exactly
`$0.00` provider API cost. W3-M2 is rejected at preflight and no scenario set
was created or evaluated. See the
[W3-M2 failure record](results/local_writer_w3_m2_preflight_failure.md) and
[provenance sidecar](results/local_writer_w3_m2_preflight_failure.provenance.json).

W3-M2-T1 then isolated the transport failure without changing the model,
prompt, schema, fixture, or token budget. Forwarding
`reasoning_effort: "none"` in every raw Ollama request eliminated the hidden
reasoning exhaustion: all eight compiler calls produced final, non-truncated
content, and total setup time fell from about 41.15 to 8.14 minutes. However,
all eight outputs still failed the frozen compiler wire schema by inventing
alternate mutation fields or copying internal stored-revision structures. D1
no-action passed. The single run used 17,653 input and 2,616 output tokens at
exactly `$0.00` provider API cost. T1 is rejected at preflight with no retry or
scenario call. See the
[W3-M2-T1 failure record](results/local_writer_w3_m2_t1_preflight_failure.md)
and [provenance sidecar](results/local_writer_w3_m2_t1_preflight_failure.provenance.json).

An additive architecture-v2 hardening track now addresses the concrete failure
mechanisms without rewriting those frozen results. The compiler receives a
closed, wire-aligned semantic state instead of reducer revisions; fact identity
is collision-safe; dead or past triggers and semantic no-op updates are rejected;
condition transitions are correctly baselined; units match exactly; and the
execution ledger uses only store-derived causal evidence. An optional external
vLLM adapter adds pinned JSON-schema constrained generation with independent
JSON, wire, domain, reducer, finish-reason, and usage gates. A separate optional
OpenMemory adapter provides namespaced retrospective recall only—its IDs and
scores can never become Anamnesis truth, triggers, executions, or evidence.
The additive `anamnesis_openmemory_recall` diagnostic strategy now connects
that boundary to the runner as a search-only sidecar: every scenario receives
a fresh caller-supplied snapshot, recalled text is serialized into a separately
labelled untrusted prompt section, and only the deterministic Anamnesis store
may commit state or evidence. OpenMemory usage remains deliberately incomplete
because the upstream interface has no provider-neutral token/cost accounting.
The first complete paired OpenMemory-style recall diagnostic has now run on a
fresh frozen v2 case set. Recall remained non-authoritative and caused no
false-action, no-hit, safety, or evidence-contamination regression, but it
produced zero helpful gains: baseline and recall each scored 4/8. The frozen
gate therefore failed, and the current prompt/model recall cell is rejected.
See the [v2 diagnostic record](results/local_openmemory_diagnostic_v2.md) and
[provenance sidecar](results/local_openmemory_diagnostic_v2.provenance.json).
The follow-up v3 cell corrected the discovered temporal-reminder versus
immediate-action prompt mismatch on fresh cases. The model then attempted all
positive actions and helpful recall resolved the intended recipient/address,
but 12/16 responses violated the closed response schema. V3 therefore also
failed, isolating structured-output enforcement as the next bottleneck. See the
[v3 record](results/local_openmemory_diagnostic_v3.md) and
[provenance sidecar](results/local_openmemory_diagnostic_v3.provenance.json).
These changes define a new experiment identity; they are not a rerun or
reinterpretation of W1-W3. A fresh v4 cell has now run with an immutable
Qwen3.5 4B MLX artifact because the prior Q4_K_M bytes are incompatible with
vllm-metal. The explicitly pinned xgrammar backend and neutral canary worked,
and recall caused no safety or evidence contamination regression. The gate
still failed: six measured calls repeated a schema-allowed action object until
the 256-token limit, and one more passed JSON/wire validation but failed a
domain subject invariant. Baseline and recall each scored 4/8 with zero helpful
gain. This isolates a JSON-Schema-to-domain alignment weakness, not a vLLM
transport failure, and it is not a causal comparison with Ollama. See the
[strict v4 result](results/local_openmemory_vllm_v4.md),
[forensic analysis](results/local_openmemory_vllm_v4_analysis.md), and
[architecture comparison and integration contract](VLLM_OPENMEMORY_ARCHITECTURE.md).
The discovered mismatch is now fixed additively for the next cell:
`VllmAlignedDecisionWire` encodes `actions.maxItems=1` and a two-or-more-token
lowercase subject directly in the JSON Schema, with separate contract/schema
hashes. The published v4 wire and hashes remain unchanged. This correction has
now passed its one authorized, fresh two-call v5 compatibility gate: both the
`emit` and `no_action` cases finished normally and passed JSON, wire, domain,
usage, and semantic validation. The run used 913 input and 132 output tokens
at `$0.00` provider API cost. This establishes schema compatibility only; it is
not a v4 rerun, recall-quality result, or hypothesis test. See the
[strict v5 record](results/local_openmemory_vllm_v5.md) and
[provenance sidecar](results/local_openmemory_vllm_v5.provenance.json).

A subsequent real indexed-memory cell replaced fixture-injected hits with
runtime add/embed/search operations through the non-authoritative OpenMemory
boundary. The pinned local index retrieved the intended record in 8/8 cases;
all 16 paired vLLM calls were accepted; exact accuracy improved from 3/8 without
recall to 4/8 with recall; and there were no safety regressions. Recall supplied
the intended missing value in all four helpful cases, but only one produced the
exact canonical action—the other three exposed payload-slot or subject
normalization errors. See the [strict v6 result](results/local_openmemory_vllm_v6.md),
[analysis](results/local_openmemory_vllm_v6_analysis.md), and
[provenance sidecar](results/local_openmemory_vllm_v6.provenance.json). This
tests the OpenMemory-compatible architecture using FastEmbed, not the upstream
Cavira SDK.

The source-grounded canonicalizer was then tested prospectively on six new v7
cases. Retrieval was 6/6, all 12 constrained calls were accepted, and exact
accuracy improved from 2/6 without recall to 5/6 with recall. All three helpful
opportunities became exact actions, with zero safety regressions. The sole miss
occurred identically in both arms because the model retained an article in an
otherwise correct subject; it was not a memory failure. See the
[strict v7 result](results/local_openmemory_vllm_v7.md),
[analysis](results/local_openmemory_vllm_v7_analysis.md), and
[provenance sidecar](results/local_openmemory_vllm_v7.provenance.json).

Both local latency sets are diagnostic only. In D0, the Ollama server was
restarted before Anamnesis to clear an 8+ GiB warm prompt cache after macOS swap
pressure. In D1, all four tasks stayed in one process, whose prompt cache grew
to roughly 7 GiB and again coincided with disk/swap pressure. Accuracy, token,
cost, prompt, schema, and raw-call accounting pass the strict reporters, but
cross-system latency should not be interpreted as a fair benchmark.

## Architecture

```text
ObservableEvent(id, at, kind, text)
  -> LLM MemoryCompiler
  -> strict MemoryDelta
  -> deterministic temporal store and trigger engine
  -> compact MemoryView
  -> shared decision LLM
  -> occurrence/execution ledger
```

Every system receives the same sanitized event stream and invokes the same
decision model once per authored checkpoint. Anamnesis alone invokes the same
frozen model once more for each non-clock event to compile memory. All compiler
usage is included in its headline totals. Gold actions, forbidden traps,
`supersedes`, scenario annotations, and future events never cross the runtime
boundary.

The store is pure in-memory Python. It keeps immutable fact and intent
revisions, stable action keys, per-occurrence recurring state, raw-event
provenance, and an execution ledger. It does not use a database, embeddings,
LangChain/LlamaIndex, a scheduler service, PyTorch, a UI, or model training.

## Open-source stack

| Role | Tool |
|---|---|
| Evaluation tasks and raw logs | [Inspect AI](https://inspect.aisi.org.uk/) |
| Strict data contracts | [Pydantic](https://docs.pydantic.dev/) |
| Vector-RAG baseline only | [FastEmbed](https://github.com/qdrant/fastembed) |
| Retrieval and report math | [NumPy](https://numpy.org/) |
| Tests and linting | [pytest](https://pytest.org/) and [Ruff](https://docs.astral.sh/ruff/) |
| Reproducible environment | [uv](https://docs.astral.sh/uv/) |

AutoResearch is intentionally not a core dependency. After the development
split and experiment configuration are frozen, its experiment-loop pattern may
be used for declared prompt/config ablations on the 35 development scenarios
only. It must never receive the final sealed data.

## Setup and local verification

Use Python 3.11, 3.12, or 3.13:

```bash
uv sync --frozen --extra dev --no-editable
source .venv/bin/activate

anamnesis-validate eval/scenarios/dev.jsonl
pytest
ruff check src tests eval
ruff format --check src tests eval
inspect list tasks eval/anamnesis_eval.py
```

The current candidate validates as 35 development scenarios with 219
checkpoints, and 50 combined scenarios with 296 checkpoints. Dataset manifests
record canonical and per-record hashes, family counts, origin, and review
state. All current cases are locally authored; no LongMemEval or TriggerBench
content is silently included.

## Freeze before measuring

1. Verify the tracked Inspect model-cost file for the exact hosted snapshot and
   materialize the preregistered FastEmbed revision locally. Confirm that its
   canonical artifact-tree hash matches the values in the manifest template.
2. Commit the source tree and confirm the worktree is clean. The preflight log
   must attest to this exact clean Git revision.
3. Run the synthetic `model_preflight` task on the hosted immutable snapshot.
   It must pass both strict schemas and complete usage/cost accounting without
   a repair call. Pin the resulting `.eval` file and its byte SHA-256 as
   `model.preflight` in the experiment manifest.
4. Copy
   [`eval/experiment_manifest.template.json`](eval/experiment_manifest.template.json)
   to the ignored local path `eval/experiment.baseline.json` (or
   `eval/experiment.final.json` for the final phase). Fill every required
   hash/configuration field, set `git_commit` to that clean `HEAD`, and set
   `status` to `frozen`. Do not add this generated manifest to the commit it
   identifies: a Git commit cannot contain its own hash. Once the first measured
   task starts, do not modify it; archive and publish those exact frozen bytes,
   rather than a regenerated copy, alongside the resulting `.eval` logs and
   result table.
5. Keep temperature at zero, response cache disabled, `max_samples=1`,
   `max_tasks=1`, and model concurrency at one for every compared task.

Model-selection rules are detailed in
[`eval/MODEL_SELECTION.md`](eval/MODEL_SELECTION.md). The task factory and
strict report fail closed when the active model pricing, prompt/schema, dataset,
git revision, RAG configuration, or embedding artifact differs from the frozen
manifest. They also re-open the pinned preflight log and verify its exact two
prompts, strict schemas, non-cached/no-retry calls, token usage, and cost against
the pinned pricing table.

Example preflight:

```bash
inspect eval eval/anamnesis_eval.py@model_preflight \
  --model openai/gpt-4.1-mini-2025-04-14 \
  -M responses_api=false \
  --model-cost-config eval/model_costs.json \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api \
  --log-format eval --log-dir results/runs --json \
  -T seed=101
```

Set `OPENAI_API_KEY` in the local shell before running this command; never add
the key to a manifest, log, source file, or commit.

## Development baseline gate

The first measured milestone is exactly three systems × 35 development
scenarios × one declared repetition (`seed=101`). Run all tasks from the same
clean frozen baseline manifest:

```bash
ANAMNESIS_FASTEMBED_PATH="$PWD/results/runs/fastembed/bge-small-en-v1.5-onnx-q/52398278842ec682c6f32300af41344b1c0b0bb2"

inspect eval eval/anamnesis_eval.py@no_memory \
  --model openai/gpt-4.1-mini-2025-04-14 -M responses_api=false \
  --model-cost-config eval/model_costs.json \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api \
  --log-format eval --log-dir results/runs/baseline \
  -T manifest=eval/experiment.baseline.json -T seed=101 -T repetition=1

inspect eval eval/anamnesis_eval.py@full_context \
  --model openai/gpt-4.1-mini-2025-04-14 -M responses_api=false \
  --model-cost-config eval/model_costs.json \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api \
  --log-format eval --log-dir results/runs/baseline \
  -T manifest=eval/experiment.baseline.json -T seed=101 -T repetition=1

inspect eval eval/anamnesis_eval.py@vector_rag \
  --model openai/gpt-4.1-mini-2025-04-14 -M responses_api=false \
  --model-cost-config eval/model_costs.json \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api \
  --log-format eval --log-dir results/runs/baseline \
  -T manifest=eval/experiment.baseline.json -T seed=101 -T repetition=1 \
  -T embedding_revision=52398278842ec682c6f32300af41344b1c0b0bb2 \
  -T embedding_snapshot_path="$ANAMNESIS_FASTEMBED_PATH"
```

Then generate the explicitly non-final development table:

```bash
anamnesis-report \
  --mode baseline \
  --manifest eval/experiment.baseline.json \
  --scenarios eval/scenarios/dev.jsonl \
  --runs results/runs/baseline/*.eval \
  --csv results/development.csv \
  --markdown results/development.md
```

Only after this table exists should the Anamnesis compiler be evaluated or
tuned on the same development split. The repository contains the implementation
scaffolding now, but this chronological research gate has not yet been
satisfied and must not be rewritten as if it had.

## Final experiment

After freezing the compiler prompt, schema, reducer, trigger behavior, and
system configuration, replace the contaminated holdout candidate with a fresh
15-scenario independently reviewed set. Its adjacent release manifest must mark
human review as passed and `preregistered_final_eligible` as true; otherwise
the `dataset=all` tasks refuse to start.

Run all four systems over all 50 scenarios for repetitions 1/2/3 with seeds
101/202/303, respectively. A strict final report requires the complete matrix
and applies the preregistered gate independently to each repetition:

- Anamnesis F1 exceeds the best simple baseline by at least 0.05;
- Anamnesis total logical input tokens are at least 30% below full-context; and
- Anamnesis has no more false-alarm checkpoints than that same comparator.

`--allow-incomplete` exists only for diagnostics. It labels output as incomplete
and suppresses every hypothesis-support claim. No LLM judge participates in
headline scoring.
