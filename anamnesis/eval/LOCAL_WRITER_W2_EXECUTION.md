# Local W2 writer diagnostic

W2 is one preregistered, zero-provider-cost diagnostic of the sparse-payload
compiler prompt. It is not a baseline comparison or a hypothesis test. W1
artifacts remain immutable and are never accepted by the W2 phase.

The frozen matrix is exactly:

- phase: `writer_diagnostic_w2`
- preflight task: `local_model_preflight_w2`
- scenario task: `local_anamnesis_writer_diagnostic_w2`
- system: `anamnesis`
- dataset: `eval/scenarios/writer_diagnostic.v3.jsonl`
- dataset raw SHA-256:
  `34e2e8751bf32a3a2e29ac75d727f2b5cf73aaba13ccc9ba1d9fdf00bf7eaf4f`
- scenarios: 10
- authored checkpoints: 69
- scenario compiler calls: 46
- setup policy literal: `frozen_w2_semantic_gate_c1_c2_c3_d1` — one frozen
  ordered semantic gate `C1,C2,C3,D1` (three compiler calls and one decision
  call), not one call per schema
- seed: 101
- repetitions: 1
- compiler prompt version: `local.v0.3`
- compiler prompt SHA-256:
  `024641f8d0ec16168eb9b7d8dbee67f92b7049fe6d35b604495aba273319d9dd`
- unchanged compiler schema SHA-256:
  `8871ff344eb3a2e88a53b964ef2f24f089a72507c69073ec323cf26a428c3030`
- compiler transport SHA-256:
  `5187889e0b2bb998d73857f9ad6c0b252141cf07b3c387567fa9e25bfb7f9a89`
- frozen preflight fixture: `eval/preflight/local_writer_w2.v1.json`
- preflight fixture raw SHA-256:
  `3b82128bab1d801d073118488aa4f0a0a662603b98325f5c9d7dad497f026057`
- reporter-only reference:
  `eval/oracle/writer_diagnostic_memory_deltas.v3.json`
- reference raw SHA-256:
  `7adb64eda15daf5351260933fbd0625fbc13c6899361735a9bf0ce13c063f857`

The W2 prompt was committed only after the v3 diagnostic set was frozen. The
synthetic preflight fixture was then frozen without running the model. Its
categories, acceptance projections, and valid examples are local validation
data only. The model receives only each case's `input`, in exact order C1, C2,
C3, D1. It never receives category names, acceptance data, or valid examples.

The `writer_reference` remains declarative and reporter-only. Scenario task
construction and execution never resolve, open, hash, read, deserialize, or
pass it to task metadata, the solver, compiler, decision model, or runtime
state. The preflight fixture is not benchmark gold and is a required,
phase-exclusive measured-protocol input.

## Freeze and stopping order

1. Run all offline tests and Ruff without starting Ollama.
2. Commit the complete W2 task, validator, reporter, tests, and documentation;
   require a clean worktree.
3. Start the pinned Ollama server with the five local-only declarations below.
4. Run `local_model_preflight_w2` exactly once. It makes three compiler calls
   and one decision call in C1,C2,C3,D1 order, with no cache, retry, or repair.
5. If any of the four calls fails parsing, semantic acceptance, complete usage,
   exact zero-cost accounting, model identity, or residency, preserve the log,
   record W2 as rejected at preflight, and stop. Do not change the prompt,
   fixture, schema, or acceptance projection and do not run scenarios.
6. Only after a passing preflight, copy the W2 manifest template to the W2 run
   directory. Pin the exact preflight `.eval` path/hash, the clean 40-character
   commit, prompt/schema hashes, and W2 system configuration hash; set status
   to `frozen`.
7. Run `local_anamnesis_writer_diagnostic_w2` exactly once. Before its first
   measured scenario call, the task repeats the same four-call W2 gate once as
   setup. Setup tokens, cost, and latency are recorded separately and excluded
   from measured scenario headline usage.
8. Run the strict phase-aware writer reporter once on that exact `.eval` log.

No preflight or scenario retry is allowed. A failed preflight or measured gate
is the W2 result. A prompt correction would be W3 and requires a newly frozen,
independent diagnostic set and preflight protocol.

## Local environment

Use one dedicated Ollama server and export the same values in the Inspect
shell:

```bash
export OLLAMA_NO_CLOUD=1
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
```

Run the one standalone W2 preflight:

```bash
inspect eval eval/anamnesis_local_eval.py@local_model_preflight_w2 \
  --model ollama/qwen3:4b-instruct \
  --model-base-url http://127.0.0.1:11434/v1 \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api \
  --log-format eval \
  --log-dir results/runs/local/writer_diagnostic_w2/preflight --json \
  -T seed=101
```

The strict W2 validator requires exactly four raw `ModelEvent` records, the
exact prompts and response schemas, the pinned model/route/configuration,
positive token usage, zero provider cost, a clean matching Git revision, and a
passing result whose usages and semantic projections match the raw outputs.

## One measured run

After freezing
`results/runs/local/writer_diagnostic_w2/experiment.writer_w2.json`, set:

```bash
ANAMNESIS_WRITER_W2_MANIFEST="$PWD/results/runs/local/writer_diagnostic_w2/experiment.writer_w2.json"
ANAMNESIS_OLLAMA_MODELS="$HOME/.ollama/models"
```

Run the singleton task once:

```bash
inspect eval eval/anamnesis_local_eval.py@local_anamnesis_writer_diagnostic_w2 \
  --model ollama/qwen3:4b-instruct \
  --model-base-url http://127.0.0.1:11434/v1 \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api \
  --log-format eval \
  --log-dir results/runs/local/writer_diagnostic_w2/runs --json \
  -T seed=101 -T repetition=1 \
  -T manifest="$ANAMNESIS_WRITER_W2_MANIFEST" \
  -T ollama_models_dir="$ANAMNESIS_OLLAMA_MODELS"
```

Do not pass the writer reference, oracle annotations, embeddings, an alternate
fixture, or a model-cost configuration. Preserve the single exact
`log_location` printed by Inspect; a directory glob is not a reporter input.

The reporter uses the same multiset-replay shape as W1 with the separately
frozen W2 candidate key, which deliberately excludes free-form summary text.
Phase dispatch must select the W2 prompt, dataset, fixture, task identity,
raw-call pattern, and v3 reporter-only reference. A valid FAIL is preserved and
ends W2; integrity errors are not experimental results.

Run the strict reporter on the one exact measured log:

```bash
ANAMNESIS_WRITER_W2_LOG="/absolute/path/from/Inspect/log_location.eval"

uv run anamnesis-writer-report \
  --manifest "$ANAMNESIS_WRITER_W2_MANIFEST" \
  --scenarios eval/scenarios/writer_diagnostic.v3.jsonl \
  --run "$ANAMNESIS_WRITER_W2_LOG" \
  --csv results/local_writer_w2.csv \
  --markdown results/local_writer_w2.md \
  --provenance results/local_writer_w2.provenance.json
```

Exit status `0` means the fully validated W2 gate passed. Exit status `2`
means the log was valid, all three result artifacts were written, and the
preregistered W2 gate failed. Any exception is an integrity failure and is not
an experimental result.
