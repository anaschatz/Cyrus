# Local W1 writer diagnostic

This is a single-system, 10-scenario diagnostic of the W1 LLM memory compiler.
It is not a baseline comparison or a hypothesis test. It must not reuse smoke
logs, preflight logs, manifests, metrics, or output files.

The frozen matrix is exactly:

- phase: `writer_diagnostic`
- task: `local_anamnesis_writer_diagnostic`
- system: `anamnesis`
- dataset: `eval/scenarios/writer_diagnostic.v1.jsonl`
- scenarios: 10
- seed: 101
- repetitions: 1
- compiler mode: `llm`
- reporter-only reference: `eval/oracle/writer_diagnostic_memory_deltas.v1.json`
- reference raw SHA-256:
  `93c24d604b32c838d635f9c9ed4fea20f770da254f522db6962b6bc57a232057`

The manifest declares a required `writer_reference` artifact pin. That artifact
is gold-derived and reporter-only. Measured task construction validates only
the pin declaration's schema as part of the manifest. It never resolves, opens,
hashes, reads, or deserializes the referenced artifact, and it never passes the
reference path or content to the solver, task metadata, compiler, decision
model, or runtime state. The strict writer reporter is solely responsible for
resolving and validating the reference after the measured run.

## Freeze order

1. Run offline tests and Ruff. Do not start Ollama as part of this step.
2. Commit the implementation and require a clean worktree.
3. Start Ollama with the five exact local-only declarations below.
4. Run a fresh two-call `local_model_preflight` from that clean commit. Do not
   reuse the smoke or oracle preflight.
5. Copy `local_writer_experiment_manifest.template.json` to
   `results/runs/local/writer_diagnostic/experiment.writer.json`.
6. Pin the fresh preflight path and SHA-256, clean 40-character Git commit,
   current decision/compiler prompt and schema hashes, and the exact
   `anamnesis` system configuration hash. Change `status` to `frozen`.
7. Run `local_anamnesis_writer_diagnostic` exactly once.
8. Give that one exact `.eval` log, the frozen manifest, dataset, and separately
   supplied reference artifact to the strict writer reporter.

The ordinary `local_anamnesis` task accepts the smoke phase only and must reject
a `writer_diagnostic` manifest. This prevents the fresh diagnostic from being
silently mixed with the prior smoke matrix.

## Local environment and preflight

Use a dedicated Ollama server and export the same values in the Inspect shell:

```bash
export OLLAMA_NO_CLOUD=1
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
```

Run the fresh semantic preflight:

```bash
inspect eval eval/anamnesis_local_eval.py@local_model_preflight \
  --model ollama/qwen3:4b-instruct \
  --model-base-url http://127.0.0.1:11434/v1 \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api \
  --log-format eval \
  --log-dir results/runs/local/writer_diagnostic/preflight --json \
  -T seed=101
```

The preflight must succeed without retry or repair and must be pinned before
the scenario task is constructed. Runtime validation binds the pinned log bytes
and the revision recorded inside that log to the frozen manifest commit. The
requirement to create it freshly at the writer-specific path is an
operator-attested procedure; the task does not independently prove that the
path was newly created rather than copied from another same-commit run.

## One measured run

After freezing the manifest, set only the local paths:

```bash
ANAMNESIS_WRITER_MANIFEST="$PWD/results/runs/local/writer_diagnostic/experiment.writer.json"
ANAMNESIS_OLLAMA_MODELS="$HOME/.ollama/models"
```

Then run the singleton task once:

```bash
inspect eval eval/anamnesis_local_eval.py@local_anamnesis_writer_diagnostic \
  --model ollama/qwen3:4b-instruct \
  --model-base-url http://127.0.0.1:11434/v1 \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api \
  --log-format eval \
  --log-dir results/runs/local/writer_diagnostic/runs --json \
  -T seed=101 -T repetition=1 \
  -T manifest="$ANAMNESIS_WRITER_MANIFEST" \
  -T ollama_models_dir="$ANAMNESIS_OLLAMA_MODELS"
```

Do not pass `writer_reference`, `oracle_annotations_path`, an embedding path,
or a model-cost config to this task. Preserve the one exact `log_location`
printed by Inspect; a directory glob is not an acceptable reporter input.

Set that exact path and run the strict reporter once:

```bash
ANAMNESIS_WRITER_LOG="/absolute/path/from/Inspect/log_location.eval"

uv run anamnesis-writer-report \
  --manifest "$ANAMNESIS_WRITER_MANIFEST" \
  --scenarios eval/scenarios/writer_diagnostic.v1.jsonl \
  --run "$ANAMNESIS_WRITER_LOG" \
  --csv results/local_writer_w1.csv \
  --markdown results/local_writer_w1.md \
  --provenance results/local_writer_w1.provenance.json
```

Exit status `0` means the measured log was valid and the frozen writer gate
passed. Exit status `2` means the measured log was valid, all three report
artifacts were written, and the writer gate failed; preserve that FAIL report
and stop. An integrity or provenance error raises instead and must not be
interpreted as an experimental FAIL result. The reporter validates the full
measured log before opening the reference and rejects any output path that
collides with the manifest, dataset, reference, input log, reporter source,
task source, model artifact pin, dependency lock, research contract,
architecture contract, preflight, or pricing configuration before writing
results. All three outputs must be below `results/` and outside `results/runs/`;
the latter remains an immutable raw-run archive.

## Frozen acceptance and stopping rule

The reporter compares due candidates as canonical multisets. Each candidate is
matched exactly on:

```text
(checkpoint, action_key, due_at, kind, canonical_payload, summary,
 sorted_evidence_event_ids)
```

`canonical_payload` is deterministic canonical JSON. Evidence event IDs are
sorted before matching. `intent_id` and `occurrence_id` are deliberately
excluded because they are runtime-local identities. Duplicate canonical tuples
remain duplicates and are counted with multiset intersection when deriving
candidate TP, FP, and FN.

The strict writer report must fail unless all of these hold:

- invalid compiler deltas: 0
- accepted compiler deltas: 100%
- due-candidate false positives: 0
- due-candidate false negatives: 0
- complete compiler token, latency, and exact zero provider-API-cost accounting
- no retry, structured-output repair, cache use, manifest drift, or reference
  coverage gap

Final decision actions, precision, recall, and F1 are not part of this writer
gate. The reporter compares deterministic due candidates after replaying the
measured deltas and reporter-only reference deltas through independent fresh
memory instances.

This is one measured W1 run. If any gate fails, record the failure and stop.
Do not create or test W2 on these same 10 cases. Any later prompt revision
requires a newly versioned, freshly authored diagnostic dataset and reference;
otherwise it is post-hoc tuning on the evaluation cases.
