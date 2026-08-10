# Local W3 writer diagnostic execution

W3 is one diagnostic-only bundled compiler repair. It is not a baseline, a
single-factor ablation, or hypothesis-test evidence. The blind v4 dataset was
committed before the prompt, and the neutral nine-call fixture was committed
after the prompt and before this runtime.

The frozen matrix is exactly:

- phase: `writer_diagnostic_w3`
- standalone task: `local_model_preflight_w3`
- measured task: `local_anamnesis_writer_diagnostic_w3`
- dataset: `eval/scenarios/writer_diagnostic.v4.jsonl`
- scenarios/checkpoints/compiler calls: `10 / 62 / 39`
- seed/repetition/temperature: `101 / 1 / 0`
- model: `ollama/qwen3:4b-instruct`
- compiler prompt/schema SHA-256:
  `412a63d6b42ea6b5e294401cabbcbacf5a6b7facddbd8fe04ca7b91914c141e5` /
  `8871ff344eb3a2e88a53b964ef2f24f089a72507c69073ec323cf26a428c3030`
- compiler transport SHA-256:
  `57d4c0a6152c5319fcd1adab4071ad010d107f9e65d987c1740fa47adaca1bcc`
- fixture/protocol SHA-256:
  `5628c3c1d7f8e1a5da43d6e567d55ac8e4fbabd8b9c4054325de6f4def1da30c` /
  `7f63c156a8af74ced2d5e5530b3e8083da95c7f46e14e1afafaaf864b3ce1915`
- setup policy: `frozen_w3_semantic_gate_c1_to_c8_d1`
- reference: `eval/oracle/writer_diagnostic_memory_deltas.v4.json`, reporter-only

The gate is exactly compiler cases C1 through C8 followed by decision case D1.
It runs once as a standalone preflight and, only after a pass, once before the
first scenario in the singleton measured task. Cache, retry, repair, and
parallel execution are forbidden. Any standalone failure stops before scenario
execution. A valid measured FAIL is final; no second W3 or W4 run may use v4.

## Commands

Run from a clean source commit after exporting:

```bash
export OLLAMA_NO_CLOUD=1
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
```

Standalone preflight, exactly once:

```bash
inspect eval eval/anamnesis_local_eval.py@local_model_preflight_w3 \
  --model ollama/qwen3:4b-instruct \
  --model-base-url http://127.0.0.1:11434/v1 \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api --log-format eval \
  --log-dir results/runs/local/writer_diagnostic_w3/preflight --json \
  -T seed=101
```

On PASS only, copy the W3 manifest template into
`results/runs/local/writer_diagnostic_w3/experiment.writer_w3.json`. Pin the
clean 40-character source commit, exact standalone `.eval` path/hash, four
prompt/schema hashes, and the W3 system configuration hash; set `status` to
`frozen`.

Measured task, exactly once:

```bash
ANAMNESIS_WRITER_W3_MANIFEST="$PWD/results/runs/local/writer_diagnostic_w3/experiment.writer_w3.json"
ANAMNESIS_OLLAMA_MODELS="$HOME/.ollama/models"

inspect eval eval/anamnesis_local_eval.py@local_anamnesis_writer_diagnostic_w3 \
  --model ollama/qwen3:4b-instruct \
  --model-base-url http://127.0.0.1:11434/v1 \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api --log-format eval \
  --log-dir results/runs/local/writer_diagnostic_w3/runs --json \
  -T seed=101 -T repetition=1 \
  -T manifest="$ANAMNESIS_WRITER_W3_MANIFEST" \
  -T ollama_models_dir="$ANAMNESIS_OLLAMA_MODELS"
```

Strict reporter, once on the single exact `log_location`:

```bash
ANAMNESIS_WRITER_W3_LOG="/absolute/path/from/Inspect/log_location.eval"

uv run anamnesis-writer-report \
  --manifest "$ANAMNESIS_WRITER_W3_MANIFEST" \
  --scenarios eval/scenarios/writer_diagnostic.v4.jsonl \
  --run "$ANAMNESIS_WRITER_W3_LOG" \
  --csv results/local_writer_w3.csv \
  --markdown results/local_writer_w3.md \
  --provenance results/local_writer_w3.provenance.json
```

Reporter exit `0` is a fully validated PASS. Exit `2` is a fully validated
FAIL whose three outputs are preserved. An exception is an integrity failure
and writes no result. The writer gate requires invalid `0`, accepted `39/39`,
candidate FP `0`, and candidate FN `0`; decision actions are diagnostic-only.
