# W3-M2 local model-only preflight

This is a diagnostic model-only ablation. It keeps the frozen W3 compiler
prompt, response schema, decision contract, C1-C8/D1 fixture, seed, and local
execution policy unchanged. The sole treatment is replacing
`ollama/qwen3:4b-instruct` with the byte-pinned
`ollama/qwen3.5:9b-q4_K_M` artifact.

The run is not a hypothesis test and does not reuse the v4 scenario dataset.
Exactly one nine-call preflight is authorized. Cache, retry, and structured
repair are forbidden. Failure is terminal for W3-M2 and permits no scenario
calls. A pass permits only the later freeze of a new blind dataset; it does not
itself authorize a measured scenario run.

The provider API price is exactly USD 0. Electricity, hardware amortization,
and human work remain unmeasured. The old 4B model was removed locally with
user authorization to make room; its historical results and content pins stay
in Git and the artifact remains recoverable from its frozen Ollama manifest.

The task must be executed from a clean commit with the five local-only Ollama
environment variables, one sample/task/connection, raw API logging, and the
absolute Ollama models directory. The strict artifact validator must accept the
result before pass/fail is interpreted.

```bash
export OLLAMA_NO_CLOUD=1
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export ANAMNESIS_OLLAMA_MODELS="$HOME/.ollama/models"

inspect eval eval/anamnesis_local_eval.py@local_model_preflight_w3_m2 \
  --model ollama/qwen3.5:9b-q4_K_M \
  --model-base-url http://127.0.0.1:11434/v1 \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api --log-format eval \
  --log-dir results/runs/local/writer_w3_m2/preflight --json \
  -T seed=101 -T ollama_models_dir="$ANAMNESIS_OLLAMA_MODELS"
```
