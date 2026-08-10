# W3-M2-T1 transport-only local preflight

W3-M2-T1 is a diagnostic, not a hypothesis test. It responds to the terminal
W3-M2 failure in which all eight compiler calls consumed the full output budget
in model reasoning and returned no final JSON. It changes exactly one effective
request field: the OpenAI-compatible Ollama request must contain top-level
`reasoning_effort: "none"`.

The model bytes, W3 prompt, response schemas, nine synthetic cases, context and
output budgets, seed, temperature, concurrency, pricing, and zero-retry policy
remain unchanged. The intervention is encoded through Inspect
`GenerateConfig.extra_body`; the strict validator requires both the effective
config and every retained raw API request to carry the exact field. Merely
declaring `reasoning_effort=none` is insufficient because the installed Inspect
Ollama provider omits that value instead of forwarding it.

## Frozen stopping rule

- Make one clean source commit before the call.
- Run exactly one ordered C1-C8,D1 preflight attempt.
- Use no cache, retry, structured-output repair, duplicate attempt, or selected
  rerun.
- If any case fails, publish the exact log and stop with zero scenario calls.
- If all cases pass, publish the preflight only. Scenario execution still
  requires a separately frozen fresh protocol; this transport cell does not
  authorize reuse of any writer diagnostic dataset.

## Command

Start the separately pinned Ollama server with the five local environment pins,
then run from the clean source commit:

```bash
PYTHONPATH=src:. uv run --frozen --no-sync inspect eval \
  eval/anamnesis_local_eval.py@local_model_preflight_w3_m2_t1 \
  --model ollama/qwen3.5:9b-q4_K_M \
  --model-base-url http://127.0.0.1:11434/v1 \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api --log-format eval \
  --log-dir results/runs/local/writer_w3_m2_t1/preflight --json \
  -T seed=101 -T ollama_models_dir="$OLLAMA_MODELS"
```

Provider API cost is exactly zero under the pinned local pricing artifact.
Electricity, hardware ownership, and human research effort remain unmeasured.
