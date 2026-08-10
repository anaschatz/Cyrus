# OpenMemory + vLLM v4 execution contract

This is one development-only joint model-artifact + structured-runtime
compatibility cell. It is not a transport-only comparison with the prior
Ollama cell, because vllm-metal cannot load that hybrid Q4_K_M GGUF. V4 uses
the fresh cases frozen in commit `19027c0` and the immutable MLX snapshot in
`eval/openmemory/vllm_v4_runtime.pin.json`.

## Frozen identity

- Model: `mlx-community/Qwen3.5-4B-MLX-4bit` at commit
  `32f3e8ecf65426fc3306969496342d504bfa13f3`.
- Exact artifact manifest: `1563d753ccd22c5b0e43dd0aa2a452452d04c2b3cdbf5d10b15187926069db7e`.
- vllm-metal source tag `v0.2.0-20260604-074434`, source commit
  `ef776cacac8f8a4219e5e23fc0b50fa72d37d22c`, package version `0.2.0`.
- vLLM package `0.22.0+cpu`; live `/version` must report `0.22.0`.
- MLX `0.31.2`, MLX-LM `0.31.3`, MLX-VLM `0.5.0`, xgrammar `0.2.4`.
- Exact loopback route `http://127.0.0.1:18000/v1`, served alias
  `anamnesis-openmemory-v4`, seed 101, temperature 0, 4096 context, 256 output
  tokens, one sequence, no retries, no speculation, thinking disabled.
- Decision contract `fb35d772872ce518c18b1c86577a2d4062f158b5c91eb079cac381ee574b48b5`.
- Exact JSON schema `dad9152ff0a16ccea5b0fbeb45249e21beb1665e204b2a7247b6e66e1d71ccc8`.
- Schema canary bytes `1172ca39801ff19ad06ff066c4efa431e7b0dd5f4bf0c205650f5a44cad5409c`.

The local API key is a loopback access guard, not a paid-provider credential;
only its SHA-256 is tracked. Provider API cost is exactly $0. Electricity,
hardware, model download time and human review remain unmeasured.

Standard vLLM endpoints attest health, version and served alias. They do not
expose loaded-weight hashes or every launch argument. The runner therefore
rehashes every local model/tokenizer file and binds an operator-declared server
configuration to the same live loopback process. This is explicit evidence,
not a claim that upstream `/v1/models` cryptographically attests model bytes.

## One-attempt procedure

From the clean source commit that contains this contract, start the separately
pinned runtime environment:

```bash
VLLM_PLUGINS=metal \
VLLM_METAL_USE_PAGED_ATTENTION=1 \
VLLM_METAL_MULTIMODAL_MODE=text-only-compat \
/private/tmp/vllm-metal-v0.2.0-20260604/.venv-vllm-metal/bin/vllm serve \
  /private/tmp/anamnesis-vllm-models/Qwen3.5-4B-MLX-4bit-32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --host 127.0.0.1 --port 18000 \
  --served-model-name anamnesis-openmemory-v4 \
  --api-key local-v4-loopback-20260809 \
  --generation-config vllm --max-model-len 4096 --max-num-seqs 1 \
  --structured-outputs-config '{"backend":"xgrammar"}' \
  --no-enable-log-outputs --no-enable-log-deltas
```

Then run exactly once:

```bash
PYTHONPATH=src /private/tmp/anamnesis-test-venv/bin/python \
  -m anamnesis.openmemory_vllm_run \
  --artifact-root /private/tmp/anamnesis-vllm-models/Qwen3.5-4B-MLX-4bit-32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --api-key local-v4-loopback-20260809 \
  --source-commit "$(git rev-parse HEAD)" \
  --output results/runs/local/openmemory_vllm_v4/run.json
```

The first and only setup call is a fresh neutral no-action canary. A transport,
schema, accounting or semantic failure writes `preflight_failed`, exits 2 and
permits zero scenario calls. A pass permits exactly one ordered 8×2 matrix.
There is no retry, repair, cache, alternate artifact/backend/limit or selected
duplicate. Any further intervention requires fresh v5 cases.

Generate the strict report while still checked out at the clean source commit:

```bash
PYTHONPATH=src /private/tmp/anamnesis-test-venv/bin/python \
  -m anamnesis.openmemory_vllm_report \
  --run results/runs/local/openmemory_vllm_v4/run.json \
  --csv results/local_openmemory_vllm_v4.csv \
  --markdown results/local_openmemory_vllm_v4.md \
  --provenance results/local_openmemory_vllm_v4.provenance.json
```

The report exits 0 only when the frozen gate passes and exits 2 for a valid
negative result. Integrity failures raise before outputs. Publication must
include the exact raw run artifact or disclose that a Git-only checkout cannot
independently reconstruct it.
