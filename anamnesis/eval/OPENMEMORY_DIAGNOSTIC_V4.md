# OpenMemory structured-generation diagnostic v4 freeze

V4 is a fresh development-only case set frozen before selecting or installing a compatible vLLM model artifact and before writing the v4 task. It responds to the terminal v3 finding: immediate-action semantics appeared in raw outputs, but 12/16 Ollama responses violated the closed decision schema.

## Frozen scope

- Eight fresh cases, seven recall hits, four emit and four no-action expectations.
- One helpful resolution opportunity, six forbidden-influence cases and one empty-recall control.
- Exact IDs, authored surfaces and named entities are disjoint from v1-v3.
- The v3 immediate-action contract and decision schema are the intended semantic interface; evaluator fields remain hidden.
- Recall remains frozen, untrusted and non-authoritative.

The raw/canonical hashes, record hashes and counts are pinned in `decision_diagnostic.v4.manifest.json`. Independent human review remains pending.

## Planned cell

The current Qwen3.5 9B Q4_K_M GGUF cannot be reused by vllm-metal because the model is hybrid and the artifact is a K-quant. Any compatible MLX/safetensors snapshot changes both model bytes and runtime. V4 must therefore be labelled a joint artifact + structured-runtime compatibility diagnostic, never a transport-only comparison with v3.

After this commit, the exact external runtime version, immutable model revision and file hashes, served alias, loopback endpoint, context/output limits, structured backend, request schema, API-key fingerprint policy and server arguments must be frozen before calls. A schema canary precedes the sole 8×2 run. Failure stops the cell; no artifact, prompt, backend or limit may be changed on v4. A new intervention requires fresh v5 cases.

No model or OpenMemory call occurred before this freeze. This is not a hypothesis test, performance benchmark or general vLLM/OpenMemory result.
