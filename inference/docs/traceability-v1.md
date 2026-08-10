# Frozen contract traceability

| Contract requirement | Implementation | Automated verification |
|---|---|---|
| Strict peer, Host, auth, media and JSON admission order | `gateway.py`, `strict_json.py` | `test_gateway_integration.py`, `test_contracts.py` |
| Duplicate keys, UTF-8/BOM, non-finite and bounded I-JSON | `strict_json.py` | `test_contracts.py`, `test_strict_json_canonical.py` |
| Exact tools, messages and response formats | `contracts.py` | `test_contracts.py`, `test_gateway_lifecycle.py` |
| Draft 2020-12 CommonSchema, local refs and bounds | `schema_subset.py` | `test_schema.py` |
| RFC 8785 manifest/public canonicalization | `strict_json.py` | `test_strict_json_canonical.py` |
| Same immutable preflight/generation token IDs | `llama_engine.py`, `mlx_engine.py`, `mlx_worker_runtime.py` | `test_llama_engine.py`, `test_gateway_lifecycle.py` |
| Observable-output boundary and one safe retry | `gateway.py` | `test_gateway_lifecycle.py` |
| Canonical non-stream/SSE/tool/usage shapes | `gateway.py` | `test_gateway_integration.py`, `test_gateway_lifecycle.py`, `sdk_conformance.py` |
| Post-output abort; cleanup before success/terminal | `gateway.py` | `test_gateway_lifecycle.py` |
| One request/no queue, byte/event backpressure, disconnect cancel | `gateway.py` | `test_gateway_lifecycle.py` |
| Monotonic circuit breaker and bounded engine termination | `circuit.py`, engine adapters | `test_circuit.py`, `test_gateway_lifecycle.py` |
| llama.cpp frozen flags, environment secret, slot erase | `llama_engine.py` | `test_llama_engine.py` |
| MLX inherited pipes and per-request parser/KV state | `mlx_engine.py`, `mlx_worker_runtime.py` | compile/import suite; native qualification is mandatory |
| Strict manifest, artifacts, DLL allowlist, wheel RECORD | `manifest.py` | `test_manifest.py`; native platform qualification |
| Durable report replacement and fail-closed readiness | `durable.py`, `gateway.py`, `qualification.py` | `test_qualification.py`, `test_gateway_lifecycle.py` |
| Challenge-bound same-session native evidence | `qualification.py` | `test_qualification.py`; native fixture runner |
| Official sync/async OpenAI SDK subset | `qualification.py` | `sdk_conformance.py`; repeated during native qualification |
| Numerical streaming/tool/schema/cancel/crash/isolation/platform gates | `qualification.py` | `test_qualification.py`; exact native fixtures |

Native qualification remains authoritative for model-specific token accounting,
chat-template/tool encoding, Outlines integration, GPU/ASR coexistence, and
unified-memory stability. Unit tests cannot mint a production pass report.
