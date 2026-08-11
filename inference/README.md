# JARVIS Stateless Inference Appliance v1

This directory is a standalone Python 3.11 service. It does not import or modify
`jarvis_voice`, and it contains no orchestrator, memory, research, conversation
store, model selector, or tool executor.

The public boundary is a documented subset of OpenAI Chat Completions:

- `GET /readyz` is unauthenticated and returns exactly `{"ready":true}` or
  `{"ready":false}` with `Cache-Control: no-store`.
- `GET /v1/models`, `GET /jarvis/v1/capabilities`, and
  `POST /v1/chat/completions` require `Authorization: Bearer ...`.
- Production binds only `127.0.0.1:8766`, accepts only the exact loopback Host
  allowlist, and admits one request without queueing.

The request and response shapes intentionally follow only the frozen appliance
subset, not the entire OpenAI API. The upstream reference used for the wire
shape is the [Chat Completions API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
and its [streaming event reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events).

## Package layout

- `contracts.py`: exact nested message/tool/response-format validation.
- `strict_json.py`: strict UTF-8/I-JSON parsing plus an auditable RFC 8785
  canonicalizer (UTF-16 key ordering and ECMAScript binary64 numbers). The
  supported numeric domain rejects integers outside `±(2^53-1)`.
- `schema_subset.py`: Draft 2020-12 CommonSchema normalization, local-ref
  inlining, bounds, and instance validation.
- `manifest.py`: strict manifest, safe path resolution, artifact/DLL/wheel
  identity and installed wheel `RECORD` verification.
- `durable.py`: exclusive qualification lock and platform-specific durable
  atomic report replacement.
- `gateway.py`: loopback security, one-request admission, deadline/backpressure,
  cancellation, canonical buffered and SSE responses.
- `llama_engine.py`: managed, authenticated `llama-server` process with the
  frozen no-cache/no-agent flags and explicit slot erase/idle verification.
- `mlx_engine.py` and `mlx_worker_runtime.py`: owner-only managed pipe worker;
  no MLX HTTP server is exposed.
- `qualification.py`: fail-closed report state, runtime identity, official
  OpenAI Python SDK bootstrap through an ephemeral gateway, and numerical gate
  evaluation.

## Commands

Install the local package from an already provisioned Python 3.11 environment:

```text
python -m pip install --no-deps ./inference
```

Strictly check a manifest and all artifacts:

```text
jarvis-inference verify-manifest --manifest /absolute/path/manifest.json
```

Run native qualification. The runner must be a fully local regular executable
under the manifest root. It is launched inside the exclusive qualification
session and receives a fresh challenge, exact identity digest, output path and
manifest path through the `JARVIS_QUAL_*` environment described in
`docs/qualification-evidence-v1.md`. A pre-existing evidence file is not an
input and cannot produce `pass`. The command also exercises a private ephemeral
gateway with the pinned official OpenAI Python SDK:

`JARVIS_GATEWAY_WHEEL` must name the safe relative path of the exact installed
gateway wheel beneath the manifest root. The qualifier records and rechecks its
size and SHA-256, requires installed `jarvis-inference==1.0.0` and
`openai==2.46.0`, and includes all three in runtime identity.

```text
JARVIS_GATEWAY_WHEEL='dist/jarvis_inference-1.0.0-py3-none-any.whl' \
JARVIS_QUAL_GPU_DEVICE='exact-device-identity' \
JARVIS_QUAL_GPU_MEMORY_BYTES='exact-positive-byte-count' \
JARVIS_QUAL_GPU_DRIVER='exact-driver-version' \
JARVIS_QUAL_COMPUTE_RUNTIME='exact-CUDA-or-Metal-runtime' \
JARVIS_QUAL_COMPUTE_CAPABILITY='exact-compute-capability-or-Metal-family' \
jarvis-inference qualify \
  --manifest /absolute/path/manifest.json \
  --fixture-runner /absolute/path/manifest-root/native-fixture-runner
```

Start production only after a matching pass report exists:

```text
JARVIS_INFERENCE_API_KEY='at-least-32-utf8-bytes-of-random-secret' \
JARVIS_GATEWAY_WHEEL='dist/jarvis_inference-1.0.0-py3-none-any.whl' \
JARVIS_QUAL_GPU_DEVICE='same-qualified-device-identity' \
JARVIS_QUAL_GPU_MEMORY_BYTES='same-qualified-positive-byte-count' \
JARVIS_QUAL_GPU_DRIVER='same-qualified-driver-version' \
JARVIS_QUAL_COMPUTE_RUNTIME='same-qualified-runtime' \
JARVIS_QUAL_COMPUTE_CAPABILITY='same-qualified-capability' \
  jarvis-inference serve --manifest /absolute/path/manifest.json
```

On Windows, set the same names with `$env:NAME='value'` in the service account
before invoking the unchanged `jarvis-inference` command. Secrets are never
accepted as command-line arguments.

Missing/corrupt/in-progress/failed/mismatched qualification keeps the process
alive but `/readyz` false. A pass is never bundled or fabricated by this repo.
Engine loading and identity monitoring run in a background supervisor;
`/readyz` only reads cached state and never waits for model load.

## Verification commands

```text
python -m compileall -q inference/src inference/tests
PYTHONPATH=inference/src python -m unittest discover -s inference/tests -v
PYTHONPATH=inference/src python inference/tests/sdk_conformance.py -v
```

The last command requires the pinned `openai==2.46.0` qualification extra and
opens only an ephemeral loopback test server. Native Windows llama.cpp and Mac
MLX/Outlines qualification are separate hardware runs; no repository test may
stand in for those gates.

## Native runtime notes

The manifest identifies every runtime artifact. On Windows it includes the
exact llama.cpp commit/build, executable, sorted complete bundled-DLL allowlist,
GGUF artifacts, and typed parameters. On Apple Silicon it includes the exact
Python executable, all transitive wheels, installed site-packages, model files,
and worker. The MLX environment is expected to be installed offline with
`pip --no-index --require-hashes`; startup verifies both wheel artifacts and all
installed `RECORD` hashes.

`JARVIS_QUAL_GPU_DEVICE`, `JARVIS_QUAL_GPU_MEMORY_BYTES`,
`JARVIS_QUAL_GPU_DRIVER`, `JARVIS_QUAL_COMPUTE_RUNTIME`, and
`JARVIS_QUAL_COMPUTE_CAPABILITY` are captured as native qualification identity.
Unknown required GPU identity keeps qualification fail-closed.
The same `JARVIS_GATEWAY_WHEEL` and the five `JARVIS_QUAL_GPU_*` /
`JARVIS_QUAL_COMPUTE_*` hardware identity variables listed above must be
supplied to production `serve`; otherwise the pass identity does not match and
readiness remains false.

### Local Apple M3 profiles

The local `native/macos-m3` bundle keeps model qualifications independent:

- `manifest.json` / `qualification-report.json`: qualified 0.5B fallback.
- `manifest-qwen3-4b.json` / `qualification-report-qwen3-4b.json`: qualified
  Qwen3 4B primary profile.

Regenerate either manifest without changing the other profile:

```text
native/macos-m3/python/bin/python3.11 native/macos-m3/build_manifest.py --profile qwen2.5-0.5b
native/macos-m3/python/bin/python3.11 native/macos-m3/build_manifest.py --profile qwen3-4b
```

The 4B profile is capped at a qualified `32768`-token context window even
though the upstream model advertises a larger architectural maximum. Serving
always uses the chosen profile's manifest and its matching pass report; a pass
from one profile is never accepted for the other.

The example manifests in `examples/` deliberately contain `REPLACE_*` strings
and negative sizes. They are documentation templates and are intentionally
invalid until every artifact is locally measured and hashed.

The implementation-to-test mapping is in
[`docs/traceability-v1.md`](docs/traceability-v1.md).
