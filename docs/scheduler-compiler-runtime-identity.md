# Scheduler compiler runtime identity — 2026-09-10

The scheduler now checks the qualified compiler package at load time, checks
the authenticated inference identity at startup, and binds every proposal
request to that identity. A different model, tokenizer bundle, profile or
runtime cannot produce an accepted proposal under the previous qualification.

## Contract

Authenticated `GET /jarvis/v1/compiler-identity` returns exactly
`schema_version: jarvis-inference-compiler-identity/1` and the four component
digests below. It is available only on a ready appliance. The existing
`/jarvis/v1/identity` contract remains version 1.

`SHA(value)` below means SHA-256 of the inference package's RFC 8785 canonical
JSON bytes. Artifact entries include their verified relative path, byte length
and SHA-256. These definitions are bundle fingerprints, not hashes of arbitrary
human-readable model names.

| Qualification input | Exact source |
| --- | --- |
| `model_artifact_sha256` | `SHA(manifest.model["artifacts"])`, the ordered model artifact list. |
| `model_profile_sha256` | `SHA({"prompt_profile": manifest.prompt_profile, "context_window": manifest.context_window, "capabilities": manifest.raw["capabilities"], "backend": manifest.backend})`. |
| `runtime_sha256` | `SHA(production.identity)`, the identity associated with the loaded engine, including manifest, gateway code, effective configuration and runtime environment. |
| `tokenizer_sha256` | `SHA({"model": manifest.model, "backend": manifest.backend, "python": manifest.python})`. GGUF embeds its tokenizer; MLX loads from its complete verified model directory and pinned Python environment. |
| `compiler_artifact_sha256` | SHA-256 over all top-level `.py` files with valid Python module names in the installed `jarvis_orchestrator` package, sorted by filename, feeding UTF-8 filename, one NUL byte, then file bytes. `COMPILER_ARTIFACT_SHA256` captures this at import. |

The complete tokenizer bundle deliberately causes requalification when its
model or implementation changes. It avoids guessing which tokenizer filenames
a backend might load. Compiler package files must remain unchanged while the
process runs; install a new package and restart after a code change.

The native engines already verify manifest artifact bytes before starting,
launch the verified local executable/worker and model paths, and establish
their ready handshake. The identity route describes that loaded bundle. This
uses the existing trusted local operator and stable installation assumptions;
it is not hardware-backed process attestation or protection against a malicious
local operator rewriting a verified installation during loading.

## Per-request enforcement

The compiler sends `X-Jarvis-Compiler-Identity: SHA(identity_document)` on each
nonstreaming completion. The gateway compares it while selecting the engine
under the same state lock, before `prepare()` or generation. Missing runtime
state, a mismatch, duplicate/malformed pins or a streaming pinned request is
rejected. A pinned request has no automatic retry on a replacement engine,
matching the compiler's zero-retry qualification contract.

A successful response echoes the verified digest. The compiler requires that
exact response header before parsing a proposal. Consequently an older gateway
that ignores the request header cannot silently pass. Startup identity failure
leaves the schedule compiler unavailable while basic assistant composition
continues.

## Wire compatibility and validation

The integrated HTTP regression also exposed an unsupported `tools: []` and an
output schema incompatible with the gateway. Tool omission now means no tools.
Output enum/const nodes declare scalar types; UUID correctness remains enforced
by the deterministic proposal parser rather than an unsupported schema format.

The gateway accepts only the nullable-scalar union
`{"anyOf": [<string/number/integer/boolean schema>, {"type": "null"}]}`.
General unions, nullable objects, unknown keywords and sibling keywords remain
rejected. Existing node/depth/byte bounds apply to both alternatives. There are
at most 64 values per enum and 128 across the expanded schema, still within the
8 KiB enum byte budget; the scheduler schema needs 66 total values.

Regression coverage includes real authenticated HTTP between compiler and
gateway with a synthetic engine, successful proposal compilation, each identity
component mismatch, replacement after discovery, missing response binding,
startup admission, no retry, and nullable schema validation/budgets.

Portable validation: Orchestrator 1,117 tests (1,112 passed, 5 skipped), inference
279 tests (271 passed, 8 skipped); Ruff F821/F401 and `git diff --check` passed.

## Qualification rollout

Deploy and restart both updated packages. Generate fresh inference qualification
for the exact installed gateway/backend, including the nullable proposal schema
on the target backend. Read the new authenticated identity using
`InferenceControlClient(...).compiler_identity()` and record its four digests
with `COMPILER_ARTIFACT_SHA256`, `PROMPT_TEMPLATE_SHA256`, and SHA-256 of
`jcs_bytes(INPUT_JSON_SCHEMA)` / `jcs_bytes(OUTPUT_JSON_SCHEMA)` from that installed
compiler. Bind a newly measured compiler qualification report using its actual
file SHA-256 as `qualification_report_sha256`.

Build `qualification_inputs` from those nine values and compute
`qualification_sha256` with `compiler_qualification_digest(inputs)`. Pin the
actual serialized qualification file in
`scheduler_compiler_qualification_file` / `scheduler_compiler_qualification_sha256`.
Do not relabel old measurements with current hashes: the output schema,
compiler package and gateway identity changed. An identity document alone does
not establish compiler accuracy or approve a release.

This checkpoint implements and tests the identity binding. It does not supply
fresh native/model measurements or activate an unqualified deployment. The
2026-09-08 review's production qualification requirement remains until those
measurements are obtained and approved.
