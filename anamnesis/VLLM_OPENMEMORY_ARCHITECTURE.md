# Optional vLLM and OpenMemory architecture

Status: implementation-ready boundary, not an executed experiment cell.

The integration keeps Anamnesis authoritative. vLLM is an optional inference
transport for the existing memory compiler contract. OpenMemory is exposed
through an optional semantic-recall sidecar. Neither component may alter the
identity or results of the frozen Ollama W1-W3 cells.

## Roles and authority

| Component | Useful capability | Allowed authority | Explicitly forbidden |
| --- | --- | --- | --- |
| Anamnesis | Typed facts, prospective intents, occurrences, executions, provenance, atomic reducer and audit | Sole authority for temporal state, due actions and execution evidence | Delegating reducer decisions to either sidecar |
| vLLM | Efficient model execution behind an OpenAI-compatible API and constrained decoding | Produce an untrusted codec-selected wire candidate; the local W3 path uses `LocalMemoryDeltaWire` | Storing memory, accepting a delta, selecting thresholds, or changing frozen Ollama cells |
| OpenMemory | Persistent semantic storage, sector embeddings, similarity/salience/recency/waypoint recall | Return non-authoritative retrospective context | Creating/cancelling triggers, declaring facts current, marking executions, or supplying authoritative evidence IDs |

This separation reflects what the projects actually provide. vLLM documents an
OpenAI-compatible structured-output API, while OpenMemory describes a
Hierarchical Memory Decomposition store with five sectors, SQLite persistence,
embeddings, decay, waypoints and composite recall ranking. Those are useful but
different concerns from Anamnesis's deterministic temporal reducer.

Primary references:

- [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [vLLM OpenAI protocol source](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/completion/protocol/)
- [OpenMemory architecture](https://github.com/CaviraOSS/OpenMemory/blob/main/ARCHITECTURE.md)

## Data flow

```text
ObservableEvent + closed CompilerStateView
                  |
                  v
 W3 rules in system role + canonical JSON data in user role
                  |
                  v
 external loopback vLLM server  (untrusted inference)
                  |
                  v
 exact response_format json_schema
                  |
                  v
 envelope -> finish_reason -> JSON -> wire -> domain -> reducer dry-run -> usage
                  |
          all gates must pass
                  |
                  v
      authoritative Anamnesis reducer + audit
                  |
       accepted projections only
                  v
 OpenMemory recall sidecar (optional, non-authoritative)
```

OpenMemory recall may be appended to a dedicated diagnostic prompt section, but
it must be visibly labelled non-authoritative. The decision path must prefer the
Anamnesis structured view whenever the two disagree. A sidecar outage, empty
recall, malformed response or version mismatch degrades to no sidecar context;
it must never change the authoritative reducer state.

## Implemented vLLM boundary

`src/anamnesis/vllm_runtime.py` is intentionally a library adapter, not a
server launcher:

- It imports no `vllm` package and installs nothing.
- It accepts an injected OpenAI-compatible client and an injected live probe,
  so all tests run offline.
- The supplied concrete client uses the project's existing `openai` dependency
  and talks only to an exact `http://127.0.0.1:<port>/v1` URL.
- It requires an API key and compares its SHA-256 fingerprint without storing
  the secret in the pin.
- The injected transport must expose the same pinned endpoint and key
  fingerprint, and the live probe must independently report that endpoint and
  the loaded artifact-manifest digest.
- It hashes the exact model/tokenizer snapshot, rejects moving revisions,
  missing files, extra files, changed sizes, changed bytes and escaping
  symlinks.
- It requires exactly one served-model alias and exact vLLM/package versions.
- It fingerprints the complete JSON server configuration and independently
  compares the settings that most directly affect this cell: structured-output
  backend, generation-config policy, context length, sequence concurrency and
  speculative decoding.
- It pins the complete current `anamnesis_runtime_contract_v2()` digest, the
  selected codec identity, codec contract and exact codec schema.
- The concrete local path uses `LocalMemoryDeltaWire` and all W3 instructions.
  Instructions occupy the system message; the observable event and validated
  `CompilerStateView` occupy a separately serialized canonical JSON user
  message. Event text can therefore contain quotes, delimiters or role-like
  strings without creating a new chat message or altering the system content.
- The former hosted `MemoryDeltaWire`/combined-prompt codec remains available
  only as an explicitly pinned compatibility path; it is not the local W3
  architecture correction.
- It re-probes live server identity before every model request. Multi-gigabyte
  artifact bytes are hashed once per runtime instance.
- The exact request timeout is pinned and checked against the injected client;
  SDK retries remain disabled.
- It never emits an accepted `MemoryDelta` unless envelope, response model,
  `finish_reason`, JSON, wire schema, domain conversion, reducer dry-run and
  usage are independently valid.
- A transport or attestation failure raises before a candidate delta exists.
- Every completion outcome carries the same-request attestation and canonical
  request SHA-256; the compiler exposes that outcome for a future strict
  reporter instead of requiring a racy second probe.

The authoritative reducer probe is required. `AnamnesisReducerProbe` calls
`InMemoryAnamnesis.validate_delta(event, delta)`, which forks the exact live
store and dry-runs the real reducer without modifying facts, intents, events or
audit state. There is deliberately no "allow all" default.

## Implemented OpenMemory boundary

`src/anamnesis/openmemory_recall.py` is a deliberately narrow async adapter:

- the caller injects an already constructed client plus declared revision,
  database path and local embedding policy; there is no implicit import or
  construction with pre-check side effects;
- the concrete main-branch shim translates `metadata` to upstream `meta`,
  normalizes SQLite row objects, and hydrates search hits through `get` so the
  namespace/user ownership can be verified before provider IDs are stripped;
- callers must supply an explicit namespace and user partition;
- `local_only=True` requires positive, inspectable locality evidence and fails
  before the first provider call when that evidence is absent or contradictory;
- provider record IDs are replaced with opaque process-local handles;
- normalized search/get outputs are closed, immutable, labelled
  `authoritative=false`, and structurally incapable of carrying action evidence;
- malformed envelopes, scores, content, IDs, or deletion acknowledgements fail
  closed, and content/query/metadata/result byte caps bound the recall surface;
  and
- the module neither imports nor receives `InMemoryAnamnesis`.

`src/anamnesis/openmemory_strategy.py` is the additive runner integration. It
wraps the ordinary authoritative `AnamnesisMemoryStrategy`, opens a fresh
caller-supplied recall snapshot on every reset, performs search only, and puts
the returned text in a separately labelled untrusted JSON prompt section. It
rejects any index that does not explicitly declare itself non-authoritative,
evidence-free, and unable to mutate Anamnesis. Its `commit` method delegates
only to the deterministic store; OpenMemory receives no decision or execution
write. Because the provider exposes no neutral usage accounting, this arm is
diagnostic-only and marks usage and cost completeness false.

This is an integration seam, not a claim that the current upstream rewrite is
stable enough for a frozen benchmark. The revision/path/provider carried by the
shim are caller attestations, not proof of installed bytes or live database
configuration. A measured OpenMemory arm would still need independent
verification of an exact upstream commit, package lock, embedding artifact,
database snapshot/reset policy, recall thresholds, and reinforcement/decay
policy.

## Architecture-v2 core hardening

The comparison also exposed bugs below the provider layer. They are corrected
under `anamnesis_runtime_contract_v2()`, which is now the only current
executable identity. The byte-stable v1 dictionary remains available only as
explicit historical artifact metadata; current task and fallback hashes cannot
claim it:

- `CompilerStateView` exposes only active, wire-aligned facts, conditions,
  triggers, templates, and stable intent IDs—not revision IDs, status,
  validity intervals, action keys, or provenance internals;
- fact identity uses a lossless length-prefixed key, so dotted entity and
  attribute boundaries cannot alias;
- create/update rejects past absolute triggers, expired condition windows,
  recurrences with no future valid occurrence, and ranges over 366 days;
- condition truth is tracked before `active_from`, contract changes establish a
  fresh baseline, and unitless predicates no longer wildcard unit-bearing facts;
- semantic no-op intent updates are rejected atomically;
- due-candidate and execution evidence is produced by the deterministic store
  and always includes the checkpoint; arbitrary model citations never poison
  the execution ledger;
- wrong payloads no longer mark an occurrence executed, duplicate evidence IDs
  cannot score as exact provenance, and reducer rejection reasons are retained
  in new checkpoint audits; and
- `InMemoryAnamnesis.validate_delta()` provides the side-effect-free reducer
dry-run required by the vLLM gate.

### Post-v4 schema alignment

The first frozen OpenMemory + vLLM cell showed that grammar enforcement is
only as strong as the supplied schema: the published v4 `actions` array had no
upper bound, so six outputs legally repeated an action until `max_tokens`, and
one additional wire-valid output failed the domain's two-token subject rule.

The additive `VllmAlignedDecisionWire` now moves both discovered invariants
into the constrained-decoding surface:

- `actions` has `maxItems: 1`; combined with the existing mode validator this
  permits zero actions for `no_action` and exactly one for `emit`;
- `payload.subject` must contain at least two lowercase, space-separated tokens
  using the closed action vocabulary character set; and
- `VllmOpenMemoryAlignedDecisionModel` binds a separately versioned contract,
  schema name and schema hash while the published v4 hashes remain byte-stable.

The exact aligned schema compiles under the pinned xgrammar 0.2.4 backend.
This is an engineering correction, not a post-hoc v4 result. A separately
frozen two-call v5 compatibility cell has now exercised it once: the constrained
`emit` and `no_action` outputs both passed every structural and semantic gate.
That result validates the schema/runtime boundary but does not measure recall
quality or reopen the frozen v4 cases.

These corrections are an architecture revision, not a silent mutation of the
published experiments. Any result using them needs a new preregistered cell,
clean source commit, v2 contract hash, fresh preflight, and new result identity.

### Post-v6 action canonicalization

The real indexed-memory diagnostic showed that retrieval could supply the
correct missing value while an otherwise valid action still failed exact
scoring because the value occupied the wrong optional slot, the subject used a
generic object, or a redundant slot repeated the subject. The additive
`immediate-action-canonicalizer.v1` is a conservative post-validator for a
future cell. It may only move or rewrite values already present in the current
event or retrieved text, preserves action identity and evidence, leaves
no-action and noncanonical-provenance decisions untouched, and records every
change. It never changes the published v4-v6 artifacts.

Applied post hoc to v6 for diagnosis only, it canonicalizes all eight recall
decisions to their exact expected payloads. That is not a result or promotion:
prospective evidence requires a separately frozen v7 dataset with the same
versioned rules and no reuse of v6 cases.

No `pyproject.toml` entry point was added. Launching a vLLM service inside the
evaluation process would mix dependency installation, model loading and
inference identity with the measured code. The external deployment must own its
launcher and implement `VllmRuntimeProbe`; Anamnesis only consumes attested
evidence.

## Exact structured-output request

The adapter sends one and only one constraint mechanism:

The primary local W3 request has this role boundary:

```json
{
  "model": "<pinned-served-model-alias>",
  "messages": [
    {
      "role": "system",
      "content": "<exact W3 instructions plus pinned data-boundary rule>"
    },
    {
      "role": "user",
      "content": "{\"active_state\":<canonical CompilerStateView>,\"current_event\":<canonical ObservableEvent>}"
    }
  ],
  "temperature": 0.0,
  "seed": 101,
  "max_tokens": 512,
  "n": 1,
  "stream": false,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "anamnesis_vllm_local_w3_memory_delta",
      "schema": "<exact LocalMemoryDeltaWire.model_json_schema()>"
    }
  },
  "chat_template_kwargs": {"enable_thinking": false}
}
```

The actual seed and token limit come from the immutable runtime pin; the numbers
above are illustrative. The event and state are validated typed objects before
canonical serialization; event text never participates in message construction
and cannot create a `system` role. This structural boundary reduces formatting
and role-injection collisions, but model behavior remains untrusted and still
passes through every validation gate. vLLM's official example uses this
`response_format` shape. The adapter does not also send `structured_outputs`, legacy
`guided_json`, tools, regex or grammar. It omits a nested `strict` option because
the current vLLM example does not require it; the exact closed Pydantic schema
is the constraint, and Anamnesis still performs every post-generation gate.
The vLLM docs state that the legacy guided fields were removed in v0.12.0.

The server backend must be explicitly pinned to `xgrammar` or `guidance`; `auto`
is rejected. vLLM says that `auto` makes opinionated choices based on request
and backend-library support and can change between releases. Likewise,
`--generation-config vllm` is mandatory because the CLI otherwise defaults to
loading generation defaults from the model directory.

Primary references:

- [Structured-output `response_format` example](https://docs.vllm.ai/en/latest/features/structured_outputs/#online-serving-openai-api)
- [Structured backend configuration and `auto` warning](https://docs.vllm.ai/en/stable/api/vllm/config/structured_outputs/)
- [`vllm serve` configuration reference](https://docs.vllm.ai/en/latest/cli/serve/)

## Required external launch cell

A real cell must be declared before use, with concrete values replacing every
placeholder. A representative command is:

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve /absolute/path/to/pinned/snapshot \
  --host 127.0.0.1 \
  --port <dedicated-port> \
  --api-key <run-secret> \
  --served-model-name <pinned-alias> \
  --generation-config vllm \
  --structured-outputs-config.backend xgrammar \
  --max-model-len <pinned-length> \
  --max-num-seqs 1
```

That command is not sufficient by itself. Before any benchmark call, archive:

1. the full immutable model repository commit;
2. the sorted path/size/SHA-256 manifest of every model and tokenizer file;
3. exact versions for vLLM and every runtime/plugin package in the pin;
4. the canonical `/server_info?config_format=json` fingerprint (the current
   endpoint is exposed in server dev mode, hence the pinned environment flag);
5. `/health`, `/version` and `/v1/models` evidence;
6. the exact local-W3 schema, codec contract, Anamnesis runtime-v2 contract,
   API-key fingerprint and request-generation settings;
7. a synthetic structured-output canary outside measured latency.

Do not infer the server environment from the client Python environment. The
injected probe must collect package versions inside the process/container that
actually hosts vLLM.

## OpenMemory integration policy

OpenMemory's query flow classifies a query, embeds it, searches sectors, expands
one graph hop, ranks with similarity/salience/recency/waypoint terms and then
reinforces recalled memories. Its own architecture also includes scheduled
decay and waypoint pruning. Therefore recall is stateful and order-sensitive by
default, not a neutral replacement for Anamnesis's versioned state.

The only acceptable first integration is a separately named recall diagnostic:

- ingest only a projection of already accepted Anamnesis audit records;
- retain the originating Anamnesis event/revision IDs as opaque metadata;
- query only at a declared checkpoint and cap returned bytes/tokens;
- never convert a recall hit directly into a fact, intent, cancellation,
  occurrence, action or execution record;
- never accept evidence IDs invented or inferred by OpenMemory;
- pin its repository commit, package lock, embedding provider/model/artifact,
  sector rules, score weights, thresholds, top-k, decay, reinforcement and
  waypoint settings before evaluation;
- clone/reset the OpenMemory database for every arm or disable query
  reinforcement in a preregistered fork; otherwise scenario order changes the
  retrieval state;
- record recall IDs and scores for audit, but keep them outside authoritative
  action provenance;
- fail closed to "no recall context" on any mismatch.

OpenMemory is therefore **recall-only and non-authoritative**. It is useful for
fuzzy retrospective context that Anamnesis does not currently target, but its
salience decay and similarity score are not evidence that a prospective trigger
is current or due.

Primary reference: [OpenMemory query, scoring, reinforcement and decay](https://github.com/CaviraOSS/OpenMemory/blob/main/ARCHITECTURE.md#22-memory-operations).

## Architectural weaknesses found and corrected

| Weakness | Consequence | Correction |
| --- | --- | --- |
| A provider adapter can silently change a frozen experiment identity | Comparisons combine architecture and runtime changes | New vLLM module and cell only; no edits to Ollama W1-W3 |
| The first adapter reused the hosted flat wire and combined hosted prompt | It did not exercise the local W3 architecture and let event formatting collide with instructions | Injectable codecs; primary local path pins `LocalMemoryDeltaWire`, puts W3 rules in `system`, and canonical typed data in one `user` message |
| vLLM defaults depend on model files and release | Hidden sampling/backend drift | Exact package/config pins, `generation_config=vllm`, explicit `xgrammar` or `guidance`, never `auto` |
| A valid grammar proves syntax, not Anamnesis semantics | Invalid updates can reach the store | Separate JSON, codec-selected wire, domain and exact reducer dry-run gates |
| A mock-only reducer probe is easy to misconfigure | Integration is not plug-and-play and can accidentally skip semantic validation | Concrete `AnamnesisReducerProbe` over the non-mutating authoritative reducer fork; no permissive default |
| Truncated output can still be valid JSON | Partial delta could be accepted | Require and record `finish_reason == "stop"` |
| A model alias does not identify model bytes | Tokenizer/weights can change under the same name | Commit plus exact sorted file/size/hash manifest and exact served alias set |
| Client environment pins do not identify a server/container | False reproducibility | Injected probe reports the serving environment and exact configuration |
| OpenMemory retrieval mutates salience/waypoints | Scenario-order leakage | Snapshot/reset or preregistered no-reinforcement mode |
| Similarity recall can look authoritative | False facts, stale triggers or invalid evidence | Recall-only labelled context; reducer and execution ledger remain authoritative |
| One generic parse-error bit hides failure cause | Diagnostics cannot distinguish model, schema and semantics | Expose independent validation layers plus raw `finish_reason`; a measured cell must persist the report rather than only the legacy `CompilerCall` bit |

## Apple M3 feasibility and live blocker

The development host is an Apple M3 with 16 GB unified memory. Upstream vLLM
currently labels native Apple-Silicon CPU support experimental, requires a
source build and lists FP32/FP16 support. GPU acceleration is provided by the
community-maintained `vllm-metal` plugin. Its package metadata currently marks
itself `Development Status :: 3 - Alpha` and pins ABI-sensitive dependencies,
including an exact MLX version and temporary compatibility caps. These are
material experimental-cell variables, not invisible installation details.

More importantly, the frozen Ollama artifacts cannot be reused byte-for-byte.
The local cells pin Qwen3 4B and Qwen3.5 9.7B GGUF artifacts at `Q4_K_M`.
vllm-metal's support matrix explicitly rejects GGUF K-quants; its accepted GGUF
scope is per-tensor Q8_0/Q4_0/Q4_1. Converting or downloading an MLX,
Safetensors, Q4_0 or Q4_1 artifact changes model bytes and often the loader and
tokenizer stack. Such a run is a joint model-format/runtime intervention, not a
transport-only replacement and not the same W1-W3 identity.

Accordingly, **live vLLM execution on this M3 remains blocked for the current
frozen cells**. The code and offline tests are ready, but no installation,
conversion, download or live request is authorized by this integration. A
future Apple diagnostic needs its own preregistration, model artifact, runtime
pins, canary and label. A remote Linux/CUDA loopback deployment would likewise
be a new cell and would need the same attestation contract.

"Zero model cost" in such a cell means no metered provider invoice. It does not
mean zero electricity, hardware, engineering or hosting cost.

Primary references:

- [Upstream vLLM Apple-Silicon CPU installation status](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html?device=apple)
- [vLLM Metal project and community-maintained status](https://github.com/vllm-project/vllm-metal)
- [vLLM Metal package version, alpha classifier and dependency pins](https://github.com/vllm-project/vllm-metal/blob/main/pyproject.toml)
- [vLLM Metal supported models and GGUF restrictions](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md)

## Test and promotion gates

The current offline suite covers endpoint rejection, API-key mismatch, artifact
tamper/missing/extra files, schema drift, `auto` backend rejection, every live
attestation mismatch, exact request shape, transport failure, non-stop finish,
invalid JSON, invalid wire, wire-valid/domain-invalid output, reducer rejection,
zero or inconsistent usage, response-model mismatch, local-W3 schema selection,
canonical system/user separation under adversarial event text, codec mismatch,
Anamnesis runtime-v2 drift and side-effect-free concrete reducer validation.

A new vLLM cell may advance to smoke evaluation only after:

1. focused and full repository tests pass;
2. all static pins are committed before model output is observed;
3. the external server attests exactly and the structured canary passes without
   retry, repair or fallback;
4. the new cell receives a distinct experiment identity;
5. any OpenMemory thresholds and state-reset policy are preregistered;
6. acceptance/rejection criteria remain unchanged after observing results.
