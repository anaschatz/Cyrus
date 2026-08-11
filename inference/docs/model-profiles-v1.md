# Model profiles v1

This document freezes the model roles and their production admission gates for
the JARVIS Stateless Inference Appliance. It adds profile qualification on top
of the wire, lifecycle, isolation, and platform gates in
`qualification-evidence-v1.md`; it does not weaken or replace them.

The words MUST, MUST NOT, SHOULD, and MAY are normative. A selected model is not
a production-ready model. Production accepts a profile only after the exact
local artifacts, runtime, configuration, hardware identity, and profile suite
produce a matching `pass` report.

## Frozen selection

| Profile ID | Role | Selected model | Runtime | Production capabilities | Status |
|---|---|---|---|---|---|
| `local-primary` | Daily bilingual conversation, strict structured output, and tool planning | `google/gemma-4-12B-it-qat-q4_0-gguf`, text-only `gemma-4-12b-it-qat-q4_0.gguf` | Pinned `llama-server` on Windows x86-64 / RTX 5080 | `text`, `stream`, `tools`, `json_object`, `json_schema` | selected, not qualified |
| `local-fallback` | Useful degraded bilingual conversation when the PC is unavailable | `mlx-community/Qwen3-4B-Instruct-2507-4bit` candidate | Pinned MLX-LM worker on the exact MacBook Air M3 / 16 GB identity | `text`, `stream` only | deterministic bilingual quality failed; not production ready |

The primary artifact is the official Google QAT Q4_0 GGUF at repository commit
`29d097773436b69ff9feafd636ab4cf873786537`:

```text
path:   gemma-4-12b-it-qat-q4_0.gguf
size:   6975879296 bytes
sha256: 93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b
```

The profile does not include `mmproj`, audio, vision, MTP/draft weights, or any
other repository artifact. It starts `llama-server` with `--no-mmproj` and the
existing appliance hardening flags. The v1 qualified context cap is `32768`
rendered tokens and the per-request completion cap is `4096` tokens. A larger
architectural context is not a qualified capability.

The fallback reuses the already materialized local source revision
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. Its current core identities are:

```text
model.safetensors
  size:   2263022417 bytes
  sha256: 2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f
tokenizer.json
  size:   11422654 bytes
  sha256: aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
chat_template.jinja
  size:   4040 bytes
  sha256: 40c21f34cf67d8c760ef72f8ad3ae5afad514299d4b06e91dd9a8d705af7b541
config.json
  size:   938 bytes
  sha256: 574349e5a343236546fda55e4744a76e181f534182d7dc60ff1bad7e7a502849
```

Its final manifest MUST continue to enumerate and hash every weight,
tokenizer, configuration, generation configuration, chat template, runtime,
wheel, and worker artifact; the four entries above are not a substitute for
the complete manifest. The v1 fallback context cap is `8192` rendered tokens
and the per-request completion cap is `1024` tokens. Changing the current local
`32768` manifest cap therefore requires a new manifest and qualification.

`Qwen/Qwen3-14B-GGUF` file `Qwen3-14B-Q4_K_M.gguf` is the frozen comparison
control for the first Windows qualification. It is not a hidden fallback and
MUST NOT be activated if Gemma fails. Selecting it would require a new profile
revision, complete artifact lock, and native qualification.

The local Qwen2.5 0.5B bundle is a development/transport probe only. It MUST
NOT be advertised or routed as `local-fallback` regardless of whether a legacy
protocol report says `pass`.

## Profile ID and public model boundary

`local-primary` and `local-fallback` are operator profile IDs, not OpenAI model
names. Exactly one profile is selected when an appliance process starts.

The public API continues to accept and emit exactly the logical model alias
`jarvis-local`. `/v1/models` exposes only that alias. A request containing a
profile ID, upstream repository name, branch, tag, filesystem path, URL, or any
other model value is rejected. The gateway never selects or downloads a model
from request data.

The orchestrator chooses which already-running appliance endpoint to use. The
gateway does not perform PC-to-Mac failover and does not gain memory,
conversation, research, or routing state. Automatic failover, if later added,
is an orchestrator milestone and MUST preserve the same capability checks.

## Exact profile lock

Before qualification, each profile MUST have one strict, RFC 8785-digested
profile lock bound into the manifest/report identity. Unknown fields are
rejected. The lock records at least:

- profile schema version and exact profile ID;
- upstream repository and full 40-hex source revision used only at intake;
- local relative path, exact byte size, and SHA-256 of every artifact;
- model format and quantization, tokenizer/config/template fingerprints, and
  adapter identities;
- exact context and completion caps, sampling defaults, stop behavior, KV
  cache types, batch settings, GPU-layer placement, and thinking mode;
- exact renderer ID/hash, tool parser ID/hash, schema constraint implementation,
  and canonical-output implementation;
- gateway wheel, engine build/commit, DLL or Python/wheel identities, OS,
  driver/runtime, and hardware identity already required by manifest v1;
- qualification suite version, fixture digests, deterministic seeds, and the
  durable qualification-report digest.

For GGUF, hashing the file locks embedded metadata but does not by itself prove
that the appliance renderer and parser match the model protocol. The Gemma 4
text renderer and function-call parser MUST be explicit, hashed appliance code
and MUST pass the exact rendered-token and tool-history fixtures. No mutable
Jinja template is fetched or interpreted at runtime.

No placeholder, short revision, `main`, `latest`, Ollama tag, model hub alias,
or remote URL is a valid production identity. Intake is a separate explicit
operator action. Qualification and serving run offline from a content-addressed
local directory, fail closed on any hash mismatch, and record zero model
resolution or network-download attempts.

## Functional scope

`local-primary` is the only v1 profile allowed to plan tool calls or produce
strict structured output. Tool execution remains entirely in the orchestrator.
Its target manifest capabilities become true only after the exact model/runtime
combination passes every capability-specific gate.

`local-fallback` is useful for short Greek/English conversation,
clarification, bounded summarization, rewriting, intent restatement, and simple
offline questions. It is not a second planner. Its manifest keeps `tools`,
`json_object`, and `json_schema` false. Requests requiring tools, actions,
research, large context, or primary-only reasoning are not silently attempted;
the orchestrator returns or handles a deterministic `primary_required` outcome.
The gateway enforces declared capabilities and limits but does not infer user
intent or execute this routing policy.

## Frozen quality and structured gates

All corpora, expected results, rubric versions, evaluator code, adjudications,
and case order are hashed qualification fixtures. A model-generated judge is
not the sole authority. Closed tasks use deterministic scoring; open answers
use a pre-registered bilingual rubric and blinded human review. A critical
failure is a wrong action/tool, invented completion of an action, privacy leak,
failure to escalate an explicitly primary-only case, or output outside the
declared wire/schema contract.

| Gate | `local-primary` | `local-fallback` |
|---|---:|---:|
| Bilingual quality corpus | 60 Greek + 60 English | 40 Greek + 40 English |
| Minimum accepted cases per language | 90% | 85% |
| Maximum Greek/English pass-rate gap | 10 percentage points | 10 percentage points |
| Critical failures | 0 | 0 |
| Existing streaming protocol cases | 20 Greek + 20 English, all pass | 20 Greek + 20 English, all pass |
| Strict JSON-Schema cases when declared | 20/20 parse and validate, no extra text | capability disabled; 20/20 requests rejected deterministically |
| Named/required tool cases when declared | 20/20 wire/schema valid; at least 19/20 exact tool-and-argument match | capability disabled; 20/20 requests rejected deterministically |
| Out-of-scope fallback routing cases | not applicable | 20/20 produce the expected `primary_required` route before inference |

Any malformed structured output is a failed case even if a retry later repairs
it. The production request contract still permits only its single safe
pre-output retry for transport/backend failure; profile evaluation MUST NOT add
a hidden semantic regeneration loop.

## Frozen performance fixture and gates

Performance uses 100 hashed, production-shaped prompts: 50 Greek and 50
English, including real chat history, system instructions, and representative
structured/tool schemas for the primary. Rendered prompt bins are `512`,
`2048`, and `4096` tokens; the requested output is `128` tokens, concurrency is
one, and the same tokenizer/rendered token sequence used for generation is
recorded. Results report p50/p95 time to first token, p50/p95 end-to-end latency,
and p5/p50/p95 accepted decode tokens per second. Load time is reported
separately and is never included in warm request latency.

| Metric | `local-primary` on the qualified RTX 5080 | `local-fallback` on the qualified M3 / 16 GB Mac |
|---|---:|---:|
| p95 time to first token | `<= 3.0 s` | `<= 2.0 s` |
| p95 end-to-end for 128 accepted tokens | `<= 10.0 s` | `<= 8.0 s` |
| p5 accepted decode rate | `>= 25 tokens/s` | `>= 20 tokens/s` |
| p95 cold load-to-ready | `<= 120 s` | `<= 30 s` |
| Timed request success | `100/100` | `100/100` |

Thresholds are evaluated per profile on the exact power/runtime identity. A
mean or a synthetic one-line prompt cannot substitute for the percentiles and
fixture above.

## Eight-hour soak and memory headroom

Each profile runs for at least `28800` monotonic seconds and completes at least
`96` load -> request set -> cleanup -> unload cycles. At least every twelfth
cycle includes a deliberate full worker/process restart, for at least eight
restart cycles. The run requires:

- all 96 cycles and all admitted requests complete;
- zero unexpected OOM, crash, hang, cleanup failure, stale request state,
  circuit-breaker opening, download attempt, or identity drift;
- zero prompt/output/schema/tool bytes in logs or durable files;
- memory after cleanup returns to the profile's pre-registered steady-state
  envelope without monotonic growth.

The Windows primary additionally repeats the frozen ASR coexistence fixture. It
requires at least `2048 MiB` minimum observed free VRAM throughout active
ASR/prefill/decode overlap, all model layers assigned as declared with no
unplanned CPU offload or WDDM eviction, zero throttle/OOM/crash, no lost final
transcript, and ASR p95 degradation no greater than `15%`.

The Mac fallback retains the existing 20-warmup/200-request unified-memory
drift gates. During the eight-hour soak it additionally requires at least the
greater of `4096 MiB` or `25%` of installed unified memory available at the
minimum sampled headroom, and no monotonic swap growth after warm-up.

The RTX 5080's nominal capacity or a model-card memory estimate is not evidence
for these gates. Native measurements include model weights, KV state, CUDA or
Metal runtime, kernels, buffers, the gateway, and concurrently active ASR.

## Ready decision

A profile is ready only when all of the following are true at the same time:

1. every model/config/tokenizer/template/runtime artifact has a complete local
   hash/size identity and matches it immediately before spawn;
2. the profile lock, effective configuration, renderer/parser, fixture corpus,
   driver/runtime, OS, and hardware identity exactly match a durable report;
3. all appliance v1 protocol, privacy, cancellation, isolation, crash, platform,
   quality, structured, performance, soak, and headroom gates pass;
4. qualification observed zero unexpected download/model-resolution attempts;
5. the report status is `pass`, not a legacy, copied, candidate, failed,
   in-progress, stale, or identity-mismatched result.

The current Mac Qwen3 4B report was produced on this Apple M3 / 16 GB hardware
and is useful protocol/isolation evidence, but it predates this profile suite.
It therefore does not yet make `local-fallback` production ready on this Mac.
There is currently no Windows bundle or native report for `local-primary`. No
pass report may be fabricated to bridge either gap.

On 2026-08-11 the exact Qwen3 4B candidate ran the frozen deterministic corpus
`cc3814b05807908f1235db815f078f5ac2a4034ad3844e9dcc03a85c5d1fdd71` on
this M3 / 16 GB identity. All 80 protocol streams completed and performance was
acceptable (`0.735 s` p95 TTFT, `1.851 s` p95 end-to-end, `23.34` p5 decode
tokens/s), but deterministic quality failed: Greek `14/40`, English `26/40`,
with a 30-percentage-point gap. This candidate MUST NOT proceed to the
eight-hour soak unless a new frozen configuration independently clears the
quality gate.

## First native qualification order

1. Materialize the selected Gemma GGUF and pinned `llama-server` bundle on the
   Windows machine, verify every byte/hash, and create the exact profile lock.
2. Implement and hash the Gemma 4 text/tool renderer/parser adapter, keeping the
   public appliance wire contract unchanged.
3. Run the primary functional, bilingual, structured, performance, ASR
   coexistence, and eight-hour soak gates.
4. Keep the failed Qwen3 4B result as a baseline and qualify a separately pinned
   Mac fallback challenger on the exact MacBook Air M3 / 16 GB identity. Do not
   spend an eight-hour soak on any candidate before it clears bilingual quality.
5. Enable a profile only after its native report passes. If Gemma fails, keep
   the primary unready and evaluate the frozen Qwen3-14B control under a new
   explicit profile revision; never switch silently.

## Selection evidence

- The official [Gemma 4 QAT GGUF model card](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)
  publishes the 12B Q4_0 artifact, Apache-2.0 license, and broad local runtime
  support. Google's [Gemma 4 function-calling documentation](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4)
  defines model-specific function-call behavior that still requires exact
  appliance qualification.
- NVIDIA specifies [16 GB GDDR7 for the RTX 5080](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/),
  which is why the primary decision reserves measured headroom instead of
  selecting a model that merely consumes the whole card.
- The official [Qwen3 4B Instruct model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
  and the selected [MLX 4-bit conversion](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit)
  support the fallback candidate, but model-card claims do not replace the
  bilingual usefulness and M3 / 16 GB qualification gates.
- The official [Qwen3 14B GGUF](https://huggingface.co/Qwen/Qwen3-14B-GGUF)
  is retained only as a controlled comparison artifact.
