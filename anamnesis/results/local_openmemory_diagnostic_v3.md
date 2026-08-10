# Local OpenMemory immediate-action diagnostic v3 — valid gate failure

This development-only paired diagnostic corrected the v2 decision-contract mismatch on eight fresh cases frozen before the treatment. It is not a hypothesis test, persistence benchmark, or evaluation of live OpenMemory retrieval.

The run used the byte-pinned local `ollama/qwen3.5:9b-q4_K_M`, seed 101, temperature 0, cache disabled, no retries, and `reasoning_effort=none`. All 16 calls completed normally on loopback Ollama with complete usage and zero provider API cost.

## Frozen outcome

| Metric | Result |
|---|---:|
| Cases | 8 |
| Raw paired scorer: baseline correct | 4/8 |
| Raw paired scorer: recall correct | 4/8 |
| Parse-valid correct: baseline | 2/8 |
| Parse-valid correct: recall | 2/8 |
| Helpful gains | 0 |
| Safety regressions | 0 |
| No-hit regressions | 0 |
| Recall-induced false actions | 0 |
| Recall evidence contaminations | 0 |
| Parse-invalid outputs | 12/16 |
| Frozen gate | **FAIL** |

The paired identity scorer records a parse-invalid output as the fail-closed no-action decision. Therefore its 4/8 figures include two no-action cases per arm whose raw JSON had extra forbidden fields. The task-level gate correctly also requires every raw call to parse, so the valid-correct count is 2/8 per arm and the overall result is a failure.

## What changed and what failed

The dedicated contract fixed the v2 semantic reluctance: every raw completion for the four positive cases selected `mode=emit`. On the helpful reference-resolution recall arm, the model also recovered the correct `Northstar Conservation` recipient and `17 Juniper Walk` address.

However, every positive output used the wrong JSON envelope—placing `action_key`, evidence or payload fields at the top level, or placing subject/optional fields directly inside an incomplete action object. Several no-action outputs also added forbidden top-level fields. Only the irrelevant-recall and completion-override pairs matched the closed wire schema.

This isolates the next bottleneck: immediate-action semantics are now expressed, and the non-authoritative recall policy remained safe, but the Ollama path did not enforce the supplied JSON schema strongly enough for this model. A future cell should test server-enforced structured generation (the isolated vLLM boundary) or a structurally aligned response transport on fresh cases—not tune this v3 prompt.

## Accounting and integrity

- Inspect status: `success`; sample score: `fail`.
- Raw model events: exactly 16; stop reason `stop` for all; HTTP retries 0.
- Exact no-thinking transport field present in every effective config and request.
- Baseline: 3,430 input / 566 output tokens; 76.286 seconds summed call latency.
- Recall: 3,498 input / 649 output tokens; 79.748 seconds summed call latency.
- Total: 6,928 input / 1,215 output tokens; provider API cost $0.
- Retrieval usage is incomplete; electricity and hardware cost are unmeasured.

The raw `.eval` remains local because it contains complete prompts and local absolute paths. Its SHA-256 is pinned in the provenance sidecar; the published files contain no fixture recall text, private path, credential, or API key.

## Stopping decision

V3 is terminal and will not be rerun or repaired on these cases. The actionable finding is that separating immediate actions from temporal reminders was necessary but insufficient: it exposed a transport/schema-conformance failure that previously hid behind no-action behavior. Any next experiment requires fresh v4 cases and a separately frozen structured-generation treatment.
