# macOS M3 endpoint quality evidence — 2026-08-13

This is sanitized, non-authoritative deployment and model-quality evidence. It
contains no bearer secret, prompt, completion, tool payload, or personal data.

## Endpoint

- LaunchAgent: `com.jarvis.shadow.inference`
- Bind: `127.0.0.1:8766`
- Public model alias: `jarvis-local`
- Profile: qualified Qwen3 4B MLX profile for Apple Silicon with 16 GB unified
  memory
- `/readyz`: `200 {"ready":true}` after startup and after the quality run
- Advertised capabilities: `text`, `stream`, `tools`, `json_object`, and
  `json_schema` are all enabled
- Single active request with no queue remains enforced

The persistent service loaded the identity-matching capable manifest and
qualification report. The endpoint is operational, but endpoint readiness is
not a declaration of conversational quality.

## Frozen daily-quality corpus

Command:

```text
PYTHONPATH=inference/src python3 inference/scripts/run_profile_quality.py \
  --orchestrator-config <private-config> \
  --corpus inference/fixtures/local-fallback-quality-v1.json \
  --summary-only
```

Corpus SHA-256:
`cc3814b05807908f1235db815f078f5ac2a4034ad3844e9dcc03a85c5d1fdd71`

Result: `deterministic_quality_failed`.

| Gate or metric | Observed | Required | Result |
|---|---:|---:|---|
| Greek quality | 20/40 (50%) | at least 85% | fail |
| English quality | 24/40 (60%) | at least 85% | fail |
| Language gap | 10 percentage points | at most 10 | pass |
| Protocol completion | 78/80 | 80/80 | fail |
| Critical unsafe-content failures | 0 | 0 | pass |
| TTFT p95 | 1.245 s | informational | — |
| End-to-end p95 | 2.625 s | informational | — |
| Decode p50 | 16.154 tokens/s | informational | — |
| Decode p5 | 9.691 tokens/s | informational | — |

Category scores were instruction 16/16, extraction 7/16, summary 3/16,
reasoning 12/16, and knowledge 6/16.

Two generations terminated with incomplete SSE responses and correctly counted
as protocol failures. A separate, concurrent orchestrator test process also
occupied the single-request endpoint during initial attempts; the recorded run
started only after the endpoint became idle. No process was terminated to obtain
the measurement.

## Decision

The Mac endpoint is suitable for continued integration and structured/tool
testing, but this Qwen3 4B profile is **not accepted as the useful conversational
`local-fallback`**. Its bilingual daily-use quality and protocol-completion gates
remain red. It must not be promoted merely because qualification and readiness
are green.

The next model-profile step is to evaluate a stronger M3/16 GB candidate with
the same frozen corpus and the same strict structured/tool qualification, while
retaining this profile as a technical integration baseline.
