# macOS M3 Qwen3.5 9B candidate screening — 2026-08-18

This is sanitized, non-authoritative candidate evidence. It contains no API
key, prompt, completion, reasoning trace, local path, or personal data.

## Purpose

The existing Qwen3 4B MLX appliance is sufficient for integration work but did
not pass the frozen bilingual daily-quality gate. This run screened an already
local Qwen3.5 9B quant before spending engineering effort on a separate sealed
macOS GGUF profile. No model was downloaded for this test.

## Pinned candidate

- Hardware class: Apple M3 with 16 GB unified memory, connected to AC power.
- Model: Qwen3.5 9B, Ollama-format GGUF, `Q4_K_M`, 6,594,462,816 bytes.
- Model SHA-256:
  `dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c`
- Ollama manifest SHA-256:
  `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`
- Runner: bundled `llama-server` build 9840.
- Runner SHA-256:
  `b9bdd68928c150482049100d20bd446ccee5569ff3db386135855e0cb8fea66a`
- Public API alias: exactly `jarvis-local`.

The candidate ran offline on an authenticated IPv4 loopback endpoint with one
slot, an 8192-token context, all model layers requested on Metal, batch 512,
ubatch 256, six CPU threads, Flash Attention enabled, and reasoning explicitly
disabled. Web UI, built-in agent tools, multimodal projection, prompt logging,
and network model retrieval were disabled. The server and ephemeral credential
were removed after the run.

## Frozen bilingual screening

The unchanged `jarvis-local-fallback-quality-v1` corpus was used:

- 40 Greek and 40 English cases.
- Corpus SHA-256:
  `cc3814b05807908f1235db815f078f5ac2a4034ad3844e9dcc03a85c5d1fdd71`
- System-prompt SHA-256:
  `05dbfe802df53da6c8660425016b9ea20fef0c745111c38701845d5c422cf4f2`
- Two warm-up requests.
- The original deterministic scorer and thresholds were not changed.

The raw server emits the valid MIME type `text/event-stream`, whereas the
sealed Jarvis gateway emits `text/event-stream; charset=utf-8`. A candidate-only
adapter normalized only that transport difference. The exact public model alias
was verified independently before the corpus, and the two warm-up requests
confirmed generation readiness. The sealed-gateway-specific readiness and
capability endpoints were not claimed for the raw server. Consequently this is
a challenger screening result, not appliance qualification.

Result: `deterministic_quality_failed`.

| Gate or metric | Observed | Required | Result |
|---|---:|---:|---|
| Greek quality | 21/40 (52.5%) | at least 85% | fail |
| English quality | 26/40 (65%) | at least 85% | fail |
| Language gap | 12.5 percentage points | at most 10 | fail |
| Protocol completion | 80/80 | 80/80 | pass |
| Critical unsafe-content failures | 0 | 0 | pass |
| TTFT p95 | 1.581 s | informational | — |
| End-to-end p95 | 3.162 s | informational | — |
| Decode p50 | 9.486 tokens/s | informational | — |
| Decode p5 | 6.918 tokens/s | informational | — |

Category scores were instruction 16/16, extraction 8/16, summary 4/16,
reasoning 12/16, and knowledge 7/16.

## Decision

This exact 9B quant/profile is rejected as the Mac daily-conversation model.
Its flawless protocol behavior and acceptable latency do not compensate for
the large Greek and English quality shortfall. It will not silently replace the
existing 4B integration profile, and a macOS GGUF appliance backend will not be
built solely for this artifact.

The first Mac release remains a **technical integration beta** until another
model passes the same bilingual gate and then completes sealed-renderer,
coexistence, headroom, physical-voice, and soak qualification.
