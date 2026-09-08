# macOS beta first-turn latency

The foreground beta session performs two bounded local streaming inference
warm-ups after Inference, Orchestrator, Pocket TTS, Whisper and the native Voice
runtime are ready, but before activating the foreground lease and declaring
the stack ready. Running them last prevents the Voice-side unified-memory and
Metal initialization from perturbing the warmed generation state. This moves
one-time MLX generation setup out of the first user turn without changing the
model, real-turn prompts, sampling parameters, output limits or sentence/TTS
streaming policy.

The warm-up:

- targets only the configured `127.0.0.1` `/v1/chat/completions` endpoint;
- requests at most 16 tokens with deterministic sampling;
- uses the same selected production system prompt as real final responses so
  the first production prompt shape does not trigger a cold prefill;
- accepts only a complete SSE stream whose final text, after trimming whitespace
  and case-folding, is `ready`;
- bounds each line, the complete response and generated UTF-8 output;
- has a ten-second deadline and never logs or persists generated content;
- fails beta startup closed with `inference_warmup_failed` on any transport,
  schema, output or timeout failure.

On the qualified M3 runtime on 2026-08-23, three identical local requests
measured 1260 ms first-token latency for the cold request and 376/372 ms for
the next two requests. The promoted startup warm-up therefore moved roughly
0.88 seconds of one-time work out of the first real response. Two warm requests
are required because a single production-shape warm-up left the first
post-ready request at 757 ms while subsequent requests measured 464/461 ms.
With both warm-ups moved after Voice readiness, the first installed post-ready
request measured 523 ms and the next two measured 463/461 ms. This cuts about
737 ms from the 1260-ms cold baseline. All measured requests returned the
required exact output.

This optimization is independent of the Voice-side eager Whisper/Pocket TTS
loads and the duration-gated Whisper audio-context profile. A physical PTT turn
is still required to qualify the combined end-to-end latency after deployment.
