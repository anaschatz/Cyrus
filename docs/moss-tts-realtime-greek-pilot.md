# MOSS-TTS-Realtime Greek qualification

Status: rejected for the JARVIS local Greek TTS backend.

## Candidate

The qualification used the official Apache-2.0 multilingual
`OpenMOSS-Team/MOSS-TTS-Realtime` model at revision
`75682787d8e2fcc73faca37ba2931453ca9c4022`. The official model card lists
Greek among the twenty supported languages.

The M3-native pilot used the community GGUF conversion from
`BricksDisplay/MOSS-TTS-Realtime-GGUF` at revision
`7394ff5cf5c53e7dbc82de2bd91c5b5916e69c6c` and the `mybigday/codec.cpp`
runtime at revision `63ade8bf2fa3451f9eca7ebaa39f792cc529438e`.
The two Q5_K_M artifacts were pinned and verified after download:

- backbone: 1,257,874,944 bytes, SHA-256
  `a0056016a6d195a848c09e556067e7f974e15b4c907a5250d8da9bad8b911087`;
- codec plus residual depth decoder: 1,831,441,856 bytes, SHA-256
  `229611078af6cfc2a31b2c5a68c1a28a1fc0defe9c4ad1e28307c46f64c46c58`.

The downloaded model files and generated WAVs remain gitignored
qualification artifacts. No model was added to the product package.

## Native runtime probe

The upstream codec runtime was built as native arm64 with its codec on Metal.
Its isolated llama.cpp backbone wrapper initially exposed a Linux-only link
recipe and forced the backbone onto CPU. The qualification added a Darwin
wrapper link, kept all `ggml_*` symbols private, embedded a second isolated
Metal backend for the backbone, and reduced the oversized inference context
from 4,096 to 1,024 tokens. The final wrapper exported zero `ggml_*` symbols,
so the codec's ggml 0.9.6 and llama.cpp's ggml 0.15.3 remained ABI-isolated.

The real Apple M3 load probe confirmed all 29 backbone layers on Metal. It
reduced the backbone KV cache from 448 MiB to 112 MiB and its compute buffers
from approximately 2.44 GiB on CPU to about 314 MiB across Metal and CPU.

## Result

The fixed input was:

> Καλησπέρα. Είμαι έτοιμη να σε βοηθήσω.

With the official realtime sampling defaults (`temperature=0.8`, `top_p=0.6`,
`top_k=30`, repetition penalty `1.1`) and seed `12345`:

| Runtime | Output | Wall time | Peak footprint | Offline Whisper transcript |
| --- | ---: | ---: | ---: | --- |
| CPU backbone, Metal codec | 4.96 s | 140.45 s | 5.49 GiB | `Τα τσ, και λυσπέρα, είμαι έτοιμη να σε βοηθήσω.` |
| Metal backbone and codec | 3.36 s | 109.03 s | 3.59 GiB | `Τα λυσπέρα. Είμαι έτοιμη να σε βοηθήσω.` |

The scorer was the already-local, manifest-pinned
`Systran/faster-whisper-base` model, forced to Greek and run offline with CPU
int8. No ASR model or dependency was downloaded for this test.

The Metal path therefore ran at approximately 32.4 times slower than its
generated audio and corrupted the first common Greek word. It fails both the
interactive latency requirement and the strict near-perfect pronunciation
gate. A twenty-sentence run was intentionally not performed after these two
independent hard failures; at the measured speed it would only consume more
time without changing the promotion decision.

## Decision

Do not integrate this GGUF/runtime combination into the JARVIS scheduler. The
model's multilingual capability is real, but this native M3 implementation is
neither realtime nor sufficiently reliable in Greek. Voice cloning was not
pursued because changing timbre cannot repair the failed pronunciation and
latency gates. The existing production voice selection remains unchanged
while the local Greek search moves to the next candidate.
