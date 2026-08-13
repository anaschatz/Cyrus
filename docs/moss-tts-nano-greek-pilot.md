# MOSS-TTS-Nano Greek qualification

Status: rejected for the JARVIS local Greek TTS backend.

## Candidate

The qualification used the official Apache-2.0 MOSS-TTS-Nano ONNX CPU
runtime. The source and model revisions were pinned before downloading:

- `OpenMOSS/MOSS-TTS-Nano-Reader` at
  `c3b2333b88e0f062ca49d403540a169609354d93`;
- `OpenMOSS/MOSS-TTS-Nano` submodule at
  `7928ec16500378de31d17bc86ce719cc9fd7b84f`;
- `OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX` at
  `f52645cb467506d8e18e746ddd59482685b74e58`;
- `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX` at
  `ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae`.

Only the ONNX inference allowlist was downloaded. The sixteen runtime files
occupy approximately 728 MiB. Their downloaded sizes and SHA-256 hashes were
checked directly against the pinned revisions before inference. The source
repository contains the full Apache License 2.0 text; the downloaded model
artifacts remain subject to release and dependency license review before any
future distribution decision.

## Method

The run used an Apple Silicon M3 with four ONNX Runtime CPU threads. It used
the official realtime streaming decode path, fixed sampling, seed `1`, and no
network access or tokenizer fallback during synthesis. `sentencepiece==0.2.1`
was installed only in the isolated qualification environment.

Voice cloning used the existing real STOMA F1 recording
`Greek-F1-B2-03-14.wav` (SHA-256
`72826663025609f540344a4adbab494d89405d0dfc44fe6202e124e7b6a1c192`).
The same local `faster-whisper-base` CPU scorer and twenty fixed Greek
fixtures used by the prior candidates measured intelligibility. The real F1
control recording previously scored 0/11 errors with this scorer.

## Result

The complete twenty-fixture run produced all outputs and measured:

- model/runtime load: 1.987 seconds;
- mean realtime factor: 0.387;
- peak resident memory: 2,532,589,568 bytes;
- 111 word errors across 186 reference words;
- 59.677% micro-WER, versus the strict 5% near-perfect gate.

The errors were pervasive, not isolated scoring noise. Examples included
malformed first syllables in `Καλημέρα`, corrupted consonant clusters,
repeated words, and substitutions throughout common assistant sentences.
A preliminary sampling sweep also showed unstable pronunciation across seeds.
Greedy decoding failed to terminate naturally and reached the 220-frame test
limit, so it is not a viable deterministic repair.

## Decision

MOSS-TTS-Nano is fast enough for local CPU streaming and its footprint can
coexist with a larger language model, but its Greek pronunciation is far below
the acceptance threshold. It must not be integrated into the scheduler or
selected as a production backend. The current F1 path remains unchanged while
the search moves to the next local candidate.
