# JOY Greek Piper pilot

This isolated pilot evaluated the 2026 JOY Greek Piper voice. It did not
replace the active JARVIS TTS backend or the previously selected F1 profile.

## Pinned candidate

- upstream: `Efso/joy-greek-tts` revision
  `4189a6e0a0e3e1636aaf83fcd736c9266657af89`
- ungated official-community mirror: `rhasspy/piper-voices` revision
  `ea046e8458f6acd997706d6e6066a022b42f6fb1`
- model: `el_GR-joy-medium.onnx`, 63,516,050 bytes,
  SHA-256 `31b1d51dc72dd43a5429b67798e27599f8cdcfce8a80265bc6f50d932bf144bc`
- config: `el_GR-joy-medium.onnx.json`, 7,367 bytes,
  SHA-256 `451a8885a0ea72e4432fffe22e65a75bbd2e65f6757273d1a5a9ed8a3419852e`
- runtime: `piper-tts==1.6.0`, CPU ONNX
- license: CC-BY-NC-4.0; commercial use requires separate permission
- training claim: 3,216 human recordings from a native Greek speaker

The direct upstream weight URL required gated Hugging Face authentication and
returned HTTP 401. No credential or API key was requested or used. The same
model was available from the public Rhasspy Piper Voices repository; its
content hash exactly matched the LFS object recorded for JOY.

All synthesis used explicit local paths with `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1` and a process-level fail-closed socket guard.

## M3 result

The complete fixed twenty-sentence set passed the resource gates:

- cold load: 0.608 seconds
- mean RTF: 0.0259
- peak RSS: 405,716,992 bytes
- model plus config: approximately 63.5 MB

The published config uses `length_scale=0.847`, not a neutral 1.00. At that
default, pinned local `faster-whisper-base` scoring produced 92 word errors
across 186 reference words: 49.462% micro-WER. The errors contained omitted or
malformed syllables, including words such as `Καλημέρα`, `αποκαταστάθηκε`,
`απώλεια`, `αρχιτεκτονική` and `άρθρωση`.

A targeted A/B repeated the greeting and articulation sentences with:

- `length_scale=1.0` and published noise values;
- `length_scale=1.0`, `noise_scale=0.3`, `noise_w_scale=0.3`;
- `length_scale=1.0`, both noise values zero.

Slowing the model to neutral pacing did not repair pronunciation. Representative
round trips still included `Καλημέρα` → `Καίλαια` and `άρθρωση` →
`άφροση`/`εράχθρωση`. Reduced and zero noise changed the defects rather than
eliminating them.

## Decision

JOY is rejected for JARVIS. It is the strongest candidate so far on footprint,
speed and training provenance, but it fails the strict near-perfect Greek
pronunciation gate and is non-commercial without extra permission. It is not
integrated and the active voice remains unchanged.

Sources:

- <https://huggingface.co/Efso/joy-greek-tts>
- <https://huggingface.co/rhasspy/piper-voices>
- <https://creativecommons.org/licenses/by-nc/4.0/legalcode>
