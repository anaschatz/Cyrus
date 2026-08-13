# Supertonic 3 native-Greek pilot

This path evaluates [Supertonic 3](https://github.com/supertone-inc/supertonic) without changing the active
JARVIS TTS backend. Supertonic 3 is a 99M-parameter, CPU-only ONNX model with
official Greek (`el`) support and ten included voices. It needs no API key and
does not consume the GPU/Metal memory reserved for the main LLM.

## Pinned profile

- runtime: `supertonic==1.3.1`
- model: `Supertone/supertonic-3`
- revision: `724fb5abbf5502583fb520898d45929e62f02c0b`
- model license: OpenRAIL-M; runtime code license: MIT
- voice: `F5`
- language: `el`
- diffusion steps: `12`
- speed: `1.00`
- output: mono 44.1 kHz WAV

The harness validates the exact size and SHA-256 digest of all four ONNX files,
the two runtime JSON files, the repository license/config and all ten voice
styles. Synthesis runs with `auto_download=False`, an explicit local model
directory, offline environment flags and a process-level socket guard.

## Isolated environment

```bash
uv venv --python 3.11 /absolute/private/supertonic3-pilot/.venv
uv pip install \
  --python /absolute/private/supertonic3-pilot/.venv/bin/python \
  --require-hashes \
  -r requirements/supertonic3-pilot.lock
```

Model acquisition is a separate, network-enabled operation. Download only the
pinned revision's `LICENSE`, `config.json`, `onnx/*` and
`voice_styles/*.json`, then run preflight offline:

```bash
/absolute/private/supertonic3-pilot/.venv/bin/python \
  scripts/run_supertonic3_pilot.py preflight \
  --model-dir /absolute/private/models/supertonic-3 \
  --report /absolute/private/evidence/supertonic3-preflight.json
```

## Qualification

```bash
/absolute/private/supertonic3-pilot/.venv/bin/python \
  scripts/run_supertonic3_pilot.py qualify \
  --model-dir /absolute/private/models/supertonic-3 \
  --output-dir /absolute/private/evidence/supertonic3-audio \
  --report /absolute/private/evidence/supertonic3-generation.json

/absolute/macos-cpu/venv/bin/python \
  scripts/run_supertonic3_pilot.py score \
  --output-dir /absolute/private/evidence/supertonic3-audio \
  --whisper-model-dir /absolute/private/models/faster-whisper-base \
  --report /absolute/private/evidence/supertonic3-asr.json
```

The fixed suite contains twenty Greek assistant phrases covering numbers,
names, consonant clusters, questions and long sentences. The technical pilot
requires all files, fixed-seed determinism, load below 2 seconds, mean RTF at
or below 0.6 and peak RSS at or below 1 GiB. Near-perfect pronunciation uses a
strict ASR screening limit of 5% micro-WER, followed by native-speaker listening.

Production promotion remains blocked until human listening, sentence-unit
streaming, hard cancellation and concurrent main-LLM qualification all pass.

## Early comparison on the M3

The rejected 36.3M-parameter VITS candidates were very light but failed the
same four-phrase Greek screen: `facebook/mms-tts-ell` measured 38–41% micro-WER
and `vlntlbr/vits-greek-female` measured 65–76%. Supertonic's F5 voice reduced
that screen to 14.7% at 12 steps and speed 1.00, while using about 572 MB peak
RSS and remaining faster than real time. This is promising but not yet the
near-perfect production gate.
