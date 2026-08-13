# Coqui Greek VITS pilot

This isolated pilot evaluated the official `tts_models/el/cv/vits` model. It
did not change the active JARVIS TTS backend.

## Pinned candidate

- model archive: Coqui release `v0.8.0_models`
- archive size: 100,857,273 bytes
- archive SHA-256:
  `7ef4f5ad94a660f6977427db3a7949354aa47d6a83e85f0e5c2e7ba360c987bf`
- checkpoint size: 108,565,327 bytes
- checkpoint SHA-256:
  `32424e7ec50c8d0020638aa2d360faa9e21c77b28b523ca2c8461eb85af2338c`
- model license: BSD-3-Clause
- runtime: `coqui-tts==0.27.5`, `torch==2.6.0`,
  `torchaudio==2.6.0`, `transformers==4.57.6`
- output: mono 22.05-kHz PCM

The historical config contains the training host's absolute `speakers_file`
path. The qualification copy replaced only that field with the accompanying,
hash-verified `speaker_ids.json`. Greek phonemization used the existing pinned
Piper eSpeak-NG 1.52 runtime through a qualification-only CLI bridge; no
Homebrew or system package was installed.

Inference was run with explicit local paths and a process-level fail-closed
socket guard. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` were also set.

## M3 CPU result

The first sentence generated in 0.606 seconds for 4.680 seconds of audio
(RTF 0.129), after a 0.573-second cold load. It used 1,071,644,672 bytes peak
RSS. The complete fixed twenty-sentence set was faster after warm-up:

- cold load: 0.331 seconds
- mean RTF: 0.0583
- peak RSS: 1,449,492,480 bytes
- all 20 fixtures generated

The speed is suitable for sentence-first playback and CPU inference avoids
Metal contention with the main LLM. Pronunciation, however, failed decisively.
The pinned local `faster-whisper-base` scorer reported 119 word errors across
186 reference words: 63.978% micro-WER, versus the strict 5% near-perfect gate.
Representative failures include:

- `είμαι έτοιμη` → `η μέτυμη`
- `αποκαταστάθηκε χωρίς απώλεια δεδομένων` →
  `από καταστάθηκε χωρίς απόλια δεδομένον`
- `πολύπλοκη αρχιτεκτονική του κτιρίου` →
  `πολύ πρωί και αχητικτονική του θυρίου`

These are audible omissions and malformed syllables, not punctuation or accent
normalization differences.

## Decision

The official Coqui Greek VITS model is rejected. It passes the offline,
licensing, latency and practical-memory checks but fails Greek pronunciation
by a wide margin. It must not be integrated, and the active F1 profile remains
unchanged.

Sources:

- <https://github.com/idiap/coqui-ai-TTS/blob/dev/TTS/.models.json>
- <https://github.com/coqui-ai/TTS/releases/tag/v0.8.0_models>
- <https://pypi.org/project/coqui-tts/>
