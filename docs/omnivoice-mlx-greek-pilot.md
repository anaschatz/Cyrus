# OmniVoice MLX Greek qualification

Status: rejected for the JARVIS local Greek TTS backend.

## Candidate

The qualification evaluated the Apple Silicon MLX port of OmniVoice, a 2026
zero-shot TTS model that explicitly lists Greek among 646 supported languages.
The following source and model revisions were pinned:

- `ailuntx/OmniVoice-MLX` source revision
  `21de4138cc8fc0ef88fad2d8e755973ee1982b97`;
- `mlx-community/OmniVoice-4bit` revision
  `e568702be8aa470656cbaa27e33281b3e272b729`;
- `mlx-community/OmniVoice-8bit` revision
  `b222e4d82e7bc20323be736ef05fcfc50defb71b`.

The source and main model are Apache-2.0. The required bundled Higgs Audio 2
tokenizer is not Apache-2.0: its Boson Community License adds redistribution,
attribution, acceptable-use, model-improvement and commercial-scale terms.
That dependency would require explicit legal review before distribution even
if the technical qualification had passed.

The 4-bit package used 1,142,674,262 bytes on the Hub. The three largest local
artifacts were verified directly after download:

- 4-bit main weights: 325,584,648 bytes, SHA-256
  `36ff0344d5e890a63ed087a5834fd436e71bf6a7a127d7a8a48b9364227d109d`;
- text tokenizer: 11,423,986 bytes, SHA-256
  `408f669b7e2b045fdf54201d815bd364e6667dbd845115da81239c40bc6dcfd1`;
- Higgs audio tokenizer: 805,665,628 bytes, SHA-256
  `fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d`.

The 8-bit main weights were 631,840,520 bytes with SHA-256
`6ff9d1a74a6d8c07700627661d2254968e5a322a63a8d5759c0906e1818f1000`.
The 8-bit staging reused the identical verified audio-tokenizer inode rather
than storing a second 806 MB copy.

## Method

All synthesis used the real M3 Metal device through `mlx==0.32.0`, local-only
model directories, `language="el"`, and the existing STOMA F1 recording
`Greek-F1-B2-03-14.wav` plus its exact transcript as the voice-cloning prompt.
The reference is a 5.44-second recording by a native Greek female speaker and
has SHA-256
`72826663025609f540344a4adbab494d89405d0dfc44fe6202e124e7b6a1c192`.

The same local `faster-whisper-base` CPU scorer used for earlier candidates
measured intelligibility. Its real F1 control previously scored 0/11 word
errors. A fast first probe used 16 diffusion steps. Eight difficult fixed
Greek fixtures then used 16 steps, followed by controlled 16/32-step and
4-bit/8-bit comparisons on the two worst sentences.

## Results

The first simple greeting was promising: the scorer produced only one
inflection error in seven words. It nevertheless required 8.57 seconds to
generate 2.84 seconds of audio (RTF 3.019) and reached 5,238,784,000 bytes peak
RSS.

Across the eight harder fixtures, the 4-bit/16-step profile produced 20 errors
in 86 reference words (23.256% micro-WER), with mean RTF 1.987 and peak RSS
4,631,150,592 bytes. Errors included malformed syllables in common words,
broken consonant clusters and multiple substitutions in the microphone
recovery sentence.

The official 32-step quality setting did not provide a general repair. In a
controlled 4-bit comparison it reduced one consonant fixture from seven to
four errors but left the connection fixture at seven errors out of seven.
Deterministic reveal did not help. Repeated 32-step testing reached RTF 3.99
and peak RSS 6,530,662,400 bytes.

The 8-bit comparison showed that this is not merely a 4-bit quantization
artifact. At 32 steps the connection fixture improved to four errors out of
seven, while the consonant fixture regressed to six errors out of nine. RTF
reached 4.186 and peak RSS was 6,073,532,416 bytes.

## Decision

OmniVoice is materially more intelligible than several smaller multilingual
candidates, but it does not meet the strict 5% near-perfect Greek gate. Its
latency is incompatible with first-sentence streaming on this M3 profile, and
its 4.6--6.5 GB measured peak memory is not a safe coexistence budget for the
user's 16 GB machine and primary LLM. The separate Higgs license also adds a
distribution review burden.

Do not integrate OmniVoice into the scheduler or select it as a runtime
backend. The production F1 path remains unchanged while qualification moves to
a stronger Greek-capable model with a bounded Apple Silicon profile.
