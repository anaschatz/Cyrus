# VoxCPM2 MLX Greek qualification

Status: rejected for the JARVIS local Greek TTS backend.

## Candidate

The qualification evaluated VoxCPM2, a 2B-parameter multilingual TTS model
that explicitly lists Greek among thirty supported languages. The following
Apple Silicon artifacts were pinned:

- `mlx-community/VoxCPM2-4bit` revision
  `dc9e5c187858da5f4a13dc4c247e297339216381`;
- `mlx-community/VoxCPM2-8bit` revision
  `d52725898a0675703f7f9ddc5a4d1a3cdbb99032`;
- `Blaizzy/mlx-audio` runtime revision
  `76f8363b28e82656d537ea17a38dc97e9ed9a0f8`.

The VoxCPM2 model is Apache-2.0 and the evaluated MLX runtime is MIT. The
4-bit weight was 2,297,215,329 bytes with SHA-256
`25bddc99a7cc6d28214f12dd676ed0afa9b0a805d6477f85c275bb113cb8acee`.
The 8-bit weight was 3,221,772,935 bytes with SHA-256
`c07c59aca742fe8f53384a3da4d8a3b3fd98ddc77066dd4bd35b51e08c34a72f`.
Both hashes and byte counts were verified directly after download.

## Method

All synthesis used the real M3 Metal device through `mlx==0.32.0`, explicit
local model directories and offline Transformers/Hugging Face settings. No API
key or hosted inference was used. The speaker reference was the real native
Greek STOMA F1 recording `Greek-F1-B2-03-14.wav`, SHA-256
`72826663025609f540344a4adbab494d89405d0dfc44fe6202e124e7b6a1c192`.

The first probe used ordinary reference cloning at seven diffusion steps. The
second used the stronger continuation path at the official ten-step setting,
supplying both the F1 audio and its exact transcript so the model received an
aligned Greek text/audio example. The same aligned profile then synthesized
eight fixed difficult Greek fixtures. The existing pinned local
`faster-whisper-base` CPU scorer measured intelligibility; its real F1 control
previously produced zero errors across eleven words.

The 8-bit comparison changed only quantization. It reused the same runtime,
F1 conditioning, ten diffusion steps, seeds and the two worst 4-bit fixtures.

## Results

The initial 4-bit reference-cloning greeting generated 2.08 seconds of audio
in 5.20 seconds and reached 3,970,398,704 bytes peak MLX memory. The scorer
heard two errors across six words. Aligned Greek continuation improved the
same greeting to one orthographic/accent disagreement across six words, but it
needed 9.56 seconds for 4.00 seconds of audio and reached 4,741,082,804 bytes
peak MLX memory.

That encouraging greeting did not generalize. Across the eight difficult
fixtures, the 4-bit profile produced 22 errors in 86 reference words: 25.581%
micro-WER against the strict 5% near-perfect gate. The connection sentence
alone produced six errors in seven words, including broken syllables in
`ασύρματη`, `αποκαταστάθηκε` and `απώλεια`. The consonant-cluster fixture
produced five errors in nine words. Mean generation RTF was 2.316, peak MLX
memory was 5,870,525,628 bytes and process peak RSS was 2,884,288,512 bytes.

The 8-bit A/B did not repair the failure. It produced the same six errors in
seven words on the connection fixture and regressed the cluster fixture to six
errors in nine words, for 75.0% micro-WER across the comparison. Generation
RTF was 2.835--2.990 and peak MLX memory reached 5,687,582,396 bytes. This
rules out 4-bit quantization as the main source of the Greek corruption.

## Decision

VoxCPM2 is rejected. Its aligned continuation mode can produce a convincing
isolated sentence, but common Greek consonants and word boundaries are not
stable across the fixed system set. Both quantizations fail pronunciation by
a wide margin, generation is more than twice slower than real time, and the
measured memory profile does not leave a safe coexistence budget for the main
LLM on this 16 GB Mac.

Do not integrate VoxCPM2 into the scheduler or retain either weight pack as a
production artifact. The active voice remains unchanged while qualification
moves to a stronger Greek-capable candidate with explicit memory and license
gates.

Sources:

- <https://huggingface.co/openbmb/VoxCPM2>
- <https://huggingface.co/mlx-community/VoxCPM2-4bit>
- <https://huggingface.co/mlx-community/VoxCPM2-8bit>
- <https://github.com/Blaizzy/mlx-audio>
