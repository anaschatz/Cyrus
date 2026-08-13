# Fish Audio S2 Pro Greek pilot on the M3 laptop

Status: rejected for the JARVIS local Greek TTS backend.

The pilot used no API key, installation, hosted inference or production backend
change. Network access was blocked during load and synthesis. The local runtime
was the exact `mlx-audio` revision
`76f8363b28e82656d537ea17a38dc97e9ed9a0f8`.

## Pinned model pack

The text/acoustic model was the 4-bit MLX conversion
`majentik/fishaudio-s2-pro-MLX-4bit` at revision
`dda28aa6e33fe946826792e3feb643e9c463b75f`. Its
`model.safetensors` is 2,566,443,153 bytes with SHA-256
`1cbb30334f1b265a601dd46a8429c7889d240095cf2a39e55e82d4a2c0e934c2`.

The conversion repository ships a legacy `codec.pth`. The qualification did
not deserialize it. It combined the 4-bit model with the safe MLX
`codec.safetensors` from `mlx-community/fish-audio-s2-pro-bf16` at revision
`eccd57bf5c1ebc13cb2f993df867f4e49931a36a`. The codec is 1,870,921,167
bytes with SHA-256
`c2096e27a1bb69a05151ccb502649a88744712270d9b029a68a3c9305f2b1edf`.
The hash matches the repository's published Xet metadata. Strict local load
accepted all weights, tokenizer and codec.

Fish S2 Pro is under the Fish Audio Research License. Personal evaluation and
non-commercial use are allowed subject to that license, but commercial use
requires a separate written agreement. This is an independent blocker for a
commercial default even if technical quality improves later.

## F1 reference

Two original STOMA F1 recordings and their exact transcripts were joined
without resampling into a 15.1595-second, 44.1-kHz, mono PCM16 reference. The
reference SHA-256 is
`a22adf7c5650d6c2029c24229a7a1a216927a71baa578f60aae6f43df736fcbe`.
The original clips remain individually hash-verifiable. The reference length
is within Fish Audio's recommended 10--30-second range.

## M3 measurements

A strict load-only probe completed in 6.611 seconds. It reported 5,171,123,280
MLX peak bytes, 2,355,380,224 bytes maximum resident set size and
6,186,832,552 bytes macOS peak memory footprint, without swap.

The first decisive Greek fixture was:

`Καλημέρα. Είμαι έτοιμη να σε βοηθήσω.`

The generated audio was 3.1579 seconds long and required 22.1952 seconds of
generation, a conventional real-time factor of 7.0284. MLX peak memory rose to
8,162,311,347 bytes. `/usr/bin/time -l` measured 2,724,102,144 bytes maximum
RSS and 12,627,481,832 bytes peak memory footprint, with no swap. This is far
outside the latency and main-LLM coexistence requirements of the 16-GB M3.

The pinned local `faster-whisper-base`, running CPU `int8` with Greek fixed,
transcribed the sample as:

`Καλημέρα, είμαι η τυμήνα σε βοηθίζω.`

That is three word errors across six reference words, or 50.0% WER. It fails
the qualification pronunciation ceiling of 5%. The wider eight-fixture suite
was intentionally not run after this decisive failure, avoiding unnecessary
compute and additional artifacts.

## Decision

Fish S2 Pro is rejected. It fails Greek pronunciation, real-time speed, unified
memory coexistence, incremental audio streaming and interruption during
generation. Its model and license are also too heavy for the target client.
The active TTS backend remains unchanged.

The next candidate should keep linguistic realization in a small native-Greek
synthesizer and treat timbre separately, or use a newer Greek-specific female
checkpoint. This avoids asking a broad multilingual acoustic language model to
learn Greek pronunciation from the F1 reference alone.

## Storage cleanup

Before this pilot, the rejected Piper MPS training directory was removed. It
contained the rejected 0--30 epoch, low-learning-rate recovery and protected
200-step runs and reduced the old workspace from about 26 GiB to 3.5 GiB. The
two pinned F1 reference clips were copied into the independent pilot directory.
The source dataset and the non-training remainder were not removed because the
destructive-action guard requires a separate, explicit user approval for those
files.

The rejected MOSS qualification directory was also removed. The `uv`, Xcode
DerivedData, pip and Homebrew download caches were cleared; these are
reproducible caches and do not contain project source. APFS reclaim later raised
free space from 2.7 GB to about 30 GiB.
