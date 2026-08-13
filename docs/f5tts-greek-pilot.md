# F5-TTS Greek pilot

This pilot evaluated `PetrosStav/F5-TTS-Greek` as an isolated, local macOS
candidate. It did not change the active JARVIS TTS backend or the accepted F1
profile.

## Pinned inputs

- runtime: `f5-tts==1.1.21`, `torch==2.6.0`, `torchaudio==2.6.0`
- model revision: `b38beef017e790294e272772ac69e9ad8f8b1320`
- checkpoint: `model_325000.safetensors`, 1,348,443,953 bytes,
  SHA-256 `3051d025dc0fbb70733098fc9fd0d0a516bb404921002c4c1b9bf7283a00effa`
- vocabulary: `vocab.txt`, 13,818 bytes,
  SHA-256 `7eaef18c7b89ae0c6ac52a1d6ff6e121e396cc87827f88eaf032f44150842a78`
- vocoder: `charactr/vocos-mel-24khz` revision
  `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21`
- F5 model license: `CC-BY-NC-4.0`

All model and vocoder paths were explicit. The process set
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, replaced outbound socket
connection methods with a fail-closed guard and supplied reference text
explicitly. The probe therefore could not transcribe the reference through a
downloaded fallback.

The cleanest supplied reference was `greek_ref_04.wav` with the text:

> πρέπει να το κάνω εγώ αφού δεν είναι άλλος να το κάνει.

The generated lower-case qualification sentence was:

> καλημέρα. η σύνδεση είναι έτοιμη και το σύστημα λειτουργεί κανονικά.

The seed was `20260813`, the speed was `1.0` and inference used 32 function
evaluations.

## M3 result

The sandboxed process correctly failed to expose MPS, so the real Metal probe
was repeated outside the sandbox while retaining the process-level network
guard. This distinction is explicit instead of silently selecting CPU.

| device | load | inference | audio | RTF | peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 3.872 s | 65.328 s | 3.797 s | 17.204 | 3,315,122,176 B |
| MPS | 3.104 s | 16.169 s | 3.797 s | 4.258 | 2,175,500,288 B |

The pinned local `faster-whisper-base` scorer, forced to Greek, produced:

- CPU output: `Κάχερο, η δίση είναι έκειο ψιστημεία, τεκαρακαχακά.`
- MPS output: `Κολχέρω, η βύστησε βενκιατώσει η λίτα ταρικά μαγνού.`

These are gross pronunciation failures on a simple sentence, not minor
orthographic disagreements. The MPS latency is also incompatible with
sentence-first streaming and the memory cost competes directly with the main
LLM on the 16-GB M3.

## Decision

`PetrosStav/F5-TTS-Greek` is rejected for JARVIS. It fails the pronunciation,
latency, memory-coexistence and commercial-distribution gates. No integration
work is justified and the active F1 backend remains unchanged.

Sources:

- <https://huggingface.co/PetrosStav/F5-TTS-Greek>
- <https://github.com/SWivid/F5-TTS>
- <https://pypi.org/project/f5-tts/>
- <https://huggingface.co/charactr/vocos-mel-24khz>
