# macOS Whisper latency qualification

The production model remains the manifest-pinned `large-v3-turbo-q5_0`
artifact. Latency candidates must use the same model, language auto-detection,
temperature and verbose response path as the Voice runtime. A candidate is
rejected when any labelled utterance has a higher word edit distance than the
full-context request, regardless of its speed.

`whisper-server` v1.9.2 accepts `audio_ctx` as a multipart request field, so an
experiment does not require restarting or replacing the model. The behaviour
is visible in the pinned upstream
[`server.cpp`](https://raw.githubusercontent.com/ggml-org/whisper.cpp/v1.9.2/examples/server/server.cpp).

Run the offline corpus benchmark with the pinned server and model paths:

```zsh
python3 scripts/benchmark_macos_whisper_audio_ctx.py \
  --server /path/to/whisper-server \
  --model /path/to/ggml-large-v3-turbo-q5_0.bin \
  --metadata /path/to/metadata.csv \
  --audio-root /path/to/audio \
  --sample-count 32 \
  --audio-contexts 0 1280 \
  --output /tmp/jarvis-whisper-audio-ctx.json
```

The script deterministically spans the eligible corpus by duration, resamples
each source to the runtime's mono 16-kHz input format, rotates profile order to
reduce thermal-order bias, and reports latency, WER, CER, transcript changes,
improvements and degradations.

## 2026-08-23 screening

Eight labelled Greek recordings between 1.85 and 11.95 seconds produced this
initial result:

| audio_ctx | Mean latency | Changed vs full | Degraded vs full | Decision |
| ---: | ---: | ---: | ---: | --- |
| 0 (full) | 3.671 s | 0 | 0 | Baseline |
| 384 | 6.489 s | 8/8 | 8/8 | Rejected |
| 512 | 2.060 s | 6/8 | 6/8 | Rejected |
| 768 | 1.902 s | 2/8 | 2/8 | Rejected |

A second screening pass found identical normalized transcripts for `1024` and
`1280` on the initial 8/8 set. The faster `1024` candidate was then rejected
during the expanded run because it changed the correctly decoded
`κομπάρσου` into `κομπάρ σου`. `1408` was also rejected after three regressions
in its 8-sample screen. `1280` remains experimental until it passes the larger
Greek set, English/code-switch fixtures and a physical user-voice comparison.

No reduced-context profile is enabled in the installed runtime until those
gates pass. Eager model warm-up remains the only promoted ASR latency change.
