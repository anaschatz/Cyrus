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
  --threads 4 \
  --audio-contexts 0 1280 \
  --output /tmp/jarvis-whisper-audio-ctx.json
```

The script deterministically spans the eligible corpus by duration, resamples
each source to the runtime's mono 16-kHz input format, rotates profile order to
reduce thermal-order bias, and reports latency, WER, CER, transcript changes,
improvements and degradations.

Use separate runs with `--threads 4`, `6` and `8` to qualify CPU parallelism.
Thread-count promotion requires identical normalized transcripts on every
fixture as well as lower latency; run only one benchmark sidecar at a time.
Use `--no-timestamps` for the matching candidate run; it may be promoted only
if every normalized transcript and confidence-bearing verbose response remains
valid while aggregate latency improves.

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
in its 8-sample screen.

`1280` then passed 24 labelled Greek recordings without a single higher word
edit distance. Aggregate WER improved from 9.34% to 8.17%, mean latency fell
from 4.77 s to 4.19 s, and p95 fell from 6.15 s to 4.99 s. A ten-fixture
English/code-switch screen retained the same aggregate WER and reduced mean
latency from 3.58 s to 3.15 s. One 2.91-second code-switch fixture did change a
proper name from `ZARVIS` to `service`, so reduced context is not qualified for
input below 3.0 seconds.

The 3.0-to-4.0-second boundary was then checked separately across 14 labelled
Greek, English and code-switch fixtures. All 14 candidate transcripts retained
the baseline word edit distance, one Greek transcript improved, and individual
latency fell by roughly 0.4 to 0.6 seconds. The shortest qualified fixture was
3.118 seconds; the known 2.911-second regression remains on full context.

The promoted policy therefore requests context `1280` only for trimmed audio
between 3.0 and 12.0 seconds inclusive. Shorter input, unqualified long input,
and any profile with `qualified_audio_context: 0` use full context. This keeps
all observed regressions on the full-quality path while retaining the measured
gain for the qualified duration window.

The installed runtime still requires a physical user-voice comparison before
the private profile is promoted. Eager model warm-up remains active regardless
of the audio-context setting.

## CPU thread and timestamp screening

A 16-fixture paired run compared the current four-thread server with six and
eight threads while holding the model, request order and `audio_ctx` values
fixed. All normalized transcripts and aggregate WER remained identical. For
the production `1280` path, mean/p95 latency was 3.209/3.731 seconds with four
threads, 3.672/4.598 seconds with six, and 4.028/5.288 seconds with eight. A
four-thread repeat after the higher-thread runs showed thermal drift, but no
higher-thread profile demonstrated a consistent advantage. Four threads remain
the qualified M3 setting.

The same corpus was screened with `--no-timestamps`. WER was unchanged, but
mean `1280` latency was 3.341 seconds versus the 3.209-second clean baseline,
and raw segment formatting changed on one fixture. Because the candidate did
not produce a repeatable latency reduction, timestamp decoding remains
unchanged in the production profile.
