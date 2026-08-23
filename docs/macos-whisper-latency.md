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

The installed runtime passed the authenticated physical-turn transport gate on
2026-08-23 with the duration-gated profile active. Eager model warm-up remains
active regardless of the audio-context setting.

## Physical first-turn VAD warm-up

The first physical turn initially spent 421.1 ms between audio normalization
and VAD completion, even though a warmed 16-fixture Silero run measured a
10.47-ms mean and 24.27-ms p95. The native runtime now executes one deterministic
512-sample, 16-kHz silent frame through the pinned VAD before opening CoreAudio
or accepting PTT. Startup fails closed if that silence is classified as speech.

After installation, the next physical turn measured 25.76 ms from normalization
to VAD completion; a second turn in the same session measured 10.02 ms. The
authenticated gate observed exactly one admitted turn, one local final-model
request, ordered speech/text completion, retired playback, no duplicate or
stale output, and a continuous Voice connection. The first post-warm turn took
5.89 seconds from PTT release to TTS start, versus 6.85 seconds in the earlier
turn; because the spoken inputs differed in duration, only the isolated
395.3-ms VAD reduction is attributed to this change.

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

A separate 16-fixture Greek run screened the decoder's `best-of` setting.
Reducing it from two to one retained identical normalized transcripts and the
same 1.63% aggregate WER, but mean `1280` latency increased from 3.892 to 4.566
seconds. The sequential runs can include thermal drift, but the candidate did
not demonstrate the required repeatable speedup and is therefore rejected.

The same physical qualification pass found eleven old `whisper-server`
children whose Voice parents had already exited. The sidecar now inherits the
Voice process group so supervisor shutdown also contains parent crashes, and
the explicit close path waits again after a forced kill to reap a stubborn
child. All observed orphan sidecars were terminated before benchmarking.

## Core ML / ANE qualification

The M3 candidate uses the upstream v1.9.2 `WHISPER_COREML=1` server, the same
`large-v3-turbo-q5_0` decoder model, and the compiled Core ML encoder derived
from Hugging Face revision
`3e251d613ae82c33b1dcca9f0406c697c3f9d083`. The runtime manifest pins the
server, decoder and every file below `ggml-large-v3-turbo-encoder.mlmodelc` by
relative path, byte size and SHA-256. It rejects symlinks, extra files, missing
files and hash or size mismatches before the sidecar starts. The load path is
fully local and permits no tokenizer or network fallback.

An expanded 34-fixture run (24 Greek and 10 English/code-switch) qualified only
full context. Core ML full context measured 1.572-second mean latency,
1.682-second p95 and 8.99% aggregate WER, compared with 9.28% WER for the Metal
baseline. Five transcripts changed: two fixtures improved and one degraded.
The previously promoted Metal `audio_ctx=1280` policy is therefore explicitly
disabled whenever the Core ML encoder is active; `qualified_audio_context` must
be zero and every request uses full context.

The standalone sidecar stabilized around 708 MB footprint (approximately
744 MB RSS). Cold startup peaked around 1.96 GB, including roughly 1.2 GB of
reclaimable ANE neural allocation. A simultaneous local-LLM coexistence smoke
retained about 65% system memory availability and completed without memory
failure. Cold startup may require up to 120 seconds; that allowance does not
change the per-turn deadline.

The installed profile then passed real CoreAudio/PTT qualification. The first
measured Core ML turn started TTS 3.66 seconds after PTT release, versus the
earlier 5.7-to-5.9-second Metal observations. In the final warm session, two
ordinary sequential turns started TTS after 4.59 and 2.87 seconds; their ASR
stages took 1.92 and 1.55 seconds. A separate acoustic barge-in stopped active
playback 6.6 ms after PTT down, admitted the new transcript, completed ASR in
1.51 seconds and started the replacement response 2.89 seconds after PTT
release. Both turns completed with distinct epochs and no duplicate transcript,
stale playback, reconnect, stuck PTT or gateway 1011 failure.

The Voice transport now also performs one bounded reconnect for a pending
transcript while preserving its `event_id` and allocating a new connection
sequence. The gateway records only a fixed phase, bounded exception category
and optional numeric SQLite/status code on internal failure; transcript text
and exception messages never enter the operational log.
