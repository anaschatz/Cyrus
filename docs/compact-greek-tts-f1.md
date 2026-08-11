# Compact Greek TTS — STOMA F1 preflight

This milestone records a no-install, no-download plan for the F1 voice selected
by listening. It does not replace the active TTS backend.

## Decision

Fine-tune a Piper medium voice on the native-Greek STOMA F1 speaker, then export
it to ONNX for local inference. The existing Greek `rapunzelina` checkpoint is
only a Greek pronunciation and model-shape warm start; its speaker identity is
not the target voice.

STOMA's repository includes ESPnet FastSpeech2 and HiFi-GAN recipes, but not the
trained weights. The project explicitly asks users to contact the authors for
pretrained models. Because there is no downloadable official F1 checkpoint, the
compact Piper path remains the reproducible default. If the authors later
provide an F1 checkpoint, it should be evaluated before training anything.

## Exact, minimal inputs

- STOMA dataset revision: `ea413ab51a4fc3b702f93afe26a4f4a2eb7d6a18`
- STOMA recipe tree: `92b1967ad5ab59c400ec7ee27b5bc07920ec7bc2`
- speaker: `F1`, 1,301 clips, 1:46:58
- source: 44.1 kHz, 16-bit mono PCM, CC-BY-4.0
- Piper: `v1.6.0`, commit `f04d52c5528ac7cf2d73757f57990ff490f75005`
- Greek medium warm start revision: `78b42203befdcf7c5c35fbe00aef5796e03aab44`
- target sample rate: 22.05 kHz

Only the first two pinned Parquet shards are required. They total 913,827,604
bytes, rather than the full 7.61 GB corpus. Together they contain 1,942 rows:
all 1,301 F1 rows plus 641 F2 rows. Extraction must select `speaker_id == "F1"`,
validate the exact per-section counts, and fail if any F2 row reaches training.
Every downloaded artifact is size- and SHA-256-pinned in
`compact_greek_tts_pilot.py`.

## Read-only check

No dependency installation or model download is needed to run:

```sh
PYTHONPATH=src python3 scripts/run_compact_greek_tts_preflight.py \
  --storage-root "$PWD"
```

The report always states whether network access, installation, model download,
or production-backend mutation occurred. For this preflight all four must be
false.

## Training and promotion gates

Training is planned for a separate CUDA host. Piper reports successful training
with as little as 8 GiB VRAM, while its documented reference machines use much
larger NVIDIA GPUs. The M3 remains the inference and listening-qualification
host; the plan does not reserve unified memory for training alongside the main
LLM.

The exported F1 candidate is rejected unless all of these pass:

- ONNX model at most 100 MiB;
- measured peak process RSS at most 1 GiB on the M3;
- warm time to first audio at most 600 ms;
- cancellation to silence at most 100 ms;
- synthesis passes with network disabled;
- matched Greek pronunciation fixtures pass automated checks;
- the user explicitly approves the F1 timbre after listening.

Only after those gates pass may a streaming `TtsBackend` adapter be added. The
current production TTS remains unchanged until then.

### Mac qualification harness

The qualification harness is implemented now, but it deliberately does not
install the already downloaded Piper wheel and cannot run real synthesis until
the trained export exists. It accepts only these two regular, non-symlinked
files:

- `el_GR-jarvis-f1-medium.onnx`
- `el_GR-jarvis-f1-medium.onnx.json`

The JSON must describe one Greek eSpeak speaker at 22.05 kHz with the expected
Piper medium symbol table. The harness computes both hashes at qualification
time; they cannot be pinned before training produces the files.

Run the read-only readiness check at any time:

```sh
python3 scripts/run_piper_f1_qualification.py preflight \
  --candidate-root "$PWD/artifacts/compact-greek-tts-f1/candidate"
```

After Windows/WSL2 training and ONNX export, create a separate Python 3.11
environment, install only the already pinned `piper-tts==1.6.0` wheel there, and
run:

```sh
python3 scripts/run_piper_f1_qualification.py qualify \
  --candidate-root "/absolute/path/to/candidate" \
  --output-directory "/absolute/new/empty/qualification-output" \
  --report "/absolute/new/qualification-report.json" \
  --exercise-playback-cancel
```

The real Piper runtime lives in a child process. Cancellation aborts the
PortAudio output first and then hard-stops that process, so a long ONNX sentence
cannot trap cancellation in a Python thread. Without
`--exercise-playback-cancel`, the cancellation gate is intentionally unmeasured
and cannot pass. `RawOutputStream.abort()` latency is software evidence; the
native acceptance run must separately measure acoustic silence through a real
microphone, complete 200 physical turns and the eight-hour soak. Four fixed
Greek fixtures are written as WAV files for listening. Even if every candidate
metric passes, promotion remains blocked until both native evidence and explicit
user voice approval exist. The command never switches the active backend.

## Pinned acquisition and extraction

Inspect the exact byte plan on the Mac without network or filesystem writes:

```sh
python3 scripts/run_compact_greek_tts_acquisition.py plan \
  --storage-root "/absolute/existing/pilot-root" \
  --group all
```

The actual acquisition is deliberately a separate command. It refuses to start
unless `--confirm-download-bytes` exactly matches the plan. Downloads use an
allow-list of pinned HTTPS URLs, resumable `.part` files, exact size/SHA-256
validation, and atomic publication. It never installs Piper:

```sh
python3 scripts/run_compact_greek_tts_acquisition.py acquire \
  --storage-root "/absolute/existing/pilot-root" \
  --group all \
  --confirm-download-bytes 1793832034
```

After acquisition, extraction requires PyArrow only in an isolated data-prep
environment. It reads the embedded WAV bytes directly, accepts exactly 1,301 F1
rows with the pinned B2/C1/C2/Harvard distribution, rejects all 641 F2 rows, and
atomically emits `metadata.csv`, `audio/`, and `bundle-manifest.json`. No audio
is resampled at this stage: pinned Piper uses Librosa to resample each source WAV
to the configured 22.05 kHz while preparing its training cache.

The Mac data-prep dependency is exactly `pyarrow==25.0.1`, installed from the
CPython 3.11/macOS arm64 wheel with SHA-256
`51093dd9e10325fbdb3c10a2ae7c4806e5c822d94e74ae4938b26524a3323fee`.
It lives in an isolated `.venv-data-prep` under the pilot artifact directory and
is not part of the JARVIS runtime dependency set.

Observed extraction on the pinned shards produced 1,301 WAV files totaling
6,678.371 seconds of raw 44.1 kHz PCM audio and occupying about 565 MiB. This is
260.371 seconds above STOMA's published 6,418-second F1 speech duration, or about
0.2 seconds per clip. The raw recordings are intentionally preserved; Piper's
pinned data module performs Silero-based silence trimming and keeps its configured
speech margins while building the training cache.

The generated training plan targets WSL2 Ubuntu plus one NVIDIA GPU, starts with
a conservative batch size of four, and caps fine-tuning at 1,000 epochs. That is
a ceiling, not a requirement: listen at epochs 100, 250, 500, and 1,000 and stop
when quality no longer improves. Before training, the Windows host still needs a
CUDA/VRAM probe, an exact dependency lock, and a CPU checkpoint-load probe.

## Licensing gate

The STOMA speech data is CC-BY-4.0 and the Greek warm-start dataset is documented
as CC0. Current Piper code is GPL-3.0-or-later and includes eSpeak-NG. That is an
explicit distribution/compliance review gate, not a permissive dependency. A
personal local pilot can remain isolated, but no packaged or proprietary release
is approved by this document.
