# Chatterbox Multilingual V3 macOS pilot

This is an isolated qualification path. It does not replace `MacSystemTtsBackend`,
change `jarvis.voice/1`, install packages or download models.

## Pinned candidate

- dependency wheel: `chatterbox-tts==0.1.7`
- V3 source overlay: official commit `65b18437192794391a0308a8f705b1e33e633948`
- model: `ResembleAI/chatterbox`
- revision: `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`
- multilingual T3: `v3`
- language: `el`
- device: Apple Silicon MPS
- watermarker override: `resemble-perth==1.0.1` from the official PyPI wheel

The PyPI `0.1.7` wheel predates the V3 source API even though the upstream
project kept the same package version. It is therefore used only to resolve a
fully hashed binary dependency closure, then replaced with the exact official
V3 release commit in `requirements/chatterbox-pilot-source.txt`. The override
also replaces the package metadata's moving `git+...@master` Perth dependency.
Resolve `requirements/chatterbox-pilot.in` with
`requirements/chatterbox-pilot.overrides`, generate hashes, acquire wheels,
install them into a dedicated venv, then install the exact source overlay with
`--no-deps`. Do not mutate the existing
`macos-cpu-v1` environment.

`setuptools==80.9.0` is intentional: Perth `1.0.1` imports the legacy
`pkg_resources` module, which is no longer present in newer setuptools builds.

The four pinned binary artifacts total about 3.21 GB. The acquisition gate
requires 8 GiB free to leave room for the isolated Python 3.11 environment,
temporary package files and at least 2 GiB of post-install headroom.

## Read-only preflight

Run with the already materialized CPython 3.11 interpreter:

```sh
"/absolute/path/to/python3.11" scripts/run_chatterbox_v3_pilot.py preflight \
  --storage-root "/absolute/existing/storage/root" \
  --model-dir "/absolute/future/model/directory"
```

Exit `2` is an expected, typed blocked result when disk space, the isolated
dependencies or the pinned model directory are unavailable. Preflight performs
no network access, installation, microphone capture or runtime start.

## Pinned model acquisition

After the exact source overlay is installed and preflight reports
`acquisition_ready: true`, acquire only the five required model files:

```sh
"/absolute/path/to/pilot/python" scripts/run_chatterbox_v3_pilot.py acquire \
  --storage-root "/absolute/private/pilot/root" \
  --model-dir "/absolute/private/pilot/root/model" \
  --report "/absolute/private/pilot/root/evidence/acquisition.json"
```

The command pins the Hugging Face commit, uses an exact allow-list, limits
download concurrency, validates binary sizes and SHA-256 values, and refuses a
model directory outside the private pilot root.

## Offline pilot

After a separately approved acquisition has produced a hash-verified local
model directory and an isolated Python 3.11 environment, run:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"/absolute/path/to/pilot/python" scripts/run_chatterbox_v3_pilot.py run \
  --model-dir "/absolute/local/model/directory" \
  --output-dir "/absolute/private/output/directory" \
  --reference-audio "/absolute/private/reference-voice.flac" \
  --report "/absolute/private/evidence/chatterbox-v3-pilot.json"
```

The pilot emits three fixed Greek fixtures and content-free latency, audio
duration, real-time factor and peak-memory evidence. Output above 0.95 peak is
attenuated before saving to preserve playback headroom. Production integration,
streaming playback and worker-process cancellation remain later gates.

For this Greek-only qualification, the harness disables Chatterbox's eager
Chinese Cangjie/PKUSeg initialization before constructing the tokenizer. That
upstream initializer otherwise performs an unrelated network download even
when `language_id="el"`; Chinese synthesis is deliberately outside this pilot.
The run is additionally wrapped in a fail-closed Python socket guard, restored
when the pilot returns, so unexpected runtime downloads fail the qualification.

The official pinned demo maps Greek to the male reference
`mtl_prompts/el_m.flac`. The qualified local copy is 162,538 bytes with SHA-256
`0c2a9aac8fb4671c05562cda22ac8a4de38c72f561f72a8c3da9f416e5b58923`.

## Observed M3 pilot (2026-08-10)

- model load: 8.16 s
- peak resident memory: 4.91 GiB
- three generated clips: 3.44 s, 2.36 s and 2.84 s
- generation real-time factors: 2.81, 3.31 and 2.03 after model load
- output: mono 24 kHz float WAV, finite samples and no clipping

The candidate reaches the local quality-listening gate but does not pass the latency gate
for default conversational streaming on this machine. It remains an optional
high-quality backend candidate until a worker/streaming optimization reaches a
real-time factor below 1 with acceptable time-to-first-audio.

## Isolated voice-conversion comparison

The already pinned S3Gen component can convert a validated source WAV to a
target speaker reference. This lets the pilot retain GreekTTS-1.5's linguistic
audio while changing speaker timbre, without asking Chatterbox's text model to
pronounce the Greek sentence:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_chatterbox_v3_pilot.py voice-convert \
  --model-dir "/absolute/private/chatterbox/model" \
  --source-audio "/absolute/private/greektts15-8bit.wav" \
  --target-voice-audio "/absolute/private/target-reference.wav" \
  --label serious-male \
  --output-dir "/absolute/private/output" \
  --report "/absolute/private/evidence/serious-male.json"
```

The conversion is offline, watermarked by the upstream Chatterbox component and
kept outside the production backend. A listening and ASR round-trip gate is
required because voice conversion can still alter consonants or introduce
artifacts.

### Observed serious-voice A/B (2026-08-11)

- the male conversion produced 2.22 s at median F0 140.9 Hz, with 6.07 s
  conversion time (RTF 2.74)
- the female conversion produced 2.24 s at median F0 195.9 Hz, with 3.86 s
  conversion time (RTF 1.73)
- both outputs are mono 24 kHz Float32 WAVs, finite, watermarked and generated
  with the Python network guard active
- local faster-whisper-base round trips showed more word errors than the source
  GreekTTS-1.5 clip, so these samples are listening candidates only and cannot
  be promoted without a better target reference or voice-conversion model

A second local-reference pair was also retained for listening: `serious-male-2`
uses the macOS Daniel voice (median output F0 107.7 Hz) and
`serious-female-2` uses the macOS Moira voice (169.6 Hz). Both preserve the
2.22 s source duration, but their faster-whisper-base round trips still show
consonant loss, so they remain non-production candidates.

The third listening pair uses the macOS Reed and Samantha references. The
converted outputs measure 115.1 Hz and 167.6 Hz median F0 respectively, retain
the 2.24 s source duration and remain subject to the same consonant-loss gate.
