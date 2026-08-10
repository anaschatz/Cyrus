# Chatterbox Multilingual V3 macOS pilot

This is an isolated qualification path. It does not replace `MacSystemTtsBackend`,
change `jarvis.voice/1`, install packages or download models.

## Pinned candidate

- package: `chatterbox-tts==0.1.7`
- model: `ResembleAI/chatterbox`
- revision: `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`
- multilingual T3: `v3`
- language: `el`
- device: Apple Silicon MPS
- watermarker override: `resemble-perth==1.0.1` from the official PyPI wheel

The override replaces the package metadata's moving `git+...@master` Perth
dependency. Resolve `requirements/chatterbox-pilot.in` with
`requirements/chatterbox-pilot.overrides`, generate hashes, acquire wheels,
then install offline into a dedicated venv. Do not mutate the existing
`macos-cpu-v1` environment.

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

## Offline pilot

After a separately approved acquisition has produced a hash-verified local
model directory and an isolated Python 3.11 environment, run:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"/absolute/path/to/pilot/python" scripts/run_chatterbox_v3_pilot.py run \
  --model-dir "/absolute/local/model/directory" \
  --output-dir "/absolute/private/output/directory" \
  --report "/absolute/private/evidence/chatterbox-v3-pilot.json"
```

The pilot emits three fixed Greek fixtures and content-free latency, audio
duration, real-time factor and peak-memory evidence. Production integration,
streaming playback and worker-process cancellation remain later gates.
