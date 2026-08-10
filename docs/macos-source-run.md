# Milestone 5 — macOS Apple Silicon source run

M5 adds source-run parity only. It does not install dependencies, build `whisper.cpp`, convert or download a model, create an `.app`, request permissions, sign or notarize anything.

## Locked native profile

- macOS 14 or later, Apple Silicon `arm64`, CPython 3.11 x64.
- `whisper.cpp` v1.9.2 `whisper-server`, built locally as native arm64 with Metal and an embedded Metal library. Core ML remains off for v1.
- An explicit local GGML/GGUF-compatible Whisper model. `ggml-large-v3-turbo-q5_0.bin` is the target profile, but M5 does not fetch it and does not replace an existing model.
- The existing Silero ONNX artifact, `sounddevice`/CoreAudio, SoXR and loopback WebSocket transport.
- `/usr/bin/say` with explicit installed Greek and English voice names.

The existing Windows `faster-whisper` model directory is a different runtime format. Pointing the macOS sidecar at that directory is rejected; M5 never attempts implicit conversion or a tokenizer/model fallback.

## Prepare explicit manifests

Keep `macos.whisper_cpp.revision` at the approved `v1.9.2`. Fill `macos.whisper_cpp.executable` and `macos.whisper_cpp.model` in a private config outside a future application bundle. Each requires an exact `path`, byte `size` and lowercase SHA-256. On macOS, the values can be inspected without changing the artifacts:

```zsh
stat -f %z /explicit/path/to/whisper-server
shasum -a 256 /explicit/path/to/whisper-server
stat -f %z /explicit/path/to/ggml-model.bin
shasum -a 256 /explicit/path/to/ggml-model.bin
```

Do the same for the existing `vad.artifact`. Configure installed system voice identifiers under `macos.system_voices`, for example `el-GR` and `en-US`. The doctor checks the names against `/usr/bin/say -v ?` but never includes voice sample phrases in its report.

## Offline doctor

Run the content-free diagnostic before opening the microphone or starting the sidecar:

```zsh
PYTHONPATH=src python3.11 -m jarvis_voice doctor-macos \
  --config /explicit/path/to/voice.json \
  --output /explicit/path/to/macos-doctor.json
```

The parent output directory must already exist. The report contains no configured paths, hashes, transcript, audio, prompts, usernames, device identifiers or subprocess output. A missing permission, dependency or artifact produces exit code `2` and `ready: false`.

Input Monitoring and microphone access must be granted manually to the exact source interpreter during development. The doctor only calls the non-prompting CoreGraphics preflight API. A future signed `.app` will have a different TCC identity and must be qualified again.

## Source runtime

Start the existing loopback orchestrator separately, then run:

```zsh
PYTHONPATH=src python3.11 -m jarvis_voice run-macos --config /explicit/path/to/voice.json
```

The runtime starts the owned `whisper-server` lazily on first speech turn. It binds only to `127.0.0.1`, generates a fresh unreported request-path token, disables context carry-over, never enables ffmpeg conversion and sends an in-memory 16-kHz WAV to that private path's `/inference` endpoint. The upstream server example itself warns that file-upload servers need sandboxing; this adapter therefore keeps the process unprivileged, loopback-only, bounded and supervised.

The PTT event tap is listen-only and does not suppress the configured F1–F20 key. `PTT_DOWN` tombstones stale output, terminates active `say` or ASR work and begins the next capture. Ambient speech without PTT does not interrupt in v1.

## Deferred native evidence

Before M5 can be called native-complete, exercise Greek/English turns, silence, first/last-unit barge-in, permission denial, event-tap disable, sidecar crash/timeout and sleep/resume on the explicit Mac profile. Packaging, Developer ID signing, notarization, hardened runtime entitlements and packaged-app TCC testing belong to the next milestone.
