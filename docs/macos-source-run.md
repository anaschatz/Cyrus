# Milestone 5 — macOS Apple Silicon source run

M5 adds source-run parity only. It does not install dependencies, build `whisper.cpp`, convert or download a model, create an `.app`, request permissions, sign or notarize anything.

## Locked native profile

- macOS 14 or later, Apple Silicon `arm64`, CPython 3.11 x64.
- `whisper.cpp` v1.9.2 `whisper-server`, built locally as native arm64 with Metal and an embedded Metal library. Core ML remains off for v1.
- An explicit local GGML/GGUF-compatible Whisper model. `ggml-large-v3-turbo-q5_0.bin` is the target profile, but M5 does not fetch it and does not replace an existing model.
- The existing Silero ONNX artifact, `sounddevice`/CoreAudio, SoXR and loopback WebSocket transport.
- `/usr/bin/say` with explicit installed Greek and English voice names.
- One external `jarvis.voice-auth-key/1` file shared by the Voice and
  Orchestrator configs on this Mac.

The existing Windows `faster-whisper` model directory is a different runtime format. Pointing the macOS sidecar at that directory is rejected; M5 never attempts implicit conversion or a tokenizer/model fallback.

## Provision the authenticated local boundary

Create one per-install key outside the repository. Its private parent directory
must already exist and the output path must not exist:

```zsh
PYTHONPATH=src python3.11 -m jarvis_voice create-auth-key \
  --output /absolute/private/path/voice-auth.json
```

The command creates the file exclusively with mode `0600` and does not print
the secret. Set this exact path in both private configs:

```json
"voice_auth_key_file": "/absolute/private/path/voice-auth.json"
```

Keep the Voice URL, Orchestrator Voice listener and inference appliance on
loopback on this Mac. A macOS Voice client with a Windows/PC Orchestrator is not
a supported split; do not copy the key to another host.

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

The ready report must also identify `jarvis.voice-auth/1` and include the
`transport.loopback_authenticated` check. A checked-in config with
`voice_auth_key_file: null`, an unreadable key or a non-private key is expected
to remain blocked.

Input Monitoring and microphone access must be granted manually to the exact source interpreter during development. The doctor only calls the non-prompting CoreGraphics preflight API. A future signed `.app` will have a different TCC identity and must be qualified again.

## Source runtime

Start the existing loopback Orchestrator with its matching key separately,
then run:

```zsh
PYTHONPATH=src python3.11 -m jarvis_voice run-macos --config /explicit/path/to/voice.json
```

The runtime starts the owned `whisper-server` lazily on first speech turn. It binds only to `127.0.0.1`, generates a fresh unreported request-path token, disables context carry-over, never enables ffmpeg conversion and sends an in-memory 16-kHz WAV to that private path's `/inference` endpoint. The upstream server example itself warns that file-upload servers need sandboxing; this adapter therefore keeps the process unprivileged, loopback-only, bounded and supervised.

The PTT event tap is listen-only and does not suppress the configured F1–F20 key. `PTT_DOWN` tombstones stale output, terminates active `say` or ASR work and begins the next capture. Ambient speech without PTT does not interrupt in v1.

## Native qualification

Use `scripts/run_macos_native_collection.py calibrate` first, then run the
collector with the emitted threshold and `--threshold-calibrated`. The runtime
automatically reopens a dead CoreAudio stream and refreshes CoreAudio plus the
event tap after a wake-sized heartbeat gap. The strict gate still requires a
real disconnect, a real sleep/resume, 200 physical turns, 20 acoustic samples
below 200 ms and the full eight-hour soak; unit tests cannot satisfy it.

On a memory-constrained qualification machine, add
`--maximum-real-voice-turns 1` through `5` to produce a bounded partial run.
The collector exits only after the requested number of physical transcripts
also have matching completed playbacks, then unloads the native runtime. A
bounded batch is never reported as the 200-turn or eight-hour production gate,
and separate soak durations must not be added together.

The approved F1 can be qualified without installing MLX into the voice
environment. Supply the complete `--f1-*` override set to `preflight` and
`run`: the existing voice interpreter remains responsible for capture, VAD,
ASR and transport, while the digest-bound F1 interpreter runs MLX, SNAC and
CoreAudio in a local process group. The parent accepts only bounded JSONL
control events and content-free callbacks. A missing field, changed interpreter
digest, malformed callback order or dead pipe fails closed. The private config
is not modified by these overrides.

If the configured loopback orchestrator is already running, pass an unused
`--loopback-port` between 1024 and 65535. The qualification server binds only
to `127.0.0.1`; the runner never terminates or replaces the existing service.
Use `--ptt-key CONTROL_OPTION` when the function row is mapped to media keys;
the override applies only to that qualification process.

Packaging, Developer ID signing, notarization, hardened runtime entitlements
and packaged-app TCC testing remain a separate qualification profile.

The authentication boundary changes the native runtime profile fingerprint.
Pre-auth native evidence cannot qualify Voice 0.8.0 or a current scheduler-TTS
promotion; recollect it with the authenticated runtime. Packaged 0.7 artifacts
must likewise be rebuilt, re-signed and re-notarized before qualification. A
key-only rotation with unchanged code/profile still requires a fresh
authenticated connection smoke, but does not by itself replace the full
acoustic evidence set.

Rotation has no dual-key window: stop Voice and Orchestrator, create a new key
at a new absolute path, update both configs, restart Orchestrator followed by
Voice, and rerun the private-stack `status` and `e2e` gates before retiring the
old key. Because those gates use their own protocol client, also complete one
native Voice PTT turn to prove the live microphone-to-speaker path.

The shared-key HMAC handshake authenticates the two local peers and binds the
exact hello frame. It is not TLS or payload encryption, does not grant remote
transport, does not authorize application actions and does not defend against
malware already running as the same macOS user.
