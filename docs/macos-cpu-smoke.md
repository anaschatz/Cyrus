# macOS faster-whisper CPU smoke

This is a temporary source-run profile for the already-local CTranslate2 model. It does not replace the `whisper.cpp`/Metal target and does not make the same model directory compatible with `whisper.cpp`.

## Authenticated setup

Create one key outside the repository; its private parent directory must already
exist and the destination must not exist:

```zsh
PYTHONPATH=src python3.11 -m jarvis_voice create-auth-key \
  --output /absolute/private/voice-auth.json
```

Set the exact same absolute path in the private Voice and Orchestrator configs:

```json
"voice_auth_key_file": "/absolute/private/voice-auth.json"
```

The key uses schema `jarvis.voice-auth-key/1`, is created exclusively with mode
`0600` and is never printed. Keep Voice, the Orchestrator and inference on
loopback on this same Mac. Copying the key to a PC does not make a Mac-to-PC
split supported.

## 1. Passive offline doctor

Use CPython 3.11 and an absolute model directory:

```zsh
PYTHONPATH=src python3.11 -m jarvis_voice doctor-macos-cpu \
  --config /absolute/private/voice.json \
  --model-dir /absolute/path/to/the/local/faster-whisper-base/snapshot \
  --output /absolute/private/output/macos-cpu-doctor.json
```

The command performs no installation, download, model load, inference, microphone capture, TTS playback or permission request. It hashes the four expected model files to enforce the pinned revision, checks installed package metadata, uses only the non-prompting Input Monitoring preflight, queries the configured input device if `sounddevice` is already available and inventories `/usr/bin/say` voice identifiers.

Exit `0` means all prerequisites are ready. Exit `2` is an expected blocked result until Python 3.11, dependencies, the Silero ONNX manifest, permissions and explicit Greek/English voices are available.

The checked-in example also remains deliberately blocked because its
`voice_auth_key_file` is `null`. A ready report must name
`jarvis.voice-auth/1` and include `transport.loopback_authenticated`.

## 2. Interactive native smoke

Create a private config outside the repository containing the external auth
key, exact Silero artifact manifest and installed voice identifiers. Start the
loopback Orchestrator with the matching key, then deliberately run:

```zsh
PYTHONPATH=src python3.11 -m jarvis_voice run-macos-cpu \
  --config /absolute/private/voice.json \
  --model-dir /absolute/path/to/the/local/faster-whisper-base/snapshot
```

This command opens CoreAudio, starts the passive PTT event tap and can speak through `/usr/bin/say`; unlike the doctor it is intentionally interactive. The ASR adapter still validates the exact model manifest before constructing faster-whisper and passes the local directory with `device="cpu"`, the configured CPU compute type and `local_files_only=True`.

Exercise Greek, English, silence, forced release, first/last-unit barge-in, reconnect deduplication and shutdown. Do not call this milestone complete merely because the passive model check passes.

When the function row must remain in multimedia mode, macOS also accepts the
passive `CONTROL_OPTION` PTT chord. Hold Control and Option together while
speaking and release either modifier to end capture. The chord is listen-only,
does not suppress either modifier and does not change the Windows F-key path.

## Deferred

Dependency installation, permission changes, downloading/converting models, obtaining Silero, acquiring/building `whisper-server`, Metal latency qualification, `.app` packaging, signing and notarization remain explicit later actions.

Pre-auth native/runtime evidence cannot qualify Voice 0.8.0 because
`jarvis.voice-auth/1` is now part of the runtime profile fingerprint. Recollect
the authenticated smoke and rebuild/re-sign/re-notarize any packaged 0.7
artifact before using it as current evidence. For key-only rotation, stop both
Voice and Orchestrator, create a new key at a new path, update both configs,
restart Orchestrator then Voice, and rerun the authenticated smoke; there is no
dual-key or unauthenticated transition.

The handshake authenticates local peers but does not encrypt loopback traffic,
add TLS, sandbox processes or protect against same-user compromise.
