# macOS faster-whisper CPU smoke

This is a temporary source-run profile for the already-local CTranslate2 model. It does not replace the `whisper.cpp`/Metal target and does not make the same model directory compatible with `whisper.cpp`.

## 1. Passive offline doctor

Use CPython 3.11 and an absolute model directory:

```zsh
PYTHONPATH=src python3.11 -m jarvis_voice doctor-macos-cpu \
  --config config.example.json \
  --model-dir /absolute/path/to/the/local/faster-whisper-base/snapshot \
  --output /absolute/private/output/macos-cpu-doctor.json
```

The command performs no installation, download, model load, inference, microphone capture, TTS playback or permission request. It hashes the four expected model files to enforce the pinned revision, checks installed package metadata, uses only the non-prompting Input Monitoring preflight, queries the configured input device if `sounddevice` is already available and inventories `/usr/bin/say` voice identifiers.

Exit `0` means all prerequisites are ready. Exit `2` is an expected blocked result until Python 3.11, dependencies, the Silero ONNX manifest, permissions and explicit Greek/English voices are available.

## 2. Interactive native smoke

Create a private config outside the repository containing the exact Silero artifact manifest and installed voice identifiers. Start the loopback orchestrator, then deliberately run:

```zsh
PYTHONPATH=src python3.11 -m jarvis_voice run-macos-cpu \
  --config /absolute/private/voice.json \
  --model-dir /absolute/path/to/the/local/faster-whisper-base/snapshot
```

This command opens CoreAudio, starts the passive PTT event tap and can speak through `/usr/bin/say`; unlike the doctor it is intentionally interactive. The ASR adapter still validates the exact model manifest before constructing faster-whisper and passes the local directory with `device="cpu"`, the configured CPU compute type and `local_files_only=True`.

Exercise Greek, English, silence, forced release, first/last-unit barge-in, reconnect deduplication and shutdown. Do not call this milestone complete merely because the passive model check passes.

## Deferred

Dependency installation, permission changes, downloading/converting models, obtaining Silero, acquiring/building `whisper-server`, Metal latency qualification, `.app` packaging, signing and notarization remain explicit later actions.
