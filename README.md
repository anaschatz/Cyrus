# JARVIS — Voice, Local Inference and Orchestrator Core v1.0

This repository implements three explicit sibling boundaries: the `jarvis.voice/1` capture/ASR/TTS client, the frozen OpenAI-compatible local inference appliance, and the durable Orchestrator that alone owns task lifecycle, conversations, routing, budgets, approvals, cancellation epochs, canonical history and output delivery.

`function-key PTT → WASAPI capture → Silero VAD → local faster-whisper → loopback WebSocket → SAPI5`

The voice layer only captures, validates, transcribes, transports, speaks, and cancels. It does not interpret requests or contain orchestrator/agent logic. Its only Orchestrator-related persistence is a bounded, content-free server-event dedup journal used to prevent duplicate TTS after a voice process restart. Authoritative `response.speech.commit` events remain the only TTS input.

The new [`orchestrator/`](orchestrator/) package is Python 3.11 source-run code with a hash-bearing cross-platform `uv.lock`. It uses `pydantic-ai-slim[openai]==2.27.0`, but its public and durable domain schemas remain framework-independent. See the [Orchestrator runbook](orchestrator/README.md) and [Core v1 contract map](docs/orchestrator-core-v1.md).

A strict `jarvis-private-stack/2` manifest now provides one private operator
entry point for the real voice, Orchestrator and qualified inference configs.
It validates their cross-component bindings without printing secrets and can
run a real local-model WebSocket-to-speech-commit probe. See the
[private stack runbook](docs/private-stack-run.md). For the bounded,
foreground-only Mac hardware turn, use the
[macOS technical-beta operator runbook](docs/macos-technical-beta-operator.md).

## Authenticated single-host boundary

Voice 0.8.0 and the Orchestrator require `jarvis.voice-auth/1` before the first
`jarvis.voice/1` frame. Create one per-install key outside the repository; the
parent directory must already exist and the destination must not exist:

```zsh
PYTHONPATH=src python3.11 -m jarvis_voice create-auth-key \
  --output /absolute/private/path/voice-auth.json
```

Set the exact same absolute path in both private component configs:

```json
"voice_auth_key_file": "/absolute/private/path/voice-auth.json"
```

The creator uses schema `jarvis.voice-auth-key/1`, creates the file
exclusively, applies mode `0600` on POSIX or a restricted owner DACL on
Windows, and never prints the secret. The checked-in examples deliberately use
`null` and therefore fail closed until materialized.

Voice, the Orchestrator and the local inference appliance form one machine-local
deployment. The Voice WebSocket and inference endpoints stay on loopback; a Mac
Voice client must not be paired with a PC Orchestrator, or the reverse. The
shared-key handshake authenticates the two local Voice peers and binds the
exact hello frame. It does not encrypt WebSocket traffic, provide TLS, authorize
later application operations, sandbox processes or protect against malware
already running as the same user.

There is no dual-key/live rotation: stop Voice and Orchestrator, create a new
key at a new path, update both configs, restart Orchestrator then Voice, and
require authenticated `status`, `e2e` and one native PTT turn before retiring
the old key. This is
also a qualification boundary: pre-auth Voice 0.7 packages/evidence and `/1`
operational transport evidence do not qualify the current runtime. See the
[private stack runbook](docs/private-stack-run.md) for the exact rotation and
evidence compatibility rules.

An off-by-default Anamnesis adapter now provides qualified local retrospective
recall for explicit completed statements. Durable memory remains in
Orchestrator SQLite, is isolated per conversation and is removed by logical
conversation deletion. See the
[Anamnesis integration qualification](docs/anamnesis-memory-integration.md).

Milestone 5 adds macOS Apple Silicon source adapters targeting `whisper.cpp`/Metal. Milestone 5A audits existing local artifacts. Milestone 5B adds an explicit temporary `faster_whisper_cpu_smoke` profile so the already-local pinned CTranslate2 model can be used for a native macOS source smoke without downloading or converting a model. The default macOS backend remains `whisper_cpp`.

## Current verification boundary

Development and dependency-free/fake verification occurred on macOS ARM64. The M4 harness correctly produces a blocked result on this host; Windows, Shell notification behavior, PyInstaller output, Authenticode, WASAPI, SAPI, NVIDIA, CUDA, cuDNN, and local-LLM coexistence have not been hardware-verified. No dependency, driver, model, voice, packaging tool, certificate, or artifact was installed or downloaded.

```sh
python3 -m unittest discover -s tests -v
python3 -m compileall src tests packaging
PYTHONPATH=src python3 -m jarvis_voice doctor --config config.example.json

cd orchestrator
uv sync --frozen --python 3.11
PYTHONPATH=src:../src:../inference/src uv run --frozen python -m unittest discover -s tests -v
```

The safe example intentionally selects CPU and contains no machine-specific
paths, authentication key or GPU identifier. Its doctor result is expected to
remain blocked; use a private materialized config for a readiness claim.

## macOS CPU readiness

The passive command below accepts the already-local CTranslate2 model through
an absolute path, validates its exact four-file manifest and never loads it.
For a readiness result, replace the example config with a private config that
also contains the external `voice_auth_key_file`:

```zsh
PYTHONPATH=src python3 -m jarvis_voice doctor-macos-cpu \
  --config /absolute/private/path/voice.local.json \
  --model-dir /absolute/local/model/snapshot \
  --output /absolute/private/macos-cpu-doctor.json
```

It does not install, download, infer, capture audio, play speech or request permission. A real interactive run is a separate deliberate `run-macos-cpu` command and requires a private config with Silero and voice manifests. See the [Milestone 5B implementation prompt](docs/milestone-5b-prompt.md) and [macOS CPU smoke runbook](docs/macos-cpu-smoke.md).

## Runtime dependencies

The existing proposal is unchanged:

- `faster-whisper==1.2.1`
- `sounddevice==0.5.3`
- `soxr==1.1.0`
- `onnxruntime==1.27.0`
- `websockets>=15,<16`
- `pywin32>=311`

`CTranslate2` is brought by faster-whisper. Its exact resolved version and the compatible NVIDIA driver/CUDA/cuDNN matrix must be inspected and pinned on the target Windows machine before installation approval. This repository does not install NVIDIA components or modify machine-wide environment variables.

`PyInstaller==6.21.0` is a build-only proposal. It is not a runtime dependency and was not installed on the macOS host. A Windows bundle must be built on native Windows 11 x64 with the approved CPython 3.11 environment.

The ASR model remains `Systran/faster-whisper-base` revision `ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66`. Its four files are validated by exact size and SHA-256 in both the parent and CUDA worker before model construction. No model name or network fallback is permitted.

## CPU and CUDA profiles

CPU remains the default and never imports the CUDA worker module. CUDA requires an explicit device index and supports:

- `turn_scoped`: spawn, load, transcribe, and exit before `input.transcript.final` is created. Process exit is the CUDA-context/VRAM release boundary.
- `warm`: retain one supervised worker between turns, with a bounded restart count. Use only after native headroom/coexistence measurement.

GPU admission reads only index, name, driver, total VRAM, and free VRAM through `nvidia-smi`; it never enumerates GPU processes. Admission subtracts the configured reserve for other workloads. This is a point-in-time heuristic, not a cross-process reservation.

CUDA failure can fall back once to the persistent CPU backend using the same validated audio. A partial CUDA result is never submitted, and late results from cancelled turns are rejected by turn identity.

## Windows tray and packaging

The windowed entry point defaults to `tray`. The tray uses native `Shell_NotifyIconW`, restores itself after Explorer's `TaskbarCreated`, routes only lifecycle/folder/exit commands, and keeps voice work off its message thread. A `Local\...` named mutex prevents a second copy in the same Windows session.

The release shape is PyInstaller `onedir`, `asInvoker`, no UPX and no bundled mutable data. Config, models, VAD artifacts, logs and state stay outside the application directory; the default per-user root is `%LOCALAPPDATA%\JARVIS Voice`. A frozen runtime hook narrows DLL lookup without changing `PATH`.

The offline manifest commands inventory every distributed file by relative path, size and SHA-256:

```powershell
py -3.11 -m jarvis_voice dist-manifest --directory ".\dist\JARVIS Voice"
py -3.11 -m jarvis_voice verify-dist --directory ".\dist\JARVIS Voice"
```

The manifest detects changes but does not authenticate the publisher. Authenticode signing/timestamping remains an explicit release gate. See the [Milestone 3 implementation prompt](docs/milestone-3-prompt.md) and [native Windows packaging procedure](docs/windows-packaging.md).

## Native macOS HUD

The initial macOS operator UI lives in [`macos/JarvisUI`](macos/JarvisUI). It is
a native SwiftUI menu-bar app with an optional floating cyan HUD inspired by the
project's JARVIS video references. The Voice runtime publishes only a strict,
owner-only state enum (`idle`, `listening`, `processing`, `speaking`, `error`,
or `offline`) with timestamps, a generation and a bounded input-activity scalar
while listening. The reactor, waveform and processing rail react to that live
signal; raw audio is never written to the UI state file. The UI never receives
transcripts, prompts or responses. Build, test and demo instructions are in the
[macOS UI README](macos/JarvisUI/README.md).

## Pocket TTS candidate

The optional English Pocket TTS adapter is CPU-only, streams validated local
PCM and keeps the platform system voice as its fail-safe. Runtime downloads and
model-name fallbacks are forbidden: the model, tokenizer and public `george`
voice state must be absolute and SHA-256-bound. Two offline M3 source smokes
reached their first chunk in 76.8–93.9 ms and 4.42–6.84x realtime, but the
frozen technical beta remains on `system_say` until physical cancellation,
coexistence and soak qualification pass. See
[Pocket TTS English candidate](docs/pocket-tts-candidate.md).

## Native release qualification

M4 adds strict, content-free evidence rather than another runtime feature. It requires a green source doctor, green doctor output produced by the packaged EXE, a valid post-sign distribution manifest, SignTool exit code `0` for every PE artifact, and sufficient passing CPU smoke samples. SignTool warnings and missing native evidence are release-blocking.

The sample schema permits only an enumerated case, `pass`/`fail`, a UUID-v4 replay guard and approved numeric latency metrics. Final evidence contains aggregate counts and p50/p95 only—no sample IDs, transcripts, audio, paths or command output. See the [Milestone 4 implementation prompt](docs/milestone-4-prompt.md) and [native qualification runbook](docs/native-windows-qualification.md).

## Diagnostics

Passive doctor never loads a CUDA model:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
py -3.11 -m jarvis_voice doctor --config C:\path\to\voice.local.json
```

The explicit active probe is permitted only after dependency approval on native Windows NVIDIA hardware:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
py -3.11 -m jarvis_voice doctor --config C:\path\to\voice.local.json --active-cuda-probe
```

It uses only the validated local model, exposes no transcript, performs no network access, and reports sampled—not absolute—VRAM observations. The doctor also reports source/frozen packaging state, tray/native boundary, mutable-path policy and DLL-hook state. See [CUDA coexistence verification](docs/cuda-coexistence.md) and the earlier [Windows CPU smoke procedure](docs/windows-smoke.md).

## Implementation map

- `orchestrator/src/jarvis_orchestrator/`: reducer, SQLite WAL store, transactional epoch CAS, canonical history, budgeted Pydantic/OpenAI dispatch, CommonSchema planner, approvals, outbox, voice gateway and authenticated control API/CLI.
- `orchestrator/uv.lock`: complete, hash-bearing dependency resolution for Python 3.11 on Windows and macOS.
- `src/jarvis_voice/delivery_journal.py`: bounded persisted logical-event deduplication without transcript or speech content.

- `protocol.py`, `state.py`, `core.py`: unchanged wire semantics, epochs, tombstones, speech completion, and turn-safe ASR completion.
- `asr.py`, `asr_factory.py`: persistent CPU backend, explicit profile factory, and one-shot CPU fallback.
- `gpu.py`: redacted NVIDIA inventory and deterministic admission policy.
- `cuda_worker.py`: strict bounded IPC, spawned worker supervision, residency, cancellation, timeouts, and local-only CUDA construction.
- `transcription.py`: non-blocking transcription and stale-result retirement.
- `cuda_probe.py`, `doctor.py`: opt-in active probing and passive redacted diagnostics.
- `runtime.py`: Windows event pump with cancellable asynchronous CUDA transcription.
- `app_paths.py`, `safe_log.py`, `lifecycle.py`: external mutable-data boundary, content-free local diagnostics, and bounded/idempotent runtime control.
- `distribution.py`: deterministic onedir integrity manifest generation and strict verification.
- `windows/tray.py`, `windows/single_instance.py`, `windows/hardening.py`: native tray recovery, per-session guard, and frozen DLL policy.
- `packaging/`, `scripts/build_windows.ps1`: windowed onedir definition and offline native-Windows build recipe.
- `qualification.py`, `scripts/qualify_windows.ps1`: strict smoke schema, percentile aggregation, frozen-doctor/distribution/Authenticode gates and the native evidence runner.

## Explicitly deferred

Native Windows M4 execution and human latency approval, native tray/build/signing approval, native CUDA compatibility, real LLM coexistence results, installer/MSIX, autostart, updater, service, bundled CUDA, newer Whisper model adoption, Piper/Kokoro, wake word, satellites, AEC, ambient interruption, telemetry server, prospective/automatic memory, research, cloud and side-effect tool integrations remain outside the verified result. Explicit Anamnesis retrospective recall is qualified but remains off by default; all model work still uses local inference.
