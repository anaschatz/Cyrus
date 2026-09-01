# Native Windows 11 packaging, tray and hardening gate

Milestone 3 is source-complete and fake-verified on the development host. A release artifact is valid only after this procedure runs on native Windows 11 x64. PyInstaller packages the active OS, Python and architecture; the macOS build host is not a Windows cross-compiler.

## Approved shape

- CPython 3.11 x64 and `PyInstaller==6.21.0` in a pre-approved offline build environment.
- The final Voice app is windowed `onedir`, `asInvoker`, with no UPX. The three
  earlier qualification runners are deliberately separate console `onefile`
  artifacts and are never rebuilt by the Voice-package stage.
- Native `Shell_NotifyIconW` tray with `NIM_SETVERSION` and `TaskbarCreated` recovery.
- Config, model, Silero artifact, logs and state remain outside the application directory. The default root is `%LOCALAPPDATA%\JARVIS Voice`.
- The runtime hook narrows the DLL search to Windows' default safe directories plus the frozen `_internal` directory. It does not mutate machine or process `PATH`.
- The distribution manifest is a deterministic hash inventory. It detects accidental or malicious file changes but is not an authenticity root.

CUDA is an explicit private qualification input, not a machine-`PATH`
assumption. Approve the exact CUDA/cuDNN versions, licenses and rollback first;
then create the canonical bundle manifest described in
[the native runner contract](../inference/docs/windows-native-fixture-runner.md).
No proprietary DLL or host-specific hash is committed to this repository.

## Offline development build

Do not let the script install anything. Qualification runners are built and
sealed before model qualification. The later Voice package verifies their
existing hashes and cannot rebuild them:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1 `
  -Stage QualificationRunners `
  -CudaRuntimeManifest C:\CYRUS\private\cuda-runtime-manifest.json `
  -RunnerOutputDirectory C:\CYRUS\staging\qualification-runners

powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1 `
  -Stage VoicePackage `
  -RunnerBuildManifest C:\CYRUS\staging\qualification-runners\qualification-runners-manifest.json
```

Every real PyInstaller process runs with Python isolated mode. Source/spec,
runtime-hook, interpreter, CUDA and output inputs must be regular paths on a
fixed local drive with no reparse component. The script also refuses any
tracked or untracked source-tree change before it executes a build. The ASR
collector embeds only the exact manifest-bound CUDA/cuDNN DLLs and boot-smokes
without ambient `PATH`.
The Voice spec deliberately includes no config, model, ONNX VAD file, SAPI
voice, certificate, installer, updater or telemetry endpoint. Inspect
`dist\JARVIS Voice`, the PyInstaller warning file and
`distribution-manifest.json` before proceeding.

Before distribution, archive the exact dependency lock, licenses and required notices/SBOM. PyInstaller's documented bootloader exception permits shipping generated bundles under another license, but it does not override the licenses of Python, faster-whisper/CTranslate2, PortAudio, SoXR, ONNX Runtime, WebSockets, pywin32 or any other collected dependency.

## Native tray smoke matrix

1. Start the EXE with no arguments. It must use `%LOCALAPPDATA%\JARVIS Voice\config\voice.json` and create only per-user application directories/logs.
2. Start a second copy in the same Windows session. It must exit with `application.already_running`; the first instance remains healthy.
3. Verify the icon menu: Start Voice, Stop Voice, both folder commands and Exit. Repeated Start/Stop/Exit must be idempotent.
4. Kill and restart Explorer. `TaskbarCreated` must restore exactly one icon and its version/status tooltip.
5. Exercise logoff/shutdown and runtime failure. No transcript, audio bytes, exception text, model path, username, device identifier or signing data may appear in `voice-events.jsonl`.
6. Run the full Milestone 1 CPU smoke matrix, including PTT barge-in while SAPI is speaking. The tray must not alter `jarvis.voice/1` events.
7. Verify `dist-manifest` then modify, add and remove one copied artifact in separate disposable copies; every mutation must fail `verify-dist` with a typed code.
8. Confirm ProcMon shows no network model/tokenizer fallback and no DLL load from the current working directory or arbitrary `PATH` entries.

## Authenticode publication gate

Signing is deliberately not automated by the repository and requires explicit certificate approval. In the controlled release job, sign the final PE files with SHA-256 and an approved RFC 3161 timestamp service, regenerate the distribution manifest after every binary mutation, then verify at minimum:

```powershell
signtool verify /pa /v /tw ".\dist\JARVIS Voice\JARVIS Voice.exe"
.\.venv\Scripts\python.exe -I -m jarvis_voice verify-dist --directory ".\dist\JARVIS Voice"
```

Record SignTool's exit code, certificate subject/thumbprint, timestamp, Windows version, Python/PyInstaller lock and final manifest hash in the release evidence. Do not put a PFX, password or certificate material in this repository or command history.

After signing and post-sign manifest regeneration, execute the separate [native Windows release qualification](native-windows-qualification.md). Publication requires its machine-readable report to be `ready: true` plus explicit human approval of the recorded latency distributions.

## Deferred

MSIX/installer, autostart, Windows service, auto-update, crash upload, telemetry,
enterprise deployment policy, CUDA DLLs in the final Voice `onedir`,
Piper/Kokoro, wake word, satellites, AEC and hands-free interruption remain
separate milestones. The sealed CUDA bundle here belongs only to the ASR
qualification collector.
