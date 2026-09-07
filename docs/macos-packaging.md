# Stable macOS packaged runtime

The release profile is a native arm64 `.app` built on macOS 14 or later with
CPython 3.11.14 and PyInstaller 6.21.0. The build script never installs or
downloads anything. It requires an already-built native `whisper-server`, a
Developer ID Application identity and an existing `notarytool` keychain
profile.

The app embeds the static helper and an immutable profile containing its exact
size and SHA-256. It deliberately excludes Whisper models, VAD models, user
configuration, logs and scheduler state. Those remain under
`~/Library/Application Support/JARVIS Voice/`; the default config is
`config/voice.json`. At every launch the signed profile and helper are checked
before the configured external model can be loaded.

```zsh
export JARVIS_MACOS_WHISPER_SERVER=/absolute/path/to/whisper-server
export JARVIS_MACOS_CODESIGN_IDENTITY='Developer ID Application: …'
export JARVIS_MACOS_NOTARY_PROFILE='jarvis-notary'
./scripts/build_macos.sh
```

The script fails closed unless the host, Python, PyInstaller and helper
architecture match the locked profile. It runs the Voice and Orchestrator test
suites, builds the onedir app, verifies hardened-runtime signing, submits the app for notarization
and staples and validates the accepted ticket before assessing the app with
Gatekeeper. This follows [Apple's notarization workflow](https://developer.apple.com/videos/play/wwdc2019/703/);
a newly signed app can fail Gatekeeper assessment before it is notarized.
Microphone and Input Monitoring permissions
must then be granted to this exact signed bundle identity. Source-interpreter
TCC grants do not transfer to the app.

A successful build is still not native acceptance evidence. Run the same 200
turn, disconnect, sleep/resume, acoustic and eight-hour qualification using the
signed package profile before setting `runtime_profile.mode` to
`signed_package`.
