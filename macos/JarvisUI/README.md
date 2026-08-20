# JARVIS macOS UI v0

Native SwiftUI operator surface for the local JARVIS stack. Its visual language
uses a restrained cyan HUD, an arc-reactor-inspired state core, thin technical
grids and monospaced operational labels. It intentionally does not imitate the
video thumbnails literally or use copyrighted Iron Man artwork.

The UI is observational. It reads only the owner-only, content-free snapshot:

`~/Library/Application Support/JARVIS Voice/state/ui-state.json`

The Voice runtime publishes one of `starting`, `idle`, `listening`,
`processing`, `speaking`, `error` or `offline`, plus a generation, state timing
and a bounded `0...1000` input-activity scalar while listening. The scalar drives
the reactor and waveform at up to 10 Hz. Raw audio is never persisted or sent to
the app. No transcript, response, prompt, locale, identifier, model path or error
message is accepted by the strict `jarvis-voice-ui-state/2` schema.

## Build and run

```zsh
cd macos/JarvisUI
swift build
swift test
swift run JarvisUI
```

For a visual-only review without starting the Voice stack:

```zsh
swift run JarvisUI --demo
```

`--demo` is visibly labelled and never becomes the default. A custom snapshot
can be selected explicitly with `--state-file /absolute/path/ui-state.json`.

This version provides the main HUD, reactive live-input visualization, explicit
hear/reason/voice progression, state timing, a menu-bar refresh command and an
optional keep-above-windows toggle. Starting/stopping the privileged stack,
settings, transcripts and agent actions are deliberately not owned by the UI.
