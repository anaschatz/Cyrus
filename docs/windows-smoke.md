# Native Windows 11 smoke procedure

Milestone 1 can be approved only on native Windows 11 x64. Do not install the optional dependencies until the proposed versions in the README have been explicitly approved. Do not download a model, Silero artifact, or SAPI language pack as part of this procedure.

## Preconditions

1. Use CPython 3.11+ x64 and a local copy of `config.example.json` outside version control.
2. Point `whisper_model_directory` at the already-present pinned four-file model pack.
3. Point the VAD artifact fields at an already-present Silero ONNX file with independently verified size and SHA-256.
4. Configure exact installed SAPI voice descriptions for `el-GR` and `en-US`.
5. Keep `orchestrator_url` on loopback and keep the PTT mode passive.

## Automated checks

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
py -3.11 -m unittest discover -s tests -v
py -3.11 -m compileall src tests
py -3.11 -m jarvis_voice doctor --config C:\path\to\voice.local.json
```

The doctor must report Windows 11 x64, optional dependency versions, exact Whisper and Silero manifests, offline CPU model loading, WASAPI input/default rate, SAPI initialization and both configured voices, Raw Input availability, loopback transport, `cuda.not_required`, `network_used: false`, and `gpu_used: false`.

## Manual sample matrix

Run the fake server and runtime in separate terminals. Record pass/fail and sample count for:

- Greek utterance.
- English utterance.
- Press/release with no speech (`input.no_speech`).
- Hold beyond `maximum_hold_seconds` (forced release).
- PTT_DOWN while SAPI is speaking the first unit.
- PTT_DOWN while SAPI is speaking the last unit.
- Keyboard removal or a controlled synthetic equivalent while held.
- Suspend/resume while held.
- Fake-server disconnect/reconnect; verify same transcript `event_id`, new connection `seq`, and no repeated turn processing.

Confirm that speech begins from an early `response.speech.commit`, before `response.text.done`, and that a cancelled/tombstoned response never emits `playback.completed`.

## Timing boundaries

Use the runtime/SAPI monotonic timestamps and report distributions, not guarantees:

- WindowProc QPC receipt of PTT_DOWN → SAPI purge dispatch.
- PTT_UP receipt → creation of `input.transcript.final`.
- `response.speech.commit` receipt → SAPI `Speak` dispatch.
- SAPI backend stream completion → creation of `playback.completed`.

The WindowProc timestamp is software receipt time, not physical switch activation. Purge API return and backend stream completion do not prove acoustic silence. PTT_DOWN → actual silence remains deferred until measured with WASAPI loopback capture or external audio equipment.
