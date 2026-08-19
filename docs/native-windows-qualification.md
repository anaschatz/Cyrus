# Milestone 4 — Native Windows release qualification

This gate does not build, install, download or sign anything. It evaluates one already-built, already-signed `onedir` artifact on native Windows 11 x64 with CPython 3.11 and the approved local model/VAD/config. A release now also requires the strict native-hardware evidence described in `native-voice-qualification.md`; the smaller case table below is scenario coverage, not the 200-turn reliability gate.

## Evidence boundary

The sample and final report schemas permit only enumerated case names, pass/fail, UUID-v4 replay protection and four numeric latency metrics. They have no notes or content fields. Do not put transcripts, audio, prompts, paths, exception messages, hostnames, usernames, device IDs or certificate details in qualification files.

The required sample counts are:

| Case | Minimum passing samples |
|---|---:|
| Greek turn | 10 |
| English turn | 10 |
| No speech | 3 |
| Forced release | 3 |
| Barge-in during first speech unit | 5 |
| Barge-in during last speech unit | 5 |
| Keyboard removal while held | 2 |
| Suspend/resume while held | 2 |
| Reconnect/deduplication | 3 |

Any failed sample blocks release, even when the minimum passing count is met.

## Finalize the artifact, then create and record the sample file

First finish signing/timestamping and regenerate the distribution manifest.
Then create the file once outside the application bundle, bound to that exact
verified manifest:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
py -3.11 -m jarvis_voice qualification-template `
  --distribution C:\JARVIS-Dist\JARVIS-Voice `
  --output C:\JARVIS-Evidence\cpu-samples.json
```

Create the Windows native template with the same distribution binding:

```powershell
py -3.11 .\scripts\run_native_voice_qualification.py template `
  --platform windows `
  --distribution C:\JARVIS-Dist\JARVIS-Voice `
  --output C:\JARVIS-Evidence\native-hardware.json
```

Every package or manifest change invalidates both files; regenerate and rerun
the physical cases instead of editing their identity fields.

Record content-free outcomes. Greek and English turn samples require all three shown latency metrics:

```powershell
py -3.11 -m jarvis_voice qualification-record `
  --samples C:\JARVIS-Evidence\cpu-samples.json `
  --case greek_turn --outcome pass `
  --metric ptt_up_to_transcript_final_ms=412.5 `
  --metric speech_commit_to_sapi_dispatch_ms=18.2 `
  --metric stream_complete_to_playback_completed_ms=7.1
```

Barge-in samples require `ptt_down_to_purge_dispatch_ms`. Recovery/no-speech cases carry no metric. Use `outcome fail` for every observed functional failure; never discard a failed run to make the report pass.

## Frozen and final qualification

After the samples and native collection are complete for that exact package,
run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\qualify_windows.ps1 `
  -Config C:\JARVIS-Config\voice.json `
  -Distribution C:\JARVIS-Dist\JARVIS-Voice `
  -Samples C:\JARVIS-Evidence\cpu-samples.json `
  -NativeEvidence C:\JARVIS-Evidence\native-hardware.json `
  -Output C:\JARVIS-Evidence\release-qualification.json
```

The script first asks the packaged EXE to write its own redacted doctor report. The source harness then checks:

1. native Windows 11 x64 and Python 3.11 x64;
2. source doctor with the exact local configuration;
3. strict M3 distribution manifest and its SHA-256;
4. `signtool verify /pa /all /tw` exit code `0` for every `.exe`, `.dll` and `.pyd`;
5. matching frozen product/protocol/auth version, authenticated loopback transport, `onedir`, DLL hardening, external mutable paths and tray/host readiness;
6. required functional samples, latency distributions and exact distribution-manifest binding;
7. at least 200 real hardware turns, zero duplicate transcript/playback, zero stuck PTT, acoustic barge-in below 200 ms, microphone/sleep recovery and an eight-hour stable-memory soak.

SignTool output is captured and discarded; it is never copied into the report. Exit code `2` is a warning according to Microsoft and is deliberately treated as release-blocking. The report contains no artifact paths or signer identity.

## Approval rule

`ready: true` means every mechanical M4 gate and the strict native-hardware acceptance gate passed for this artifact and evidence set. Software-only purge latency never substitutes for the acoustic result.

`ready: false` is final for that evidence run. Correct the underlying artifact/config/signature/sample problem and create a fresh run; do not edit the report.
