# JARVIS Operational Ready Gate v1

This gate qualifies the existing adapters as one long-running local system. It
does not introduce a new lifecycle framework.

## Source-run command

From the repository root, define the source path for the current shell:

```bash
export PYTHONPATH="orchestrator/src:src:inference/src:anamnesis/src:research/src"
```

Use the Python 3.11 environment already locked by the Orchestrator:

```bash
orchestrator/.venv/bin/python -m jarvis_orchestrator --config /absolute/path/config.json serve
```

`SIGINT`, `SIGTERM`, or the background-manager stop request now performs the
same controlled shutdown: stop admission, cancel active work, tombstone stale
output, close the WebSocket gateway, stop the control server, close cloud
clients, checkpoint/close SQLite, and release the single-process lock.

## Background process

The manager installs a per-user `launchd` LaunchAgent on macOS or a least-
privilege Task Scheduler task on Windows. The generated runner stores only
absolute source/config paths; it never copies configuration values or secrets.
Crash restart is bounded by the operating-system supervisor. A controlled exit
is not restarted until `start` is requested.

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json background install
python -m jarvis_orchestrator --config /absolute/path/config.json background status
python -m jarvis_orchestrator --config /absolute/path/config.json background stop
python -m jarvis_orchestrator --config /absolute/path/config.json background start
python -m jarvis_orchestrator --config /absolute/path/config.json background uninstall
```

Installation is an explicit OS mutation. It is not performed by tests or
ordinary `serve` startup.

## SQLite integrity, backup and restore

Diagnostics are safe while the service is running:

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json operations diagnostics
```

The report contains only schema/resource counts and fixed invariant codes. It
contains no transcript, response, prompt, memory, UUID, approval arguments, or
provider payload.

An online backup uses the SQLite backup API, verifies integrity and lifecycle
invariants on the temporary copy, and atomically renames it into place:

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations backup /absolute/path/backups/jarvis.sqlite3
```

Corruption/restore procedure:

1. Stop the background service and verify `running:false`.
2. Preserve the damaged database and its WAL/SHM as incident evidence.
3. Run diagnostics against the selected backup.
4. Restore only with the exact configured database path as confirmation.
5. Run diagnostics again before startup.
6. Start the service and run the restart and short live gates.

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json background stop
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations restore /absolute/path/backups/jarvis.sqlite3 \
  --confirm-database /absolute/path/state/orchestrator.sqlite3
python -m jarvis_orchestrator --config /absolute/path/config.json operations diagnostics
```

Restore fails closed when the service lock is held. Replacement is atomic and
stale WAL/SHM files are removed only after the lock proves quiescence.

## Qualification commands

Fast deterministic lifecycle gate:

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations synthetic-gate --turns 1000 \
  --output /absolute/path/evidence/synthetic.json
```

Real sequential local-model gate through the voice WebSocket and finalized
speech commits:

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations live-gate --turns 100 --maximum-database-bytes 536870912 \
  --output /absolute/path/evidence/live.json
```

Safe corruption/backup/restore drill against disposable copies of the live
database (the configured database is never replaced by this command):

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations recovery-gate --output /absolute/path/evidence/recovery.json
```

Installed-daemon restart gate:

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations restart-gate --output /absolute/path/evidence/restart.json
```

It performs one controlled restart immediately after transcript admission and
one after durable `response.speech.done` but before `playback.completed`. The
first response must become `service_interrupted`; the completed response must
replay with identical logical event IDs, timestamps and payloads.

Full soak gate:

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations soak-gate --duration-seconds 86400 --batch-turns 10 \
  --maximum-database-bytes 536870912 \
  --output /absolute/path/evidence/soak.json
```

The gate reconnects between bounded batches and fails immediately on an
invariant violation, duplicate logical publication, failed turn, or database
size overflow. A 24-hour pass may be claimed only when the report contains
`duration_seconds >= 86400` and `ready:true`.

## Bounded resources

- WebSocket inbound queue: 32 frames.
- WebSocket frame size: 64 KiB base transport; cloud extension remains 16 KiB.
- Publication write/lock deadline: 1 second.
- Operational JSONL: 1 MiB per file, three rotated backups.
- Voice event journal: 4096 live base events and 64 live cloud-approval records.
- Database: explicit gate bound, default 512 MiB during qualification.
- No SDK retry or provider fallback is introduced by operational supervision.

The live transport gate proves transcript-to-finalized-speech-commit behavior.
Actual speaker playback remains a voice-runtime observation and must be present
in the 100-turn and 24-hour evidence collected on the target machine.

## Physical voice evidence and final decision

The real macOS collector runs the actual microphone, PTT, ASR, network client
and selected TTS backend. Its threshold is deliberately stricter than this
gate: at least 200 real turns and an eight-hour native runtime sample. Run it
while the real background Orchestrator and local inference appliance are active:

```bash
python scripts/run_macos_native_collection.py preflight \
  --config /absolute/path/voice-config.json --audio-path built_in \
  --silence-threshold-dbfs -45 --threshold-calibrated

python scripts/run_macos_native_collection.py run \
  --config /absolute/path/voice-config.json --audio-path built_in \
  --silence-threshold-dbfs -45 --threshold-calibrated \
  --duration-seconds 28800 \
  --output /absolute/path/evidence/native-voice.json
```

The Orchestrator is declared operationally ready only by the aggregate command.
It validates all evidence independently and emits no prompt, transcript,
response, UUID or device identifier:

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations evaluate-gate \
  --synthetic /absolute/path/evidence/synthetic.json \
  --live /absolute/path/evidence/live.json \
  --restart /absolute/path/evidence/restart.json \
  --recovery /absolute/path/evidence/recovery.json \
  --soak /absolute/path/evidence/soak.json \
  --native-voice /absolute/path/evidence/native-voice.json \
  --output /absolute/path/evidence/operational-ready.json
```

Only `operational-ready.json` with `"ready":true` is the full ready gate. The
individual transport reports intentionally state
`"physical_speaker_playback_observed":false`; physical playback is accepted
only from the native voice evidence.
