# macOS technical-beta operator runbook

This runbook operates one bounded, foreground-only CYRUS session on one
Apple-silicon Mac. Voice, Orchestrator and local inference all run on that Mac
over loopback. It does not install a LaunchAgent, enable autostart, detach a
process or authorize a Mac/PC split.

The current gate proves a **technical integration beta**, not a daily-owner or
production release. A passing preflight deliberately still reports
`daily_owner_beta_ready:false`,
`daily_owner_blocker:"bilingual_conversational_quality_not_qualified"` and
`production_ready:false`.

## 1. Operator prerequisites

Use a bounded test window, preferably while the Mac is connected to power. The
operator must have:

- an Apple-silicon Mac with a working selected microphone and speaker;
- Microphone and Input Monitoring permission granted to the terminal/runtime
  that will own the session;
- one already-materialized, owner-only `jarvis-private-stack/2` config whose
  Voice and Orchestrator configs share the same external authentication key;
- the exact local inference runner, manifest, qualification report, ASR/VAD
  artifacts and system voices referenced by those private configs;
- an absolute Python 3.11 runtime containing the current installed
  `jarvis_orchestrator`, `jarvis_voice` and `anamnesis` distributions. The
  supervisor removes `PYTHONPATH` before starting children, so a source-path
  workaround is not sufficient;
- ports `127.0.0.1:8765`, `:8766` and `:8767` free. Do not start the old manual
  services or a LaunchAgent alongside this flow.

Set only path placeholders in the shell; never put a bearer token or Voice key
in an environment variable:

```zsh
JARVIS_REPO="/absolute/path/to/JARVIS-MARK-0"
JARVIS_PRIVATE="/absolute/path/to/private-jarvis-state"
JARVIS_STACK_CONFIG="$JARVIS_PRIVATE/jarvis.private.json"
JARVIS_SESSION_PARENT="$JARVIS_PRIVATE/sessions"
JARVIS_EVIDENCE="$JARVIS_PRIVATE/evidence"
JARVIS_RUNTIME_PYTHON="$JARVIS_REPO/.venv/bin/python"

umask 077
mkdir -p "$JARVIS_SESSION_PARENT" "$JARVIS_EVIDENCE"
chmod 700 "$JARVIS_SESSION_PARENT" "$JARVIS_EVIDENCE"
```

Both directories must be real, owner-controlled directories, not symlinks.
Confirm that the runtime works without `PYTHONPATH`:

```zsh
"$JARVIS_RUNTIME_PYTHON" --version
"$JARVIS_RUNTIME_PYTHON" -c \
  'import jarvis_orchestrator, jarvis_voice, anamnesis; print("runtime imports: ready")'
```

The version must be Python 3.11.x. If the import check fails, repair the
installed runtime before proceeding; do not add source directories to
`PYTHONPATH` for `beta-run`.

## 2. Validate and preflight while stopped

No CYRUS service should be running yet. Validate the private binding:

```zsh
"$JARVIS_RUNTIME_PYTHON" -m jarvis_orchestrator.stack_cli \
  --config "$JARVIS_STACK_CONFIG" validate
```

Success is exit code `0` with `ready:true` and
`schema_version:"jarvis-private-stack/2"`. Then run the passive Mac gate:

```zsh
"$JARVIS_RUNTIME_PYTHON" -m jarvis_orchestrator.stack_cli \
  --config "$JARVIS_STACK_CONFIG" beta-preflight \
  --output "$JARVIS_EVIDENCE/beta-preflight.json"
```

Proceed only on exit code `0` with
`technical_beta_preflight_ready:true`. Every value under `checks` must be
`true`: deployment identity, qualified inference appliance, owner-only state,
safe feature scope, selected Voice profile and Voice environment. The report's
`content_captured:false` and `network_used:false` mean this step neither records
a turn nor contacts a remote model.

Run this command from the same ordinary Terminal identity that will own
`beta-run`. A Codex Seatbelt/sandbox process deliberately reports
`environment.restricted_native_io_probe` and an inconclusive Voice environment,
because that process cannot authoritatively inspect CoreAudio or Input
Monitoring. Do not unset a sandbox marker or copy a green report from another
process. If a real Terminal reports missing permission, grant that Terminal
Microphone and Input Monitoring access in System Settings, fully quit and
reopen it, and rerun the live preflight.

The safe first-beta scope is intentional:

- cloud routing and cloud credentials/qualification residue are absent;
- research is disabled;
- tools and Action Host execution are disabled;
- scheduler generation and native delivery are disabled;
- prospective/shadow memory is disabled;
- explicit retrospective memory is either `READY` or deliberately `DISABLED`,
  exactly as listed under `capabilities`.

Do not broaden that scope merely to make preflight pass.

## 3. Start the foreground stack

In Terminal A, run the supervisor in the foreground:

```zsh
"$JARVIS_RUNTIME_PYTHON" -m jarvis_orchestrator.stack_cli \
  --config "$JARVIS_STACK_CONFIG" beta-run \
  --runtime-python "$JARVIS_RUNTIME_PYTHON" \
  --session-parent "$JARVIS_SESSION_PARENT" \
  --startup-timeout-seconds 600
```

Leave Terminal A open. Do not add `&`, `nohup`, LaunchAgent or autostart. The
command re-runs preflight, requires the three fixed ports to be free, then
starts inference, Orchestrator and Voice in that order. It creates a private
mode-`0700` session directory with mode-`0600` component logs. It prints its
final session report only after shutdown.

In Terminal B, wait for model startup and run:

```zsh
"$JARVIS_RUNTIME_PYTHON" -m jarvis_orchestrator.stack_cli \
  --config "$JARVIS_STACK_CONFIG" status
```

Retry this read-only command until it exits `0` and reports
`version:"jarvis-stack-status/5"` with `ready:true`. Version 5 means:

- `inference_ready`: the live appliance advertises text and streaming;
- `inference_identity_ready`: its exact identity is bound either by the live
  identity endpoint or a valid supervisor-owned foreground lease;
- `inference_identity_source`: `runtime_endpoint` or `foreground_lease`, never
  `none` for a ready stack;
- `orchestrator_ready`: `/readyz` passed;
- `voice_auth_ready`: the Voice gateway passed an authenticated probe;
- `services_ready`: inference, identity, Orchestrator core and authenticated
  gateway are ready. This field alone does **not** include Voice-client
  liveness;
- `voice_client_connected`: the real authenticated Voice client is connected;
- `ready`: both `services_ready` and `voice_client_connected` are true.

`foreground_session_ready` is specifically the legacy-appliance lease bridge;
it may be false when identity comes from `runtime_endpoint`. An invalid stack
does not produce `config_valid:false`; it fails closed with an error envelope.

## 4. Prove exactly one physical turn

Keep Terminal A running. In Terminal B, arm the content-free gate **before**
speaking:

```zsh
"$JARVIS_RUNTIME_PYTHON" -m jarvis_orchestrator.stack_cli \
  --config "$JARVIS_STACK_CONFIG" physical-turn \
  --timeout-seconds 180 \
  --poll-interval-seconds 0.1 \
  --output "$JARVIS_EVIDENCE/physical-turn-001.json"
```

After pressing Return, wait one second for the database baseline to be armed,
then perform exactly one request:

1. Hold the configured PTT chord (`Control`+`Option` for
   `CONTROL_OPTION`, or right `Option` for `RIGHT_OPTION`).
2. After the key-down, say one short request, for example: “Πες μου σε μία
   πρόταση τι είναι η τεχνητή νοημοσύνη.”
3. Release PTT only after finishing the sentence.
4. Do not start another turn. Wait until the spoken response fully ends and the
   gate returns.

The machine gate passes only with `version:"jarvis-macos-physical-turn/2"` and
`ready:true`. It requires one new admission, one local final-model dispatch,
ordered text/speech events, completed playback, retired delivery, continuous
Voice connection, stable deployment identity and no duplicate, stale, cloud,
cancel or failure path.

Voice continuity is bound to a content-free, process-lifetime gateway tracker
and connection generation captured at gate baseline and compared through the
final settle. A detach/reconnect or Orchestrator service restart therefore
fails the gate even if it happens completely between polling samples. The raw
tracker and generation values are not included in the gate report.

The operator must separately confirm that capture started/stopped on the PTT
chord, the request was understood, speech was audible and intelligible, and no
duplicate or late speech occurred. The report proves authenticated transport
and a persisted playback-completed receipt; it explicitly does not prove
remote native-binary attestation or human-perceived acoustic quality.

Run `status` once more after the gate. It must still report `ready:true`.

## 5. Stop cleanly and prove no orphan remains

Return to Terminal A and press `Ctrl-C` once. Wait for the supervisor to stop
Voice, Orchestrator and inference in reverse order. Normal completion is:

```json
{"reason_code":"signal","shutdown_complete":true,"startup_completed":true,"status":"stopped","version":"jarvis-macos-beta-session/1"}
```

Do not close Terminal A before that report appears. Then verify that all fixed
listeners are gone:

```zsh
for JARVIS_PORT in 8765 8766 8767; do
  if lsof -nP -iTCP:"$JARVIS_PORT" -sTCP:LISTEN; then
    echo "BLOCKED: listener remains on $JARVIS_PORT"
  else
    echo "free: $JARVIS_PORT"
  fi
done

pgrep -fl 'run_qualified_inference|jarvis_orchestrator|jarvis_voice'
```

All three ports must print `free`. The `pgrep` command should print nothing
belonging to this beta session. `shutdown_complete:true` is emitted only after
the supervisor has attempted bounded TERM/KILL and confirmed that every owned
component process group has no live member. The supervisor keeps an exited
group leader unreaped until that check finishes, so a native helper cannot be
missed and a recycled unrelated PGID is never signalled. `/usr/bin/say`
intentionally inherits the Voice group for crash containment; ordinary
barge-in and close still target only the exact `say` child. No name-based
`pkill` is used. Before the first component spawn, the supervisor durably
creates an owner-only `.jarvis-macos-beta-unclean.json` interlock in the exact
session parent. It stores only stack/deployment/session digests and PID start
tokens, never paths, secrets, prompts, transcripts or responses. The interlock
is removed only after all owned groups and all three loopback listeners are
conclusively drained. Any retained interlock blocks another launch.

## 6. Evidence handling and failure recovery

Preflight, status and physical-turn reports are deliberately content-free:
they contain fixed codes, booleans, counts, durations, timestamps and digests,
not audio, transcripts, prompts, responses, tokens or bearer secrets. Keep
them under the private evidence directory anyway. The supervisor intentionally
does not print the session-log path. Treat all retained component logs as
sensitive local diagnostics; do not commit, upload or paste them verbatim.

After any startup/runtime/gate failure:

1. Let `beta-run` finish its reverse shutdown; do not immediately relaunch.
2. Run the no-orphan checks above. If a port remains occupied, identify its
   owner with `lsof`; do not kill an unrelated process blindly.
3. Inspect only the newest private `.jarvis-macos-beta-*` session directory
   locally. Preserve it until the failure is understood and share only the
   sanitized fixed error code.
4. Re-run `validate` and `beta-preflight`. Fix the failed prerequisite rather
   than bypassing it, then start a new foreground session.

If the previous supervisor was terminated before it could remove its
interlock, run the explicit verification-only recovery after the checks above:

```bash
cyrus-stack --config "$JARVIS_STACK_CONFIG" beta-recover \
  --session-parent "$JARVIS_SESSION_PARENT"
```

Recovery never signals or kills a PID. It clears the interlock only when the
recorded supervisor identity is gone, every recorded process group is empty,
and ports 8765--8767 can all be bound locally. A PID with a different start
token is treated as a reused numeric identity, not as the old process; an
occupied recycled group still fails closed. `session_recovery_unsafe` means
manual investigation is still required.

Common fixed codes are self-scoping: `required_port_occupied` means an old or
unrelated listener must be stopped; `*_startup_failed` or
`*_readiness_timeout` points to that component's local log;
`*_runtime_failed` means a child died after readiness;
`physical_turn_timeout` usually means no single complete PTT/playback turn was
observed; `multiple_new_voice_turns` requires a fresh gate with exactly one
request. A generic `stack_operation_failed` still requires local diagnostic
inspection and is not permission to weaken the private-stack contract.

## Qualification boundary

A green run establishes one real, local Mac PTT → ASR → authenticated
Orchestrator → qualified local inference → TTS/playback integration turn and a
clean process lifecycle. It does not establish bilingual conversational
quality, production ASR/TTS quality, latency/SLOs, long soak or restart
reliability, battery/thermal behavior, packaging/notarization, autostart,
updating, telemetry, side-effect safety, cloud/research/tool readiness or
protection from malware running as the same OS user. Those remain separate
qualification gates; do not label this result daily-owner-ready or
production-ready.
