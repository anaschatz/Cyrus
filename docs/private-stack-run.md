# Private stack config and real local E2E

`jarvis-private-stack/2` is the single operator entry point for a real local
CYRUS run. It does not copy secrets or model settings into the repository. It
binds the already-private voice and Orchestrator configs to one exact qualified
inference manifest, qualification report and sealed runner. Both component
configs must reference the same external `jarvis.voice-auth-key/1` key file;
the stack validator rejects missing, mismatched, symlinked or non-private keys.

For the exact no-autostart Mac flow—preflight, foreground supervision,
`jarvis-stack-status/5`, one physical PTT turn, graceful shutdown and orphan
checks—use the
[macOS technical-beta operator runbook](macos-technical-beta-operator.md).

## Provision the Voice authentication key

Create one key per installation, on the same machine that will run Voice and
the Orchestrator. The parent directory must already exist, must be private and
must not be a symlink. The output path must not already exist:

```zsh
PYTHONPATH=src python3.11 -m jarvis_voice create-auth-key \
  --output /absolute/private/path/voice-auth.json
```

The command creates a `jarvis.voice-auth-key/1` document exclusively with POSIX
mode `0600`, or a restricted owner DACL on Windows, and prints no secret. Set
the exact same absolute path in both private component configs:

```json
"voice_auth_key_file": "/absolute/private/path/voice-auth.json"
```

Do not place the key in the stack manifest, copy it into the repository or
copy it to another host. The Voice URL, Orchestrator Voice listener and
inference URL remain loopback-only. The supported deployment is all-local on
one Mac or all-local on one PC; splitting Voice and Orchestrator between a Mac
and PC is outside this contract and fails closed.

## Materialize and validate the stack manifest

Create the Orchestrator component config through its executable materializer;
do not copy the example's placeholder tokens into a runtime file:

```zsh
PYTHONPATH=orchestrator/src:src orchestrator/.venv/bin/python \
  scripts/materialize_orchestrator_config.py \
  --output /absolute/private/path/orchestrator.local.json \
  --database /absolute/private/path/state/orchestrator.sqlite3 \
  --voice-auth-key /absolute/private/path/voice-auth.json
```

The output must not exist. The helper writes it privately, generates canonical
tokens from 48 operating-system CSPRNG bytes and verifies a strict production
loader round trip without printing values or paths. The token validator checks
canonical format and a decoded capacity of 32–384 bytes; it cannot prove
randomness from the value itself.

After both component configs reference the same key, create the private stack
file outside version control. The command writes it atomically with mode `0600`
and validates every referenced file, endpoint, model alias and inference
qualification digest:

```zsh
PYTHONPATH=orchestrator/src:src:inference/src:research/src \
orchestrator/.venv/bin/python -m jarvis_orchestrator.stack_cli init \
  --output /absolute/private/path/jarvis.private.json \
  --voice-config /absolute/private/path/voice.local.json \
  --orchestrator-config /absolute/private/path/orchestrator.local.json \
  --inference-runner /absolute/private/path/run_qualified_inference.py \
  --inference-manifest /absolute/private/path/manifest.json \
  --inference-qualification-report /absolute/private/path/qualification-report.json
```

The content-free status gate checks the Orchestrator `/readyz`, qualified
inference capability and exact identity, the authenticated Voice gateway and
the real Voice-client connection. In `jarvis-stack-status/5`, `ready:true`
requires `services_ready:true` and `voice_client_connected:true`; it still does
not prove physical microphone capture, audible speaker playback or acoustic
quality:

```zsh
PYTHONPATH=orchestrator/src:src:inference/src:research/src \
orchestrator/.venv/bin/python -m jarvis_orchestrator.stack_cli \
  --config /absolute/private/path/jarvis.private.json status
```

The E2E gate admits a real transcript through `jarvis.voice/1`, waits for the
guarded qualified local-model request, verifies ordered text and speech commits,
sends playback completion and checks SQLite invariants. It does not claim
physical microphone capture or speaker playback:

```zsh
PYTHONPATH=orchestrator/src:src:inference/src:research/src \
orchestrator/.venv/bin/python -m jarvis_orchestrator.stack_cli \
  --config /absolute/private/path/jarvis.private.json e2e \
  --turns 1 --output /absolute/private/path/e2e.json
```

All command responses and evidence are content-free. Private paths, bearer
tokens, transcripts and model output are never printed.

## Local model failure gate

Stop only the qualified local inference service, leave the Orchestrator
running, and execute:

```zsh
PYTHONPATH=orchestrator/src:src:inference/src:research/src \
orchestrator/.venv/bin/python -m jarvis_orchestrator.stack_cli \
  --config /absolute/private/path/jarvis.private.json model-failure \
  --timeout-seconds 5 \
  --output /absolute/private/path/model-failure.json
```

The report passes only for the exact order `server.ack`,
`server.error(code=backend_failure)`, `response.cancelled(reason_code=backend_failure)`.
It also requires bounded completion, zero text/speech output and zero duplicate
publication. Restart inference afterwards and require both `status` and the
normal `e2e` gate to pass. This local-only fault test must not be run with cloud
routing enabled.

## Migrate a legacy v1 stack

Stop the legacy Voice and Orchestrator before switching configurations. First
attempt the one-way migration without authorizing token replacement:

```zsh
PYTHONPATH=orchestrator/src:src:inference/src:research/src \
orchestrator/.venv/bin/python -m jarvis_orchestrator.stack_cli \
  prepare-auth-v2 \
  --legacy-config /absolute/private/path/jarvis-v1.private.json \
  --output-directory /absolute/private/path/prepared-v2
```

If it returns `legacy_bearer_token_rotation_required`, the legacy inline bearer
tokens do not meet the canonical 32–384-byte production contract. Treat that as
a hard migration stop. To authorize replacement, rerun against a fresh,
nonexistent output directory with:

```zsh
PYTHONPATH=orchestrator/src:src:inference/src:research/src \
orchestrator/.venv/bin/python -m jarvis_orchestrator.stack_cli \
  prepare-auth-v2 \
  --legacy-config /absolute/private/path/jarvis-v1.private.json \
  --output-directory /absolute/private/path/prepared-v2 \
  --rotate-weak-bearer-tokens
```

The flag rotates the configured inline Orchestrator bearer tokens only in the
fresh v2 bundle. It never edits the v1 manifest or component configs. Start all
consumers from the generated v2 paths, validate the new stack, then require
authenticated `status`, `e2e` and one physical PTT turn before retiring the
legacy material. Never copy old token strings into the generated bundle.

## Rotate the key

There is no dual-key grace period or live rotation in v1. Rotate only through a
bounded outage:

1. Stop Voice, then stop the Orchestrator.
2. Create a new key at a new absolute path. The creator intentionally refuses
   to overwrite the old file.
3. While both processes are stopped, change `voice_auth_key_file` in both
   private component configs to that exact new path.
4. Run `jarvis_orchestrator.stack_cli ... validate` against the existing stack
   manifest.
5. Start the Orchestrator, then start Voice.
6. Require both `status` and the normal `e2e` gate to pass, then complete one
   real native Voice PTT turn before retiring the old key through the
   operator's approved secret-destruction procedure. `status` and `e2e` do not
   themselves prove that the Voice process, microphone or speaker is working.

Any transient path, version or secret mismatch is intentionally unavailable;
there is no downgrade to unauthenticated `jarvis.voice/1`.

## Evidence compatibility and security limits

The authentication change is a qualification boundary. Voice 0.8.0 emits
release qualification schema `2`, and source/frozen doctor evidence must name
`jarvis.voice-auth/1` with check code `transport.loopback_authenticated`.
Pre-auth Voice 0.7 artifacts, packages and doctor/release reports cannot
qualify the current runtime. Rebuild and re-sign/re-notarize packaged Voice
binaries, and regenerate native runtime evidence whose profile fingerprint was
created before the authentication protocol became part of that profile.

Operational live, model-failure, restart and soak reports now use their `/2`
schemas and bind `voice_auth_protocol`; the aggregate is
`operational-ready-gate/2`. Older `/1` transport, restart, soak or aggregate
evidence is rejected. A secret-only rotation with unchanged binaries and
runtime profile does not retroactively invalidate component acoustic evidence,
but fresh `status` and `e2e` results are required to prove that both components
were updated consistently.

`jarvis.voice-auth/1` is a bounded shared-secret HMAC handshake that
authenticates the local peers and binds the exact first hello. Loopback traffic
is still plain WebSocket: this is not TLS or payload encryption, it does not
authorize later application actions, it does not sandbox either process and it
does not protect a key from malware already running as the same user.
