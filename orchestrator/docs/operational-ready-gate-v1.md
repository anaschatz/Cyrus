# JARVIS Operational Ready Gate v4 (runtime-bound evidence revision)

This gate qualifies the existing adapters as one long-running local system. It
does not introduce a new lifecycle framework.

## Source-run command

From the repository root, define the source path for the current shell:

```bash
export PYTHONPATH="orchestrator/src:src:inference/src:research/src"
```

Use the Python 3.11 environment already locked by the Orchestrator:

```bash
PYTHONPATH=src python3.11 -m jarvis_voice create-auth-key \
  --output /absolute/private/path/voice-auth.json
```

The private parent directory must already exist and the output must not. Set
the exact same absolute path in both the Voice and Orchestrator configs:

```json
"voice_auth_key_file": "/absolute/private/path/voice-auth.json"
```

The key is an external `jarvis.voice-auth-key/1` document; it is created
exclusively with mode `0600` on POSIX or a restricted owner DACL on Windows and
is never printed. Voice, Orchestrator and inference must run on the same host
with loopback endpoints. This gate does not support a Mac-to-PC split and the
key must not be copied between hosts.

Start the Orchestrator only after both configs are materialized:

```bash
orchestrator/.venv/bin/python -m jarvis_orchestrator --config /absolute/path/config.json serve
```

`SIGINT`, `SIGTERM`, or the background-manager stop request now performs the
same controlled shutdown: stop admission, cancel active work, tombstone stale
output, close the WebSocket gateway, stop the control server, close cloud
clients, checkpoint/close SQLite, and release the single-process lock.

## Readiness signing authority and pinned identities

Operational gate output is eligible only when it is an Ed25519-signed
`jarvis.readiness-evidence/2` envelope. Generate the operator key offline in an
owner-private directory; the private key must remain mode `0600` on POSIX (or
have the equivalent owner-only DACL on Windows):

```bash
openssl genpkey -algorithm Ed25519 -out /absolute/private/path/readiness-signing.pem
chmod 600 /absolute/private/path/readiness-signing.pem
openssl pkey -in /absolute/private/path/readiness-signing.pem -pubout \
  -out /absolute/path/readiness-public.pem
python -c 'import hashlib; from cryptography.hazmat.primitives import serialization; p=serialization.load_pem_public_key(open("/absolute/path/readiness-public.pem","rb").read()); print(hashlib.sha256(p.public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)).hexdigest())'
```

Pin the printed public-key digest, the key ID, and two real artifact files in
the Orchestrator config. The configured build/model SHA-256 values are checked
against those files every time evidence is signed or evaluated:

```bash
shasum -a 256 /absolute/path/qualified-build.manifest
shasum -a 256 /absolute/path/qualified-model.manifest
```

```json
"readiness_signing_key_file": "/absolute/private/path/readiness-signing.pem",
"readiness_trusted_public_key_file": "/absolute/path/readiness-public.pem",
"readiness_signer_key_id": "operator-2026-01",
"readiness_trusted_public_key_sha256": "64-lowercase-hex-characters",
"readiness_build_artifact_file": "/absolute/path/qualified-build.manifest",
"readiness_build_artifact_sha256": "64-lowercase-hex-characters",
"readiness_model_artifact_file": "/absolute/path/qualified-model.manifest",
"readiness_model_artifact_sha256": "64-lowercase-hex-characters",
"readiness_inference_manifest_sha256": "64-lowercase-hex-characters",
"readiness_inference_runtime_identity_sha256": "64-lowercase-hex-characters"
```

The model artifact file SHA is checked independently. The inference manifest
pin must equal the active appliance's authenticated `/jarvis/v1/identity`
`manifest_digest` (the manifest's canonical semantic digest, which need not be
the raw file-byte digest). The runtime-identity pin must equal that endpoint's
`runtime_identity_sha256`, and the authenticated capabilities response must
name the configured `jarvis-local` alias. Locally selected files without these
live exact matches are not identity evidence. Every collection command checks
the live identity before and after its gate; aggregate evaluation checks it
again at the final signing boundary.

No key embedded in an evidence file is a trust root. Evaluation uses only the
config-pinned public key. The config-profile hash includes every configuration
field, but bearer values are domain-separated and hashed before that projection
is hashed; raw bearer values are never written to evidence.

The signing key is operator authority, not a daemon secret. The long-running
`serve` path does not load it: keep it offline or otherwise unavailable except
while an operator is collecting/signing gates, and remove access again
afterward. The configured path may be absent outside those commands. Compromise
of the OS user while this key is readable can forge readiness attestations, so
mode `0600` alone is necessary but not a substitute for offline custody.

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
invariants on the temporary copy, and atomically renames it into place. For a
schema-v10 `ENCRYPTED` saved-memory database, the library additionally needs a
narrow live provider attestor. Provider calls happen only after SQLite readers
are closed. The current CLI does not wire that attestor yet, so it deliberately
refuses encrypted-memory backup with `memory_backup_provider_required` rather
than creating an unverifiable artifact. Pre-v10 and empty `LEGACY_PENDING`
databases retain the existing command behavior:

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations backup /absolute/path/backups/jarvis.sqlite3
```

The encrypted-memory library path publishes
`jarvis.sqlite3.memory-backup-v1.json` after the database rename. This
owner-private canonical sidecar binds the exact database SHA-256 and byte
length, schema/storage state, provider installation ID, active/high-water
versions, every referenced epoch ID and key version, and the memory revision.
It contains no saved content, search tokens, confirmation HMACs, or key bytes.
Its fixed labels are:

- `same_install_only:true`
- `portable:false`
- `transcript_encrypted:false`
- `physical_erasure_claimed:false`

`MIGRATING`, any non-`COMPLETE` crypto operation, and `LEGACY_PENDING` with
legacy saved-memory rows are not backup-eligible. In particular, a v10 backup
cannot silently preserve plaintext saved memory while claiming encrypted
memory protection.

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

Restore fails closed when the service lock is held. Replacement is atomic;
destination WAL/SHM/journal files must first be preserved as incident evidence
and removed from the restore path, otherwise restore refuses to publish. An
`ENCRYPTED` restore also requires the exact canonical sidecar, an exact database
hash match, the same provider installation, the recorded active epoch, and
every referenced provider key. It never bootstraps a replacement provider.
SQLite WAL/SHM/journal files beside a committed backup are rejected as an
ambiguous artifact.

Saved-memory envelope encryption does not encrypt canonical transcripts,
prompts, responses, or historical SQLite pages outside the encrypted memory
tables. Old backups remain independent retained copies and are outside any
later key-destruction or physical-erasure claim; they must be inventoried and
removed under the operator's retention policy.

## Qualification commands

Create one UUIDv4 for the qualification cohort and record one UTC-second start
time before the first gate. Reuse both values unchanged for all six leaves:

```bash
python -c 'import uuid; print(uuid.uuid4())'
```

The examples use `COHORT-UUID-V4` and `YYYY-MM-DDTHH:MM:SSZ` as placeholders.
The closed contract assigns fixed sequences 1 through 6 to the leaf kinds and
sequence 7 to the aggregate. All captures must be chronological within the
same 72-hour window; mixed cohorts, duplicate/replayed sequences, changed
window bounds, or aggregation after the window fail closed.

Every signing command requires `--output`. Before an envelope is published,
the writer atomically claims its fixed sequence in the owner-private cohort
receipt under `<database-parent>/operations/readiness-cohorts/`. A receipt
binds the cohort/window, pinned identity digest, signer key ID, evidence hash
and canonical destination hash. The same evidence at the same destination is
an idempotent crash-recovery retry; a changed hash, a second destination, or
reuse of one destination for another sequence fails closed. The receipt is
written and fsynced before the evidence artifact, while the artifact itself is
atomically replaced, so interruption can be recovered only by the exact retry.
Keep this state until the cohort is complete; deleting or splitting it does
not create eligible aggregate evidence.

Fast deterministic lifecycle gate:

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations synthetic-gate --turns 1000 \
  --qualification-cohort-id COHORT-UUID-V4 \
  --qualification-window-started-at-utc YYYY-MM-DDTHH:MM:SSZ \
  --output /absolute/path/evidence/synthetic.json
```

Real sequential local-model gate through the voice WebSocket and finalized
speech commits:

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations live-gate --turns 100 --maximum-database-bytes 536870912 \
  --qualification-cohort-id COHORT-UUID-V4 \
  --qualification-window-started-at-utc YYYY-MM-DDTHH:MM:SSZ \
  --output /absolute/path/evidence/live.json
```

The live client completes `jarvis.voice-auth/1` before sending the first
`jarvis.voice/1` hello. Missing, malformed or mismatched keys fail the gate;
there is no unauthenticated fallback.

Installed-daemon restart gate:

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations restart-gate \
  --qualification-cohort-id COHORT-UUID-V4 \
  --qualification-window-started-at-utc YYYY-MM-DDTHH:MM:SSZ \
  --output /absolute/path/evidence/restart.json
```

It performs one controlled restart immediately after transcript admission and
one after durable `response.speech.done` but before `playback.completed`. The
first response must become `service_interrupted`; the completed response must
replay with identical logical event IDs, timestamps and payloads.

Safe corruption/backup/restore drill against disposable copies of the live
database (the configured database is never replaced by this command):

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations recovery-gate \
  --qualification-cohort-id COHORT-UUID-V4 \
  --qualification-window-started-at-utc YYYY-MM-DDTHH:MM:SSZ \
  --output /absolute/path/evidence/recovery.json
```

Full soak gate:

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations soak-gate --duration-seconds 86400 --batch-turns 10 \
  --maximum-database-bytes 536870912 \
  --qualification-cohort-id COHORT-UUID-V4 \
  --qualification-window-started-at-utc YYYY-MM-DDTHH:MM:SSZ \
  --output /absolute/path/evidence/soak.json
```

The gate reconnects between bounded batches and fails immediately on an
invariant violation, duplicate logical publication, failed turn, or database
size overflow. A 24-hour pass may be claimed only when the report contains
`duration_seconds >= 86400` and `ready:true`.

## Bounded resources

- WebSocket inbound queue: 32 frames.
- Authentication frames: 4 KiB maximum and a bounded two-second handshake.
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

First project the raw native collector result into a closed, content-free,
signed native leaf. The expected runtime-profile SHA must come from the
separately qualified Voice build/manifest, not from the evidence file itself:

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations sign-native-gate /absolute/path/evidence/native-voice.json \
  --expected-runtime-profile-sha256 64-lowercase-hex-characters \
  --qualification-cohort-id COHORT-UUID-V4 \
  --qualification-window-started-at-utc YYYY-MM-DDTHH:MM:SSZ \
  --output /absolute/path/evidence/native-voice.signed.json
```

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations evaluate-gate \
  --synthetic /absolute/path/evidence/synthetic.json \
  --live /absolute/path/evidence/live.json \
  --restart /absolute/path/evidence/restart.json \
  --recovery /absolute/path/evidence/recovery.json \
  --soak /absolute/path/evidence/soak.json \
  --native-voice /absolute/path/evidence/native-voice.signed.json \
  --output /absolute/path/evidence/operational-ready.json
```

Only a signature-verified `operational-ready.json` with `"ready":true` is the
full ready gate. It binds exactly one hash to each of the six required leaf
kinds; a missing, duplicated, swapped, stale, cross-build or cross-config leaf
fails closed. The individual transport reports intentionally state
`"physical_speaker_playback_observed":false`; physical playback is accepted
only from the native voice evidence.

Sequence 7 uses the deterministic UTC second immediately after the latest
leaf capture, never the wall-clock time chosen by a caller. After the final
live runtime/database state check returns, the evaluator reads the clock,
requires every leaf to be no later than that instant, re-runs freshness at the
exact sequence-7 instant, and requires that instant to remain strictly before
the cohort expiry. Identical six parents and identical current state therefore
produce byte-identical aggregate JSON, hash and Ed25519 signature; delayed
collection across the window boundary or changed current state is rejected.

Every signed envelope contains a closed, content-free current SQLite
diagnostics snapshot and its domain-separated digest. Signing requires
`ready:true`, `integrity_ok:true`, zero integrity issues and zero invariant
violations. Aggregate evaluation recomputes diagnostics immediately before
the aggregate signature, so a database that becomes unhealthy after leaf
collection cannot produce a ready aggregate. Native projection also retains
the expected runtime-profile SHA and runtime mode and requires exact agreement
with the signed evidence host OS/mode; macOS evidence cannot qualify Windows.

## Evidence compatibility, rotation and limits

The closed leaf payloads retain `live-sequential-ready-gate/2`,
`live-restart-ready-gate/2` and `live-soak-ready-gate/2`, each with
`voice_auth_protocol: "jarvis.voice-auth/1"`; the aggregate is
`operational-ready-gate/4` inside a signed `jarvis.readiness-evidence/2`
envelope. The model-failure gate likewise uses
`live-model-failure-ready-gate/2`. Pre-auth `/1` live, restart, soak and
aggregate reports are rejected and must be regenerated.

All `jarvis.readiness-evidence/1` files and `operational-ready-gate/3`
aggregates are now legacy-ineligible because they do not bind current database
health, authenticated inference identity, native runtime profile, or a closed
qualification cohort.

All historical caller-authored or unsigned readiness JSON is legacy and
explicitly ineligible; parsing it can never produce `ready:true`. Do not wrap
an old report and call it fresh evidence. After rotating the readiness key,
build manifest, model manifest, config profile or database schema, update the
pins while stopped and rerun all six leaf gates. Leaf envelopes are valid for
at most seven days, and the aggregate expires after 24 hours by default.

Voice 0.8.0 also changes the release boundary: source and frozen doctor evidence
must contain the authenticated transport marker, and release qualification uses
schema `2`. Pre-auth Voice 0.7 packages, doctor reports and release evidence do
not qualify the current runtime. Because the auth protocol is part of the macOS
runtime profile fingerprint, pre-auth native evidence cannot qualify the 0.8
runtime or a current scheduler-TTS promotion. Rebuild and re-sign/re-notarize
packaged binaries, then recollect the affected whole-runtime evidence.

There is no dual-key/live rotation. Stop Voice and Orchestrator, create a new
key at a new absolute path, update both configs while stopped, restart
Orchestrator followed by Voice, and require fresh private-stack `status` and
`e2e` results plus one native Voice PTT turn before retiring the old key. A
secret-only rotation with unchanged
binaries/profile does not erase historical component acoustic results, but it
does require fresh authenticated connectivity evidence.

`jarvis.voice-auth/1` is a shared-secret HMAC handshake that authenticates both
local peers and binds the exact first hello. It does not encrypt the loopback
WebSocket, provide TLS, authorize later application actions, sandbox either
process or protect against malware already running as the same OS user.
