# S2 native scheduler delivery

This increment connects the durable `jarvis.scheduler/1` outbox to a native
desktop notification adapter without giving the Voice process time authority.
The Orchestrator remains authoritative for due instants, recurrence, retry,
expiry, tombstones, and the execution ledger.

## Enablement gate

The Voice client advertises the scheduler protocols only when its configuration
enables the extension and supplies exact SHA-256 bindings for both:

- a `voice-scheduler-compatibility/1` manifest; and
- a platform-specific `scheduler-native-qualification/1` report.

Both files are parsed with closed field sets. Their protocol, CPython 3.11.14,
Unicode DB 14.0.0, AEAD, journal, platform, adapter, lock provider, artifact,
permission, and native test bindings must match the running artifacts. A missing
key, mismatched digest, unsupported platform, missing permission, or failed
qualification prevents scheduler capability advertisement. There is no downgrade
to `jarvis.voice/1` response events.

No repository fixture grants production qualification. A release must generate
the two signed/hashed artifacts on its native target after exercising post,
replace, remove, silent delivery, restart recovery, privacy, and idempotency.

## Durable dispatch order

For `scheduler.notification.requested`, the client:

1. encrypts the complete payload in its FULL-sync SQLite journal;
2. sends the durable scheduler event ACK;
3. writes a `DISPATCHING` attempt fence;
4. sends the `DISPATCHING` receipt;
5. calls the native API; and
6. persists and sends `OS_ACCEPTED`, `OS_REJECTED`, or
   `AMBIGUOUS_OS_DISPATCH`.

On restart, any attempt left at `DISPATCHING` becomes
`AMBIGUOUS_OS_DISPATCH`. The same logical delivery can then receive a new
attempt, so a crash after an OS call may produce a duplicate native dispatch.
The guarantee is durable retry and at-least-once native dispatch attempts while
a qualified capability is available. Physical display is zero-or-more and is
not observable as a reliable receipt.

Tombstones are persisted before native removal is attempted. An accepted
delivery or tombstoned notification is not redispatched from the encrypted
inbox.

## Privacy and content handling

Scheduler frames, previews, notification bodies, and speech payloads are stored
only through `scheduler-content-aead/1`, using a Voice-specific key. The macOS
source-run uses a dedicated versioned current-user login-Keychain service and
Windows uses current-user DPAPI. Notification
content is sent to native helper processes over stdin, never through command-line
arguments or a plaintext temporary file.

At dispatch, `unknown` lock state always becomes `private`; `locked` becomes at
most `title_only` unless the manifest-qualified global full-body opt-in and the
job policy both allow full content.

Scheduler TTS remains off unless the compatibility manifest includes
`scheduler_notification_speech` and its `tts_adapter_qualification_sha256`
matches an additional strict native Voice qualification file configured as
`tts_qualification_path`/`tts_qualification_sha256`. That report must pass the
200-turn, acoustic barge-in, recovery and soak gates, and its runtime-profile
fingerprint must equal the currently running macOS profile. Scheduler speech
uses its isolated encrypted journal and receipts; it cannot fall back to normal
response playback.

## Deliberate remaining gate

The callback core accepts only exact `jarvis.scheduler-native-callback/1`
envelopes authenticated with HMAC-SHA256 under a distinct 256-bit helper key.
It rejects unknown fields, invalid UUID/SHA/timestamps, invalid action/due-time
combinations, inactive or tombstoned notification IDs, and conflicting replay.
The accepted payload is encrypted in the Voice scheduler journal before it can
produce `scheduler.notification.action`; job, revision, occurrence and snooze
generation are always recovered from the authoritative notification frame and
never trusted from native input.

The Orchestrator applies the action and records its inbox idempotency row in the
same SQLite transaction. A durable `scheduler.notification.action_result`
outbox frame closes the Voice callback. If a connection is lost before that
result, Voice retransmits the same action `event_id` on reconnect and the
Orchestrator returns `DUPLICATE` without repeating the mutation.

The signed macOS callback boundary is implemented as the separate
`jarvis-scheduler-native` process. Voice starts the
exact digest- and code-identity-pinned executable directly with `shell=False`;
notification content travels only over inherited pipes and never through argv
or environment values. The helper owns `UNUserNotificationCenter`, registers
fixed Snooze-10-minutes and Cancel actions, and signs ASCII-only callback cores
with an ephemeral 256-bit launch key transferred through the private stdin
pipe. Voice still verifies the HMAC and persists the callback before emitting
an action. The helper never receives job IDs, revisions, occurrence IDs or
Scheduler authority.

Until the Developer-ID-signed artifact is physically qualified, `/v1/capabilities`
continues to report `notification_actions=BLOCKED` with
`native_action_callback_not_qualified`, and `reminders_overall` remains
blocked. Exact authenticated scheduler APIs remain the authority for snooze and
cancel.

Physical callback qualification is intentionally interactive and must run
against the embedded, signed helper artifact:

```bash
PYTHONPATH=src:orchestrator/src python3.11 \
  scripts/run_macos_scheduler_callback_qualification.py \
  --helper "/absolute/JARVIS Voice.app/Contents/Frameworks/helpers/jarvis-scheduler-native" \
  --artifact-manifest /absolute/jarvis-scheduler-helper-artifact.json \
  --output /absolute/scheduler-callback-qualification.json
```

The runner does not emit a manifest unless signature identity, permission,
post/replace/remove, silent delivery, real Snooze and Cancel callbacks, callback
authentication and reconnect replay all pass. Voice advertises
`scheduler_notification_actions` plus the resulting qualification digest; the
Orchestrator requires the independently pinned expected digest before tracking
that connection as action-capable.
