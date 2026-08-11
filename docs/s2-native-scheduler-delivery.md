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
only through `scheduler-content-aead/1`, using a Voice-specific key. macOS uses
the Data Protection Keychain and Windows uses current-user DPAPI. Notification
content is sent to native helper processes over stdin, never through command-line
arguments or a plaintext temporary file.

At dispatch, `unknown` lock state always becomes `private`; `locked` becomes at
most `title_only` unless the manifest-qualified global full-body opt-in and the
job policy both allow full content. Scheduler TTS is never advertised by this
runtime and cannot fall back to normal response playback.

## Deliberate remaining gate

Native notification action callbacks are not implemented by this increment.
Accordingly `/v1/capabilities` may report `notification_delivery=READY` for a
qualified connected client, but reports `notification_actions=BLOCKED` with
`native_action_callback_not_qualified`. `reminders_overall` remains blocked
until a signed native action host provides callback authentication and durable
idempotency. Exact authenticated scheduler APIs remain the authority for snooze
and cancel.
