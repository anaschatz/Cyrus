# S1 deterministic scheduler v3

Status: implemented contract candidate. Scheduler persistence schema is v5; Tool/Action schema is v6.

## Authority boundary

The model can emit one strict `schedule-proposal/1` candidate. The deterministic compiler resolves it against a persisted request timestamp, packaged `tzdata==2026.3`, CPython 3.11.14, and Unicode DB 14.0.0. Only an approved canonical proposal creates or mutates a job. The scheduler stores, materializes, fires, retries, recovers, snoozes, and cancels without inference.

`timer.*` has no operating-system scheduler path. A future timer adapter can only delegate to `SchedulerPort`. LaunchAgent, Task Scheduler, and cron are not time-authority implementations.

## Global approval channel

Cloud and scheduler approvals share `approval_arbiter_queue`. At most one row is in `PRESENTING`, `PRESENTED`, `DECIDING`, or `WAITING_FOR_OWNER`. New approvals enter durable `WAITING_FOR_CHANNEL`, ordered by `(arbiter_seq, approval_id)`, with a global limit of 64. An incompatible head remains head until a compatible client connects or the approval expires. A scheduler proposal expires after 600 seconds or when less than 10 seconds remain before its first due instant.

F10/F11 are owned by one Voice `GlobalApprovalController`, not by individual extensions. Capture disarms the current presentation; rearming requires capture completion and a newly observed both-keys-up state. Exact authenticated scheduler decisions bind `approval_id`, `proposal_id`, and `proposal_sha256` and do not arm F10/F11.

## Transport

`jarvis.voice/1`, `jarvis.cloud-approval/1`, and `jarvis.scheduler/1` share one WebSocket. Sequence scope is `(protocol, direction, websocket_connection)`, so each protocol can independently start at sequence 1. Scheduler negotiation also requires `jarvis.approval-arbiter/1` and an exact configured Voice compatibility-manifest SHA-256. An unnegotiated scheduler frame closes with WebSocket 1002. Maximum scheduler frame size is 16 KiB.

Scheduler output is never downgraded to `jarvis.voice/1`. Scheduler speech is sent only through `scheduler.notification.speech` after durable notification ACK and only when the separately qualified capability is negotiated. Its self-contained payload is `(occurrence_id, notification_id, unit_index, locale, NFC speakable_text)`. Voice keeps scheduler playback in a separate journal.

## Time and recurrence

Persisted instants have exact integer-millisecond precision and canonical RFC 3339 `.mmmZ` form. Sub-millisecond inputs are rejected, never truncated. UTC wall time owns due instants. Monotonic time owns bounded waits and lease durations. The worker wakes at least every 250 ms, rereads UTC, records detected clock jumps, and never attempts to adjust the system clock.

`count` includes DTSTART. `until_at_utc` is inclusive. Monthly missing dates use `skip`; there is no implicit clamp. Recurring DST gaps skip the invalid nominal slot without consuming count. One-shot gaps require clarification. Ambiguous folds use the earlier instant (`fold=0`). Relative day is local-calendar arithmetic; day/week duration is elapsed 24h/7×24h.

A date without a year searches only the representable years through 9999;
impossible dates fail with clarification instead of looping. A future DST gap
keeps its selected calendar date and follows the same one-shot/recurring gap
rules as an explicitly supplied year. A recurrence end cannot precede DTSTART;
an end exactly at DTSTART is valid.

Accepted due instants must also leave room for the 24-hour delivery expiry and
24-hour content-retention deadline. The last representable due instant is
`9999-12-29T23:59:59.999Z`. Proposal and snooze validation reject later instants;
recurrence materialization stops when its next slot is unrepresentable or exceeds
that bound, while preserving delivery of the current valid occurrence. Deadline
values are never silently clamped.

Occurrence identity is the domain-separated SHA-256 of `(job_id, job_revision, canonical scheduled UTC slot)`. Snooze preserves the occurrence and canonical recurrence slot, increments delivery generation, and produces a new deterministic notification identity. At most eight snoozes are accepted.

## Persistence, delivery, and recovery

Approval creates the job, recurrence cursor, first occurrence, ledger events, and approval-result outbox in one SQLite `BEGIN IMMEDIATE` transaction. It never creates a notification outbox. A due transaction creates exactly one logical delivery generation and `scheduler.notification.requested` outbox row.

The system provides durable retry and at-least-once native dispatch attempts when a qualified and available delivery capability exists. Actual physical display is zero-or-more and is not verified. A crash after the native call but before a receipt becomes `AMBIGUOUS_OS_DISPATCH`; a later attempt may produce a duplicate physical notification. The ledger never converts native API acceptance into a claim that the user saw the notification.

Restart recovery returns leased scheduler outbox rows to pending, restores presented approvals to owner-wait, and converts persisted dispatching attempts to ambiguous/retry-wait. Missed occurrences within 24 hours deterministically coalesce per job; older occurrences expire. Recurrence catch-up is bounded and cursor-driven.

## Encrypted content

All scheduler content at rest uses `scheduler-content-aead/1`: AES-256-GCM (`cryptography==49.0.0`), a random 96-bit nonce per record, a database uniqueness constraint on `(key_version, nonce)`, and JCS AAD binding record type/id, role, job/revision, content hash, and key version. Orchestrator and Voice use distinct namespaces/keys. The macOS source-run uses a dedicated versioned service in the current user's login Keychain because unsigned Python receives `-34018` from the Data Protection Keychain; Windows uses current-user DPAPI without machine scope or UI fallback.

Missing keys and invalid tags fail closed. No preview ACK, dispatch, or speech is permitted. Metadata remains listable/cancellable. Ciphertext is retained while live and until 24 hours after the terminal job, then cryptographically erased from the active database row; hashes and content-free ledger metadata remain.

## Privacy and readiness

At dispatch, `locked` permits at most title-only, `unknown` forces private, and private remains private. Scheduler TTS is forbidden while locked or unknown. A client without a matching qualified manifest advertises neither scheduler nor speech.

Legacy unauthenticated `/readyz` remains unchanged and preserves assistant/inference semantics. Authenticated `/v1/readiness` reports degraded scheduler/control availability independently. `/v1/capabilities` is content-free and returns `READY`, `BLOCKED`, or `DISABLED` per component. S1 intentionally reports native notification delivery blocked until S2 qualification.

## Rollout boundary

S1 owns deterministic time authority, recurrence, approval arbitration, encrypted persistence, recovery, delivery outbox, and exact ledger semantics. S2 adds autostart and qualified native Windows/macOS notification, lock-state, and TTS adapters. Calendar, deterministic condition triggers, and prospective Anamnesis follow. Prospective Anamnesis remains only a proposal producer and never receives clock or execution authority.
