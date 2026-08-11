# Scheduler notification playback

Scheduler speech remains a negotiated `jarvis.scheduler/1` extension. It never
enters the conversational `ResponseSession`, never produces
`playback.completed`, and cannot be enabled by an unqualified compatibility
manifest.

Accepted `scheduler.notification.speech` frames are encrypted in the Voice
scheduler journal before ACK. A separate playback controller takes one durable
unit only when capture, conversational output and the shared TTS backend are
idle. It reserves the next global output epoch solely as the backend
cancellation fence; scheduler notification state remains in its own journal.

The controller emits `scheduler.playback.receipt` for `PLAYING`, `COMPLETED`,
`CANCELLED` or `SKIPPED`. A restart converts a previously `PLAYING` unit to
`CANCELLED` and reports `restart_interrupted_playback`; it never guesses that
audio completed while the process was absent.

`PTT_DOWN` wins synchronously. The active scheduler row is cancelled, the
shared backend is purged through the new epoch, and the cancellation receipt is
sent before capture proceeds. Late start/completion callbacks are recognized
as scheduler callbacks and discarded, so they cannot mutate conversational
state or resurrect a tombstoned notification.

Speech payloads are NFC, limited to 4096 UTF-8 bytes and carry a positive
`unit_index`. Playback receipt state and reason codes are strict tokens; free
form text is forbidden.

Enablement additionally requires absolute `scheduler.tts_qualification_path`
and matching `scheduler.tts_qualification_sha256` configuration values. The
report digest must be the one bound by the compatibility manifest and the
report's runtime-profile fingerprint must match the active macOS source or
packaged profile.
