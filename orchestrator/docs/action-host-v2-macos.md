# Action Host v2 — macOS beta

`jarvis.action-host/2` is the required production protocol for every v1 tool.
There is no downgrade to the receipt-only v1 protocol. If the host, exact
registry, output schema, Keychain key, token, helper digest, or code identity is
unavailable, tool capability is blocked while the basic assistant remains live.

## Process boundary

The Orchestrator connects to the standalone `jarvis-action-host` process over a
`0600` Unix socket in a `0700` directory. Both processes load the versioned
`action-host-auth` key from the Data Protection Keychain and a per-launch
32-byte `0600` token. The host checks the peer UID, performs the v2 challenge,
uses independent C2H/H2C AES-256-GCM keys, and rejects replay or sequence gaps.

The host writes `HOST_RECEIVED` and `OS_CALL_ENTERING` under FULL-sync SQLite
transactions. It never repeats a native call after dispatch. Terminal structured
results remain encrypted in the host journal until the Orchestrator has validated
the exact registry output schema, encrypted the result with its separate
ToolStore key, committed the live task epoch, and sent `action.result.ack`.

## Native helper

The Swift package under `native/macos_action_host` provides closed adapters for:

- bounded filename/path metadata search and identity-bound file references;
- NSWorkspace app/file open with fixed app and UTI allowlists;
- EventKit Calendar read/create;
- EventKit Reminder list/create/complete; and
- Apple Mail draft creation through a code-owned Apple Events program.

`note.capture` is stored separately in the Action Host's AES-GCM encrypted
`JARVIS_PRIVATE_NOTES` database and verified by exact read-back. It does not use
Apple Notes.

The helper accepts one bounded JSON object on stdin. User content is never placed
in argv, shell commands, environment values, Voice frames, receipts, or audit.
The Python boundary invokes one fixed qualified executable with `shell=False`,
suppresses native stderr, validates the structured result, and validates it again
against the frozen registry before delivery.

`scripts/build_macos.sh` builds the helper from source with Swift 6, signs it
separately with the Hardened Runtime, embeds it in the signed JARVIS application,
and verifies the embedded copy before notarization. The inspection step rejects
unsigned and ad-hoc signatures, requires a Developer ID Application authority
and Team ID, and records both the binary SHA-256 and SHA-256 of:

```text
/usr/bin/codesign --display --requirements - <helper>
```

in the host configuration. Ad-hoc or unsigned helpers fail closed.

## Permissions

Setup may request permissions; preview must never do so. Runtime adapters only
inspect the current authorization state:

- Calendar read and verified create require EventKit full access.
- Write-only Calendar create returns `ACCEPTED/UNAVAILABLE` with
  `calendar.write_only_no_readback`.
- Reminders require full EventKit access.
- Mail drafts require Apple Events permission for the signed helper.
- Revocation is checked on every action and is never treated as a remembered
  one-time readiness result.

The v1 same-user threat limit remains explicit: Unix permissions, peer UID,
Keychain secrets, nonces, and replay protection do not protect a fully
compromised process already running as the same user. A future XPC service with
audit-token and signing-requirement enforcement is the stronger boundary.
