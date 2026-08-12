# Action Host v2 — macOS beta

`jarvis.action-host/2` is the required production protocol for every v1 tool.
There is no downgrade to the receipt-only v1 protocol. If the host, exact
registry, output schema, Keychain key, token, helper digest, or code identity is
unavailable, tool capability is blocked while the basic assistant remains live.
Both the Orchestrator and Action Host examples ship with actions disabled. Merely
installing or signing the helper does not enable any tool authority.

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
and Team ID, and records both the binary SHA-256 and SHA-256 of the single
canonical `designated => ...` requirement line returned by:

```text
/usr/bin/codesign --display --requirements - <helper>
```

Path-bearing `Executable=...` diagnostics are excluded, so installing the same
signed bytes at a different absolute application path does not change their
identity. The digest is stored in the host configuration. Ad-hoc or unsigned
helpers fail closed.

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

The native readiness probe is code-owned and is not registered as a tool. It
uses only `EKEventStore.authorizationStatus`,
`AEDeterminePermissionToAutomateTarget(..., askUserIfNeeded: false)`, an
`NSMetadataQuery` availability probe, and fixed `NSWorkspace` lookups. It must
never display a TCC prompt. Permission requests belong to a separate, explicit
setup UI and are not part of preview, planning, startup, or qualification.

After installing the signed helper, the operator may deliberately enter the
only prompt-capable setup path:

```text
PYTHONPATH=orchestrator/src python3.11 -m jarvis_orchestrator.action_host_daemon \
  --config /absolute/private/action-host.json --request-permissions
```

The flag is mutually exclusive with serving the host. It requests EventKit full
Calendar and Reminders access and Mail Automation consent, then prints only the
closed readiness states. No filenames, events, reminders, recipients, or
message content are read or emitted. Running the daemon normally never calls
this path.

## Native qualification and activation

Qualification is content-addressed and expires after at most 30 days. It binds:

- the exact `jarvis.action-host/2` contract digest and frozen registry digest;
- the signed helper binary SHA-256 and designated-requirement SHA-256;
- the exact macOS build;
- the current passive Calendar, Reminders, Mail Automation, Spotlight,
  NSWorkspace and private-notes readiness states; and
- an immutable external test manifest with one `PASS` evidence digest for every
  registered tool.

Start from `action-host-native-tests.template.json`. Native tests must run on the
signed embedded helper with dedicated Calendar/Reminder/Mail test data. They may
request permissions only during the explicit setup/qualification session. The
evidence file stores hashes and stable statuses, never event, reminder, note,
file, recipient, subject, or body content.

Generate the qualification only after all tests pass:

```text
PYTHONPATH=orchestrator/src python3.11 scripts/generate_action_host_qualification.py \
  --config /absolute/private/action-host.json \
  --test-manifest /absolute/private/native-tests.json \
  --test-manifest-sha256 <sha256-of-JCS-normalized-manifest> \
  --expires-at 2026-09-01T00:00:00.000Z \
  --output /absolute/private/action-host-qualification.json
```

The command prints the qualification digest. Activation then requires two
explicit changes using the same artifact and digest:

1. set `enabled: true`, `qualification_file`, and `qualification_sha256` in the
   Action Host configuration;
2. set `tools_enabled: true`, `action_host_qualification_file`, and
   `action_host_qualification_sha256` in the Orchestrator configuration.

On every host start, the helper signature, binary identity, macOS build,
registry, artifact expiry and passive readiness are rechecked. Revocation or
drift fails closed before the Unix socket is served. On the Orchestrator side,
an unavailable/stale/mismatched qualification disables actions only; the basic
assistant continues. Each adapter still checks its live permission immediately
before its native operation.

The v1 same-user threat limit remains explicit: Unix permissions, peer UID,
Keychain secrets, nonces, and replay protection do not protect a fully
compromised process already running as the same user. A future XPC service with
audit-token and signing-requirement enforcement is the stronger boundary.
