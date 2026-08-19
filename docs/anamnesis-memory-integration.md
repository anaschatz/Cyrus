# Anamnesis memory integration qualification

## Decision

JARVIS qualifies the real Anamnesis `OpenMemoryRecallIndex` boundary for
local, non-authoritative retrospective recall. It does **not** promote the
experimental `InMemoryAnamnesis` temporal/reminder reducer to production
durable storage.

The accepted runtime contract SHA-256 is:

`832de106ffcfa543d8544bb7b29084a141666d6a3f6e2d40e90ff03ade2b1352`

Startup fails closed if the imported Anamnesis runtime contract changes.

## Inspected Anamnesis surface

| Concern | Qualified finding |
|---|---|
| Location | `anamnesis/` inside the JARVIS workspace |
| Startup | No production daemon or HTTP API exists. Orchestrator installs the lightweight Anamnesis runtime distribution; evaluation/report dependencies remain optional. |
| Python interface | `anamnesis.runtime_contract.anamnesis_runtime_contract()` and `anamnesis.openmemory_recall.OpenMemoryRecallIndex` |
| Input | Bounded text, namespace, user partition, scalar metadata and search limit |
| Output | Closed Pydantic `RecallSearchResult`/`RecallMatch` objects; recalled text is explicitly non-authoritative and cannot carry action evidence |
| Persistence | The Anamnesis temporal store is explicitly pure in-memory. JARVIS therefore persists accepted recall records in Orchestrator SQLite. |
| Deletion | JARVIS accepts only an explicit `Forget that…` / `Ξέχασε ότι…` command whose normalized target matches one active local-owner record. No match fails closed. Logical deletion, its tombstone, content-free audit and usage/revision change commit together. Conversation deletion writes the tombstone/audit before purging records sourced by that conversation. Chat-history deletion and physical/forensic erasure are separate operations. |
| Authentication | No network service is opened, so there is no bearer-auth surface. The adapter requires positive local-only attestation and performs no network request. |
| Timeout/cancellation | Indexed recall runs through `asyncio.to_thread` on a dedicated read-only SQLite connection with a 100 ms busy timeout, 150 ms internal deadline and progress-handler cancellation. It never interrupts the shared authority connection. Explicit commands use one local SQLite transaction and make no provider/model call. |
| Idempotency | Source-event replay returns the committed outcome. Normalized duplicate remember commands return `already_exists`; they do not create another record. |

The current Orchestrator SQLite schema is version 10. The constructor performs
only SQL migration: v1/v8 data first passes the fail-closed v9 validation, then
v9 plaintext authority is renamed to inaccessible legacy tables and v10
encrypted tables are installed in one `BEGIN IMMEDIATE` transaction. The
constructor never bootstraps or loads a platform key.

`prepare_memory_storage()` is a separate, explicit, resumable cutover. It
freezes one operation cutoff and an exact temporary source digest, qualifies
one provider epoch before taking the database lock, encrypts validated rows,
and rechecks the complete source plus every decrypted target inside the final
cutover transaction. Expired legacy rows receive one tombstone and
`TTL_EXPIRED` audit rather than ciphertext. At successful cutover it drops the
legacy tables and replaces the temporary source digest with a content-free
final-operation hash. No plaintext fact, legacy fact SHA, normalized SHA or
term SHA remains in live v10 tables.

## Runtime behavior

- Memory is off by default.
- A turn is eligible for storage only when it explicitly begins with
  a phrase such as `Να θυμάσαι ότι…`, `Θυμήσου…`, `Remember that…` or
  `Please remember…`.
- Explicit remember/forget is intercepted before recall, planning or model
  dispatch. One `BEGIN IMMEDIATE` transaction binds the canonical input,
  applies the mutation, writes a content-free command audit, commits a
  deterministic localized response and fills the normal replayable outbox.
- Encrypted memory is unavailable until explicit preparation has installed a
  provider-free `MemoryEpochSessionV1`. Provider attestation/key loading occurs
  before any store lock or SQLite transaction. Remember, dedup, forget, recall
  and logical CRUD recheck the exact database install/epoch binding and use
  only the cached session; they never call the platform provider while a
  database transaction is held.
- A crash before commit leaves neither mutation nor acknowledgement. A crash
  after commit replays the same outbox and outcome without a second mutation.
- Record deletion uses deterministic exact normalized-content equality within
  the conversation's stable local-owner principal; no model chooses the target. It removes only the
  saved-memory record: canonical chat history remains unchanged and the
  deterministic reply states that limitation explicitly.
- Authenticated loopback management exposes bounded principal-wide
  `GET /v1/memory/records`, `GET /v1/memory/records/{record_id}` and logical
  `DELETE /v1/memory/records/{record_id}`. List cursors are opaque, integrity
  protected, revision-bound and limited to 50 records; `next_cursor` is passed
  back through the `after` query parameter. List/get exclude expired records
  and return only `record_id`, decrypted content, creation and expiry;
  key versions, envelope hashes, blind indexes and internal metadata never
  cross the control boundary. The existing exact loopback Host, bearer-token,
  forwarded-header and runtime-admission checks apply, and every success or
  content-free error response carries `Cache-Control: no-store`.
- Management delete requires one UUIDv4 `Idempotency-Key`, one
  `If-Match-Memory-Revision`, no `If-Match-Epoch` and an empty body. Receipt
  lookup, revision CAS, tombstone, content-free audit, row deletion, revision
  increment and exact response receipt share one `BEGIN IMMEDIATE`
  transaction. The response states both `chat_history_changed: false` and
  `physical_erasure_claimed: false`. A non-null delete request ID is globally
  bound to its record and original pre-delete revision, so conflicting reuse
  fails closed even after the 24-hour HTTP receipt expires. List, get and new
  deletes return retryable `memory_storage_unavailable` (`503`) unless the
  encrypted local-owner store and cached epoch session are ready.
- Closed outcomes additionally include `limit_exceeded`, `quota_exceeded` and
  `invalid_content`. Disabled, oversized, full or termless memory therefore
  reports the truth instead of allowing a general model to claim a mutation.
- Empty and ambiguous forget commands are audited but never delete a record.
- Partial, cancelled, interrupted, budget-exhausted and failed turns cannot
  write normal memory.
- Each record is at most 4096 UTF-8 bytes, expires exactly 365 days after its
  committed insertion, and is deduplicated by `(principal_id, key_version,
  normalized_hmac)`. It may create at most 128 unique keyed-HMAC term rows. A
  principal is capped at 2000 active records and 8 MiB. Startup and the hourly
  activity seam sweep at most 256 expired rows per pass.
- Recall accepts at most 4096 query bytes and 32 unique normalized terms. It
  joins blind HMAC terms through `memory_terms_by_principal_term`, excludes records
  at `expires_at <= now`, and orders by score descending, creation time
  descending, then record ID. There is no recall-all fallback or table scan.
- Recall runs during `ROUTING`. A non-empty qualified result selects
  `memory_local`; empty, timed-out or unavailable recall falls back to `local`.
- Recalled strings enter the model request only as labelled untrusted JSON.
  They never enter canonical history and cannot supply action evidence.
- Recall audit rows contain hashes, counts and stable error codes—not raw memory
  or provider exception text.
- Conversations on the local installation share one DB-stable owner principal,
  so a later conversation can recall an earlier saved fact. Deleting the source
  conversation removes that fact and keeps the shared principal/other bindings.

## Current startup boundary

The policy fields remain:

```json
{
  "memory_enabled": true,
  "memory_timeout_seconds": 0.25,
  "memory_recall_limit": 5
}
```

With `memory_enabled=true`, production startup selects the native macOS
current-user login Keychain or Windows DPAPI provider below the dedicated owner-private
`operations/memory-keys` root. It constructs the envelope cipher/runtime,
opens the store with encrypted-memory authority enabled, and completes
`prepare_memory_storage()` before any Anamnesis adapter, listener or model
component starts. A fresh database therefore follows the journaled bootstrap
and cutover path. An already `ENCRYPTED` database must reopen its exact bound
provider installation and write epoch; a missing or mismatched provider can
never authorize a replacement bootstrap.

Provider construction, binding and migration failures stop startup with the
stable content-free `memory_storage_unavailable` code. With
`memory_enabled=false`, startup passes no crypto runtime, disables memory
storage authority and performs no provider/native call. The service readiness
path reads only the store's cached content-free status and is false unless the
configured memory state is `ENCRYPTED` and ready; it never re-attests or loads
keys. Offline signed-readiness collection separately reads schema-10 memory
binding metadata through read-only/query-only SQLite, then attests and loads
the already-bound provider key without constructing a store, bootstrapping,
migrating or sweeping data.

The explicit macOS `beta-preflight` command performs the same non-mutating
native binding check when retrospective memory is enabled. A valid current
state digest promotes only `memory.retrospective` from `BLOCKED` to `READY`;
an unavailable provider, missing epoch, database drift or malformed digest
remains content-free and blocked. Request-time `/readyz`, `/v1/readiness` and
`/v1/beta-session` do not repeat Keychain or DPAPI calls: they use the
provider-qualified session and cached store status established before the
listeners opened.

The macOS source-run uses the dedicated, versioned service
`com.jarvis.memory-records.login-keychain-v1`. Unsigned Python processes
cannot request the Data Protection Keychain (`-34018`, missing entitlement),
so the runtime does not silently claim that stronger storage class. A future
signed and entitled bundle must use a different service namespace and an
explicit migration; it cannot reinterpret source-run keys.

## Deliberate non-goals

The qualified path does not enable prospective reminders, triggers,
OpenMemory/Cavira SDK installation, embeddings, hosted inference, automatic
fact extraction from every conversation, or multi-owner/account identity.
Those require separate policy, privacy and model-quality qualification.
Key rotation, key destruction, secure/physical erasure and portable encrypted
backup remain route-blocked. The management API performs logical saved-memory
deletion only; it does not claim that SQLite pages, WAL files, chat history,
process memory or backups were physically erased.
