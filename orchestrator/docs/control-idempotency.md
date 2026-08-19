# Control mutation idempotency boundary

The HTTP control API binds every mutation receipt to the exact
`Idempotency-Key`, method, canonical path, canonical request-body SHA-256, and
route-specific integer precondition. Most mutations use `If-Match-Epoch`.
Saved-memory deletion instead requires `If-Match-Memory-Revision` and rejects
`If-Match-Epoch`; the internal legacy `expected_epoch` receipt column holds
that route's memory revision. Receipts expire exactly 24 hours after creation.
Reusing a live key with any different binding returns
`idempotency_conflict` (`409`).

For the following Orchestrator-owned mutations, receipt lookup, state mutation,
and receipt insertion run in one SQLite `BEGIN IMMEDIATE` transaction:

- `POST /v1/conversations`
- `DELETE /v1/conversations/{conversation_id}`
- `POST /v1/tasks/{task_id}/cancel`
- `POST /v1/approvals/{approval_id}/decision` for research approvals
- `DELETE /v1/memory/records/{record_id}`

An expected validation or state error rolls the mutation savepoint back before
the closed error response is stored. An unexpected failure after mutation but
before receipt insertion rolls the entire transaction back. Concurrent callers
using the same key therefore observe one committed effect and the exact same
response. Conversation deletion may perform a best-effort WAL checkpoint after
the logical deletion and receipt have committed; that compaction is not part of
the logical API result.

Saved-memory deletion also stores the UUIDv4 request key and post-delete
revision in the logical tombstone. If its HTTP receipt has expired, an exact
replay can reconstruct the same response only when the supplied pre-delete
revision is still the tombstone revision minus one. A changed record, body,
path or precondition conflicts. The response explicitly says chat history was
not changed and physical erasure is not claimed.

Cloud-model decisions remain forbidden on this generic endpoint. Tool/action
approvals are intentionally not decided here: the endpoint returns the stable
`tool.approval_route_required` (`409`) result and leaves the approval `PENDING`.
They must use the separately bound `jarvis.action-approval/1` transport, whose
decision inbox, binding validation, and result outbox own that lifecycle. No
cross-authority atomicity is claimed.

Scheduler control mutations retain their existing scheduler-owned idempotency
ledger and are not routed through the Orchestrator command-receipt table.
