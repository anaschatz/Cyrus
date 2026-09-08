"""Schema-v6 Tool/Action persistence and atomic execution transitions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from .domain import OrchestratorError, StaleEpochError, TaskState, new_uuid4, parse_utc, utc_now, utc_text
from .jcs import domain_sha256, jcs_bytes, jcs_text
from .scheduler_store import MAXIMUM_APPROVAL_QUEUE
from .tool_contracts import ToolOutcomeV1, ToolRegistryV1, ToolStatus, arguments_sha256, canonical_uuid4
from .tool_crypto import EncryptedToolPayloadV1, ToolPayloadBindingV1, ToolPayloadCipherV1
from .tool_provenance import ContentProvenanceV1
from .tool_schema import MIGRATION_6_SQL as MIGRATION_6_SQL
from .tool_state import ToolRunEvent, ToolRunState, restart_event, transition_tool_state


ACTION_APPROVAL_PROTOCOL = "jarvis.action-approval/1"


def _sha(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise OrchestratorError("tool.binding_mismatch", status=409)
    return value


class ToolStoreV1:
    def __init__(self, orchestrator_store: Any, *, cipher: ToolPayloadCipherV1, registry: ToolRegistryV1) -> None:
        self._store = orchestrator_store
        self._connection = orchestrator_store._connection
        self._lock = orchestrator_store._lock
        self._transaction = orchestrator_store._transaction
        self.cipher = cipher
        self.registry = registry

    @staticmethod
    def _payload_binding(execution_id: str, tool_id: str, kind: str) -> ToolPayloadBindingV1:
        return ToolPayloadBindingV1(execution_id, tool_id, 6, kind)

    def _insert_payload_tx(
        self,
        connection: sqlite3.Connection,
        *,
        execution_id: str,
        tool_id: str,
        kind: str,
        content: bytes,
        created_at: str,
        purge_after: str | None,
    ) -> str:
        payload_id = new_uuid4()
        envelope = self.cipher.encrypt(content, self._payload_binding(execution_id, tool_id, kind))
        connection.execute(
            """INSERT INTO tool_sensitive_payloads(
               payload_id,execution_id,record_kind,key_version,nonce,ciphertext_and_tag,
               content_sha256,aad_sha256,created_at,purge_after
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (payload_id, execution_id, kind, envelope.key_version, envelope.nonce, envelope.ciphertext_and_tag, envelope.content_sha256, envelope.aad_sha256, created_at, purge_after),
        )
        return payload_id

    def create_tool_run(
        self,
        *,
        task_id: str,
        epoch: int,
        tool_call_id: str,
        tool_id: str,
        arguments: Mapping[str, Any],
        preview: Mapping[str, Any],
        idempotency_key: str,
        policy_sha256: str,
        scheduler_contract_sha256: str,
        user_intent_binding_sha256: str,
        now: datetime | None = None,
        approval_ttl: timedelta = timedelta(minutes=10),
    ) -> dict[str, str]:
        canonical_uuid4(tool_call_id)
        canonical_uuid4(idempotency_key)
        for digest in (policy_sha256, scheduler_contract_sha256, user_intent_binding_sha256):
            _sha(digest)
        definition = self.registry.resolve(tool_id)
        canonical_arguments = self.registry.validate(tool_id, arguments)
        selected = now or utc_now()
        execution_id = new_uuid4()
        approval_id = new_uuid4()
        presentation_id = new_uuid4()
        decision_nonce = new_uuid4()
        preview_nonce = new_uuid4()
        stamp = utc_text(selected)
        args_digest = arguments_sha256(canonical_arguments)
        preview_bytes = jcs_bytes(preview)
        preview_digest = hashlib.sha256(preview_bytes).hexdigest()
        approval_binding = domain_sha256(
            "jarvis.action-approval-binding/1",
            {
                "approval_id": approval_id,
                "arguments_sha256": args_digest,
                "execution_id": execution_id,
                "policy_sha256": policy_sha256,
                "preview_sha256": preview_digest,
                "registry_sha256": self.registry.digest,
                "scheduler_contract_sha256": scheduler_contract_sha256,
                "tool_call_id": tool_call_id,
                "tool_id": tool_id,
                "tool_schema_sha256": definition.schema_sha256,
                "user_intent_binding_sha256": user_intent_binding_sha256,
            },
        )
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM tool_executions WHERE task_id=? AND epoch=? AND tool_call_id=?",
                (task_id, epoch, tool_call_id),
            ).fetchone()
            if existing is not None:
                if any(existing[key] != value for key, value in {
                    "arguments_sha256": args_digest, "idempotency_key": idempotency_key,
                    "tool_name": tool_id, "preview_sha256": preview_digest,
                    "policy_sha256": policy_sha256, "registry_sha256": self.registry.digest,
                    "tool_schema_sha256": definition.schema_sha256,
                    "scheduler_contract_sha256": scheduler_contract_sha256,
                    "user_intent_binding_sha256": user_intent_binding_sha256,
                }.items()):
                    raise OrchestratorError("tool.idempotency_conflict", status=409)
                return {"execution_id": existing["execution_id"], "approval_id": existing["approval_id"], "status": existing["status"]}
            queue_count = connection.execute(
                """SELECT COUNT(*) FROM approval_arbiter_queue WHERE state IN (
                   'WAITING_FOR_CHANNEL','PRESENTING','PRESENTED','DECIDING','WAITING_FOR_OWNER')"""
            ).fetchone()[0]
            if queue_count >= MAXIMUM_APPROVAL_QUEUE:
                raise OrchestratorError("APPROVAL_QUEUE_FULL", status=409)
            task = self._store._guarded_task_tx(connection, task_id, epoch, (TaskState.EXECUTING,))
            expires = min(selected + approval_ttl, parse_utc(task["absolute_expires_at"]))
            # The existing tool_executions.approval_id foreign key requires the
            # approval to exist first.  The approval deliberately stores only an
            # encrypted-payload pointer; it is completed after the payload rows
            # have been written in this same transaction.
            pointer = {"encrypted": True, "execution_id": execution_id, "payload_id": ""}
            connection.execute(
                """INSERT INTO approvals(
                   approval_id,conversation_id,task_id,epoch,kind,name,arguments_json,
                   arguments_sha256,status,created_at,expires_at
                   ) VALUES(?,?,?,?, 'tool',?,?,?,'PENDING',?,?)""",
                (approval_id, task["conversation_id"], task_id, epoch, tool_id, jcs_text(pointer), args_digest, stamp, utc_text(expires)),
            )
            connection.execute(
                """INSERT INTO tool_executions(
                   execution_id,task_id,epoch,tool_call_id,tool_name,arguments_sha256,status,
                   approval_id,tool_version,verification,reason_code,idempotency_key,
                   registry_sha256,tool_schema_sha256,policy_sha256,scheduler_contract_sha256,
                   user_intent_binding_sha256,preview_sha256,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,'PLANNED',?,1,'','',?,?,?,?,?,?,?,?,?)""",
                (execution_id, task_id, epoch, tool_call_id, tool_id, args_digest, approval_id, idempotency_key, self.registry.digest, definition.schema_sha256, policy_sha256, scheduler_contract_sha256, user_intent_binding_sha256, preview_digest, stamp, stamp),
            )
            purge_after = utc_text(expires)
            args_payload_id = self._insert_payload_tx(connection, execution_id=execution_id, tool_id=tool_id, kind="ARGUMENTS", content=jcs_bytes(canonical_arguments), created_at=stamp, purge_after=purge_after)
            preview_payload_id = self._insert_payload_tx(connection, execution_id=execution_id, tool_id=tool_id, kind="PREVIEW", content=preview_bytes, created_at=stamp, purge_after=purge_after)
            connection.execute(
                "UPDATE tool_executions SET arguments_payload_id=?,preview_payload_id=?,status='AWAITING_APPROVAL',updated_at=? WHERE execution_id=?",
                (args_payload_id, preview_payload_id, stamp, execution_id),
            )
            pointer = {"encrypted": True, "execution_id": execution_id, "payload_id": args_payload_id}
            connection.execute(
                "UPDATE approvals SET arguments_json=? WHERE approval_id=?",
                (jcs_text(pointer), approval_id),
            )
            connection.execute(
                """INSERT INTO approval_arbiter_queue(
                   approval_id,producer_protocol,binding_sha256,state,created_at,expires_at,updated_at
                   ) VALUES(?,?,?,'WAITING_FOR_CHANNEL',?,?,?)""",
                (approval_id, ACTION_APPROVAL_PROTOCOL, approval_binding, stamp, utc_text(expires), stamp),
            )
            connection.execute(
                """INSERT INTO action_approval_presentations(
                   presentation_id,approval_id,execution_id,decision_nonce,
                   preview_fetch_nonce,preview_fetch_nonce_sha256,state,created_at,expires_at
                   ) VALUES(?,?,?,?,?,?,'PENDING',?,?)""",
                (presentation_id, approval_id, execution_id, decision_nonce, preview_nonce, hashlib.sha256(preview_nonce.encode("ascii")).hexdigest(), stamp, utc_text(expires)),
            )
            elapsed = float(task["active_elapsed_seconds"])
            if task["active_started_at"] is not None:
                elapsed += max(0.0, (selected - parse_utc(task["active_started_at"])).total_seconds())
            connection.execute(
                """UPDATE tasks SET state='AWAITING_APPROVAL',active_elapsed_seconds=?,
                   active_started_at=NULL,updated_at=? WHERE task_id=? AND epoch=? AND state='EXECUTING'""",
                (elapsed, stamp, task_id, epoch),
            )
            self._store._append_task_event_tx(connection, task_id, epoch, "tool.awaiting_approval", {"approval_id": approval_id, "execution_id": execution_id, "tool_id": tool_id}, selected)
            self._audit_tx(connection, execution_id, task_id, epoch, tool_id, "PLANNED", "AWAITING_APPROVAL", "ok", args_digest, None, selected)
        return {"execution_id": execution_id, "approval_id": approval_id, "presentation_id": presentation_id, "preview_fetch_nonce": preview_nonce, "approval_binding_sha256": approval_binding, "status": "AWAITING_APPROVAL"}

    def task_run(self, task_id: str, epoch: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM tool_executions WHERE task_id=? AND epoch=? ORDER BY created_at LIMIT 1",
                (task_id, epoch),
            ).fetchone()
        if row is None:
            return None
        return {
            "approval_id": row["approval_id"],
            "arguments_sha256": row["arguments_sha256"],
            "execution_id": row["execution_id"],
            "idempotency_key": row["idempotency_key"],
            "reason_code": row["reason_code"],
            "result_sha256": row["result_sha256"],
            "status": row["status"],
            "tool_call_id": row["tool_call_id"],
            "tool_id": row["tool_name"],
        }

    def issue_execution_permit(
        self,
        execution_id: str,
        *,
        task_id: str,
        epoch: int,
        policy_sha256: str,
        scheduler_contract_sha256: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Atomically revalidate the approved run and bind one dispatch permit."""

        selected = now or utc_now()
        with self._transaction() as connection:
            self._store._guarded_task_tx(
                connection, task_id, epoch, (TaskState.EXECUTING,)
            )
            row = connection.execute(
                "SELECT * FROM tool_executions WHERE execution_id=? AND task_id=? AND epoch=?",
                (execution_id, task_id, epoch),
            ).fetchone()
            if row is None or row["status"] != ToolRunState.APPROVED.value:
                raise StaleEpochError()
            approval = connection.execute(
                "SELECT status,expires_at FROM approvals WHERE approval_id=?", (row["approval_id"],)
            ).fetchone()
            if approval is None or approval["status"] != "APPROVED":
                raise OrchestratorError("tool.binding_mismatch", status=409)
            if parse_utc(approval["expires_at"]) <= selected:
                raise OrchestratorError("tool.approval_expired", status=410)
            definition = self.registry.resolve(row["tool_name"])
            if any(
                (
                    row["registry_sha256"] != self.registry.digest,
                    row["tool_schema_sha256"] != definition.schema_sha256,
                    row["policy_sha256"] != policy_sha256,
                    row["scheduler_contract_sha256"] != scheduler_contract_sha256,
                )
            ):
                raise OrchestratorError("tool.readiness_changed", status=503)
            payload = connection.execute(
                "SELECT * FROM tool_sensitive_payloads WHERE payload_id=? AND purged_at IS NULL",
                (row["arguments_payload_id"],),
            ).fetchone()
            if payload is None:
                raise OrchestratorError("tool.payload_unavailable", status=503)
            envelope = EncryptedToolPayloadV1(
                payload["key_version"],
                payload["nonce"],
                payload["ciphertext_and_tag"],
                payload["content_sha256"],
                payload["aad_sha256"],
            )
            arguments = json.loads(
                self.cipher.decrypt(
                    envelope,
                    self._payload_binding(execution_id, row["tool_name"], "ARGUMENTS"),
                )
            )
            canonical = self.registry.validate(row["tool_name"], arguments)
            if arguments_sha256(canonical) != row["arguments_sha256"]:
                raise OrchestratorError("tool.binding_mismatch", status=409)
            permit_id = new_uuid4()
            changed = connection.execute(
                """UPDATE tool_executions SET status='DISPATCH_COMMITTED',permit_id=?,
                   dispatch_stage='PERMIT_COMMITTED',updated_at=?
                   WHERE execution_id=? AND status='APPROVED'""",
                (permit_id, utc_text(selected), execution_id),
            )
            if changed.rowcount != 1:
                raise StaleEpochError()
            self._audit_tx(
                connection,
                execution_id,
                task_id,
                epoch,
                row["tool_name"],
                ToolRunState.APPROVED.value,
                ToolRunState.DISPATCH_COMMITTED.value,
                "ok",
                row["arguments_sha256"],
                None,
                selected,
            )
        return {
            "arguments": canonical,
            "arguments_sha256": row["arguments_sha256"],
            "definition": definition,
            "idempotency_key": row["idempotency_key"],
            "permit_id": permit_id,
            "policy_sha256": row["policy_sha256"],
            "registry_sha256": row["registry_sha256"],
            "scheduler_contract_sha256": row["scheduler_contract_sha256"],
            "tool_call_id": row["tool_call_id"],
            "user_intent_binding_sha256": row["user_intent_binding_sha256"],
        }

    def complete_execution(
        self,
        execution_id: str,
        *,
        task_id: str,
        epoch: int,
        outcome: ToolOutcomeV1,
        result: Mapping[str, Any],
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Commit a final adapter result only while the task epoch is current."""

        selected = now or utc_now()
        with self._transaction() as connection:
            self._store._guarded_task_tx(
                connection, task_id, epoch, (TaskState.EXECUTING,)
            )
            row = connection.execute(
                "SELECT * FROM tool_executions WHERE execution_id=? AND task_id=? AND epoch=?",
                (execution_id, task_id, epoch),
            ).fetchone()
            if row is None or row["status"] != ToolRunState.DISPATCH_COMMITTED.value:
                raise StaleEpochError()
            canonical = self.registry.validate_output(row["tool_name"], result)
            result_bytes = jcs_bytes(canonical)
            result_digest = hashlib.sha256(result_bytes).hexdigest()
            result_payload_id = self._insert_payload_tx(
                connection,
                execution_id=execution_id,
                tool_id=row["tool_name"],
                kind="RESULT",
                content=result_bytes,
                created_at=utc_text(selected),
                purge_after=None,
            )
            final_state = {
                ToolStatus.VERIFIED: ToolRunState.VERIFIED,
                ToolStatus.ACCEPTED: ToolRunState.ACCEPTED,
                ToolStatus.FAILED: ToolRunState.FAILED,
                ToolStatus.AMBIGUOUS: ToolRunState.AMBIGUOUS,
            }[outcome.status]
            connection.execute(
                """UPDATE tool_executions SET status=?,verification=?,reason_code=?,
                   result_sha256=?,result_payload_id=?,dispatch_stage='COMPLETED',
                   completed_at=?,updated_at=? WHERE execution_id=? AND status='DISPATCH_COMMITTED'""",
                (
                    final_state.value,
                    outcome.verification.value,
                    outcome.reason_code,
                    result_digest,
                    result_payload_id,
                    utc_text(selected),
                    utc_text(selected),
                    execution_id,
                ),
            )
            self._audit_tx(
                connection,
                execution_id,
                task_id,
                epoch,
                row["tool_name"],
                ToolRunState.DISPATCH_COMMITTED.value,
                final_state.value,
                outcome.reason_code,
                row["arguments_sha256"],
                result_digest,
                selected,
            )
        return self.registry.model_projection(row["tool_name"], canonical)

    def mark_dispatch_ambiguous(
        self,
        execution_id: str,
        *,
        reason_code: str = "tool.ambiguous_after_dispatch",
        now: datetime | None = None,
    ) -> None:
        selected = now or utc_now()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM tool_executions WHERE execution_id=?",
                (execution_id,),
            ).fetchone()
            if row is None or row["status"] != ToolRunState.DISPATCH_COMMITTED.value:
                return
            connection.execute(
                """UPDATE tool_executions SET status='AMBIGUOUS',reason_code=?,
                   completed_at=?,updated_at=? WHERE execution_id=? AND status='DISPATCH_COMMITTED'""",
                (reason_code, utc_text(selected), utc_text(selected), execution_id),
            )
            self._audit_tx(
                connection,
                execution_id,
                row["task_id"],
                row["epoch"],
                row["tool_name"],
                ToolRunState.DISPATCH_COMMITTED.value,
                ToolRunState.AMBIGUOUS.value,
                reason_code,
                row["arguments_sha256"],
                None,
                selected,
            )

    def result_projection(self, execution_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                """SELECT x.tool_name,x.result_payload_id,p.* FROM tool_executions x
                   JOIN tool_sensitive_payloads p ON p.payload_id=x.result_payload_id
                   WHERE x.execution_id=? AND p.purged_at IS NULL""",
                (execution_id,),
            ).fetchone()
        if row is None:
            raise OrchestratorError("tool.payload_unavailable", status=503)
        envelope = EncryptedToolPayloadV1(
            row["key_version"],
            row["nonce"],
            row["ciphertext_and_tag"],
            row["content_sha256"],
            row["aad_sha256"],
        )
        result = json.loads(
            self.cipher.decrypt(
                envelope,
                self._payload_binding(execution_id, row["tool_name"], "RESULT"),
            )
        )
        return self.registry.model_projection(row["tool_name"], result)

    def promote_next_action_approval(self, owner_session_id: str, *, now: datetime | None = None) -> dict[str, Any] | None:
        selected = now or utc_now()
        stamp = utc_text(selected)
        with self._transaction() as connection:
            active = connection.execute(
                """SELECT 1 FROM approval_arbiter_queue
                   WHERE state IN ('PRESENTING','PRESENTED','DECIDING','WAITING_FOR_OWNER')
                   LIMIT 1"""
            ).fetchone()
            if active is not None:
                return None
            row = connection.execute(
                """SELECT q.arbiter_seq,q.binding_sha256,a.approval_id,a.task_id,a.epoch,
                          a.expires_at,x.tool_call_id,x.tool_name,x.tool_version,
                          x.preview_sha256,x.arguments_sha256,x.user_intent_binding_sha256,
                          x.registry_sha256,x.tool_schema_sha256,x.policy_sha256,
                          p.presentation_id,p.decision_nonce,p.preview_fetch_nonce
                   FROM approval_arbiter_queue q
                   JOIN approvals a USING(approval_id)
                   JOIN tool_executions x USING(approval_id)
                   JOIN action_approval_presentations p USING(approval_id)
                   WHERE q.state='WAITING_FOR_CHANNEL' AND a.status='PENDING'
                   ORDER BY q.arbiter_seq,q.approval_id LIMIT 1"""
            ).fetchone()
            if row is None:
                return None
            if parse_utc(row["expires_at"]) <= selected:
                connection.execute(
                    "UPDATE approval_arbiter_queue SET state='EXPIRED',updated_at=? WHERE approval_id=?",
                    (stamp, row["approval_id"]),
                )
                return None
            event_id = new_uuid4()
            payload = {
                "approval_id": row["approval_id"],
                "presentation_id": row["presentation_id"],
                "task_id": row["task_id"],
                "tool_call_id": row["tool_call_id"],
                "epoch": row["epoch"],
                "tool_id": row["tool_name"],
                "tool_version": row["tool_version"],
                "approval_binding_sha256": row["binding_sha256"],
                "preview_sha256": row["preview_sha256"],
                "arguments_sha256": row["arguments_sha256"],
                "user_intent_binding_sha256": row["user_intent_binding_sha256"],
                "registry_sha256": row["registry_sha256"],
                "tool_schema_sha256": row["tool_schema_sha256"],
                "policy_sha256": row["policy_sha256"],
                "expires_at": row["expires_at"],
                "decision_nonce": row["decision_nonce"],
                "preview_fetch_nonce": row["preview_fetch_nonce"],
                "approve_key": "F10",
                "deny_key": "F11",
            }
            semantic = domain_sha256("jarvis.action-approval-event/1", {"type": "action.approval.requested", "payload": payload})
            connection.execute(
                """UPDATE approval_arbiter_queue SET state='PRESENTED',owner_session_id=?,
                   presentation_id=?,requested_event_id=?,decision_nonce=?,updated_at=?
                   WHERE approval_id=? AND state='WAITING_FOR_CHANNEL'""",
                (owner_session_id, row["presentation_id"], event_id, row["decision_nonce"], stamp, row["approval_id"]),
            )
            connection.execute(
                "UPDATE action_approval_presentations SET owner_session_id=?,requested_event_id=?,state='PRESENTED' WHERE presentation_id=?",
                (owner_session_id, event_id, row["presentation_id"]),
            )
            connection.execute(
                """INSERT INTO action_approval_outbox(
                   event_id,approval_id,presentation_id,event_type,semantic_sha256,payload_json,state,created_at,last_sent_at
                   ) VALUES(?,?,?,'action.approval.requested',?,?,'PENDING',?,NULL)""",
                (event_id, row["approval_id"], row["presentation_id"], semantic, jcs_text(payload), stamp),
            )
            return {"event_id": event_id, "payload": payload}

    def claim_action_extension_events(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT o.*,q.arbiter_seq FROM action_approval_outbox o
                   JOIN approval_arbiter_queue q USING(approval_id)
                   WHERE o.state='PENDING'
                      OR (o.state='SENT' AND o.last_sent_at<=?)
                   ORDER BY q.arbiter_seq,o.created_at,o.event_id""",
                (utc_text(utc_now() - timedelta(seconds=1)),),
            ).fetchall()
        return [
            {
                "ordinal": row["arbiter_seq"],
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "original_sent_at": row["created_at"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def acknowledge_action_extension(self, event_id: str) -> None:
        with self._transaction() as connection:
            connection.execute(
                "UPDATE action_approval_outbox SET state='ACKED' WHERE event_id=? AND state IN ('PENDING','SENT')",
                (event_id,),
            )

    def finish_action_extension_write(self, event_id: str, *, complete: bool) -> None:
        if not complete:
            return
        with self._transaction() as connection:
            connection.execute(
                "UPDATE action_approval_outbox SET state='SENT',last_sent_at=? WHERE event_id=? AND state IN ('PENDING','SENT')",
                (utc_text(), event_id),
            )

    def reset_action_extension_replay(self, owner_session_id: str, *, now: datetime | None = None) -> str | None:
        stamp = utc_text(now or utc_now())
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT q.approval_id,q.owner_session_id,q.requested_event_id
                   FROM approval_arbiter_queue q JOIN approvals a USING(approval_id)
                   JOIN action_approval_presentations p USING(approval_id,presentation_id)
                   JOIN tasks t USING(task_id) JOIN conversations c USING(conversation_id)
                   WHERE q.producer_protocol=? AND q.state='PRESENTED' AND p.state='PRESENTED'
                     AND a.status='PENDING' AND a.expires_at>? AND t.absolute_expires_at>?
                     AND t.state='AWAITING_APPROVAL' AND t.tombstoned=0 AND t.epoch=c.current_epoch
                   ORDER BY q.arbiter_seq LIMIT 1""",
                (ACTION_APPROVAL_PROTOCOL, stamp, stamp),
            ).fetchone()
            if row is not None:
                connection.execute(
                    "UPDATE approval_arbiter_queue SET owner_session_id=?,updated_at=? WHERE approval_id=?",
                    (owner_session_id, stamp, row["approval_id"]),
                )
                connection.execute(
                    "UPDATE action_approval_presentations SET owner_session_id=? WHERE approval_id=? AND state='PRESENTED'",
                    (owner_session_id, row["approval_id"]),
                )
                connection.execute(
                    """UPDATE action_approval_outbox SET state='PENDING',last_sent_at=NULL
                       WHERE event_id=? AND state IN ('PENDING','SENT','ACKED')""",
                    (row["requested_event_id"],),
                )
            connection.execute(
                "UPDATE action_approval_outbox SET state='PENDING',last_sent_at=NULL WHERE state='SENT'"
            )
        return row["owner_session_id"] if row is not None else None

    def decide_action_transport(
        self,
        *,
        decision_event_id: str,
        semantic_sha256: str,
        approval_id: str,
        presentation_id: str,
        requested_event_id: str,
        approval_binding_sha256: str,
        decision_nonce: str,
        decision: str,
        session_id: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        selected = now or utc_now()
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT semantic_sha256 FROM action_approval_inbox WHERE event_id=?",
                (decision_event_id,),
            ).fetchone()
            if existing is not None:
                if existing["semantic_sha256"] != semantic_sha256:
                    raise OrchestratorError("tool.binding_mismatch", status=409)
                return {"duplicate": True}
            row = connection.execute(
                """SELECT q.*,p.state AS presentation_state,a.epoch
                   FROM approval_arbiter_queue q
                   JOIN action_approval_presentations p USING(approval_id,presentation_id)
                   JOIN approvals a USING(approval_id)
                   WHERE q.approval_id=?""",
                (approval_id,),
            ).fetchone()
            if (
                row is None
                or row["owner_session_id"] != session_id
                or row["presentation_id"] != presentation_id
                or row["requested_event_id"] != requested_event_id
                or row["binding_sha256"] != approval_binding_sha256
                or row["decision_nonce"] != decision_nonce
                or row["state"] != "PRESENTED"
            ):
                raise OrchestratorError("tool.binding_mismatch", status=409)
            connection.execute(
                "INSERT INTO action_approval_inbox VALUES(?,?,?,?,?,?)",
                (decision_event_id, semantic_sha256, "action.approval.decision", approval_id, presentation_id, utc_text(selected)),
            )
            result = self._decide_action_approval_tx(
                connection, approval_id=approval_id, epoch=int(row["epoch"]),
                decision=decision, selected=selected,
            )
            for event_type, payload in (
                (
                    "action.approval.decision_ack",
                    {"decision_event_id": decision_event_id, "status": "accepted"},
                ),
                (
                    "action.approval.decision_result",
                    {
                        "decision_event_id": decision_event_id,
                        "approval_id": approval_id,
                        "presentation_id": presentation_id,
                        "approval_binding_sha256": approval_binding_sha256,
                        "result": "rejected" if result.get("expired") else "applied",
                        "reason_code": "approval_expired" if result.get("expired") else (
                            "approved" if decision == "approve" else "denied"),
                    },
                ),
            ):
                event_id = new_uuid4()
                connection.execute(
                    """INSERT INTO action_approval_outbox(
                       event_id,approval_id,presentation_id,event_type,semantic_sha256,payload_json,state,created_at,last_sent_at
                       ) VALUES(?,?,?,?,?,?,'PENDING',?,NULL)""",
                    (event_id, approval_id, presentation_id, event_type, domain_sha256("jarvis.action-approval-event/1", {"type": event_type, "payload": payload}), jcs_text(payload), utc_text(selected)),
                )
        return result

    def _audit_tx(self, connection: sqlite3.Connection, execution_id: str, task_id: str, epoch: int, tool_id: str, before: str | None, after: str, reason_code: str, args_sha: str, result_sha: str | None, now: datetime) -> None:
        connection.execute(
            """INSERT INTO action_audit(execution_id,task_id,epoch,tool_name,from_state,to_state,
               reason_code,arguments_sha256,result_sha256,occurred_at,retain_until)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (execution_id, task_id, epoch, tool_id, before, after, reason_code, args_sha, result_sha, utc_text(now), utc_text(now + timedelta(days=90))),
        )

    def consume_preview(self, *, approval_id: str, presentation_id: str, preview_nonce: str, epoch: int, now: datetime | None = None) -> dict[str, Any]:
        selected = now or utc_now()
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT p.*,x.tool_name,x.preview_payload_id,x.epoch,a.status,
                          s.key_version,s.nonce,s.ciphertext_and_tag,s.content_sha256,s.aad_sha256
                   FROM action_approval_presentations p
                   JOIN tool_executions x USING(execution_id)
                   JOIN approvals a USING(approval_id)
                   JOIN tool_sensitive_payloads s ON s.payload_id=x.preview_payload_id
                   WHERE p.approval_id=? AND p.presentation_id=?""",
                (approval_id, presentation_id),
            ).fetchone()
            if row is None or row["epoch"] != epoch or row["status"] != "PENDING" or row["state"] not in {"PENDING", "PRESENTED"} or parse_utc(row["expires_at"]) <= selected:
                raise OrchestratorError("tool.approval_expired", status=410)
            if row["preview_consumed_at"] is not None or hashlib.sha256(preview_nonce.encode("ascii")).hexdigest() != row["preview_fetch_nonce_sha256"]:
                raise OrchestratorError("tool.binding_mismatch", status=409)
            envelope = EncryptedToolPayloadV1(row["key_version"], row["nonce"], row["ciphertext_and_tag"], row["content_sha256"], row["aad_sha256"])
            plaintext = self.cipher.decrypt(envelope, self._payload_binding(row["execution_id"], row["tool_name"], "PREVIEW"))
            connection.execute("UPDATE action_approval_presentations SET preview_consumed_at=? WHERE presentation_id=? AND preview_consumed_at IS NULL", (utc_text(selected), presentation_id))
        return json.loads(plaintext)

    def decide_action_approval(
        self, *, approval_id: str, epoch: int, decision: str, now: datetime | None = None,
    ) -> dict[str, Any]:
        with self._transaction() as connection:
            result = self._decide_action_approval_tx(
                connection, approval_id=approval_id, epoch=epoch,
                decision=decision, selected=now or utc_now(),
            )
        if result.get("expired"):
            raise OrchestratorError("tool.approval_expired", status=410)
        return result

    def _decide_action_approval_tx(
        self, connection: sqlite3.Connection, *, approval_id: str,
        epoch: int, decision: str, selected: datetime,
    ) -> dict[str, Any]:
        if decision not in ("approve", "deny"):
            raise OrchestratorError("invalid_request", status=400)
        stamp = utc_text(selected)
        row = connection.execute(
            """SELECT a.*,x.execution_id,x.status AS execution_status,x.tool_name,
                      x.arguments_sha256,t.state AS task_state,t.absolute_expires_at
               FROM approvals a
               JOIN tool_executions x USING(approval_id)
               JOIN tasks t USING(task_id)
               JOIN conversations c USING(conversation_id)
               WHERE a.approval_id=? AND a.epoch=? AND t.epoch=?
                 AND c.current_epoch=? AND t.tombstoned=0""",
            (approval_id, epoch, epoch, epoch),
        ).fetchone()
        if row is None:
            raise StaleEpochError()
        if row["status"] != "PENDING" or row["execution_status"] != "AWAITING_APPROVAL":
            raise OrchestratorError("tool.binding_mismatch", status=409)
        expired = parse_utc(row["expires_at"]) <= selected
        if expired:
            self._store._expire_approval_tx(connection, approval_id, now=selected)
        approved = decision == "approve" and not expired
        next_state = "EXPIRED" if expired else ("APPROVED" if approved else "DENIED")
        reason = "tool.approval_expired" if expired else ("approved" if approved else "tool.approval_denied")
        connection.execute(
            "UPDATE approvals SET status=?,decided_at=? WHERE approval_id=?",
            (next_state, stamp, approval_id),
        )
        connection.execute(
            """UPDATE tool_executions SET status=?,reason_code=?,updated_at=?,
               completed_at=CASE WHEN ? IN ('DENIED','EXPIRED') THEN ? ELSE completed_at END
               WHERE execution_id=?""",
            (next_state, reason, stamp, next_state, stamp, row["execution_id"]),
        )
        connection.execute(
            "UPDATE approval_arbiter_queue SET state=?,updated_at=? WHERE approval_id=?",
            ("EXPIRED" if expired else "RESOLVED", stamp, approval_id),
        )
        connection.execute(
            "UPDATE action_approval_presentations SET state=? WHERE approval_id=? AND state!='INVALIDATED'",
            ("INVALIDATED" if expired else "RESOLVED", approval_id),
        )
        connection.execute(
            """UPDATE action_approval_outbox SET state='INVALIDATED' WHERE approval_id=?
               AND event_type='action.approval.requested' AND state IN ('PENDING','SENT')""", (approval_id,),
        )
        # A decision resumes the ordinary task loop. Approval is transport
        # state; it is not itself a claim that an OS side effect occurred.
        connection.execute(
            """UPDATE tasks SET state='EXECUTING',active_started_at=?,updated_at=?
               WHERE task_id=? AND epoch=? AND state='AWAITING_APPROVAL' AND ?=0""",
            (stamp, stamp, row["task_id"], epoch, expired),
        )
        self._store._append_task_event_tx(
            connection,
            row["task_id"],
            epoch,
            "tool.approval.expired" if expired else ("tool.approval.approved" if approved else "tool.approval.denied"),
            {"approval_id": approval_id, "execution_id": row["execution_id"]},
            selected,
        )
        self._audit_tx(
            connection,
            row["execution_id"],
            row["task_id"],
            epoch,
            row["tool_name"],
            "AWAITING_APPROVAL",
            next_state,
            reason,
            row["arguments_sha256"],
            None,
            selected,
        )
        return {
            "approval_id": approval_id,
            "execution_id": row["execution_id"],
            "task_id": row["task_id"],
            "task_state": TaskState.CANCELLED.value if expired else TaskState.EXECUTING.value,
            **({"expired": True} if expired else {}),
            "decision": decision,
        }

    def transition(self, execution_id: str, event: ToolRunEvent | str, *, reason_code: str, outcome: ToolOutcomeV1 | None = None, now: datetime | None = None) -> ToolRunState:
        selected = now or utc_now()
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM tool_executions WHERE execution_id=?", (execution_id,)).fetchone()
            if row is None:
                raise OrchestratorError("not_found", status=404)
            before = ToolRunState(row["status"])
            after = transition_tool_state(before, event)
            verification = outcome.verification.value if outcome is not None else row["verification"]
            selected_reason = outcome.reason_code if outcome is not None else reason_code
            connection.execute(
                """UPDATE tool_executions SET status=?,verification=?,reason_code=?,updated_at=?,
                   completed_at=CASE WHEN ? IN ('VERIFIED','ACCEPTED','FAILED','AMBIGUOUS','DENIED','CANCELLED','EXPIRED') THEN ? ELSE completed_at END
                   WHERE execution_id=? AND status=?""",
                (after.value, verification, selected_reason, utc_text(selected), after.value, utc_text(selected), execution_id, before.value),
            )
            self._audit_tx(connection, execution_id, row["task_id"], row["epoch"], row["tool_name"], before.value, after.value, selected_reason, row["arguments_sha256"], row["result_sha256"], selected)
            return after

    def mark_cancel_requested(self, execution_id: str, *, now: datetime | None = None) -> ToolRunState:
        selected = now or utc_now()
        with self._transaction() as connection:
            row = connection.execute("SELECT status FROM tool_executions WHERE execution_id=?", (execution_id,)).fetchone()
            if row is None:
                raise OrchestratorError("not_found", status=404)
            state = ToolRunState(row["status"])
            if state in {ToolRunState.PLANNED, ToolRunState.AWAITING_APPROVAL, ToolRunState.APPROVED}:
                pass
            else:
                connection.execute("UPDATE tool_executions SET cancel_requested=1,updated_at=? WHERE execution_id=?", (utc_text(selected), execution_id))
                return state
        return self.transition(execution_id, ToolRunEvent.CANCELLED, reason_code="tool.cancelled_pre_dispatch", now=selected)

    def recover_uncertain(self, *, now: datetime | None = None) -> int:
        selected = now or utc_now()
        with self._lock:
            rows = self._connection.execute("SELECT execution_id,status FROM tool_executions WHERE status IN ('DISPATCH_COMMITTED','DISPATCHED','VERIFYING')").fetchall()
        changed = 0
        for row in rows:
            event = restart_event(row["status"], authenticated_pre_call_rejection=False)
            if event is not None:
                self.transition(row["execution_id"], event, reason_code="tool.ambiguous_after_dispatch", now=selected)
                changed += 1
        return changed

    def record_late_receipt(self, execution_id: str, *, stage: str, semantic_sha256: str, observed_at: str, now: datetime | None = None) -> None:
        _sha(semantic_sha256)
        selected = now or utc_now()
        with self._transaction() as connection:
            row = connection.execute("SELECT status FROM tool_executions WHERE execution_id=?", (execution_id,)).fetchone()
            if row is None or row["status"] != ToolRunState.AMBIGUOUS.value:
                raise OrchestratorError("tool.binding_mismatch", status=409)
            connection.execute(
                "INSERT INTO tool_late_receipts VALUES(?,?,?,?,?,?)",
                (new_uuid4(), execution_id, stage, semantic_sha256, observed_at, utc_text(selected)),
            )

    def purge_expired_payloads(self, *, now: datetime | None = None) -> int:
        """Cryptographically invalidate expired plaintext without copying it."""

        selected = now or utc_now()
        with self._transaction() as connection:
            result = connection.execute(
                """UPDATE tool_sensitive_payloads
                   SET nonce=randomblob(12),ciphertext_and_tag=X'',purged_at=?
                   WHERE purged_at IS NULL AND purge_after IS NOT NULL AND purge_after<=?""",
                (utc_text(selected), utc_text(selected)),
            )
            connection.execute(
                "DELETE FROM action_audit WHERE retain_until<=?",
                (utc_text(selected),),
            )
            return result.rowcount

    def bind_provenance(self, *, entity_kind: str, entity_id: str, provenance: ContentProvenanceV1, now: datetime | None = None) -> None:
        selected = now or utc_now()
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO content_provenance(entity_kind,entity_id,provenance_json,provenance_sha256,
                   policy_label,egress_disposition,created_at) VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(entity_kind,entity_id) DO NOTHING""",
                (entity_kind, entity_id, jcs_text(provenance.to_dict()), provenance.sha256, provenance.policy_label.value, provenance.egress.value, utc_text(selected)),
            )

    def load_provenance(self, entity_kind: str, entity_id: str) -> ContentProvenanceV1:
        with self._lock:
            row = self._connection.execute("SELECT provenance_json,provenance_sha256 FROM content_provenance WHERE entity_kind=? AND entity_id=?", (entity_kind, entity_id)).fetchone()
        if row is None:
            raise OrchestratorError("content_no_egress", status=409)
        provenance = ContentProvenanceV1.from_dict(json.loads(row["provenance_json"]))
        if provenance.sha256 != row["provenance_sha256"]:
            raise OrchestratorError("content_no_egress", status=409)
        return provenance
