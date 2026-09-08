"""Qualified macOS adapter bridge for the out-of-process native helper."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sqlite3
import stat
import subprocess
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .action_host_protocol import strict_json
from .action_host_qualification import NativeReadinessV1
from .action_host_v2_protocol import ResultVerificationV2
from .action_host_v2_server import AdapterResultV2, HostAuthKeyProvider
from .jcs import jcs_bytes
from .macos_code_identity import code_identity_sha256
from .tool_contracts import ToolContractError, ToolRegistryV1, ToolStatus


# Native responses include the result plus reference bindings and receipt metadata.
MAX_NATIVE_RESPONSE_BYTES = 512 * 1024

PAGINATED_TOOLS = frozenset({"file.search@1", "calendar.read@1", "reminder.list@1"})
REFERENCE_ARGUMENTS = {
    "file.open@1": "file_ref",
    "reminder.complete@1": "reminder_ref",
}


def _utc_text(value: datetime | None = None) -> str:
    return (value or datetime.now(UTC)).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


class MacOSAdapterError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class MacOSActionAdapterConfig:
    helper_path: Path
    helper_sha256: str
    helper_code_identity_sha256: str
    roots: Mapping[str, str]
    app_ids: tuple[str, ...]
    state_path: Path
    notes_path: Path
    timeout_seconds: float = 25.0
    require_code_signature: bool = True

    def validate(self) -> None:
        helper = self.helper_path
        if not helper.is_absolute() or helper.is_symlink():
            raise MacOSAdapterError("host.adapter_rejected")
        selected = helper.stat()
        if not stat.S_ISREG(selected.st_mode) or not os.access(helper, os.X_OK):
            raise MacOSAdapterError("host.adapter_rejected")
        if selected.st_uid != os.getuid():
            raise MacOSAdapterError("host.adapter_rejected")
        actual_sha256 = hashlib.sha256(helper.read_bytes()).hexdigest()
        if actual_sha256 != self.helper_sha256:
            raise MacOSAdapterError("host.adapter_rejected")
        if self.require_code_signature:
            verified = subprocess.run(
                ["/usr/bin/codesign", "--verify", "--strict", os.fspath(helper)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                shell=False,
            )
            identity = subprocess.run(
                ["/usr/bin/codesign", "--display", "--requirements", "-", os.fspath(helper)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                shell=False,
            )
            if (
                verified.returncode != 0
                or identity.returncode != 0
                or code_identity_sha256(identity.stdout) != self.helper_code_identity_sha256
            ):
                raise MacOSAdapterError("host.adapter_rejected")
        if not self.roots or any(
            not Path(path).is_absolute() or Path(path) == Path(Path(path).anchor)
            for path in self.roots.values()
        ):
            raise MacOSAdapterError("file.root_unavailable")
        if self.timeout_seconds <= 0:
            raise MacOSAdapterError("host.deadline_expired")


class MacOSReferenceStoreV1:
    def __init__(self, path: Path, keys: HostAuthKeyProvider) -> None:
        self.path = Path(path)
        self.keys = keys
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        self._db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS native_refs(
                ref TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                key_version INTEGER NOT NULL,
                nonce BLOB NOT NULL,
                ciphertext BLOB NOT NULL,
                binding_sha256 TEXT NOT NULL,
                expires_at_utc TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS private_notes(
                note_ref TEXT PRIMARY KEY,
                key_version INTEGER NOT NULL,
                nonce BLOB NOT NULL,
                ciphertext BLOB NOT NULL,
                content_sha256 TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );
            """
        )
        os.chmod(self.path, 0o600)
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _encrypt(self, domain: str, identifier: str, value: Mapping[str, Any]):
        version = self.keys.active_version()
        key = self.keys.load(version)
        nonce = os.urandom(12)
        plaintext = jcs_bytes(value)
        digest = hashlib.sha256(plaintext).hexdigest()
        aad = jcs_bytes(
            {
                "contract": domain,
                "identifier": identifier,
                "key_version": version,
                "sha256": digest,
            }
        )
        return version, nonce, AESGCM(key).encrypt(nonce, plaintext, aad), digest

    def _decrypt(self, domain: str, identifier: str, row: sqlite3.Row) -> dict[str, Any]:
        aad = jcs_bytes(
            {
                "contract": domain,
                "identifier": identifier,
                "key_version": row["key_version"],
                "sha256": row["binding_sha256"] if "binding_sha256" in row.keys() else row["content_sha256"],
            }
        )
        try:
            raw = AESGCM(self.keys.load(row["key_version"])).decrypt(
                row["nonce"], row["ciphertext"] , aad
            )
        except Exception as exc:
            raise MacOSAdapterError("host.result_conflict") from exc
        digest = hashlib.sha256(raw).hexdigest()
        expected = row["binding_sha256"] if "binding_sha256" in row.keys() else row["content_sha256"]
        if digest != expected:
            raise MacOSAdapterError("host.result_conflict")
        return json.loads(raw)

    def store_bindings(self, bindings: list[Mapping[str, Any]]) -> None:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                for value in bindings:
                    if not isinstance(value, Mapping) or not isinstance(value.get("ref"), str) or not isinstance(value.get("kind"), str):
                        raise MacOSAdapterError("host.result_conflict")
                    reference = str(value["ref"])
                    uuid.UUID(reference)
                    expires = str(value.get("expires_at_utc") or "9999-12-31T23:59:59.999Z")
                    version, nonce, ciphertext, digest = self._encrypt(
                        "jarvis.native-ref/1", reference, value
                    )
                    existing = self._db.execute(
                        "SELECT binding_sha256 FROM native_refs WHERE ref=?", (reference,)
                    ).fetchone()
                    if existing is not None and existing["binding_sha256"] != digest:
                        raise MacOSAdapterError("host.result_conflict")
                    self._db.execute(
                        """INSERT OR IGNORE INTO native_refs VALUES(?,?,?,?,?,?,?,?)""",
                        (reference, value["kind"], version, nonce, ciphertext, digest, expires, _utc_text()),
                    )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def resolve(self, reference: str, kind: str) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM native_refs WHERE ref=? AND kind=?",
                (reference, kind),
            ).fetchone()
        if row is None:
            raise MacOSAdapterError(
                "file.ref_invalid" if kind == "file" else "reminder.ref_invalid"
            )
        if row["expires_at_utc"] <= _utc_text():
            raise MacOSAdapterError("file.ref_expired")
        return self._decrypt("jarvis.native-ref/1", reference, row)

    def capture_note(self, title: str, body: str) -> dict[str, Any]:
        note_ref = str(uuid.uuid4())
        created = _utc_text()
        value = {"body": body, "created_at_utc": created, "title": title}
        version, nonce, ciphertext, digest = self._encrypt(
            "jarvis.private-note/1", note_ref, value
        )
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute(
                    "INSERT INTO private_notes VALUES(?,?,?,?,?,?)",
                    (note_ref, version, nonce, ciphertext, digest, created),
                )
                row = self._db.execute(
                    "SELECT * FROM private_notes WHERE note_ref=?", (note_ref,)
                ).fetchone()
                readback = self._decrypt("jarvis.private-note/1", note_ref, row)
                if readback != value:
                    raise MacOSAdapterError("note.verify_failed")
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return {
            "note_ref": note_ref,
            "title": title,
            "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "created_at_utc": created,
            "storage_domain": "JARVIS_PRIVATE_NOTES",
        }


class MacOSNativeAdapterV1:
    """Dispatch only to a fixed, prequalified Swift executable over stdin."""

    def __init__(
        self,
        *,
        config: MacOSActionAdapterConfig,
        registry: ToolRegistryV1,
        auth_keys: HostAuthKeyProvider,
    ) -> None:
        config.validate()
        self.config = config
        self.registry = registry
        self.auth_keys = auth_keys
        self.references = MacOSReferenceStoreV1(config.state_path, auth_keys)

    def close(self) -> None:
        self.references.close()

    def _cursor_key(self) -> bytes:
        key = self.auth_keys.load(self.auth_keys.active_version())
        return hmac.new(key, b"jarvis.native-cursor/1", hashlib.sha256).digest()

    def _decode_cursor(self, tool_id: str, arguments: Mapping[str, Any]) -> int:
        cursor = arguments.get("cursor", "")
        if cursor == "":
            return 0
        try:
            encoded, signature = str(cursor).split(".", 1)
            raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            expected = hmac.new(self._cursor_key(), raw, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, signature):
                raise ValueError
            payload = json.loads(raw)
            semantic = {
                key: value for key, value in arguments.items() if key != "cursor"
            }
            if (
                payload != {
                    "expires_at": payload.get("expires_at"),
                    "offset": payload.get("offset"),
                    "semantic_sha256": hashlib.sha256(jcs_bytes(semantic)).hexdigest(),
                    "tool_id": tool_id,
                }
                or datetime.fromisoformat(payload["expires_at"].replace("Z", "+00:00")) <= datetime.now(UTC)
                or not isinstance(payload["offset"], int)
                or payload["offset"] < 0
            ):
                raise ValueError
            return payload["offset"]
        except Exception as exc:
            raise ToolContractError("tool.schema_invalid") from exc

    def _encode_cursor(self, tool_id: str, arguments: Mapping[str, Any], offset: int) -> str:
        semantic = {key: value for key, value in arguments.items() if key != "cursor"}
        payload = {
            "expires_at": _utc_text(datetime.now(UTC) + timedelta(minutes=10)),
            "offset": offset,
            "semantic_sha256": hashlib.sha256(jcs_bytes(semantic)).hexdigest(),
            "tool_id": tool_id,
        }
        raw = jcs_bytes(payload)
        encoded = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        signature = hmac.new(self._cursor_key(), raw, hashlib.sha256).hexdigest()
        selected = encoded + "." + signature
        if len(selected) > 512:
            raise MacOSAdapterError("host.internal_safe_failure")
        return selected

    def _native_context(self, tool_id: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        context: dict[str, Any] = {
            "app_ids": list(self.config.app_ids),
            # Private notes are encrypted in the qualified reference-store DB;
            # readiness therefore probes the actual durable storage path.
            "notes_path": os.fspath(self.config.state_path),
            "roots": dict(self.config.roots),
        }
        reference_field = REFERENCE_ARGUMENTS.get(tool_id)
        if reference_field is not None:
            kind = "file" if reference_field == "file_ref" else "reminder"
            context["reference_binding"] = self.references.resolve(
                str(arguments[reference_field]), kind
            )
        return context

    def _run_helper(
        self, tool_id: str, arguments: Mapping[str, Any], deadline_utc: str
    ) -> dict[str, Any]:
        selected_arguments = dict(arguments)
        if tool_id in PAGINATED_TOOLS:
            selected_arguments["_cursor_offset"] = self._decode_cursor(tool_id, arguments)
        request = {
            "arguments": selected_arguments,
            "context": self._native_context(tool_id, arguments),
            "tool_id": tool_id,
        }
        remaining = (
            datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
            - datetime.now(UTC)
        ).total_seconds()
        if remaining <= 0:
            raise MacOSAdapterError("host.deadline_expired")
        try:
            completed = subprocess.run(
                [os.fspath(self.config.helper_path)],
                input=jcs_bytes(request),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=remaining if tool_id == "host.permission-setup@1" else min(remaining, self.config.timeout_seconds),
                check=False,
                shell=False,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise MacOSAdapterError("host.deadline_expired") from exc
        if not completed.stdout or len(completed.stdout) > MAX_NATIVE_RESPONSE_BYTES:
            raise MacOSAdapterError("host.internal_safe_failure")
        try:
            response = strict_json(completed.stdout, maximum=MAX_NATIVE_RESPONSE_BYTES)
        except Exception as exc:
            raise MacOSAdapterError("host.internal_safe_failure") from exc
        if completed.returncode != 0:
            if set(response) == {"error"} and isinstance(response["error"], str):
                raise MacOSAdapterError(response["error"])
            raise MacOSAdapterError("host.internal_safe_failure")
        expected = {
            "status",
            "verification",
            "reason_code",
            "result_payload",
            "reference_bindings",
            "os_receipt_sha256",
        }
        if set(response) != expected:
            raise MacOSAdapterError("host.result_conflict")
        return response

    def readiness(self) -> NativeReadinessV1:
        """Inspect current native permissions without requesting or mutating them."""

        deadline = _utc_text(datetime.now(UTC) + timedelta(seconds=10))
        response = self._run_helper("host.readiness@1", {}, deadline)
        if (
            response["status"] != ToolStatus.VERIFIED.value
            or response["verification"] != ResultVerificationV2.EXACT_READBACK.value
            or response["reason_code"] != "ok"
            or response["reference_bindings"] != []
        ):
            raise MacOSAdapterError("host.adapter_rejected")
        try:
            return NativeReadinessV1.from_mapping(response["result_payload"])
        except Exception as exc:
            raise MacOSAdapterError("host.adapter_rejected") from exc

    def request_permissions(self) -> NativeReadinessV1:
        """Explicit setup-only TCC request; never called by startup or execution."""

        deadline = _utc_text(datetime.now(UTC) + timedelta(minutes=5))
        response = self._run_helper("host.permission-setup@1", {}, deadline)
        if (
            response["status"] != ToolStatus.VERIFIED.value
            or response["verification"] != ResultVerificationV2.EXACT_READBACK.value
            or response["reason_code"] != "ok"
            or response["reference_bindings"] != []
        ):
            raise MacOSAdapterError("host.adapter_rejected")
        try:
            return NativeReadinessV1.from_mapping(response["result_payload"])
        except Exception as exc:
            raise MacOSAdapterError("host.adapter_rejected") from exc

    def execute(
        self,
        *,
        tool_id: str,
        arguments: Mapping[str, Any],
        execution_id: str,
        deadline_utc: str,
    ) -> AdapterResultV2:
        del execution_id
        canonical_arguments = self.registry.validate(tool_id, arguments)
        if tool_id == "note.capture@1":
            result = self.references.capture_note(
                canonical_arguments["title"], canonical_arguments["body"]
            )
            canonical = self.registry.validate_output(tool_id, result)
            return AdapterResultV2(
                ToolStatus.VERIFIED,
                ResultVerificationV2.EXACT_READBACK,
                "ok",
                canonical,
                hashlib.sha256(jcs_bytes(canonical)).hexdigest(),
            )
        response = self._run_helper(tool_id, canonical_arguments, deadline_utc)
        try:
            status = ToolStatus(response["status"])
            verification = ResultVerificationV2(response["verification"])
        except ValueError as exc:
            raise MacOSAdapterError("host.result_conflict") from exc
        result = dict(response["result_payload"])
        if tool_id in PAGINATED_TOOLS:
            next_offset = result.pop("_next_offset", None)
            if result.get("truncated") is True:
                if not isinstance(next_offset, int):
                    raise MacOSAdapterError("host.result_conflict")
                result["next_cursor"] = self._encode_cursor(
                    tool_id, canonical_arguments, next_offset
                )
            else:
                result["next_cursor"] = ""
        canonical_result = self.registry.validate_output(tool_id, result)
        bindings = response["reference_bindings"]
        if not isinstance(bindings, list):
            raise MacOSAdapterError("host.result_conflict")
        self.references.store_bindings(bindings)
        return AdapterResultV2(
            status,
            verification,
            str(response["reason_code"]),
            canonical_result,
            str(response["os_receipt_sha256"]),
        )
