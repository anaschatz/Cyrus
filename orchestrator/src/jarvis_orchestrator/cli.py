"""Source-run server and authenticated control CLI."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import sqlite3
import uuid
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

from .background import BackgroundServiceManager
from .cloud_activation import cloud_qualification_status
from .config import OrchestratorConfig, _parse_strict_json
from .control import ControlApplication, ControlServer
from .domain import OrchestratorError, canonical_sha256
from .gateway import VoiceGateway
from .inference import (
    InferenceControlClient,
    InferenceIdentityEndpointUnavailable,
)
from .memory_backup import MemoryBackupProviderAttestor
from .memory_release import ShadowHistoryReaderFacade
from .memory_crypto import MemoryEnvelopeCipherV1
from .memory_key_provider import platform_memory_key_provider
from .memory_runtime import ProviderBoundMemoryCryptoRuntime
from .service import OrchestratorService
from .store import OrchestratorStore
from .scheduler_contract import PackagedTimezoneProvider
from .scheduler_crypto import SchedulerContentCipher, platform_key_provider
from .scheduler_store import SchedulerStore
from .scheduler_worker import SchedulerWorker
from .schedule_compiler import LocalScheduleCompiler, load_compiler_qualification
from .tool_contracts import ToolRegistryV1
from .tool_crypto import ToolPayloadCipherV1
from .tool_executor import ActionHostV2ToolExecutor
from .tool_store import ToolStoreV1
from .translation import OpusCTranslate2TranslationPort
from .action_host_v2_client import ActionHostV2Client
from .action_host_qualification import current_macos_build, load_native_qualification
from .operations import (
    ContentFreeOperationalLog,
    OperationalError,
    ProcessLock,
    backup_database,
    database_diagnostics,
    operational_paths,
    restore_database,
)
from .private_state import (
    PrivateStateError,
    assert_private_database,
    assert_private_regular_file,
    unlink_private_file,
)
from .readiness import (
    evaluate_operational_ready_gate,
    run_live_restart_gate,
    run_live_sequential_gate,
    run_live_soak_gate,
    run_sqlite_recovery_gate,
    run_synthetic_lifecycle_gate,
)
from .readiness_report import write_readiness_report, write_signed_readiness_report
from .readiness_evidence import (
    LIVE_KIND,
    MEMORY_CRYPTO_DISABLED_STATE_SHA256,
    MAXIMUM_FRESHNESS_SECONDS,
    NATIVE_KIND,
    RECOVERY_KIND,
    RESTART_KIND,
    SOAK_KIND,
    SYNTHETIC_KIND,
    AuthenticatedInferenceIdentityV2,
    QualificationBindingV2,
    ReadinessEvidenceV2,
    ReadinessIdentityV2,
    build_readiness_evidence,
    load_ed25519_signer,
    load_native_gate_payload,
    load_pinned_ed25519_verifier,
    qualification_binding_for_kind,
    readiness_identity_from_config,
)
from .runtime_supervisor import (
    RuntimeFailure,
    RuntimeProcessDeadline,
    RuntimeSignalGuard,
    RuntimeSupervisor,
    SupervisedControlRuntime,
)


RUNTIME_SHUTDOWN_TIMEOUT_SECONDS = 3.0
RUNTIME_TOTAL_SHUTDOWN_SECONDS = 4.0


def _memory_provider_state_root(config: OrchestratorConfig) -> Path:
    """Keep key-provider authority separate from operational lease/log files."""

    return operational_paths(config.database_path)["state"] / "memory-keys"


class _PlatformMemoryBackupAttestor:
    """Expose only provider attestation/key-presence, never bootstrap authority."""

    __slots__ = ("_provider",)

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    def attest(self) -> Any:
        return self._provider.attest()

    def exists(self, version: int) -> bool:
        return self._provider.exists(version)


def _database_has_encrypted_memory(path: Path, *, immutable: bool = False) -> bool:
    """Inspect only the selected DB; invalid/non-v10 inputs need no provider."""

    try:
        database = assert_private_database(path)
    except PrivateStateError:
        return False
    connection: sqlite3.Connection | None = None
    try:
        uri = database.resolve(strict=True).as_uri() + "?mode=ro"
        if immutable:
            uri += "&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only=ON")
        if int(connection.execute("PRAGMA query_only").fetchone()[0]) != 1:
            return False
        schema_rows = connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchall()
        if len(schema_rows) != 1 or schema_rows[0][0] != "10":
            return False
        state_rows = connection.execute(
            "SELECT storage_state FROM memory_crypto_state"
        ).fetchall()
        return len(state_rows) == 1 and state_rows[0][0] == "ENCRYPTED"
    except (sqlite3.DatabaseError, OSError, TypeError, ValueError, IndexError):
        return False
    finally:
        if connection is not None:
            connection.close()


def _memory_provider_attestor(
    config: OrchestratorConfig,
    database: Path,
    *,
    immutable: bool = False,
) -> MemoryBackupProviderAttestor | None:
    """Open a narrow provider seam only for an exact encrypted DB."""

    if not _database_has_encrypted_memory(
        database,
        immutable=immutable,
    ):
        return None
    try:
        provider = platform_memory_key_provider(
            state_root=_memory_provider_state_root(config)
        )
        return _PlatformMemoryBackupAttestor(provider)
    except Exception:  # noqa: BLE001 - expose one path-free operations code.
        raise OperationalError("memory_backup_provider_unavailable") from None


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _emit_operational(value: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        write_readiness_report(output, value)
    _print(value)


def _authenticated_readiness_runtime_identity(
    config: OrchestratorConfig,
) -> AuthenticatedInferenceIdentityV2:
    try:
        client = InferenceControlClient(
            config.inference_base_url,
            config.inference_bearer_token,
        )
        client.capabilities()
        identity = client.runtime_identity()
    except (InferenceIdentityEndpointUnavailable, OrchestratorError):
        raise OperationalError("readiness_runtime_identity_unavailable") from None
    return AuthenticatedInferenceIdentityV2(
        model_alias=config.model_alias,
        manifest_digest=identity.manifest_digest,
        runtime_identity_sha256=identity.runtime_identity_sha256,
    )


def _current_readiness_state(config: OrchestratorConfig):
    initial_memory_state = _memory_crypto_identity_state(config)
    initial_diagnostics = database_diagnostics(config.database_path)
    initial_runtime = _authenticated_readiness_runtime_identity(config)
    identity = readiness_identity_from_config(
        config,
        database_schema_version=initial_diagnostics.schema_version,
        inference_identity=initial_runtime,
        memory_crypto_state_sha256=initial_memory_state,
    )
    final_runtime = _authenticated_readiness_runtime_identity(config)
    if final_runtime != initial_runtime:
        raise OperationalError("readiness_runtime_identity_changed")
    if _memory_crypto_identity_state(config) != initial_memory_state:
        raise OperationalError("readiness_memory_crypto_identity_changed")
    final_diagnostics = database_diagnostics(config.database_path)
    if final_diagnostics.schema_version != identity.database_schema_version:
        raise OperationalError("readiness_database_identity_changed")
    return identity, final_diagnostics


def _read_memory_crypto_identity_binding(config: OrchestratorConfig) -> dict[str, Any]:
    """Read one exact schema-10 binding without opening writable authority."""

    database = assert_private_database(config.database_path)
    connection = sqlite3.connect(
        database.resolve(strict=True).as_uri() + "?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        if int(connection.execute("PRAGMA query_only").fetchone()[0]) != 1:
            raise OperationalError("readiness_memory_crypto_unavailable")
        connection.execute("BEGIN")
        schema_rows = connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchall()
        if len(schema_rows) != 1 or schema_rows[0]["value"] != "10":
            raise OperationalError("readiness_memory_crypto_unavailable")
        state_count = int(
            connection.execute("SELECT COUNT(*) FROM memory_crypto_state").fetchone()[0]
        )
        rows = connection.execute(
            """SELECT
               s.principal_id,s.storage_state,s.provider_install_id,
               s.write_epoch_id,s.active_operation_id,
               e.epoch_id,e.principal_id AS epoch_principal_id,
               e.provider_install_id AS epoch_provider_install_id,
               e.key_version,e.state AS epoch_state,
               e.confirmation_challenge,e.confirmation_hmac
               FROM memory_crypto_state s
               JOIN memory_principals p ON p.principal_id=s.principal_id
               JOIN memory_key_epochs e ON e.epoch_id=s.write_epoch_id
               WHERE p.owner_kind='local_owner'
                 AND p.owner_key='local-owner'"""
        ).fetchall()
        if state_count != 1 or len(rows) != 1:
            raise OperationalError("readiness_memory_crypto_unavailable")
        row = dict(rows[0])
        pending_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM memory_crypto_operations "
                "WHERE principal_id=? AND phase!='COMPLETE'",
                (row["principal_id"],),
            ).fetchone()[0]
        )
    finally:
        connection.close()

    install_id = row["provider_install_id"]
    key_version = row["key_version"]
    confirmation_hmac = row["confirmation_hmac"]
    if (
        row["storage_state"] != "ENCRYPTED"
        or row["active_operation_id"] is not None
        or pending_count != 0
        or row["write_epoch_id"] != row["epoch_id"]
        or row["principal_id"] != row["epoch_principal_id"]
        or row["epoch_state"] != "WRITE_ACTIVE"
        or type(install_id) is not str
        or len(install_id) != 64
        or install_id != install_id.lower()
        or any(character not in "0123456789abcdef" for character in install_id)
        or row["epoch_provider_install_id"] != install_id
        or type(key_version) is not int
        or not 1 <= key_version <= 2_147_483_647
        or type(row["confirmation_challenge"]) is not str
        or not isinstance(confirmation_hmac, bytes)
        or len(confirmation_hmac) != 32
    ):
        raise OperationalError("readiness_memory_crypto_unavailable")
    return row


def _memory_crypto_identity_state(config: OrchestratorConfig) -> str:
    if not config.memory_enabled:
        return MEMORY_CRYPTO_DISABLED_STATE_SHA256
    try:
        initial = _read_memory_crypto_identity_binding(config)
        provider = platform_memory_key_provider(
            state_root=_memory_provider_state_root(config),
        )
        cipher = MemoryEnvelopeCipherV1(provider)
        session = cipher.open_epoch_session(
            principal_id=initial["principal_id"],
            key_version=initial["key_version"],
        )
        attestation = provider.attest()
        final = _read_memory_crypto_identity_binding(config)
        if final != initial:
            raise OperationalError("readiness_memory_crypto_unavailable")
        confirmation_hmac = final["confirmation_hmac"]
        confirmation = session.blind_token(
            kind="content",
            value=final["confirmation_challenge"],
        )
        if (
            attestation.install_id != final["provider_install_id"]
            or attestation.active_version != final["key_version"]
            or attestation.high_water < attestation.active_version
            or session.install_id != final["provider_install_id"]
            or not hmac.compare_digest(confirmation, confirmation_hmac)
        ):
            raise OperationalError("readiness_memory_crypto_unavailable")
        return canonical_sha256(
            {
                "confirmation_hmac_sha256": hashlib.sha256(
                    confirmation_hmac
                ).hexdigest(),
                "contract": "jarvis.memory-crypto-readiness/1",
                "epoch_id": final["epoch_id"],
                "key_version": final["key_version"],
                "provider_high_water": attestation.high_water,
                "provider_install_id": final["provider_install_id"],
                "storage_state": final["storage_state"],
            }
        )
    except OperationalError:
        raise
    except Exception:  # noqa: BLE001 - normalize all provider/SQLite failures.
        raise OperationalError("readiness_memory_crypto_unavailable") from None


def _readiness_authority(config: OrchestratorConfig):
    key_path = config.readiness_signing_key_file
    public_path = config.readiness_trusted_public_key_file
    if key_path is None or public_path is None:
        raise OperationalError("readiness_authority_required")
    verifier = load_pinned_ed25519_verifier(
        public_path,
        key_id=config.readiness_signer_key_id,
        expected_public_key_sha256=config.readiness_trusted_public_key_sha256,
    )
    signer = load_ed25519_signer(
        key_path,
        key_id=config.readiness_signer_key_id,
        expected_public_key_sha256=config.readiness_trusted_public_key_sha256,
    )
    identity, diagnostics = _current_readiness_state(config)
    return identity, diagnostics, signer, verifier


def _emit_signed_readiness(
    config: OrchestratorConfig,
    *,
    evidence_kind: str,
    payload: dict[str, Any],
    qualification: QualificationBindingV2,
    output: Path,
    expected_identity: ReadinessIdentityV2 | None = None,
) -> ReadinessEvidenceV2:
    identity, diagnostics, signer, verifier = _readiness_authority(config)
    if expected_identity is not None and identity != expected_identity:
        raise OperationalError("readiness_identity_changed_during_gate")
    evidence = build_readiness_evidence(
        evidence_kind=evidence_kind,
        payload=payload,
        identity=identity,
        database_diagnostics=diagnostics,
        qualification=qualification,
        signer=signer,
        freshness_seconds=MAXIMUM_FRESHNESS_SECONDS,
    )
    write_signed_readiness_report(
        output,
        evidence,
        verifier=verifier,
        cohort_state_directory=(
            operational_paths(config.database_path)["state"] / "readiness-cohorts"
        ),
    )
    _print(evidence.canonical())
    return evidence


def _voice_auth_key(config: OrchestratorConfig):
    from jarvis_voice.voice_auth import VoiceAuthError, load_voice_auth_key

    if config.voice_auth_key_file is None:
        raise OperationalError("voice_auth_key_required")
    try:
        return load_voice_auth_key(config.voice_auth_key_file)
    except VoiceAuthError:
        raise OperationalError("voice_auth_key_invalid") from None


def _request(config: OrchestratorConfig, method: str, path: str, *, body: Any = None, epoch: int | None = None, key: str | None = None) -> tuple[int, Any | None]:
    import httpx

    url = f"http://{config.control_host}:{config.control_port}{path}"
    encoded = None if body is None else json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    headers = {"Authorization": f"Bearer {config.control_bearer_token}", "Accept-Encoding": "identity"}
    if encoded is not None:
        headers["Content-Type"] = "application/json"
    if epoch is not None:
        headers["If-Match-Epoch"] = str(epoch)
        headers["Idempotency-Key"] = key or str(uuid.uuid4())

    async def send():
        async with asyncio.timeout(3), httpx.AsyncClient(timeout=3, follow_redirects=False, trust_env=False) as client:
            async with client.stream(method, url, content=encoded, headers=headers) as response:
                if (300 <= response.status_code < 400
                        or response.headers.get("content-encoding", "identity").lower() != "identity"):
                    raise ValueError("invalid control response")
                raw = bytearray()
                async for block in response.aiter_bytes():
                    raw.extend(block)
                    if len(raw) > 1024 * 1024:
                        raise ValueError("control response exceeds limit")
                if raw and response.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
                    raise ValueError("invalid control response type")
                value = _parse_strict_json(bytes(raw)) if raw else None
                if (not raw and response.status_code != 204) or (raw and (not isinstance(value, dict) or response.status_code == 204)):
                    raise ValueError("invalid control response body")
                json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
                return response.status_code, value
    try:
        return asyncio.run(send())
    except (httpx.HTTPError, TimeoutError, ValueError, RecursionError):
        return 503, {"error": {"code": "control_request_failed", "retryable": True}}


async def _serve_components(
    config: OrchestratorConfig,
    *,
    voice_auth_key: Any,
    runtime_supervisor: RuntimeSupervisor | None = None,
    shutdown_deadline: RuntimeProcessDeadline | None = None,
    operational_log: ContentFreeOperationalLog | None = None,
) -> None:
    supervisor = runtime_supervisor or RuntimeSupervisor(
        shutdown_timeout_seconds=RUNTIME_SHUTDOWN_TIMEOUT_SECONDS,
    )
    deadline = shutdown_deadline or supervisor.process_deadline(
        RUNTIME_TOTAL_SHUTDOWN_SECONDS
    )
    owns_deadline = shutdown_deadline is None
    memory_runtime = None
    if config.memory_enabled:
        try:
            provider = platform_memory_key_provider(
                state_root=_memory_provider_state_root(config),
            )
            memory_runtime = ProviderBoundMemoryCryptoRuntime(
                MemoryEnvelopeCipherV1(provider)
            )
        except Exception:  # noqa: BLE001 - stable startup boundary.
            raise OperationalError("memory_storage_unavailable") from None
    store: OrchestratorStore | None = None
    try:
        try:
            store = OrchestratorStore(
                config.database_path,
                model_request_limit=config.model_request_limit,
                output_token_limit=config.output_token_limit,
                active_budget_seconds=config.active_budget_seconds,
                approval_ttl=timedelta(seconds=config.approval_ttl_seconds),
                disconnect_ttl=timedelta(seconds=config.reconnect_ttl_seconds),
                delivery_ttl=timedelta(seconds=config.delivery_ttl_seconds),
                tool_authority_enabled=config.tools_enabled,
                memory_crypto=memory_runtime,
                memory_storage_enabled=config.memory_enabled,
            )
        except Exception:  # noqa: BLE001 - preserve disabled errors; normalize enabled.
            if config.memory_enabled:
                raise OperationalError("memory_storage_unavailable") from None
            raise
        if memory_runtime is not None:
            try:
                memory_status = store.prepare_memory_storage()
            except Exception:  # noqa: BLE001 - stable migration/binding failure.
                raise OperationalError("memory_storage_unavailable") from None
            if (
                not isinstance(memory_status, dict)
                or memory_status.get("storage_state") != "ENCRYPTED"
                or memory_status.get("ready") is not True
            ):
                raise OperationalError("memory_storage_unavailable")
        await _run_components(
            config,
            store,
            voice_auth_key=voice_auth_key,
            runtime_supervisor=supervisor,
            shutdown_deadline=deadline,
            operational_log=operational_log,
        )
    finally:
        if store is not None:
            # This watchdog is outside asyncio and remains armed while the
            # synchronous SQLite close boundary runs.
            deadline.begin()
            store.close()
            if owns_deadline:
                deadline.disarm()


def _apply_native_action_qualification(
    config: OrchestratorConfig,
) -> tuple[OrchestratorConfig, ToolRegistryV1 | None]:
    if not config.tools_enabled:
        return config, None
    registry = ToolRegistryV1.build(
        root_ids=tuple((config.tool_roots or {}).keys()),
        app_ids=config.tool_app_ids,
    )
    assert config.action_host_qualification_file is not None
    try:
        native_qualification = load_native_qualification(
            config.action_host_qualification_file,
            expected_sha256=config.action_host_qualification_sha256,
            registry_sha256=registry.digest,
            expected_os_build=current_macos_build(),
        )
        if native_qualification.qualified_tools != registry.names:
            raise RuntimeError("native qualification does not cover the frozen registry")
    except Exception:
        # Native actions are an optional authority. A missing, stale, or
        # mismatched qualification never takes down the basic assistant.
        return replace(config, tools_enabled=False), None
    return config, registry


async def _run_components(
    config: OrchestratorConfig,
    store: OrchestratorStore,
    *,
    voice_auth_key: Any,
    runtime_supervisor: RuntimeSupervisor | None = None,
    shutdown_deadline: RuntimeProcessDeadline | None = None,
    operational_log: ContentFreeOperationalLog | None = None,
) -> None:
    supervisor = runtime_supervisor or RuntimeSupervisor(
        shutdown_timeout_seconds=RUNTIME_SHUTDOWN_TIMEOUT_SECONDS,
    )
    deadline = shutdown_deadline or supervisor.process_deadline(
        RUNTIME_TOTAL_SHUTDOWN_SECONDS
    )
    owns_deadline = shutdown_deadline is None
    config, tool_registry = _apply_native_action_qualification(config)
    inference_control = InferenceControlClient(
        config.inference_base_url,
        config.inference_bearer_token,
        timeout_seconds=min(2.0, config.model_timeout_seconds),
    )
    capabilities = None
    for delay in (0.0, 0.25, 0.5):
        if delay:
            await asyncio.sleep(delay)
        try:
            capabilities = await asyncio.to_thread(inference_control.capabilities)
            break
        except Exception:
            continue
    if capabilities is None:
        # Local inference is the basic assistant, not an optional capability.
        # Failing startup lets the controlled service manager retry instead of
        # advertising readiness with a permanently unavailable model object.
        raise RuntimeError("local inference capability discovery failed")
    if config.research_enabled and not (capabilities.json_schema or capabilities.tools):
        raise RuntimeError("research requires qualified structured local inference")
    if config.cloud_enabled and not (capabilities.json_schema or capabilities.tools):
        raise RuntimeError("cloud routing requires qualified structured local inference")
    if config.tools_enabled and not (capabilities.json_schema or capabilities.tools):
        raise RuntimeError("tools require qualified structured local inference")
    from .pydantic_adapter import build_local_pydantic_port

    model = build_local_pydantic_port(
        store,
        base_url=config.inference_base_url,
        bearer_token=config.inference_bearer_token,
        timeout_seconds=config.model_timeout_seconds,
        tools=capabilities.tools,
        json_schema=capabilities.json_schema,
        registered_tools=tool_registry.names if tool_registry is not None else (),
    )
    cloud_adapter = None
    try:
        if config.memory_enabled:
            from .anamnesis_adapter import AnamnesisMemoryAdapter

            active_memory = AnamnesisMemoryAdapter(store)
        else:
            active_memory = None
        shadow_memory = None
        if config.memory_shadow_enabled:
            from .anamnesis_shadow import (
                AnamnesisShadowAdapter,
                DeterministicFirstLifecycleWriter,
                LocalLifecycleCompletionClient,
                LocalStructuredLifecycleWriter,
            )

            from .lifecycle_gate import DeterministicLifecycleGateV4, DeterministicOnlyLifecycleWriterV4

            shadow_history = ShadowHistoryReaderFacade(store.memory_shadow_history)
            if config.memory_shadow_mode == "deterministic_only":
                shadow_memory = AnamnesisShadowAdapter(
                    shadow_history,
                    DeterministicOnlyLifecycleWriterV4(),
                )
            else:
                if not capabilities.json_schema:
                    raise RuntimeError(
                        "structured shadow fallback requires JSON Schema capability"
                    )
                shadow_client = LocalLifecycleCompletionClient(
                    base_url=config.inference_base_url,
                    bearer_token=config.inference_bearer_token,
                    timeout_seconds=config.memory_shadow_timeout_seconds,
                )
                shadow_memory = AnamnesisShadowAdapter(
                    shadow_history,
                    DeterministicFirstLifecycleWriter(
                        LocalStructuredLifecycleWriter(
                            shadow_client,
                            model_alias=config.model_alias,
                        ),
                        gate=DeterministicLifecycleGateV4(),
                    ),
                )
        research_adapter = None
        cloud_endpoints = ()
        runtime_config = config
        store.disable_all_endpoint_qualifications()
        if config.research_enabled:
            from .research_adapter import LoopbackResearchAdapter

            research_adapter = LoopbackResearchAdapter(
                base_url=config.research_base_url,
                bearer_token=config.research_bearer_token,
                timeout_seconds=config.research_timeout_seconds,
            )
            await asyncio.to_thread(
                research_adapter.qualify,
                {
                    "contract_version": "research-corpus/1",
                    "model_sha256": config.research_model_sha256,
                    "profile_sha256": config.research_profile_sha256,
                    "qualification_sha256": config.research_qualification_sha256,
                    "language_detector_version": config.research_language_detector_version,
                    "language_detector_sha256": config.research_language_detector_sha256,
                    "authority_registry_version": config.research_authority_registry_version,
                    "browser_qualification_sha256": (
                        config.research_browser_qualification_sha256
                        if config.research_browser_fallback
                        else ""
                    ),
                },
            )
        if config.cloud_enabled:
            from .cloud_provider import (
                OpenRouterCloudAdapter,
                load_endpoint_registry,
                load_openrouter_secret,
                revalidate_cloud_endpoints,
            )
            from .cloud_qualification import load_cloud_release_qualification

            candidate_endpoints: tuple[Any, ...] = ()
            try:
                assert config.cloud_secret_file is not None
                assert config.cloud_qualification_file is not None
                assert config.cloud_release_qualification_file is not None
                bearer_token = load_openrouter_secret(config.cloud_secret_file)
                registry = load_endpoint_registry(config.cloud_qualification_file)
                release = load_cloud_release_qualification(
                    config.cloud_release_qualification_file,
                    expected_sha256=config.cloud_release_qualification_sha256,
                    registry=registry,
                    bearer_token=bearer_token,
                    terra_enabled=config.cloud_terra_enabled,
                )
                candidate_endpoints = release.active_endpoints
                for endpoint in candidate_endpoints:
                    store.register_endpoint_qualification(endpoint, enabled=True)
                candidate_adapter = OpenRouterCloudAdapter(
                    store,
                    bearer_token=bearer_token,
                    timeout_seconds=config.cloud_timeout_seconds,
                )
                try:
                    await revalidate_cloud_endpoints(candidate_adapter, candidate_endpoints)
                except BaseException:
                    startup_cleanup_deadline = supervisor.process_deadline(
                        RUNTIME_TOTAL_SHUTDOWN_SECONDS
                    )
                    startup_cleanup_deadline.begin()
                    try:
                        await supervisor.run_bounded_cleanup(
                            "cloud_startup",
                            candidate_adapter.aclose,
                            failure_code="runtime_cloud_startup_cleanup_failed",
                            timeout_code="runtime_cloud_startup_cleanup_timeout",
                            timeout_seconds=startup_cleanup_deadline.remaining(),
                        )
                    finally:
                        startup_cleanup_deadline.disarm()
                    raise
                cloud_adapter = candidate_adapter
                cloud_endpoints = candidate_endpoints
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                    raise
                for endpoint in candidate_endpoints:
                    try:
                        store.quarantine_endpoint(endpoint.qualification_sha256)
                    except Exception:
                        pass
                cloud_adapter = None
                cloud_endpoints = ()
                runtime_config = replace(
                    config,
                    cloud_enabled=False,
                    cloud_terra_enabled=False,
                )
        await _run_servers(
            runtime_config,
            store,
            model=model,
            inference_control=inference_control,
            shadow_memory=shadow_memory,
            active_memory=active_memory,
            research_adapter=research_adapter,
            cloud_adapter=cloud_adapter,
            cloud_endpoints=cloud_endpoints,
            structured_inference_ready=capabilities.json_schema,
            tool_registry=tool_registry,
            voice_auth_key=voice_auth_key,
            runtime_supervisor=supervisor,
            shutdown_deadline=deadline,
            operational_log=operational_log,
        )
    finally:
        deadline.begin()
        try:
            try:
                if cloud_adapter is not None:
                    await supervisor.run_bounded_cleanup(
                        "cloud_adapter",
                        cloud_adapter.aclose,
                        failure_code="runtime_cloud_shutdown_failed",
                        timeout_code="runtime_cloud_shutdown_timeout",
                        timeout_seconds=deadline.remaining(),
                    )
            finally:
                await supervisor.run_bounded_cleanup(
                    "model",
                    model.aclose,
                    failure_code="runtime_model_shutdown_failed",
                    timeout_code="runtime_model_shutdown_timeout",
                    timeout_seconds=deadline.remaining(),
                )
        finally:
            if owns_deadline:
                deadline.disarm()


async def _run_servers(
    config: OrchestratorConfig,
    store: OrchestratorStore,
    *,
    model: Any,
    inference_control: InferenceControlClient,
    shadow_memory: Any,
    active_memory: Any,
    research_adapter: Any,
    cloud_adapter: Any | None,
    cloud_endpoints: tuple[Any, ...],
    structured_inference_ready: bool = True,
    tool_registry: ToolRegistryV1 | None = None,
    voice_auth_key: Any,
    runtime_supervisor: RuntimeSupervisor | None = None,
    shutdown_deadline: RuntimeProcessDeadline | None = None,
    operational_log: ContentFreeOperationalLog | None = None,
) -> None:
    supervisor = runtime_supervisor or RuntimeSupervisor(
        shutdown_timeout_seconds=RUNTIME_SHUTDOWN_TIMEOUT_SECONDS,
    )
    deadline = shutdown_deadline or supervisor.process_deadline(
        RUNTIME_TOTAL_SHUTDOWN_SECONDS
    )
    owns_deadline = shutdown_deadline is None
    scheduler: SchedulerStore | None = None
    schedule_compiler: LocalScheduleCompiler | None = None
    try:
        key_provider = platform_key_provider(
            store_role="orchestrator",
            state_root=config.database_path.parent / "scheduler-keys",
        )
        scheduler = SchedulerStore(
            store,
            cipher=SchedulerContentCipher(key_provider),
            zones=PackagedTimezoneProvider(),
        )
    except Exception:
        # The assistant remains available, while scheduler capability reports
        # fail-closed through the authenticated readiness surface.
        scheduler = None
    if (scheduler is not None and structured_inference_ready
            and config.scheduler_compiler_qualification_file is not None):
        try:
            qualification = load_compiler_qualification(
                config.scheduler_compiler_qualification_file,
                config.scheduler_compiler_qualification_sha256,
            )
            schedule_compiler = LocalScheduleCompiler(
                base_url=config.inference_base_url,
                bearer_token=config.inference_bearer_token,
                model_alias=config.model_alias,
                qualification=qualification,
                timeout_seconds=min(config.model_timeout_seconds, 30.0),
                zones=scheduler.zones,
            )
        except Exception:
            schedule_compiler = None
    tool_store: ToolStoreV1 | None = None
    tool_executor: ActionHostV2ToolExecutor | None = None
    if config.tools_enabled:
        if tool_registry is None:
            raise RuntimeError("tools_enabled requires a frozen registry")
        # Key access is a startup qualification boundary. There is no plaintext
        # fallback and independent capabilities remain available if tools are off.
        tool_key_provider = platform_key_provider(
            store_role="tool-actions",
            state_root=config.database_path.parent / "tool-keys",
        )
        tool_store = ToolStoreV1(
            store,
            cipher=ToolPayloadCipherV1(tool_key_provider),
            registry=tool_registry,
        )
        action_host_auth_keys = platform_key_provider(
            store_role="action-host-auth",
            state_root=config.database_path.parent / "action-host-auth-keys",
        )
        assert config.action_host_socket_path is not None
        assert config.action_host_token_file is not None
        tool_executor = ActionHostV2ToolExecutor(
            ActionHostV2Client(
                socket_path=config.action_host_socket_path,
                token_path=config.action_host_token_file,
                auth_keys=action_host_auth_keys,
                timeout_seconds=min(config.model_timeout_seconds, 30.0),
            )
        )
    input_translator = None
    if config.input_translation_mode == "opus_ct2":
        assert config.input_translation_model_root is not None
        assert config.input_translation_manifest_file is not None
        input_translator = OpusCTranslate2TranslationPort(
            model_root=config.input_translation_model_root,
            manifest_path=config.input_translation_manifest_file,
            manifest_sha256=config.input_translation_manifest_sha256,
            timeout_seconds=config.input_translation_timeout_seconds,
        )
    service = OrchestratorService(
        config,
        store,
        model,
        inference_control,
        memory=shadow_memory,
        active_memory=active_memory,
        research_planner=model if config.research_enabled else None,
        research=research_adapter,
        task_profiler=model if config.cloud_enabled else None,
        cloud_model=cloud_adapter,
        cloud_endpoints=cloud_endpoints,
        cloud_release_verified=(
            config.cloud_enabled
            and cloud_adapter is not None
            and bool(cloud_endpoints)
        ),
        scheduler=scheduler,
        schedule_compiler=schedule_compiler,
        tool_store=tool_store,
        tool_planner=model if config.tools_enabled else None,
        tool_executor=tool_executor,
        input_translator=input_translator,
    )
    service.startup_recovery()
    if scheduler is not None:
        scheduler.recover_after_restart()
    gateway = VoiceGateway(
        service,
        auth_key=voice_auth_key,
        business_admission=supervisor.business_admission_allowed,
        service_error_reporter=(
            None
            if operational_log is None
            else lambda code, detail: operational_log.emit(code, detail=detail)
        ),
    )
    scheduler_worker = SchedulerWorker(scheduler) if scheduler is not None else None
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    stop_path = operational_paths(config.database_path)["stop"]
    signal_guard = RuntimeSignalGuard(loop, stop_requested)
    signal_guard.install()

    async def watch_stop_request() -> None:
        acknowledged = False
        while True:
            if stop_requested.is_set():
                # The supervisor owns the watcher until shutdown begins.  A
                # controlled stop must not look like an unexpected clean exit.
                await asyncio.Event().wait()
            try:
                if assert_private_regular_file(stop_path, required=False):
                    stop_requested.set()
                    continue
            except PrivateStateError as exc:
                stop_requested.set()
                raise OperationalError(exc.code) from exc
            if not acknowledged:
                supervisor.acknowledge_ready("stop_watcher")
                acknowledged = True
            await asyncio.sleep(0.25)

    gateway_started = False
    control_runtime: SupervisedControlRuntime | None = None
    primary_failure: BaseException | None = None

    try:
        if input_translator is not None:
            await input_translator.start()
            supervisor.start_task(
                "input_translation",
                input_translator.monitor,
                failure_code="runtime_input_translation_worker_failed",
            )
            supervisor.acknowledge_ready("input_translation")
        # Treat a partial start as cleanup-owned before any await can fail.
        gateway_started = True
        await gateway.start(
            supervisor=supervisor,
            shutdown_deadline=deadline,
        )
        if scheduler_worker is not None:
            scheduler_worker.start(supervisor=supervisor)
        supervisor.start_task(
            "stop_watcher",
            watch_stop_request,
            failure_code="runtime_stop_watcher_failed",
        )
        # Both listener surfaces bind with business admission closed.  Control
        # is the final registration, so the following barrier also seals the
        # registry against late critical tasks.
        control_runtime = SupervisedControlRuntime(
            ControlServer(
                ControlApplication(
                    config,
                    service,
                    runtime_health=supervisor.health,
                    business_admission=supervisor.business_admission_allowed,
                )
            ),
            supervisor,
            shutdown_timeout_seconds=RUNTIME_SHUTDOWN_TIMEOUT_SECONDS,
        )
        control_runtime.start()
        # asyncio's timeout cannot fire while a component blocks the event
        # loop synchronously.  This independent watchdog is deliberately a
        # little later, so the stable async barrier timeout wins when the loop
        # remains responsive.
        startup_watchdog = supervisor.process_deadline(
            RUNTIME_SHUTDOWN_TIMEOUT_SECONDS + 0.25
        )
        startup_watchdog.begin()
        try:
            ready = await supervisor.finalize_readiness(
                stop_requested,
                timeout_seconds=RUNTIME_SHUTDOWN_TIMEOUT_SECONDS,
            )
        finally:
            startup_watchdog.disarm()
        if ready:
            await supervisor.wait(stop_requested)
    except BaseException as exc:  # noqa: BLE001 - preserve through full cleanup
        primary_failure = exc
    finally:
        supervisor.begin_shutdown()
        signal_guard.begin_cleanup()
        deadline.begin()
        cleanup_failure: RuntimeFailure | None = None
        try:
            try:
                if control_runtime is not None:
                    await control_runtime.stop(
                        timeout_seconds=deadline.remaining()
                    )
            except RuntimeFailure as exc:
                cleanup_failure = exc
            if gateway_started:
                try:
                    await supervisor.run_bounded_cleanup(
                        "gateway",
                        gateway.stop,
                        failure_code="runtime_gateway_shutdown_failed",
                        timeout_code="runtime_gateway_shutdown_timeout",
                        timeout_seconds=deadline.remaining(),
                    )
                except RuntimeFailure as exc:
                    cleanup_failure = cleanup_failure or exc
            if scheduler_worker is not None:
                try:
                    await supervisor.run_bounded_cleanup(
                        "scheduler_worker",
                        scheduler_worker.stop,
                        failure_code="runtime_scheduler_shutdown_failed",
                        timeout_code="runtime_scheduler_shutdown_timeout",
                        timeout_seconds=deadline.remaining(),
                    )
                except RuntimeFailure as exc:
                    cleanup_failure = cleanup_failure or exc
            if input_translator is not None:
                try:
                    await supervisor.run_bounded_cleanup(
                        "input_translation",
                        input_translator.close,
                        failure_code="runtime_input_translation_shutdown_failed",
                        timeout_code="runtime_input_translation_shutdown_timeout",
                        timeout_seconds=deadline.remaining(),
                    )
                except RuntimeFailure as exc:
                    cleanup_failure = cleanup_failure or exc
            try:
                await supervisor.shutdown(
                    timeout_seconds=deadline.remaining()
                )
            except RuntimeFailure as exc:
                cleanup_failure = cleanup_failure or RuntimeFailure(
                    exc.code,
                    component=exc.component,
                    cause=exc.__cause__,
                )
        finally:
            signal_guard.restore()
            if owns_deadline:
                deadline.disarm()
        if primary_failure is not None:
            raise primary_failure
        if cleanup_failure is not None:
            raise cleanup_failure


async def _serve(config: OrchestratorConfig) -> None:
    voice_auth_key = _voice_auth_key(config)
    paths = operational_paths(config.database_path)
    logger = ContentFreeOperationalLog(paths["log"])
    lock = ProcessLock(paths["lock"])
    supervisor = RuntimeSupervisor(
        shutdown_timeout_seconds=RUNTIME_SHUTDOWN_TIMEOUT_SECONDS,
    )
    deadline = supervisor.process_deadline(RUNTIME_TOTAL_SHUTDOWN_SECONDS)
    lifecycle_succeeded = False
    try:
        lock.acquire()
        logger.emit("service_starting")
        await _serve_components(
            config,
            voice_auth_key=voice_auth_key,
            runtime_supervisor=supervisor,
            shutdown_deadline=deadline,
            operational_log=logger,
        )
        logger.emit("service_stopped", controlled=True)
        lifecycle_succeeded = True
    except RuntimeFailure as exc:
        logger.emit(exc.code)
        raise
    except OperationalError:
        logger.emit("service_start_rejected")
        raise
    except BaseException:
        logger.emit("service_failed")
        raise
    finally:
        lock.release()
        logger.close()
        if not lifecycle_succeeded:
            deadline.disarm()
    # A marker is durable stop intent.  It is consumed only after every
    # adapter/store and the operational lock/log have closed successfully;
    # failures leave it for the service manager's next retry to observe before
    # readiness.
    try:
        unlink_private_file(paths["stop"], missing_ok=True)
    except PrivateStateError as exc:
        raise OperationalError(exc.code) from exc
    finally:
        deadline.disarm()


def _add_qualification_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--qualification-cohort-id", required=True)
    command.add_argument("--qualification-window-started-at-utc", required=True)


def _command_qualification(
    parsed: argparse.Namespace,
    evidence_kind: str,
) -> QualificationBindingV2:
    return qualification_binding_for_kind(
        evidence_kind,
        cohort_id=parsed.qualification_cohort_id,
        window_started_at=parsed.qualification_window_started_at_utc,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="cyrus-orchestrator")
    result.add_argument("--config", required=True)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("serve")
    commands.add_parser("ready")
    background = commands.add_parser("background")
    background.add_argument(
        "background_command", choices=("install", "start", "stop", "status", "uninstall")
    )
    operations = commands.add_parser("operations")
    operation_commands = operations.add_subparsers(dest="operation_command", required=True)
    operation_commands.add_parser("diagnostics")
    backup = operation_commands.add_parser("backup")
    backup.add_argument("destination", type=Path)
    restore = operation_commands.add_parser("restore")
    restore.add_argument("backup", type=Path)
    restore.add_argument("--confirm-database", required=True)
    synthetic = operation_commands.add_parser("synthetic-gate")
    _add_qualification_arguments(synthetic)
    synthetic.add_argument("--turns", type=int, default=1000)
    synthetic.add_argument("--output", type=Path, required=True)
    live = operation_commands.add_parser("live-gate")
    _add_qualification_arguments(live)
    live.add_argument("--turns", type=int, default=100)
    live.add_argument("--maximum-database-bytes", type=int, default=512 * 1024 * 1024)
    live.add_argument("--output", type=Path, required=True)
    soak = operation_commands.add_parser("soak-gate")
    _add_qualification_arguments(soak)
    soak.add_argument("--duration-seconds", type=int, default=86_400)
    soak.add_argument("--batch-turns", type=int, default=10)
    soak.add_argument("--maximum-database-bytes", type=int, default=512 * 1024 * 1024)
    soak.add_argument("--output", type=Path, required=True)
    restart = operation_commands.add_parser("restart-gate")
    _add_qualification_arguments(restart)
    restart.add_argument("--output", type=Path, required=True)
    recovery = operation_commands.add_parser("recovery-gate")
    _add_qualification_arguments(recovery)
    recovery.add_argument("--output", type=Path, required=True)
    native_sign = operation_commands.add_parser("sign-native-gate")
    _add_qualification_arguments(native_sign)
    native_sign.add_argument("evidence", type=Path)
    native_sign.add_argument("--expected-runtime-profile-sha256", required=True)
    native_sign.add_argument("--output", type=Path, required=True)
    evaluate = operation_commands.add_parser("evaluate-gate")
    evaluate.add_argument("--synthetic", type=Path, required=True)
    evaluate.add_argument("--live", type=Path, required=True)
    evaluate.add_argument("--restart", type=Path, required=True)
    evaluate.add_argument("--recovery", type=Path, required=True)
    evaluate.add_argument("--soak", type=Path, required=True)
    evaluate.add_argument("--native-voice", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    cloud_gate = operation_commands.add_parser("cloud-gate")
    cloud_gate.add_argument("--output", type=Path)
    cloud_privacy = operation_commands.add_parser("cloud-privacy-gate")
    cloud_privacy.add_argument("--output", type=Path)
    cloud_digest = operation_commands.add_parser("cloud-release-digest")
    cloud_digest.add_argument("release", type=Path)
    approvals = commands.add_parser("approvals")
    approval_commands = approvals.add_subparsers(dest="approval_command", required=True)
    approval_commands.add_parser("list")
    decide = approval_commands.add_parser("decide")
    decide.add_argument("approval_id")
    decide.add_argument("--epoch", type=int, required=True)
    decision = decide.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--deny", action="store_true")
    tasks = commands.add_parser("tasks")
    task_commands = tasks.add_subparsers(dest="task_command", required=True)
    cancel = task_commands.add_parser("cancel")
    cancel.add_argument("task_id")
    cancel.add_argument("--epoch", type=int, required=True)
    conversations = commands.add_parser("conversations")
    conversation_commands = conversations.add_subparsers(dest="conversation_command", required=True)
    conversation_commands.add_parser("active")
    create = conversation_commands.add_parser("create")
    create.add_argument("--epoch", type=int, required=True)
    delete = conversation_commands.add_parser("delete")
    delete.add_argument("conversation_id")
    delete.add_argument("--epoch", type=int, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parsed = parser().parse_args(argv)
    try:
        config = OrchestratorConfig.load(Path(parsed.config))
    except Exception:
        _print(
            {
                "error": {
                    "code": "configuration_invalid",
                    "retryable": False,
                }
            }
        )
        return 1
    if parsed.command == "serve":
        try:
            asyncio.run(_serve(config))
        except KeyboardInterrupt:
            return 0
        except OperationalError as exc:
            _print({"error": {"code": exc.code, "retryable": False}})
            return 1
        except RuntimeFailure as exc:
            _print({"error": {"code": exc.code, "retryable": False}})
            return 1
        except Exception:
            _print({"error": {"code": "service_start_failed", "retryable": False}})
            return 1
        return 0
    if parsed.command == "operations":
        try:
            if parsed.operation_command == "diagnostics":
                report = database_diagnostics(config.database_path).to_dict()
                report["scope"] = "database_only"
                report["live_memory_provider_checked"] = False
                _print(report)
                return 0 if report["ready"] else 1
            if parsed.operation_command == "backup":
                _print(
                    backup_database(
                        config.database_path,
                        parsed.destination,
                        memory_provider_attestor=_memory_provider_attestor(
                            config,
                            config.database_path,
                        ),
                    )
                )
                return 0
            if parsed.operation_command == "synthetic-gate":
                expected_identity, _, _, _ = _readiness_authority(config)
                report = run_synthetic_lifecycle_gate(turns=parsed.turns)
                _emit_signed_readiness(
                    config,
                    evidence_kind=SYNTHETIC_KIND,
                    payload=report,
                    qualification=_command_qualification(
                        parsed,
                        SYNTHETIC_KIND,
                    ),
                    output=parsed.output,
                    expected_identity=expected_identity,
                )
                return 0 if report["ready"] else 1
            if parsed.operation_command == "live-gate":
                expected_identity, _, _, _ = _readiness_authority(config)
                report = asyncio.run(
                    run_live_sequential_gate(
                        host=config.voice_host,
                        port=config.voice_port,
                        auth_key=_voice_auth_key(config),
                        turns=parsed.turns,
                        database_path=config.database_path,
                        maximum_database_bytes=parsed.maximum_database_bytes,
                    )
                )
                _emit_signed_readiness(
                    config,
                    evidence_kind=LIVE_KIND,
                    payload=report,
                    qualification=_command_qualification(parsed, LIVE_KIND),
                    output=parsed.output,
                    expected_identity=expected_identity,
                )
                return 0 if report["ready"] else 1
            if parsed.operation_command == "soak-gate":
                expected_identity, _, _, _ = _readiness_authority(config)
                report = asyncio.run(
                    run_live_soak_gate(
                        host=config.voice_host,
                        port=config.voice_port,
                        auth_key=_voice_auth_key(config),
                        database_path=config.database_path,
                        duration_seconds=parsed.duration_seconds,
                        batch_turns=parsed.batch_turns,
                        maximum_database_bytes=parsed.maximum_database_bytes,
                    )
                )
                _emit_signed_readiness(
                    config,
                    evidence_kind=SOAK_KIND,
                    payload=report,
                    qualification=_command_qualification(parsed, SOAK_KIND),
                    output=parsed.output,
                    expected_identity=expected_identity,
                )
                return 0 if report["ready"] else 1
            if parsed.operation_command == "restart-gate":
                expected_identity, _, _, _ = _readiness_authority(config)
                manager = BackgroundServiceManager(
                    config_path=Path(parsed.config), database_path=config.database_path
                )
                status = manager.status()
                if not status["installed"] or not status["running"]:
                    raise OperationalError("background_not_running")
                report = asyncio.run(
                    run_live_restart_gate(
                        host=config.voice_host,
                        port=config.voice_port,
                        auth_key=_voice_auth_key(config),
                        stop_service=manager.stop,
                        start_service=manager.start,
                    )
                )
                _emit_signed_readiness(
                    config,
                    evidence_kind=RESTART_KIND,
                    payload=report,
                    qualification=_command_qualification(parsed, RESTART_KIND),
                    output=parsed.output,
                    expected_identity=expected_identity,
                )
                return 0 if report["ready"] else 1
            if parsed.operation_command == "recovery-gate":
                expected_identity, _, _, _ = _readiness_authority(config)
                report = run_sqlite_recovery_gate(config.database_path)
                _emit_signed_readiness(
                    config,
                    evidence_kind=RECOVERY_KIND,
                    payload=report,
                    qualification=_command_qualification(parsed, RECOVERY_KIND),
                    output=parsed.output,
                    expected_identity=expected_identity,
                )
                return 0 if report["ready"] else 1
            if parsed.operation_command == "sign-native-gate":
                expected_identity, _, _, _ = _readiness_authority(config)
                report = load_native_gate_payload(
                    parsed.evidence,
                    expected_runtime_profile_sha256=(
                        parsed.expected_runtime_profile_sha256
                    ),
                    expected_host_class=expected_identity.host_class,
                )
                evidence = _emit_signed_readiness(
                    config,
                    evidence_kind=NATIVE_KIND,
                    payload=report,
                    qualification=_command_qualification(parsed, NATIVE_KIND),
                    output=parsed.output,
                    expected_identity=expected_identity,
                )
                return 0 if evidence.payload["ready"] else 1
            if parsed.operation_command == "evaluate-gate":
                identity, _, signer, verifier = _readiness_authority(config)
                evidence = evaluate_operational_ready_gate(
                    synthetic_evidence=parsed.synthetic,
                    live_evidence=parsed.live,
                    restart_evidence=parsed.restart,
                    recovery_evidence=parsed.recovery,
                    soak_evidence=parsed.soak,
                    native_voice_evidence=parsed.native_voice,
                    verifier=verifier,
                    expected_identity=identity,
                    aggregate_signer=signer,
                    cohort_state_directory=(
                        operational_paths(config.database_path)["state"]
                        / "readiness-cohorts"
                    ),
                    current_state_provider=lambda: _current_readiness_state(config),
                )
                write_signed_readiness_report(
                    parsed.output,
                    evidence,
                    verifier=verifier,
                    cohort_state_directory=(
                        operational_paths(config.database_path)["state"]
                        / "readiness-cohorts"
                    ),
                    current_state_provider=lambda: _current_readiness_state(config),
                )
                _print(evidence.canonical())
                return 0 if evidence.payload["ready"] else 1
            if parsed.operation_command == "cloud-gate":
                report = asyncio.run(cloud_qualification_status(config))
                _emit_operational(report, parsed.output)
                return 0 if report["cloud_ready"] else 1
            if parsed.operation_command == "cloud-privacy-gate":
                from .cloud_qualification import run_cloud_privacy_qualification

                report = run_cloud_privacy_qualification()
                _emit_operational(report, parsed.output)
                return 0 if report["ready"] else 1
            if parsed.operation_command == "cloud-release-digest":
                from .cloud_qualification import cloud_release_sha256

                _print(
                    {
                        "sha256": cloud_release_sha256(parsed.release),
                        "version": "cloud-release-qualification/1",
                    }
                )
                return 0
            if parsed.confirm_database != str(config.database_path):
                _print({"error": {"code": "confirmation_failed", "retryable": False}})
                return 2
            paths = operational_paths(config.database_path)
            _print(
                restore_database(
                    parsed.backup,
                    config.database_path,
                    service_lock=paths["lock"],
                    memory_provider_attestor=_memory_provider_attestor(
                        config,
                        parsed.backup,
                        immutable=True,
                    ),
                )
            )
            return 0
        except OperationalError as exc:
            _print({"error": {"code": exc.code, "retryable": False}})
            return 1
        except Exception:
            _print({"error": {"code": "operation_failed", "retryable": False}})
            return 1
    if parsed.command == "background":
        try:
            manager = BackgroundServiceManager(
                config_path=Path(parsed.config), database_path=config.database_path
            )
            action = getattr(manager, parsed.background_command)
            _print(action())
            return 0
        except OperationalError as exc:
            _print({"error": {"code": exc.code, "retryable": False}})
            return 1
        except Exception:
            _print({"error": {"code": "background_operation_failed", "retryable": False}})
            return 1
    if parsed.command == "ready":
        status, value = _request(config, "GET", "/readyz")
    elif parsed.command == "approvals" and parsed.approval_command == "list":
        status, value = _request(config, "GET", "/v1/approvals")
    elif parsed.command == "approvals":
        decision = "approve" if parsed.approve else "deny"
        status, value = _request(config, "POST", f"/v1/approvals/{parsed.approval_id}/decision", body={"decision": decision}, epoch=parsed.epoch)
    elif parsed.command == "tasks":
        status, value = _request(config, "POST", f"/v1/tasks/{parsed.task_id}/cancel", body={"reason_code": "user_cancelled"}, epoch=parsed.epoch)
    elif parsed.command == "conversations" and parsed.conversation_command == "active":
        status, value = _request(config, "GET", "/v1/conversations/active")
    elif parsed.command == "conversations" and parsed.conversation_command == "create":
        status, value = _request(config, "POST", "/v1/conversations", body={}, epoch=parsed.epoch)
    elif parsed.command == "conversations" and parsed.conversation_command == "delete":
        confirmation = input(f"Type the full conversation ID to delete {parsed.conversation_id}: ").strip()
        if confirmation != parsed.conversation_id:
            _print({"error": {"code": "confirmation_failed", "retryable": False}})
            return 2
        status, value = _request(config, "DELETE", f"/v1/conversations/{parsed.conversation_id}", epoch=parsed.epoch)
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    _print({"status": status, "body": value})
    return 0 if 200 <= status < 300 else 1
