"""Source-run server and authenticated control CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import tempfile
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

from .anamnesis_adapter import AnamnesisMemoryAdapter
from .background import BackgroundServiceManager
from .anamnesis_shadow import (
    AnamnesisShadowAdapter,
    DeterministicFirstLifecycleWriter,
    DeterministicLifecycleGateV3,
    DeterministicOnlyLifecycleWriterV3,
    LocalLifecycleCompletionClient,
    LocalStructuredLifecycleWriter,
)
from .config import OrchestratorConfig
from .control import ControlApplication, ControlServer
from .gateway import VoiceGateway
from .inference import InferenceControlClient
from .memory_release import ShadowHistoryReaderFacade
from .pydantic_adapter import build_local_pydantic_port
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
from .action_host_v2_client import ActionHostV2Client
from .research_adapter import LoopbackResearchAdapter
from .cloud_provider import (
    OpenRouterCloudAdapter,
    load_endpoint_registry,
    load_openrouter_secret,
)
from .cloud_qualification import (
    cloud_release_sha256,
    load_cloud_release_qualification,
    run_cloud_privacy_qualification,
)
from .operations import (
    ContentFreeOperationalLog,
    OperationalError,
    ProcessLock,
    backup_database,
    database_diagnostics,
    operational_paths,
    restore_database,
)
from .readiness import (
    evaluate_operational_ready_gate,
    run_live_restart_gate,
    run_live_sequential_gate,
    run_live_soak_gate,
    run_sqlite_recovery_gate,
    run_synthetic_lifecycle_gate,
    write_readiness_report,
)


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _emit_operational(value: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        write_readiness_report(output, value)
    _print(value)


async def _cloud_qualification_status(config: OrchestratorConfig) -> dict[str, Any]:
    """Run static release and live key/ZDR gates without a model dispatch."""

    if not config.cloud_enabled:
        return {
            "active_endpoint_count": 0,
            "cloud_ready": False,
            "code": "cloud_not_requested",
            "terra_route_enabled": False,
            "version": "cloud-activation-ready-gate/1",
        }
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
        with tempfile.TemporaryDirectory(prefix="jarvis-cloud-gate-") as directory:
            store = OrchestratorStore(Path(directory) / "qualification.sqlite3")
            adapter: OpenRouterCloudAdapter | None = None
            try:
                for endpoint in release.active_endpoints:
                    store.register_endpoint_qualification(endpoint, enabled=True)
                adapter = OpenRouterCloudAdapter(
                    store,
                    bearer_token=bearer_token,
                    timeout_seconds=config.cloud_timeout_seconds,
                )
                results = await asyncio.gather(
                    *(adapter.revalidate(endpoint) for endpoint in release.active_endpoints)
                )
            finally:
                if adapter is not None:
                    await adapter.aclose()
                store.close()
        if not results or any(not item.key_policy_verified for item in results):
            raise RuntimeError("cloud_live_gate_invalid")
        return {
            "active_endpoint_count": len(results),
            "cloud_ready": True,
            "code": "cloud_activation_ready",
            "deepseek_endpoint_ready": True,
            "dedicated_monthly_key_ready": True,
            "terra_route_enabled": config.cloud_terra_enabled,
            "version": "cloud-activation-ready-gate/1",
        }
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
            raise
        return {
            "active_endpoint_count": 0,
            "cloud_ready": False,
            "code": getattr(exc, "code", "cloud_activation_blocked"),
            "terra_route_enabled": False,
            "version": "cloud-activation-ready-gate/1",
        }


def _request(config: OrchestratorConfig, method: str, path: str, *, body: Any = None, epoch: int | None = None, key: str | None = None) -> tuple[int, Any | None]:
    url = f"http://{config.control_host}:{config.control_port}{path}"
    encoded = None if body is None else json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    headers = {"Authorization": f"Bearer {config.control_bearer_token}"}
    if encoded is not None:
        headers["Content-Type"] = "application/json"
    if epoch is not None:
        headers["If-Match-Epoch"] = str(epoch)
        headers["Idempotency-Key"] = key or str(uuid.uuid4())
    request = urllib.request.Request(url, data=encoded, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return exc.code, json.loads(raw) if raw else None


async def _serve_components(config: OrchestratorConfig) -> None:
    store = OrchestratorStore(
        config.database_path,
        model_request_limit=config.model_request_limit,
        output_token_limit=config.output_token_limit,
        active_budget_seconds=config.active_budget_seconds,
        approval_ttl=timedelta(seconds=config.approval_ttl_seconds),
        disconnect_ttl=timedelta(seconds=config.reconnect_ttl_seconds),
        delivery_ttl=timedelta(seconds=config.delivery_ttl_seconds),
        tool_authority_enabled=config.tools_enabled,
    )
    try:
        await _run_components(config, store)
    finally:
        store.close()


async def _run_components(
    config: OrchestratorConfig, store: OrchestratorStore
) -> None:
    tool_registry = (
        ToolRegistryV1.build(
            root_ids=tuple((config.tool_roots or {}).keys()),
            app_ids=config.tool_app_ids,
        )
        if config.tools_enabled
        else None
    )
    inference_control = InferenceControlClient(
        config.inference_base_url,
        config.inference_bearer_token,
        timeout_seconds=min(2.0, config.model_timeout_seconds),
    )
    try:
        capabilities = await asyncio.to_thread(inference_control.capabilities)
    except Exception:
        class _InferenceUnavailableModel:
            async def stream(self, _request: Any) -> Any:
                raise RuntimeError("local inference unavailable")
                yield  # pragma: no cover - makes this an async iterator

        degraded_config = replace(
            config,
            cloud_enabled=False,
            cloud_terra_enabled=False,
            research_enabled=False,
            memory_shadow_enabled=False,
            tools_enabled=False,
        )
        await _run_servers(
            degraded_config,
            store,
            model=_InferenceUnavailableModel(),
            inference_control=inference_control,
            shadow_memory=None,
            active_memory=AnamnesisMemoryAdapter(store) if config.memory_enabled else None,
            research_adapter=None,
            cloud_adapter=None,
            cloud_endpoints=(),
            structured_inference_ready=False,
            tool_registry=None,
        )
        return
    if config.research_enabled and not (capabilities.json_schema or capabilities.tools):
        raise RuntimeError("research requires qualified structured local inference")
    if config.cloud_enabled and not (capabilities.json_schema or capabilities.tools):
        raise RuntimeError("cloud routing requires qualified structured local inference")
    if config.tools_enabled and not (capabilities.json_schema or capabilities.tools):
        raise RuntimeError("tools require qualified structured local inference")
    model = build_local_pydantic_port(
        store,
        base_url=config.inference_base_url,
        bearer_token=config.inference_bearer_token,
        timeout_seconds=config.model_timeout_seconds,
        tools=capabilities.tools,
        json_schema=capabilities.json_schema,
        registered_tools=tool_registry.names if tool_registry is not None else (),
    )
    active_memory = AnamnesisMemoryAdapter(store) if config.memory_enabled else None
    shadow_memory = None
    if config.memory_shadow_enabled:
        shadow_history = ShadowHistoryReaderFacade(store.memory_shadow_history)
        if config.memory_shadow_mode == "deterministic_only":
            shadow_memory = AnamnesisShadowAdapter(
                shadow_history,
                DeterministicOnlyLifecycleWriterV3(),
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
                    gate=DeterministicLifecycleGateV3(),
                ),
            )
    research_adapter = None
    cloud_adapter = None
    cloud_endpoints = ()
    runtime_config = config
    store.disable_all_endpoint_qualifications()
    if config.research_enabled:
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
                await asyncio.gather(
                    *(candidate_adapter.revalidate(endpoint) for endpoint in candidate_endpoints)
                )
            except BaseException:
                await candidate_adapter.aclose()
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
    try:
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
        )
    finally:
        if cloud_adapter is not None:
            await cloud_adapter.aclose()


async def _run_servers(
    config: OrchestratorConfig,
    store: OrchestratorStore,
    *,
    model: Any,
    inference_control: InferenceControlClient,
    shadow_memory: Any,
    active_memory: Any,
    research_adapter: Any,
    cloud_adapter: OpenRouterCloudAdapter | None,
    cloud_endpoints: tuple[Any, ...],
    structured_inference_ready: bool = True,
    tool_registry: ToolRegistryV1 | None = None,
) -> None:
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
        assert config.action_host_socket_path is not None
        assert config.action_host_token_file is not None
        tool_executor = ActionHostV2ToolExecutor(
            ActionHostV2Client(
                socket_path=config.action_host_socket_path,
                token_path=config.action_host_token_file,
                auth_keys=tool_key_provider,
                timeout_seconds=min(config.model_timeout_seconds, 30.0),
            )
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
    )
    service.startup_recovery()
    if scheduler is not None:
        scheduler.recover_after_restart()
    control = ControlServer(ControlApplication(config, service))
    control_thread = threading.Thread(target=control.serve_forever, name="jarvis-control", daemon=True)
    control_thread.start()
    gateway = VoiceGateway(service)
    scheduler_worker = SchedulerWorker(scheduler) if scheduler is not None else None
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    for selected_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(selected_signal, stop_requested.set)
            installed_signals.append(selected_signal)
        except (NotImplementedError, RuntimeError):  # Windows event loop
            signal.signal(
                selected_signal,
                lambda _number, _frame: loop.call_soon_threadsafe(stop_requested.set),
            )
    stop_path = operational_paths(config.database_path)["stop"]

    async def watch_stop_request() -> None:
        while not stop_requested.is_set():
            if stop_path.exists():
                stop_path.unlink(missing_ok=True)
                stop_requested.set()
                return
            await asyncio.sleep(0.25)

    stop_watcher = asyncio.create_task(
        watch_stop_request(), name="jarvis-controlled-stop-watcher"
    )
    try:
        if scheduler_worker is not None:
            scheduler_worker.start()
        await gateway.start()
        await stop_requested.wait()
    finally:
        stop_watcher.cancel()
        try:
            await stop_watcher
        except asyncio.CancelledError:
            pass
        for selected_signal in installed_signals:
            loop.remove_signal_handler(selected_signal)
        await gateway.stop()
        if scheduler_worker is not None:
            await scheduler_worker.stop()
        control.shutdown()
        control_thread.join(timeout=2)


async def _serve(config: OrchestratorConfig) -> None:
    paths = operational_paths(config.database_path)
    logger = ContentFreeOperationalLog(paths["log"])
    lock = ProcessLock(paths["lock"])
    try:
        lock.acquire()
        paths["stop"].unlink(missing_ok=True)
        logger.emit("service_starting")
        await _serve_components(config)
        logger.emit("service_stopped", controlled=True)
    except OperationalError:
        logger.emit("service_start_rejected")
        raise
    except BaseException:
        logger.emit("service_failed")
        raise
    finally:
        lock.release()
        logger.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="jarvis-orchestrator")
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
    synthetic.add_argument("--turns", type=int, default=1000)
    synthetic.add_argument("--output", type=Path)
    live = operation_commands.add_parser("live-gate")
    live.add_argument("--turns", type=int, default=100)
    live.add_argument("--maximum-database-bytes", type=int, default=512 * 1024 * 1024)
    live.add_argument("--output", type=Path)
    soak = operation_commands.add_parser("soak-gate")
    soak.add_argument("--duration-seconds", type=int, default=86_400)
    soak.add_argument("--batch-turns", type=int, default=10)
    soak.add_argument("--maximum-database-bytes", type=int, default=512 * 1024 * 1024)
    soak.add_argument("--output", type=Path)
    restart = operation_commands.add_parser("restart-gate")
    restart.add_argument("--output", type=Path)
    recovery = operation_commands.add_parser("recovery-gate")
    recovery.add_argument("--output", type=Path)
    evaluate = operation_commands.add_parser("evaluate-gate")
    evaluate.add_argument("--synthetic", type=Path, required=True)
    evaluate.add_argument("--live", type=Path, required=True)
    evaluate.add_argument("--restart", type=Path, required=True)
    evaluate.add_argument("--recovery", type=Path, required=True)
    evaluate.add_argument("--soak", type=Path, required=True)
    evaluate.add_argument("--native-voice", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
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
    config = OrchestratorConfig.load(Path(parsed.config))
    if parsed.command == "serve":
        try:
            asyncio.run(_serve(config))
        except KeyboardInterrupt:
            return 0
        except OperationalError as exc:
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
                _print(report)
                return 0 if report["ready"] else 1
            if parsed.operation_command == "backup":
                _print(backup_database(config.database_path, parsed.destination))
                return 0
            if parsed.operation_command == "synthetic-gate":
                report = run_synthetic_lifecycle_gate(turns=parsed.turns)
                _emit_operational(report, parsed.output)
                return 0 if report["ready"] else 1
            if parsed.operation_command == "live-gate":
                report = asyncio.run(
                    run_live_sequential_gate(
                        host=config.voice_host,
                        port=config.voice_port,
                        turns=parsed.turns,
                        database_path=config.database_path,
                        maximum_database_bytes=parsed.maximum_database_bytes,
                    )
                )
                _emit_operational(report, parsed.output)
                return 0 if report["ready"] else 1
            if parsed.operation_command == "soak-gate":
                report = asyncio.run(
                    run_live_soak_gate(
                        host=config.voice_host,
                        port=config.voice_port,
                        database_path=config.database_path,
                        duration_seconds=parsed.duration_seconds,
                        batch_turns=parsed.batch_turns,
                        maximum_database_bytes=parsed.maximum_database_bytes,
                    )
                )
                _emit_operational(report, parsed.output)
                return 0 if report["ready"] else 1
            if parsed.operation_command == "restart-gate":
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
                        stop_service=manager.stop,
                        start_service=manager.start,
                    )
                )
                _emit_operational(report, parsed.output)
                return 0 if report["ready"] else 1
            if parsed.operation_command == "recovery-gate":
                report = run_sqlite_recovery_gate(config.database_path)
                _emit_operational(report, parsed.output)
                return 0 if report["ready"] else 1
            if parsed.operation_command == "evaluate-gate":
                report = evaluate_operational_ready_gate(
                    synthetic_evidence=parsed.synthetic,
                    live_evidence=parsed.live,
                    restart_evidence=parsed.restart,
                    recovery_evidence=parsed.recovery,
                    soak_evidence=parsed.soak,
                    native_voice_evidence=parsed.native_voice,
                )
                _emit_operational(report, parsed.output)
                return 0 if report["ready"] else 1
            if parsed.operation_command == "cloud-gate":
                report = asyncio.run(_cloud_qualification_status(config))
                _emit_operational(report, parsed.output)
                return 0 if report["cloud_ready"] else 1
            if parsed.operation_command == "cloud-privacy-gate":
                report = run_cloud_privacy_qualification()
                _emit_operational(report, parsed.output)
                return 0 if report["ready"] else 1
            if parsed.operation_command == "cloud-release-digest":
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
