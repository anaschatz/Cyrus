"""Native Windows ASR/LLM coexistence qualification collector.

The executable built from this module is intentionally not a general CLI.  It
uses the production Voice ASR factory and its turn-scoped CUDA worker, drives
the qualifier-owned loopback gateway, and publishes only fixed numeric and
identity fields.  Audio, prompts, responses, transcripts, paths, device IDs and
the bearer remain transient process memory and are never printed or persisted.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from ..asr_factory import AsrBackendFactory, FallbackAsrBackend
from ..errors import BackendError
from ..gpu import NvidiaSmiInventory
from ..transcription import AsyncTranscriptionCoordinator, TranscriptionOutcome
from .asr_collector_contract import (
    CollectorFailure,
    CollectorInputs,
    load_audio,
    load_inputs,
    load_reference_words,
    normalize_hypothesis_words,
    revalidate_inputs,
    select_coexistence_entries,
)
from .asr_collector_metrics import (
    ForcedCudaFailure,
    within_regression,
)
from .asr_collector_metrics import (
    word_errors as word_errors_metric,
)
from .asr_collector_probes import (
    GpuObservation,
    GpuProbe,
    NativeGpuProbe,
    NativePowerProbe,
    PowerProbe,
)

REPORT_FIELDS = frozenset(
    {
        "asr_runtime_identity_digest",
        "audio_corpus_digest",
        "greek_utterances",
        "english_utterances",
        "greek_language_correct",
        "english_language_correct",
        "greek_reference_words",
        "greek_word_errors",
        "english_reference_words",
        "english_word_errors",
        "forced_cpu_fallbacks",
        "overlapping_utterances",
        "baseline_p95",
        "coexistence_p95",
        "baseline_llm_p95",
        "coexistence_llm_p95",
        "minimum_free_vram_mib",
        "lost_final_transcripts",
        "duplicate_final_transcripts",
        "stale_results",
        "oom",
        "crashes",
        "throttle_reasons",
    }
)
MAX_GATEWAY_EVENT_BYTES = 1_048_576
MAX_GATEWAY_RESPONSE_SECONDS = 600.0
P95_FRACTION = 0.95


class Gateway(Protocol):
    def perform(self, case_index: int, workload_units: int, locale: str) -> float: ...


class GatewayWork(Protocol):
    started_at: float
    completed_at: float | None

    def start(self) -> None: ...
    def active(self) -> bool: ...
    def wait(self, timeout: float) -> float: ...


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_GATEWAY_EVENT_BYTES or raw.startswith(b"\xef\xbb\xbf"):
        raise CollectorFailure("gateway_protocol_failed")
    seen_duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal seen_duplicate
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                seen_duplicate = True
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CollectorFailure("gateway_protocol_failed") from exc
    if seen_duplicate or not isinstance(value, dict):
        raise CollectorFailure("gateway_protocol_failed")
    return value


def _synthetic_prompt(case_index: int, workload_units: int, locale: str) -> str:
    """Generate a deterministic, private-data-free workload only in memory."""

    if workload_units not in {512, 2048, 4096} or locale not in {"el-GR", "en-US"}:
        raise CollectorFailure("fixture_case_invalid")
    if locale == "el-GR":
        prefix = (
            "Συνθετική δοκιμή συνύπαρξης. Ανάλυσε το ακόλουθο μη ευαίσθητο "
            "πλαίσιο και δώσε εκτενή αριθμημένη σύνοψη χωρίς εξωτερικές πηγές. "
        )
        fragment = "πλαίσιο δοκιμής σταθερό δεδομένο "
    else:
        prefix = (
            "Synthetic coexistence fixture. Analyze the following non-sensitive "
            "context and provide a long numbered summary without external sources. "
        )
        fragment = "stable fixture context datum "
    return f"{prefix}Case {case_index:03d}. " + fragment * workload_units


class LoopbackGateway:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port is None
            or parsed.path.rstrip("/") != "/v1"
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise CollectorFailure("gateway_invalid")
        self.host = parsed.hostname
        self.port = parsed.port
        self.api_key = api_key
        self.clock = clock

    def perform(self, case_index: int, workload_units: int, locale: str) -> float:
        prompt = _synthetic_prompt(case_index, workload_units, locale)
        request = {
            "model": "jarvis-local",
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 512,
            "temperature": 0,
            "top_p": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        body = json.dumps(
            request,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        started = self.clock()
        connection = http.client.HTTPConnection(
            self.host,
            self.port,
            timeout=MAX_GATEWAY_RESPONSE_SECONDS,
        )
        terminal = False
        usage = False
        done = False
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            response = connection.getresponse()
            content_type = response.getheader("Content-Type", "")
            if response.status != 200 or not content_type.startswith("text/event-stream"):
                raise CollectorFailure("gateway_request_failed")
            while True:
                line = response.readline(MAX_GATEWAY_EVENT_BYTES + 1)
                if len(line) > MAX_GATEWAY_EVENT_BYTES:
                    raise CollectorFailure("gateway_protocol_failed")
                if not line:
                    break
                if not line.startswith(b"data: "):
                    if line.strip():
                        raise CollectorFailure("gateway_protocol_failed")
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = _strict_json_object(data)
                choices = event.get("choices")
                if choices == []:
                    token_usage = event.get("usage")
                    usage = (
                        isinstance(token_usage, dict)
                        and type(token_usage.get("completion_tokens")) is int
                        and token_usage["completion_tokens"] > 0
                    )
                    continue
                if not isinstance(choices, list) or len(choices) != 1:
                    raise CollectorFailure("gateway_protocol_failed")
                choice = choices[0]
                if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict):
                    raise CollectorFailure("gateway_protocol_failed")
                if choice.get("finish_reason") in {"stop", "length"}:
                    terminal = True
                # `event` may transiently contain generated text.  It is neither
                # accumulated nor returned and is dropped on the next iteration.
        except CollectorFailure:
            raise
        except (OSError, http.client.HTTPException) as exc:
            raise CollectorFailure("gateway_request_failed") from exc
        finally:
            connection.close()
        elapsed = self.clock() - started
        if not terminal or not usage or not done or not math.isfinite(elapsed) or elapsed <= 0:
            raise CollectorFailure("gateway_protocol_failed")
        return elapsed


class ThreadedGatewayWork:
    def __init__(
        self,
        gateway: Gateway,
        case_index: int,
        workload_units: int,
        locale: str,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.gateway = gateway
        self.case_index = case_index
        self.workload_units = workload_units
        self.locale = locale
        self.clock = clock
        self.started_at = 0.0
        self.completed_at: float | None = None
        self._duration: float | None = None
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        try:
            self._duration = self.gateway.perform(
                self.case_index,
                self.workload_units,
                self.locale,
            )
        except BaseException as exc:  # noqa: BLE001 - transferred to the owner thread
            self._error = exc
        finally:
            self.completed_at = self.clock()

    def start(self) -> None:
        if self._thread is not None:
            raise CollectorFailure("gateway_work_invalid")
        self.started_at = self.clock()
        self._thread = threading.Thread(
            target=self._run,
            name="cyrus-qualification-llm",
            daemon=True,
        )
        self._thread.start()

    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout: float) -> float:
        thread = self._thread
        if thread is None:
            raise CollectorFailure("gateway_work_invalid")
        thread.join(timeout)
        if thread.is_alive():
            raise CollectorFailure("gateway_request_timeout")
        if self._error is not None:
            if isinstance(self._error, CollectorFailure):
                raise self._error
            raise CollectorFailure("gateway_request_failed") from self._error
        if self._duration is None or self.completed_at is None:
            raise CollectorFailure("gateway_work_invalid")
        return self._duration


class VramMonitor:
    def __init__(
        self,
        probe: GpuProbe,
        device_index: int,
        interval_seconds: float,
    ) -> None:
        if not 0.05 <= interval_seconds <= 1.0:
            raise CollectorFailure("vram_poll_interval_invalid")
        self.probe = probe
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self._active = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._sample_count = 0
        self._minimum_free_vram_mib: int | None = None
        self._throttle_reasons = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="cyrus-qualification-vram",
            daemon=True,
        )

    def _sample(self) -> None:
        observation = self.probe.observe(self.device_index)
        with self._lock:
            self._sample_count += 1
            if self._minimum_free_vram_mib is None:
                self._minimum_free_vram_mib = observation.free_vram_mib
            else:
                self._minimum_free_vram_mib = min(
                    self._minimum_free_vram_mib,
                    observation.free_vram_mib,
                )
            self._throttle_reasons += observation.throttle_reasons

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.interval_seconds):
                if self._active.is_set():
                    self._sample()
        except BaseException as exc:  # noqa: BLE001 - transferred to the owner thread
            self._error = exc
            self._stop.set()

    def start(self) -> None:
        self._thread.start()

    def begin_overlap(self) -> None:
        if self._error is not None:
            raise CollectorFailure("vram_probe_failed") from self._error
        self._active.set()
        self._sample()

    def end_overlap(self) -> None:
        self._active.clear()

    def finish(self) -> tuple[int, int]:
        self._active.clear()
        self._stop.set()
        self._thread.join(max(2.0, self.interval_seconds * 4))
        if self._thread.is_alive() or self._error is not None:
            raise CollectorFailure("vram_probe_failed") from self._error
        with self._lock:
            if self._sample_count <= 0 or self._minimum_free_vram_mib is None:
                raise CollectorFailure("vram_samples_missing")
            return self._minimum_free_vram_mib, self._throttle_reasons


def _percentile(values: Sequence[float], fraction: float) -> float:
    if (
        not values
        or not 0 <= fraction <= 1
        or any(not math.isfinite(value) or value <= 0 for value in values)
    ):
        raise CollectorFailure("metric_invalid")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    result = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return round(result, 6)


class CollectorRunner:
    def __init__(
        self,
        inputs: CollectorInputs,
        *,
        asr_factory: Callable[[], Any] = AsrBackendFactory,
        coordinator_factory: Callable[[Any], Any] = AsyncTranscriptionCoordinator,
        gateway: Gateway | None = None,
        gpu_probe: GpuProbe | None = None,
        power_probe: PowerProbe | None = None,
        work_factory: Callable[[Gateway, int, int, str], GatewayWork] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.inputs = inputs
        self.asr_factory = (
            asr_factory
            if asr_factory is not AsrBackendFactory
            else lambda: AsrBackendFactory(
                inventory_factory=lambda: NvidiaSmiInventory(
                    executable=inputs.environment.nvidia_smi
                )
            )
        )
        self.coordinator_factory = coordinator_factory
        self.clock = clock
        self.sleep = sleep
        self.gateway = gateway or LoopbackGateway(
            inputs.environment.base_url,
            inputs.environment.api_key,
            clock=clock,
        )
        self.gpu_probe = gpu_probe or NativeGpuProbe(inputs.environment.nvidia_smi)
        self.power_probe = power_probe or NativePowerProbe(
            self.gpu_probe,
            inputs.environment.powercfg,
            inputs.environment.nvidia_smi,
        )
        self.work_factory = work_factory or (
            lambda selected, index, tokens, locale: ThreadedGatewayWork(
                selected,
                index,
                tokens,
                locale,
                clock=clock,
            )
        )
        self._seen_turns: set[str] = set()

    def _transcribe(
        self,
        coordinator: Any,
        backend: Any,
        case_id: str,
        audio: Any,
        *,
        during_poll: Callable[[], None] | None = None,
        expected_fallbacks: int = 0,
    ) -> tuple[float, float, float, str, tuple[str, ...]]:
        started = self.clock()
        coordinator.submit(case_id, audio)
        timeout = (
            self.inputs.asr.voice_config.asr.cuda.worker_start_timeout_seconds
            + self.inputs.asr.voice_config.asr.cuda.transcription_timeout_seconds
            + 30.0
        )
        deadline = started + timeout
        outcome: TranscriptionOutcome | None = None
        while self.clock() < deadline:
            if during_poll is not None:
                during_poll()
            outcome = coordinator.poll()
            if outcome is not None:
                break
            self.sleep(0.005)
        completed = self.clock()
        if outcome is None:
            coordinator.cancel_current()
            raise CollectorFailure("lost_final_transcript")
        if outcome.turn_id != case_id:
            raise CollectorFailure("stale_result")
        if outcome.turn_id in self._seen_turns:
            raise CollectorFailure("duplicate_final_transcript")
        self._seen_turns.add(outcome.turn_id)
        if outcome.error is not None:
            code = outcome.error.code.casefold()
            failure = "asr_oom" if "oom" in code else "asr_crash"
            raise CollectorFailure(failure)
        result = outcome.result
        if result is None or not result.text:
            raise CollectorFailure("lost_final_transcript")
        language_code = result.language.code
        if language_code not in {"el", "en"}:
            language_code = "other"
        if coordinator.poll() is not None:
            raise CollectorFailure("duplicate_final_transcript")
        hypothesis_words = normalize_hypothesis_words(result.text)
        if getattr(backend, "fallbacks", 0) != expected_fallbacks:
            raise CollectorFailure("unplanned_cpu_fallback")
        if not math.isfinite(completed - started) or completed <= started:
            raise CollectorFailure("metric_invalid")
        return completed - started, started, completed, language_code, hypothesis_words

    def run(self) -> dict[str, Any]:
        contract = self.inputs.fixture
        entries = self.inputs.corpus["entries"]
        selected_entries = select_coexistence_entries(self.inputs)
        if (
            len(entries) != contract.clean_case_count
            or len(selected_entries) != contract.request_count
        ):
            raise CollectorFailure("corpus_count_invalid")
        self.power_probe.assert_matches(
            self.inputs.power,
            self.inputs.asr.device_index,
        )
        try:
            backend = self.asr_factory().create(self.inputs.asr.voice_config)
        except BackendError as exc:
            raise CollectorFailure("asr_factory_failed") from exc
        cuda_backend = getattr(backend, "cuda_backend", None)
        if getattr(cuda_backend, "residency", None) != "turn_scoped":
            raise CollectorFailure("asr_residency_invalid")
        coordinator = self.coordinator_factory(backend)
        baseline_asr: list[float] = []
        baseline_llm: list[float] = []
        coexistence_asr: list[float] = []
        coexistence_llm: list[float] = []
        language_correct = {"el-GR": 0, "en-US": 0}
        reference_words = {"el-GR": 0, "en-US": 0}
        word_errors = {"el-GR": 0, "en-US": 0}
        overlaps = 0
        monitor = VramMonitor(
            self.gpu_probe,
            self.inputs.asr.device_index,
            self.inputs.asr.poll_interval_ms / 1000.0,
        )
        monitor_started = False
        try:
            for entry in entries:
                audio = load_audio(self.inputs, entry)
                duration, _started, _completed, language_code, hypothesis = self._transcribe(
                    coordinator,
                    backend,
                    str(entry["case_id"]),
                    audio,
                )
                reference = load_reference_words(self.inputs, entry)
                locale = str(entry["locale"])
                reference_words[locale] += len(reference)
                word_errors[locale] += word_errors_metric(reference, hypothesis)
                expected_code = "el" if entry["locale"] == "el-GR" else "en"
                language_correct[locale] += int(
                    language_code == expected_code
                )
                if entry in selected_entries:
                    baseline_asr.append(duration)

            for index, entry in enumerate(selected_entries):
                target = contract.prompt_workload_units[
                    index % len(contract.prompt_workload_units)
                ]
                baseline_llm.append(
                    self.gateway.perform(index, target, str(entry["locale"]))
                )

            self.power_probe.assert_matches(
                self.inputs.power,
                self.inputs.asr.device_index,
            )
            monitor.start()
            monitor_started = True
            for index, entry in enumerate(selected_entries):
                target = contract.prompt_workload_units[
                    index % len(contract.prompt_workload_units)
                ]
                work = self.work_factory(
                    self.gateway,
                    index,
                    target,
                    str(entry["locale"]),
                )
                work.start()
                self.sleep(contract.inference_lead_in_ms / 1000.0)
                active_overlap = work.active()
                if active_overlap:
                    monitor.begin_overlap()

                def during_poll(
                    overlap: bool = active_overlap,
                    current_work: GatewayWork = work,
                ) -> None:
                    if overlap and not current_work.active():
                        monitor.end_overlap()

                audio = load_audio(self.inputs, entry)
                duration, asr_started, asr_completed, _language_code, _hypothesis = self._transcribe(
                    coordinator,
                    backend,
                    "coexist-" + str(entry["case_id"]),
                    audio,
                    during_poll=during_poll,
                )
                if (
                    asr_started - work.started_at
                    < contract.inference_lead_in_ms / 1000.0
                ):
                    raise CollectorFailure("lead_in_failed")
                monitor.end_overlap()
                coexistence_asr.append(duration)
                coexistence_llm.append(work.wait(MAX_GATEWAY_RESPONSE_SECONDS))
                if (
                    active_overlap
                    and work.completed_at is not None
                    and min(asr_completed, work.completed_at)
                    > max(asr_started, work.started_at)
                ):
                    overlaps += 1

            if not isinstance(backend, FallbackAsrBackend):
                raise CollectorFailure("cpu_fallback_route_invalid")
            fallback_entries = [
                selected_entries[index] for index in (0, 1, 2, 50, 51)
            ]
            if [entry["case_id"] for entry in fallback_entries] != [
                "asr-el-001",
                "asr-el-002",
                "asr-el-003",
                "asr-en-001",
                "asr-en-002",
            ]:
                raise CollectorFailure("cpu_fallback_cases_invalid")
            original_cuda_backend = backend.cuda_backend
            backend.cuda_backend = ForcedCudaFailure()
            forced_reference_words = {"el-GR": 0, "en-US": 0}
            forced_word_errors = {"el-GR": 0, "en-US": 0}
            try:
                for fallback_index, entry in enumerate(fallback_entries, start=1):
                    audio = load_audio(self.inputs, entry)
                    (
                        _duration,
                        _started,
                        _completed,
                        language_code,
                        hypothesis,
                    ) = self._transcribe(
                        coordinator,
                        backend,
                        "forced-cpu-" + str(entry["case_id"]),
                        audio,
                        expected_fallbacks=fallback_index,
                    )
                    expected_code = "el" if entry["locale"] == "el-GR" else "en"
                    if language_code != expected_code:
                        raise CollectorFailure("cpu_fallback_language_failed")
                    reference = load_reference_words(self.inputs, entry)
                    locale = str(entry["locale"])
                    forced_reference_words[locale] += len(reference)
                    forced_word_errors[locale] += word_errors_metric(
                        reference,
                        hypothesis,
                    )
            finally:
                backend.cuda_backend = original_cuda_backend
            if (
                backend.fallbacks != 5
                or any(
                    forced_reference_words[locale] <= 0
                    or forced_word_errors[locale] * 100
                    > forced_reference_words[locale]
                    * contract.maximum_wer_percent_per_locale
                    for locale in ("el-GR", "en-US")
                )
            ):
                raise CollectorFailure("cpu_fallback_quality_failed")
            minimum_vram, throttle_reasons = monitor.finish()
            monitor_started = False
        finally:
            if monitor_started:
                try:
                    monitor.finish()
                except BaseException:  # noqa: BLE001, S110 - preserve original failure
                    pass
            coordinator.close()

        self.power_probe.assert_matches(
            self.inputs.power,
            self.inputs.asr.device_index,
        )
        revalidate_inputs(self.inputs)
        language_counts = {"el-GR": 0, "en-US": 0}
        for entry in entries:
            locale = str(entry["locale"])
            if locale not in language_counts:
                raise CollectorFailure("corpus_language_invalid")
            language_counts[locale] += 1
        if (
            language_counts != {"el-GR": 100, "en-US": 100}
            or len(self._seen_turns)
            != contract.clean_case_count + contract.request_count + 5
        ):
            raise CollectorFailure("coexistence_incomplete")
        baseline_p95 = _percentile(baseline_asr, P95_FRACTION)
        coexistence_p95 = _percentile(coexistence_asr, P95_FRACTION)
        baseline_llm_p95 = _percentile(baseline_llm, P95_FRACTION)
        coexistence_llm_p95 = _percentile(coexistence_llm, P95_FRACTION)
        if (
            len(baseline_asr) != contract.request_count
            or len(coexistence_asr) != contract.request_count
            or len(baseline_llm) != contract.request_count
            or len(coexistence_llm) != contract.request_count
            or any(
                language_correct[locale]
                < contract.minimum_language_correct_per_locale
                for locale in ("el-GR", "en-US")
            )
            or any(reference_words[locale] <= 0 for locale in ("el-GR", "en-US"))
            or any(
                word_errors[locale] * 100
                > reference_words[locale] * contract.maximum_wer_percent_per_locale
                for locale in ("el-GR", "en-US")
            )
            or overlaps < contract.minimum_overlaps
            or minimum_vram < 2048
            or throttle_reasons != 0
            or not within_regression(baseline_p95, coexistence_p95)
            or not within_regression(baseline_llm_p95, coexistence_llm_p95)
        ):
            raise CollectorFailure("coexistence_gate_failed")
        report: dict[str, Any] = {
            "asr_runtime_identity_digest": self.inputs.asr.digest,
            "audio_corpus_digest": self.inputs.corpus_digest,
            "greek_utterances": language_counts["el-GR"],
            "english_utterances": language_counts["en-US"],
            "greek_language_correct": language_correct["el-GR"],
            "english_language_correct": language_correct["en-US"],
            "greek_reference_words": reference_words["el-GR"],
            "greek_word_errors": word_errors["el-GR"],
            "english_reference_words": reference_words["en-US"],
            "english_word_errors": word_errors["en-US"],
            "forced_cpu_fallbacks": backend.fallbacks,
            "overlapping_utterances": overlaps,
            "baseline_p95": baseline_p95,
            "coexistence_p95": coexistence_p95,
            "baseline_llm_p95": baseline_llm_p95,
            "coexistence_llm_p95": coexistence_llm_p95,
            "minimum_free_vram_mib": minimum_vram,
            "lost_final_transcripts": 0,
            "duplicate_final_transcripts": 0,
            "stale_results": 0,
            "oom": 0,
            "crashes": 0,
            "throttle_reasons": throttle_reasons,
        }
        if set(report) != REPORT_FIELDS:
            raise CollectorFailure("report_invalid")
        return report


def _atomic_write_report(path: Path, report: Mapping[str, Any]) -> None:
    if set(report) != REPORT_FIELDS or path.exists() or path.is_symlink():
        raise CollectorFailure("report_invalid")
    try:
        raw = json.dumps(
            dict(report),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CollectorFailure("report_invalid") from exc
    if len(raw) > 65_536:
        raise CollectorFailure("report_invalid")
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".cyrus-asr-", dir=path.parent)
        temporary = Path(name)
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-volume hard-link publishes the complete inode atomically and
        # fails if a racing process created the destination.  No overwrite is
        # possible and the temporary name is removed immediately afterwards.
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    except (OSError, ValueError) as exc:
        raise CollectorFailure("report_write_failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    if sys.platform != "win32" or os.name != "nt":
        raise CollectorFailure("native_windows_required")
    inputs = load_inputs(os.environ if environ is None else environ)
    return CollectorRunner(inputs).run()


def main() -> int:
    try:
        inputs = load_inputs(os.environ)
        if sys.platform != "win32" or os.name != "nt":
            raise CollectorFailure("native_windows_required")
        report = CollectorRunner(inputs).run()
        _atomic_write_report(inputs.environment.output, report)
        return 0
    except BaseException:  # noqa: BLE001 - content-free executable boundary
        # The parent sees only a fixed non-zero exit.  Deliberately emit no
        # traceback, exception text, path, prompt, transcript or secret.
        return 2


__all__ = [
    "REPORT_FIELDS",
    "CollectorRunner",
    "GpuObservation",
    "LoopbackGateway",
    "NativeGpuProbe",
    "NativePowerProbe",
    "ThreadedGatewayWork",
    "VramMonitor",
    "main",
    "run",
]
