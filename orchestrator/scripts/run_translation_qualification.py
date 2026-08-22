#!/usr/bin/env python3
"""Qualify the pinned local Greek-to-English worker without human input."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - native Windows qualification
    resource = None  # type: ignore[assignment]

from jarvis_orchestrator import translation as translation_module
from jarvis_orchestrator import translation_worker as translation_worker_module
from jarvis_orchestrator.domain import canonical_json
from jarvis_orchestrator.translation import (
    TRANSLATION_CONVERSION_POLICY_SHA256,
    TRANSLATION_CTRANSLATE2_VERSION,
    TRANSLATION_MODEL_ID,
    TRANSLATION_MODEL_REVISION,
    TRANSLATION_SENTENCEPIECE_VERSION,
    OpusCTranslate2TranslationPort,
    load_translation_manifest,
    validate_translation_output,
)


QUALIFICATION_VERSION = "jarvis.translation-qualification/1"
QUALIFICATION_DETERMINISTIC_RUNS = 3
QUALIFICATION_SINGLE_TURNS = 250
_FIXTURE_ID = re.compile(r"[a-z][a-z0-9-]{0,63}")


class QualificationFailure(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationFailure("translation_qualification_duplicate_key")
        result[key] = value
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_build_lock(path: Path) -> str:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise QualificationFailure("translation_qualification_build_lock_invalid")
    raw = path.read_bytes()
    if not raw or len(raw) > 256 * 1024:
        raise QualificationFailure("translation_qualification_build_lock_invalid")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_closed_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise QualificationFailure(
            "translation_qualification_build_lock_invalid"
        ) from None
    digest = hashlib.sha256(canonical_json(document).encode()).hexdigest()
    if (
        not isinstance(document, dict)
        or document.get("version") != "jarvis.translation-build-lock/1"
        or document.get("model_id") != TRANSLATION_MODEL_ID
        or document.get("revision") != TRANSLATION_MODEL_REVISION
        or digest != TRANSLATION_CONVERSION_POLICY_SHA256
    ):
        raise QualificationFailure("translation_qualification_build_lock_invalid")
    return digest


def _string_list(value: Any) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise QualificationFailure("translation_qualification_corpus_invalid")
    return list(value)


def load_fixtures(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise QualificationFailure("translation_qualification_corpus_invalid")
    raw = path.read_bytes()
    if not raw or len(raw) > 1024 * 1024:
        raise QualificationFailure("translation_qualification_corpus_invalid")
    fixtures: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    allowed = {
        "id",
        "source",
        "exact",
        "required_all",
        "required_any",
        "preserve",
        "forbidden",
        "ordered",
    }
    for line in raw.splitlines():
        try:
            item = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_closed_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise QualificationFailure(
                "translation_qualification_corpus_invalid"
            ) from None
        if not isinstance(item, dict) or not set(item) <= allowed:
            raise QualificationFailure("translation_qualification_corpus_invalid")
        identifier = item.get("id")
        source = item.get("source")
        if (
            set(item) < {"id", "source"}
            or not isinstance(identifier, str)
            or not _FIXTURE_ID.fullmatch(identifier)
            or identifier in identifiers
            or not isinstance(source, str)
            or not source.strip()
            or len(source.encode("utf-8")) > 8_192
        ):
            raise QualificationFailure("translation_qualification_corpus_invalid")
        if "exact" in item and (
            not isinstance(item["exact"], str) or not item["exact"].strip()
        ):
            raise QualificationFailure("translation_qualification_corpus_invalid")
        for field in ("required_all", "preserve", "forbidden", "ordered"):
            if field in item:
                item[field] = _string_list(item[field])
        if "required_any" in item:
            groups = item["required_any"]
            if not isinstance(groups, list) or not groups:
                raise QualificationFailure("translation_qualification_corpus_invalid")
            item["required_any"] = [_string_list(group) for group in groups]
        identifiers.add(identifier)
        fixtures.append(item)
    if not 25 <= len(fixtures) <= 500:
        raise QualificationFailure("translation_qualification_corpus_invalid")
    return tuple(fixtures)


def _validate_pack(model_root: Path) -> None:
    try:
        import ctranslate2
        import sentencepiece
    except ImportError:
        raise QualificationFailure(
            "translation_qualification_runtime_unavailable"
        ) from None
    if (
        ctranslate2.__version__ != TRANSLATION_CTRANSLATE2_VERSION
        or sentencepiece.__version__ != TRANSLATION_SENTENCEPIECE_VERSION
        or not ctranslate2.contains_model(str(model_root))
        or "int8" not in ctranslate2.get_supported_compute_types("cpu")
    ):
        raise QualificationFailure("translation_qualification_runtime_invalid")
    try:
        config = json.loads((model_root / "config.json").read_text())
        vocabulary = json.loads(
            (model_root / "shared_vocabulary.json").read_text()
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise QualificationFailure(
            "translation_qualification_model_invalid"
        ) from None
    if config != {
        "add_source_bos": False,
        "add_source_eos": False,
        "bos_token": "<s>",
        "decoder_start_token": "</s>",
        "eos_token": "</s>",
        "layer_norm_epsilon": None,
        "multi_query_attention": False,
        "unk_token": "<unk>",
    }:
        raise QualificationFailure("translation_qualification_model_invalid")
    if (
        not isinstance(vocabulary, list)
        or len(vocabulary) != 58_828
        or len(set(vocabulary)) != 58_828
        or vocabulary[25_697] != "</s>"
        or vocabulary[25_698] != "<s>"
        or vocabulary[30_496] != "<unk>"
        or "<pad>" in vocabulary
    ):
        raise QualificationFailure("translation_qualification_vocabulary_invalid")
    for name in ("source.spm", "target.spm"):
        processor = sentencepiece.SentencePieceProcessor(
            model_file=str(model_root / name)
        )
        if (
            processor.vocab_size() != 32_000
            or processor.unk_id() != 0
            or processor.bos_id() != 1
            or processor.eos_id() != 2
            or processor.pad_id() != -1
        ):
            raise QualificationFailure("translation_qualification_tokenizer_invalid")


def _phrase_match(text: str, phrase: str, start: int = 0) -> re.Match[str] | None:
    escaped = re.escape(phrase.casefold()).replace(r"\ ", r"\s+")
    return re.search(r"(?<!\w)" + escaped + r"(?!\w)", text.casefold()[start:])


def _validate_fixture(fixture: dict[str, Any], output: str) -> None:
    try:
        validate_translation_output(output)
    except BaseException:
        raise QualificationFailure(
            f"translation_qualification_case_{fixture['id']}"
        ) from None
    if "exact" in fixture and output != fixture["exact"]:
        raise QualificationFailure(
            f"translation_qualification_case_{fixture['id']}"
        )
    if any(
        _phrase_match(output, value) is None
        for value in fixture.get("required_all", ())
    ):
        raise QualificationFailure(
            f"translation_qualification_case_{fixture['id']}"
        )
    for group in fixture.get("required_any", ()):
        if not any(_phrase_match(output, value) is not None for value in group):
            raise QualificationFailure(
                f"translation_qualification_case_{fixture['id']}"
            )
    if any(value not in output for value in fixture.get("preserve", ())):
        raise QualificationFailure(
            f"translation_qualification_case_{fixture['id']}"
        )
    if any(
        _phrase_match(output, value) is not None
        for value in fixture.get("forbidden", ())
    ):
        raise QualificationFailure(
            f"translation_qualification_case_{fixture['id']}"
        )
    position = 0
    for value in fixture.get("ordered", ()):
        match = _phrase_match(output, value, position)
        if match is None:
            raise QualificationFailure(
                f"translation_qualification_case_{fixture['id']}"
            )
        position += match.end()


def _worker_runtime_identity(worker_python: Path) -> dict[str, str]:
    if not worker_python.is_absolute() or not worker_python.is_file():
        raise QualificationFailure("translation_qualification_worker_invalid")
    probe = """
import json
import platform
import ctranslate2
import sentencepiece
import jarvis_orchestrator.translation as translation
import jarvis_orchestrator.translation_worker as worker
print(json.dumps({
    "implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "ctranslate2_version": ctranslate2.__version__,
    "sentencepiece_version": sentencepiece.__version__,
    "translation_module": translation.__file__,
    "worker_module": worker.__file__,
}, sort_keys=True, separators=(",", ":")))
"""
    environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    for key in ("PATH", "SystemRoot", "WINDIR", "TEMP", "TMP", "ComSpec"):
        if os.environ.get(key):
            environment[key] = os.environ[key]
    try:
        completed = subprocess.run(
            [str(worker_python), "-I", "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            env=environment,
        )
        lines = completed.stdout.splitlines()
        if len(lines) != 1:
            raise ValueError("unexpected worker probe output")
        document = json.loads(lines[0], object_pairs_hook=_closed_object)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
        raise QualificationFailure("translation_qualification_worker_invalid") from None
    expected_keys = {
        "implementation",
        "python_version",
        "ctranslate2_version",
        "sentencepiece_version",
        "translation_module",
        "worker_module",
    }
    if (
        not isinstance(document, dict)
        or set(document) != expected_keys
        or document["implementation"] != "CPython"
        or document["ctranslate2_version"] != TRANSLATION_CTRANSLATE2_VERSION
        or document["sentencepiece_version"] != TRANSLATION_SENTENCEPIECE_VERSION
    ):
        raise QualificationFailure("translation_qualification_worker_invalid")
    paths: dict[str, Path] = {}
    for name in ("translation_module", "worker_module"):
        value = document[name]
        if not isinstance(value, str):
            raise QualificationFailure("translation_qualification_worker_invalid")
        path = Path(value)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise QualificationFailure("translation_qualification_worker_invalid")
        paths[name] = path
    expected_translation = Path(translation_module.__file__).resolve()
    expected_worker = Path(translation_worker_module.__file__).resolve()
    identity = {
        "implementation": document["implementation"],
        "python_version": document["python_version"],
        "ctranslate2_version": document["ctranslate2_version"],
        "sentencepiece_version": document["sentencepiece_version"],
        "translation_module_sha256": _sha256_file(paths["translation_module"]),
        "worker_module_sha256": _sha256_file(paths["worker_module"]),
    }
    if (
        identity["translation_module_sha256"] != _sha256_file(expected_translation)
        or identity["worker_module_sha256"] != _sha256_file(expected_worker)
    ):
        raise QualificationFailure("translation_qualification_worker_invalid")
    return identity


def _percentile(values: list[float], percentile: float) -> float:
    selected = sorted(values)
    index = max(0, min(len(selected) - 1, math.ceil(percentile * len(selected)) - 1))
    return selected[index]


def _rss_kib(pid: int) -> int | None:
    if os.name == "posix" and Path("/bin/ps").is_file():
        try:
            result = subprocess.run(
                ["/bin/ps", "-o", "rss=", "-p", str(pid)],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            return int(result.stdout.strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    if platform.system() != "Darwin":
        return None
    try:
        import ctypes

        class ProcTaskInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("total_user", ctypes.c_uint64),
                ("total_system", ctypes.c_uint64),
                ("threads_user", ctypes.c_uint64),
                ("threads_system", ctypes.c_uint64),
                ("policy", ctypes.c_int32),
                ("faults", ctypes.c_int32),
                ("pageins", ctypes.c_int32),
                ("cow_faults", ctypes.c_int32),
                ("messages_sent", ctypes.c_int32),
                ("messages_received", ctypes.c_int32),
                ("syscalls_mach", ctypes.c_int32),
                ("syscalls_unix", ctypes.c_int32),
                ("context_switches", ctypes.c_int32),
                ("thread_count", ctypes.c_int32),
                ("running_threads", ctypes.c_int32),
                ("priority", ctypes.c_int32),
            ]

        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        function = library.proc_pidinfo
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        information = ProcTaskInfo()
        size = ctypes.sizeof(information)
        if function(pid, 4, 0, ctypes.byref(information), size) != size:
            return None
        return int(information.resident_size // 1024)
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        return None


def _children_peak_rss_kib() -> int | None:
    if os.name != "posix" or resource is None:
        return None
    value = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    if value <= 0:
        return None
    return value // 1024 if platform.system() == "Darwin" else value


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def qualify(
    *,
    model_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    build_lock_path: Path,
    corpus_path: Path,
    holdout_path: Path,
    worker_python: Path,
    network_denied: bool,
) -> dict[str, Any]:
    if network_denied is not True:
        raise QualificationFailure("translation_qualification_network_policy_required")
    build_lock_sha256 = _load_build_lock(build_lock_path)
    load_translation_manifest(model_root, manifest_path, manifest_sha256)
    _validate_pack(model_root)
    worker_identity = _worker_runtime_identity(worker_python)
    corpus_sha256 = _sha256_file(corpus_path)
    holdout_sha256 = _sha256_file(holdout_path)
    fixtures = load_fixtures(corpus_path)
    holdout_fixtures = load_fixtures(holdout_path)
    port = OpusCTranslate2TranslationPort(
        model_root=model_root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        timeout_seconds=5.0,
        worker_python=worker_python,
    )
    latencies: list[float] = []
    first_outputs: dict[str, str] = {}
    current_pid = 0
    peak_rss_kib: int | None = None
    steady_rss_baseline_kib: int | None = None
    steady_rss_growth_kib: int | None = None
    batch_seconds = 0.0
    cancellation_seconds = 0.0
    replacement_seconds = 0.0
    fresh_worker_digest = ""
    holdout_outputs: dict[str, str] = {}
    try:
        started = time.perf_counter()
        await port.start()
        cold_start_seconds = time.perf_counter() - started
        process = port._process
        if process is None or not isinstance(process.pid, int):
            raise QualificationFailure("translation_qualification_worker_missing")
        current_pid = process.pid
        peak_rss_kib = _rss_kib(current_pid)
        single_turns = 0
        for run in range(QUALIFICATION_DETERMINISTIC_RUNS):
            for fixture in fixtures:
                before = time.perf_counter()
                result = await port.translate(
                    task_id=f"qualification-{run}",
                    epoch=run + 1,
                    items={"current": fixture["source"]},
                )
                elapsed = time.perf_counter() - before
                latencies.append(elapsed)
                output = result["current"]
                _validate_fixture(fixture, output)
                if run == 0:
                    first_outputs[fixture["id"]] = output
                elif first_outputs[fixture["id"]] != output:
                    raise QualificationFailure(
                        "translation_qualification_nondeterministic"
                    )
                single_turns += 1
                selected_rss = _rss_kib(current_pid)
                if selected_rss is not None:
                    peak_rss_kib = max(peak_rss_kib or 0, selected_rss)
                    if steady_rss_baseline_kib is None:
                        steady_rss_baseline_kib = selected_rss
        while single_turns < QUALIFICATION_SINGLE_TURNS:
            fixture = fixtures[single_turns % len(fixtures)]
            before = time.perf_counter()
            result = await port.translate(
                task_id="qualification-soak",
                epoch=QUALIFICATION_DETERMINISTIC_RUNS + 1,
                items={"current": fixture["source"]},
            )
            latencies.append(time.perf_counter() - before)
            output = result["current"]
            _validate_fixture(fixture, output)
            if first_outputs[fixture["id"]] != output:
                raise QualificationFailure("translation_qualification_nondeterministic")
            single_turns += 1
            selected_rss = _rss_kib(current_pid)
            if selected_rss is not None:
                peak_rss_kib = max(peak_rss_kib or 0, selected_rss)
        final_rss_kib = _rss_kib(current_pid)
        if steady_rss_baseline_kib is not None and final_rss_kib is not None:
            steady_rss_growth_kib = max(0, final_rss_kib - steady_rss_baseline_kib)

        batch_fixtures = (
            fixtures[34],
            {
                "id": "history-narrative",
                "source": "Σταμάτησε να λειτουργεί χθες.",
                "exact": "It stopped working yesterday.",
            },
            fixtures[3],
            fixtures[29],
            fixtures[26],
            fixtures[46],
            fixtures[47],
            fixtures[15],
        )
        batch_identifiers = (
            "current",
            "history.0.user",
            "history.0.assistant",
            "recall.0",
            "recall.1",
            "recall.2",
            "recall.3",
            "recall.4",
        )
        batch_items = {
            identifier: fixture["source"]
            for identifier, fixture in zip(
                batch_identifiers,
                batch_fixtures,
                strict=True,
            )
        }
        before = time.perf_counter()
        batch_result = await port.translate(
            task_id="qualification-production-batch",
            epoch=QUALIFICATION_DETERMINISTIC_RUNS + 2,
            items=batch_items,
        )
        batch_seconds = time.perf_counter() - before
        if tuple(batch_result) != batch_identifiers:
            raise QualificationFailure("translation_qualification_batch_invalid")
        for identifier, fixture in zip(
            batch_identifiers,
            batch_fixtures,
            strict=True,
        ):
            _validate_fixture(fixture, batch_result[identifier])
            expected = first_outputs.get(fixture["id"], fixture.get("exact"))
            if batch_result[identifier] != expected:
                raise QualificationFailure("translation_qualification_batch_drift")

        for fixture in holdout_fixtures:
            before = time.perf_counter()
            result = await port.translate(
                task_id="qualification-holdout",
                epoch=QUALIFICATION_DETERMINISTIC_RUNS + 3,
                items={"current": fixture["source"]},
            )
            latencies.append(time.perf_counter() - before)
            output = result["current"]
            _validate_fixture(fixture, output)
            holdout_outputs[fixture["id"]] = output

        old_pid = current_pid
        long_source = " ".join(
            ["Μετάφρασε αυτή τη δοκιμή προσεκτικά."] * 50
        )
        pending = asyncio.create_task(
            port.translate(
                task_id="qualification-cancel",
                epoch=QUALIFICATION_DETERMINISTIC_RUNS + 4,
                items={"current": long_source},
            )
        )
        await asyncio.sleep(0.01)
        if pending.done():
            raise QualificationFailure(
                "translation_qualification_cancel_not_exercised"
            )
        cancellation_started = time.perf_counter()
        pending.cancel()
        try:
            await pending
        except asyncio.CancelledError:
            pass
        cancellation_seconds = time.perf_counter() - cancellation_started
        replacement_started = time.perf_counter()
        replacement = await port.translate(
            task_id="qualification-replacement",
            epoch=QUALIFICATION_DETERMINISTIC_RUNS + 5,
            items={"current": "Το σύστημα είναι έτοιμο."},
        )
        replacement_seconds = time.perf_counter() - replacement_started
        _validate_fixture(
            {
                "id": "replacement",
                "source": "Το σύστημα είναι έτοιμο.",
                "required_all": ["system", "ready"],
            },
            replacement["current"],
        )
        process = port._process
        if (
            process is None
            or not isinstance(process.pid, int)
            or process.pid == old_pid
            or _pid_is_alive(old_pid)
        ):
            raise QualificationFailure("translation_qualification_cancel_failed")
        current_pid = process.pid
        fresh_outputs: dict[str, str] = {}
        for fixture in fixtures[:8]:
            result = await port.translate(
                task_id="qualification-fresh-worker",
                epoch=QUALIFICATION_DETERMINISTIC_RUNS + 6,
                items={"current": fixture["source"]},
            )
            output = result["current"]
            _validate_fixture(fixture, output)
            if output != first_outputs[fixture["id"]]:
                raise QualificationFailure("translation_qualification_restart_drift")
            fresh_outputs[fixture["id"]] = output
        fresh_worker_digest = hashlib.sha256(
            canonical_json(fresh_outputs).encode()
        ).hexdigest()
        warm_p95 = _percentile(latencies, 0.95)
        warm_max = max(latencies)
        if cold_start_seconds > 2.0:
            raise QualificationFailure("translation_qualification_cold_start_failed")
        if warm_p95 > 0.65:
            raise QualificationFailure("translation_qualification_warm_p95_failed")
        if warm_max > 0.9:
            raise QualificationFailure("translation_qualification_warm_max_failed")
        if batch_seconds > 2.0:
            raise QualificationFailure(
                "translation_qualification_batch_latency_failed"
            )
        if cancellation_seconds > 0.25:
            raise QualificationFailure(
                "translation_qualification_cancel_latency_failed"
            )
        if replacement_seconds > 2.0:
            raise QualificationFailure(
                "translation_qualification_replacement_latency_failed"
            )
    finally:
        await port.close()
    child_peak_rss_kib = _children_peak_rss_kib()
    if child_peak_rss_kib is not None:
        peak_rss_kib = max(peak_rss_kib or 0, child_peak_rss_kib)
    if peak_rss_kib is not None and peak_rss_kib > 750 * 1024:
        raise QualificationFailure("translation_qualification_memory_failed")
    if (
        steady_rss_growth_kib is not None
        and steady_rss_growth_kib > 25 * 1024
    ):
        raise QualificationFailure("translation_qualification_memory_growth_failed")
    if platform.system() == "Darwin" and steady_rss_growth_kib is None:
        raise QualificationFailure("translation_qualification_memory_probe_failed")
    if current_pid and _pid_is_alive(current_pid):
        raise QualificationFailure("translation_qualification_worker_leaked")
    if _sha256_file(corpus_path) != corpus_sha256:
        raise QualificationFailure("translation_qualification_corpus_changed")
    if _sha256_file(holdout_path) != holdout_sha256:
        raise QualificationFailure("translation_qualification_holdout_changed")
    return {
        "version": QUALIFICATION_VERSION,
        "status": "passed",
        "model_id": TRANSLATION_MODEL_ID,
        "revision": TRANSLATION_MODEL_REVISION,
        "manifest_sha256": manifest_sha256,
        "build_lock_sha256": build_lock_sha256,
        "corpus_sha256": corpus_sha256,
        "fixture_count": len(fixtures),
        "holdout_sha256": holdout_sha256,
        "holdout_fixture_count": len(holdout_fixtures),
        "holdout_output_digest": hashlib.sha256(
            canonical_json(holdout_outputs).encode()
        ).hexdigest(),
        "deterministic_runs": QUALIFICATION_DETERMINISTIC_RUNS,
        "single_turns": single_turns,
        "output_digest": hashlib.sha256(
            canonical_json(first_outputs).encode()
        ).hexdigest(),
        "fresh_worker_digest": fresh_worker_digest,
        "cold_start_seconds": round(cold_start_seconds, 6),
        "warm_mean_seconds": round(statistics.fmean(latencies), 6),
        "warm_p95_seconds": round(_percentile(latencies, 0.95), 6),
        "warm_max_seconds": round(max(latencies), 6),
        "production_batch_components": len(batch_identifiers),
        "production_batch_seconds": round(batch_seconds, 6),
        "peak_worker_rss_kib": peak_rss_kib,
        "steady_rss_growth_kib": steady_rss_growth_kib,
        "cancellation_seconds": round(cancellation_seconds, 6),
        "replacement_seconds": round(replacement_seconds, 6),
        "cancellation_replaced_worker": True,
        "network_policy": "external_deny_network",
        "latency_limits_seconds": {
            "cold_start": 2.0,
            "warm_p95": 0.65,
            "warm_max": 0.9,
            "production_batch": 2.0,
            "cancellation": 0.25,
            "replacement": 2.0,
        },
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "worker_python_sha256": _sha256_file(worker_python),
        "worker_runtime": worker_identity,
        "qualification_script_sha256": _sha256_file(Path(__file__).resolve()),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--model-root", type=Path, required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--manifest-sha256", required=True)
    result.add_argument("--build-lock", type=Path, required=True)
    result.add_argument("--corpus", type=Path, required=True)
    result.add_argument("--holdout", type=Path, required=True)
    result.add_argument("--worker-python", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--network-denied", action="store_true")
    return result


def _write_report(path: Path, report: dict[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise QualificationFailure("translation_qualification_output_invalid")
    raw = canonical_json(report).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise QualificationFailure("translation_qualification_output_invalid")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parsed = parser().parse_args(argv)
    try:
        report = asyncio.run(
            qualify(
                model_root=parsed.model_root,
                manifest_path=parsed.manifest,
                manifest_sha256=parsed.manifest_sha256,
                build_lock_path=parsed.build_lock,
                corpus_path=parsed.corpus,
                holdout_path=parsed.holdout,
                worker_python=parsed.worker_python,
                network_denied=parsed.network_denied,
            )
        )
        _write_report(parsed.output, report)
    except QualificationFailure as exc:
        print(canonical_json({"status": "failed", "code": exc.code}))
        return 2
    except BaseException:
        print(
            canonical_json(
                {"status": "failed", "code": "translation_qualification_failed"}
            )
        )
        return 2
    print(
        canonical_json(
            {
                "status": "passed",
                "fixture_count": report["fixture_count"],
                "report_sha256": _sha256_file(parsed.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
