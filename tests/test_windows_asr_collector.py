from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jarvis_voice.asr_factory import FallbackAsrBackend
from jarvis_voice.backends import AsrScore, LanguageMetadata, TranscriptResult
from jarvis_voice.config import voice_config_from_mapping
from jarvis_voice.errors import BackendError
from jarvis_voice.manifest import model_manifest_for_profile
from jarvis_voice.transcription import TranscriptionOutcome
from jarvis_voice.windows.asr_collector import (
    CollectorRunner,
    GpuObservation,
    NativeGpuProbe,
    NativePowerProbe,
    REPORT_FIELDS,
    _atomic_write_report,
)
from jarvis_voice.windows.asr_collector_contract import (
    AsrContract,
    CollectorEnvironment,
    CollectorFailure,
    CollectorInputs,
    FixtureContract,
    PowerContract,
    _is_local_windows_absolute,
    _windows_drive_is_local,
    load_environment,
)
from jarvis_voice.windows.asr_collector_metrics import word_errors


class _Clock:
    def __init__(self, *, sleep_scale: float = 1.0) -> None:
        self.value = 0.0
        self.sleep_scale = sleep_scale

    def __call__(self) -> float:
        self.value += 0.01
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds * self.sleep_scale


def _result(entry: dict, *, wrong_language: bool = False, wrong_text: bool = False) -> TranscriptResult:
    expected = "el" if entry["locale"] == "el-GR" else "en"
    language = ("en" if expected == "el" else "el") if wrong_language else expected
    text = "unrelated output" if wrong_text else " ".join(entry["_words"])
    return TranscriptResult(
        text=text,
        language=LanguageMetadata(language, 0.99),
        asr_score=AsrScore(0.9, -0.1),
    )


class _CudaBackend:
    residency = "turn_scoped"

    def __init__(self, *, wrong_languages: int = 0, wrong_texts: int = 0) -> None:
        self.wrong_languages = wrong_languages
        self.wrong_texts = wrong_texts

    def transcribe(self, entry: dict) -> TranscriptResult:
        ordinal = int(entry["case_id"].rsplit("-", 1)[1])
        return _result(
            entry,
            wrong_language=entry["locale"] == "en-US" and ordinal <= self.wrong_languages,
            wrong_text=entry["locale"] == "el-GR" and ordinal <= self.wrong_texts,
        )

    def close(self) -> None:
        return None


class _CpuBackend:
    def __init__(self, *, wrong_language: bool = False, wrong_text: bool = False) -> None:
        self.wrong_language = wrong_language
        self.wrong_text = wrong_text

    def transcribe(self, entry: dict) -> TranscriptResult:
        return _result(
            entry,
            wrong_language=self.wrong_language,
            wrong_text=self.wrong_text,
        )


class _Factory:
    def __init__(
        self,
        *,
        wrong_languages: int = 0,
        wrong_texts: int = 0,
        cpu_wrong_language: bool = False,
        cpu_wrong_text: bool = False,
    ) -> None:
        self.wrong_languages = wrong_languages
        self.wrong_texts = wrong_texts
        self.cpu_wrong_language = cpu_wrong_language
        self.cpu_wrong_text = cpu_wrong_text

    def create(self, _config: object) -> FallbackAsrBackend:
        return FallbackAsrBackend(
            _CudaBackend(
                wrong_languages=self.wrong_languages,
                wrong_texts=self.wrong_texts,
            ),
            _CpuBackend(
                wrong_language=self.cpu_wrong_language,
                wrong_text=self.cpu_wrong_text,
            ),
            enabled=True,
            fallbackable_codes=frozenset({"cuda.insufficient_free_vram"}),
        )


class _Coordinator:
    def __init__(
        self,
        backend: object,
        *,
        stale: bool = False,
        duplicate: bool = False,
    ) -> None:
        self.backend = backend
        self.stale = stale
        self.duplicate = duplicate
        self.pending: TranscriptionOutcome | None = None
        self._repeat: TranscriptionOutcome | None = None

    def submit(self, turn_id: str, audio: dict) -> None:
        try:
            result = self.backend.transcribe(audio)
        except BackendError as exc:
            self.pending = TranscriptionOutcome(turn_id, error=exc)
        else:
            observed = "stale-turn" if self.stale else turn_id
            self.pending = TranscriptionOutcome(observed, result=result)

    def poll(self) -> TranscriptionOutcome | None:
        if self.pending is None and self._repeat is not None:
            result, self._repeat = self._repeat, None
            return result
        result, self.pending = self.pending, None
        if self.duplicate and result is not None:
            self._repeat = result
        return result

    def cancel_current(self) -> None:
        self.pending = None

    def close(self) -> None:
        self.backend.close()


class _Gateway:
    def __init__(self, duration: float = 1.0) -> None:
        self.duration = duration

    def perform(self, _case_index: int, _units: int, _locale: str) -> float:
        return self.duration


class _Work:
    def __init__(self, clock: _Clock, *, overlaps: bool, duration: float = 1.0) -> None:
        self.clock = clock
        self.overlaps = overlaps
        self.duration = duration
        self.started_at = 0.0
        self.completed_at: float | None = None

    def start(self) -> None:
        self.started_at = self.clock()

    def active(self) -> bool:
        return self.overlaps and self.completed_at is None

    def wait(self, _timeout: float) -> float:
        self.completed_at = self.clock()
        return self.duration


class _GpuProbe:
    def __init__(self, *, free_vram_mib: int = 4096, throttle_reasons: int = 0) -> None:
        self.free_vram_mib = free_vram_mib
        self.throttle_reasons = throttle_reasons

    def observe(self, _device_index: int) -> GpuObservation:
        return GpuObservation(
            self.free_vram_mib,
            "test-driver",
            360.0,
            "applications_graphics_mhz=2500;applications_memory_mhz=14000",
            self.throttle_reasons,
        )


class _PowerProbe:
    def assert_matches(self, _expected: object, _device_index: int) -> GpuObservation:
        return _GpuProbe().observe(0)


def _entries() -> list[dict]:
    result = []
    for language, locale in (("el", "el-GR"), ("en", "en-US")):
        for ordinal in range(1, 101):
            words = (
                ("δοκιμή", "αναφοράς", str(ordinal))
                if language == "el"
                else ("reference", "test", str(ordinal))
            )
            result.append(
                {
                    "case_id": f"asr-{language}-{ordinal:03d}",
                    "locale": locale,
                    "_words": words,
                }
            )
    return result


def _inputs(root: Path) -> CollectorInputs:
    voice_config = voice_config_from_mapping(
        {
            "maximum_utterance_seconds": 30,
            "whisper_model_directory": str(root / "model"),
            "asr": {
                "model_profile": "base",
                "execution_profile": "cuda",
                "cpu_compute_type": "int8",
                "cuda": {
                    "device_index": 0,
                    "compute_type": "int8_float16",
                    "minimum_free_vram_mb": 2048,
                    "reserve_for_other_workloads_mb": 2048,
                    "fallback_to_cpu": True,
                    "worker_start_timeout_seconds": 120,
                    "transcription_timeout_seconds": 180,
                    "warm_restart_limit": 0,
                    "vram_poll_interval_ms": 100,
                },
            },
        }
    )
    environment = CollectorEnvironment(
        root / "report.json",
        root / "corpus.json",
        root / "fixture.json",
        root / "power.json",
        root / "asr.json",
        root / "System32" / "nvidia-smi.exe",
        root / "System32" / "powercfg.exe",
        "http://127.0.0.1:8766/v1",
        "k" * 32,
        "f" * 64,
    )
    return CollectorInputs(
        environment=environment,
        fixture=FixtureContract(100, 200, 512, (512, 2048, 4096), 80, 2000, 95, 20),
        power=PowerContract(
            "12345678-1234-1234-1234-123456789abc",
            360.0,
            "test-driver",
            "CUDA 12.8",
            "applications_graphics_mhz=2500;applications_memory_mhz=14000",
        ),
        asr=AsrContract(
            {},
            "a" * 64,
            voice_config,
            model_manifest_for_profile("base"),
            root / "model",
            0,
            100,
        ),
        corpus={"entries": _entries()},
        corpus_digest="b" * 64,
        raw_identities=(),
    )


class WindowsAsrCollectorTests(unittest.TestCase):
    def _run(
        self,
        *,
        wrong_languages: int = 0,
        wrong_texts: int = 0,
        overlaps: int = 100,
        stale: bool = False,
        duplicate: bool = False,
        cpu_wrong_language: bool = False,
        cpu_wrong_text: bool = False,
        free_vram_mib: int = 4096,
        throttle_reasons: int = 0,
        baseline_llm_duration: float = 1.0,
        coexistence_llm_duration: float = 1.0,
        lead_in_scale: float = 1.0,
    ) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            clock = _Clock(sleep_scale=lead_in_scale)
            coordinator_factory = lambda backend: _Coordinator(
                backend,
                stale=stale,
                duplicate=duplicate,
            )
            work_factory = lambda _gateway, index, _units, _locale: _Work(
                clock,
                overlaps=index < overlaps,
                duration=coexistence_llm_duration,
            )
            runner = CollectorRunner(
                _inputs(root),
                asr_factory=lambda: _Factory(
                    wrong_languages=wrong_languages,
                    wrong_texts=wrong_texts,
                    cpu_wrong_language=cpu_wrong_language,
                    cpu_wrong_text=cpu_wrong_text,
                ),
                coordinator_factory=coordinator_factory,
                gateway=_Gateway(baseline_llm_duration),
                gpu_probe=_GpuProbe(
                    free_vram_mib=free_vram_mib,
                    throttle_reasons=throttle_reasons,
                ),
                power_probe=_PowerProbe(),
                work_factory=work_factory,
                clock=clock,
                sleep=clock.sleep,
            )
            with (
                patch(
                    "jarvis_voice.windows.asr_collector.load_audio",
                    side_effect=lambda _inputs, entry: entry,
                ),
                patch(
                    "jarvis_voice.windows.asr_collector.load_reference_words",
                    side_effect=lambda _inputs, entry: tuple(
                        word.casefold() for word in entry["_words"]
                    ),
                ),
                patch(
                    "jarvis_voice.windows.asr_collector.normalize_hypothesis_words",
                    side_effect=lambda text: tuple(
                        word.casefold() for word in text.split()
                    ),
                ),
                patch(
                    "jarvis_voice.windows.asr_collector.select_coexistence_entries",
                    side_effect=lambda inputs: (
                        inputs.corpus["entries"][:50]
                        + inputs.corpus["entries"][100:150]
                    ),
                ),
                patch("jarvis_voice.windows.asr_collector.revalidate_inputs"),
            ):
                return runner.run()

    def test_fake_full_run_exercises_fixed_content_free_contract(self) -> None:
        report = self._run()
        self.assertEqual(set(report), REPORT_FIELDS)
        self.assertEqual(report["greek_utterances"], 100)
        self.assertEqual(report["english_utterances"], 100)
        self.assertEqual(report["greek_language_correct"], 100)
        self.assertEqual(report["english_language_correct"], 100)
        self.assertEqual(report["greek_word_errors"], 0)
        self.assertEqual(report["english_word_errors"], 0)
        self.assertEqual(report["forced_cpu_fallbacks"], 5)
        self.assertEqual(report["overlapping_utterances"], 100)
        self.assertFalse(
            any(isinstance(value, str) and len(value) != 64 for value in report.values())
        )

    def test_language_wer_overlap_and_stale_fail_closed(self) -> None:
        for kwargs in (
            {"wrong_languages": 6},
            {"wrong_texts": 21},
            {"overlaps": 79},
            {"stale": True},
            {"duplicate": True},
            {"cpu_wrong_language": True},
            {"cpu_wrong_text": True},
            {"free_vram_mib": 2047},
            {"throttle_reasons": 1},
            {"coexistence_llm_duration": 1.151},
            {"lead_in_scale": 0.9},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(CollectorFailure):
                self._run(**kwargs)

    def test_word_error_metric_has_deterministic_edit_distance(self) -> None:
        self.assertEqual(word_errors(("a", "b", "c"), ("a", "x", "c")), 1)
        self.assertEqual(word_errors(("a", "b"), ("a", "x", "b")), 1)
        self.assertEqual(word_errors(("a", "b"), ("a",)), 1)

    def test_native_probes_bind_compute_clock_driver_power_and_throttle(self) -> None:
        gpu_csv = (
            "0, 4096, test-driver, 360, 2500, 14000, "
            "Not Active, Not Active, Not Active, Not Active\n"
        )
        nvidia_smi = Path("/trusted/System32/nvidia-smi.exe")
        powercfg = Path("/trusted/System32/powercfg.exe")
        observed_commands: list[tuple[str, ...]] = []

        def gpu_runner(arguments: object) -> str:
            observed_commands.append(tuple(arguments))
            return gpu_csv

        gpu = NativeGpuProbe(nvidia_smi, runner=gpu_runner)
        observation = gpu.observe(0)
        self.assertEqual(observed_commands[0][0], str(nvidia_smi))
        self.assertEqual(observation.free_vram_mib, 4096)
        self.assertEqual(observation.throttle_reasons, 0)
        self.assertEqual(
            observation.clock_policy,
            "applications_graphics_mhz=2500;applications_memory_mhz=14000",
        )
        throttled = NativeGpuProbe(
            nvidia_smi,
            runner=lambda _arguments: gpu_csv.replace(
                "Not Active", "Active", 1
            ),
        ).observe(0)
        self.assertEqual(throttled.throttle_reasons, 1)

        responses = {
            # Greek Windows powercfg output follows the active OEM code page,
            # not a guaranteed UTF-8 console encoding.  Only the ASCII GUID is
            # authoritative and must remain parseable from bounded bytes.
            (str(powercfg), "/getactivescheme"): (
                b"\xc4\xef\xea\xe9\xec\xde: "
                b"12345678-1234-1234-1234-123456789abc"
            ),
            (str(nvidia_smi),): b"NVIDIA-SMI 555.1 CUDA Version: 12.8",
        }
        power = NativePowerProbe(
            gpu,
            powercfg,
            nvidia_smi,
            command_runner=lambda arguments: responses[tuple(arguments)],
            ac_reader=lambda: True,
        )
        expected = PowerContract(
            "12345678-1234-1234-1234-123456789abc",
            360.0,
            "test-driver",
            "CUDA 12.8",
            "applications_graphics_mhz=2500;applications_memory_mhz=14000",
        )
        self.assertEqual(power.assert_matches(expected, 0), observation)
        throttled_power = NativePowerProbe(
            NativeGpuProbe(
                nvidia_smi,
                runner=lambda _arguments: gpu_csv.replace(
                    "Not Active", "Active", 1
                ),
            ),
            powercfg,
            nvidia_smi,
            command_runner=lambda arguments: responses[tuple(arguments)],
            ac_reader=lambda: True,
        )
        with self.assertRaises(CollectorFailure):
            throttled_power.assert_matches(expected, 0)
        with self.assertRaises(CollectorFailure):
            power.assert_matches(
                PowerContract(
                    expected.scheme_guid,
                    expected.power_limit_watts,
                    expected.driver_version,
                    "CUDA 12.9",
                    expected.gpu_clock_policy,
                ),
                0,
            )

    def test_unknown_throttle_state_cannot_be_reported_as_inactive(self) -> None:
        for position in range(6, 10):
            for unknown in ("", "N/A", "[N/A]", "Unsupported", "Unknown"):
                with self.subTest(position=position, unknown=unknown):
                    fields = [
                        "0", "4096", "test-driver", "360", "2500", "14000",
                        "Not Active", "Not Active", "Not Active", "Not Active",
                    ]
                    fields[position] = unknown
                    gpu = NativeGpuProbe(
                        Path("/trusted/System32/nvidia-smi.exe"),
                        runner=lambda _arguments: ",".join(fields) + "\n",
                    )
                    with self.assertRaisesRegex(CollectorFailure, "gpu_probe_invalid"):
                        gpu.observe(0)

    def test_default_collector_inventory_uses_the_validated_absolute_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = _inputs(Path(directory))
            runner = CollectorRunner(
                inputs,
                gateway=_Gateway(),
                gpu_probe=_GpuProbe(),
                power_probe=_PowerProbe(),
            )
            factory = runner.asr_factory()
            inventory = factory.inventory_factory()
            self.assertEqual(
                inventory._query[0],
                str(inputs.environment.nvidia_smi),
            )
            self.assertNotEqual(inventory._query[0], "nvidia-smi")

    def test_environment_is_exact_and_requires_a_32_byte_bearer(self) -> None:
        self.assertTrue(_is_local_windows_absolute(r"C:\CYRUS\qualification"))
        self.assertFalse(_is_local_windows_absolute(r"\\server\share\qualification"))
        self.assertFalse(_is_local_windows_absolute(r"\\?\C:\qualification"))
        self.assertFalse(
            _windows_drive_is_local(
                r"Z:\qualification",
                drive_type_reader=lambda _root: 4,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = {}
            for name in (
                "CORPUS",
                "FIXTURE",
                "POWER_PROFILE",
                "ASR_CONFIG",
                "NVIDIA_SMI",
                "POWERCFG",
            ):
                paths[name] = root / (name.casefold() + ".json")
                paths[name].write_text("{}", encoding="utf-8")
            environment = {
                "CYRUS_ASR_QUALIFICATION_OUTPUT": str(root / "report.json"),
                **{
                    "CYRUS_ASR_QUALIFICATION_" + name: str(path)
                    for name, path in paths.items()
                },
                "CYRUS_ASR_QUALIFICATION_BASE_URL": "http://127.0.0.1:8766/v1",
                "CYRUS_ASR_QUALIFICATION_API_KEY": "k" * 32,
                "CYRUS_ASR_QUALIFICATION_FIXTURE_DIGEST": "f" * 64,
            }
            self.assertEqual(load_environment(environment).api_key, "k" * 32)
            environment["CYRUS_ASR_QUALIFICATION_API_KEY"] = "x"
            with self.assertRaises(CollectorFailure):
                load_environment(environment)
            environment["CYRUS_ASR_QUALIFICATION_API_KEY"] = "k" * 32
            environment["CYRUS_ASR_QUALIFICATION_EXTRA"] = "x"
            with self.assertRaises(CollectorFailure):
                load_environment(environment)

    def test_report_is_atomic_no_overwrite_and_source_is_sealed(self) -> None:
        report = self._run()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            _atomic_write_report(output, report)
            first = output.read_bytes()
            with self.assertRaises(CollectorFailure):
                _atomic_write_report(output, report)
            self.assertEqual(output.read_bytes(), first)
        root = Path(__file__).resolve().parents[1]
        entry = (root / "packaging" / "windows_asr_collector_entry.py").read_text()
        spec = (root / "packaging" / "windows_asr_collector.spec").read_text()
        build = (root / "scripts" / "build_windows.ps1").read_text()
        self.assertIn("len(sys.argv) == 1", entry)
        self.assertIn("console=True", spec)
        self.assertIn('"jarvis_inference.windows_asr_corpus"', spec)
        self.assertIn("CYRUS_WINDOWS_CUDA_BUNDLE_MANIFEST", spec)
        self.assertIn("is_cuda_runtime_name", spec)
        self.assertIn('binaries += [(str(item.path), ".")', spec)
        self.assertIn("REQUIRED_CUDA_DLLS", entry)
        self.assertIn("ctypes.WinDLL(str(candidate))", entry)
        self.assertIn("get_cuda_device_count", entry)
        self.assertIn("except Exception:", entry)
        self.assertIn("return 2", entry)
        self.assertIn("1.2.1:4.8.1:6.21.0", build)
        contract = (
            root / "src" / "jarvis_voice" / "windows" / "asr_collector_contract.py"
        ).read_text()
        self.assertIn("GetWindowsDirectoryW", contract)
        self.assertIn("SHGetKnownFolderPath", contract)
        self.assertNotIn('os.environ.get("ProgramW6432")', contract)
        self.assertNotIn("status\": \"pass", entry + spec)


if __name__ == "__main__":
    unittest.main()
