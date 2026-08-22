from __future__ import annotations

import asyncio
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_orchestrator.context_assembler import ContextAssemblyError
from jarvis_orchestrator.domain import canonical_json
from jarvis_orchestrator.translation import (
    TRANSLATION_BACKEND,
    TRANSLATION_COMPUTE_TYPE,
    TRANSLATION_CONVERSION_POLICY_SHA256,
    TRANSLATION_CTRANSLATE2_VERSION,
    TRANSLATION_LICENSE,
    TRANSLATION_MANIFEST_VERSION,
    TRANSLATION_MODEL_ID,
    TRANSLATION_MODEL_REVISION,
    TRANSLATION_READY_VERSION,
    TRANSLATION_SENTENCEPIECE_VERSION,
    OpusCTranslate2TranslationPort,
    contains_greek,
    load_translation_manifest,
)
from jarvis_orchestrator.translation_worker import translate_document
from jarvis_orchestrator import translation_worker


class FakeStdin:
    def __init__(self, process) -> None:
        self.process = process

    def write(self, raw: bytes) -> None:
        self.process.raw = raw

    async def drain(self) -> None:
        return None


class FakeStdout:
    def __init__(self, process) -> None:
        self.process = process
        self.reads = 0

    async def readline(self) -> bytes:
        self.reads += 1
        if self.reads == 1:
            if isinstance(self.process.ready, BaseException):
                raise self.process.ready
            return self.process.ready
        self.process.entered.set()
        if self.process.block:
            await asyncio.Event().wait()
        return self.process.responder(self.process.raw)


class FakeProcess:
    def __init__(
        self,
        responder,
        *,
        block: bool = False,
        ready: bytes | BaseException | None = None,
    ) -> None:
        self.responder = responder
        self.block = block
        self.returncode = None
        self.entered = asyncio.Event()
        self.terminated = False
        self.killed = False
        self.raw = b""
        self.stdin = FakeStdin(self)
        self.stdout = FakeStdout(self)
        self.spawned = 0
        self.ready = ready or (
            canonical_json(
                {"version": TRANSLATION_READY_VERSION, "status": "ready"}
            )
            + "\n"
        ).encode()

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class SlowStopProcess(FakeProcess):
    def __init__(self, responder) -> None:
        super().__init__(responder)
        self.wait_entered = asyncio.Event()
        self.release = asyncio.Event()

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.release.set()

    async def wait(self) -> int:
        self.wait_entered.set()
        await self.release.wait()
        assert self.returncode is not None
        return self.returncode


class TranslationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model_root = self.root / "model"
        self.model_root.mkdir()
        self.worker_python = self.root / "python"
        self.worker_python.write_bytes(b"python")
        self.files: list[dict[str, object]] = []
        for name in (
            "config.json",
            "model.bin",
            "shared_vocabulary.json",
            "source.spm",
            "target.spm",
        ):
            path = self.model_root / name
            if name == "shared_vocabulary.json":
                path.write_bytes(
                    canonical_json(
                        [
                            "<unk>",
                            "</s>",
                            "▁Το",
                            "▁σύστημα",
                            "▁δοκιμή",
                            "▁Ρα",
                            "ντε",
                            "β",
                            "ού",
                            "▁αύριο",
                            ".",
                        ]
                    ).encode()
                )
            else:
                path.write_bytes(("artifact:" + name).encode())
            raw = path.read_bytes()
            self.files.append(
                {
                    "path": name,
                    "size": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        self.manifest_path = self.root / "manifest.json"
        self.manifest_document = {
            "version": TRANSLATION_MANIFEST_VERSION,
            "backend": TRANSLATION_BACKEND,
            "model_id": TRANSLATION_MODEL_ID,
            "revision": TRANSLATION_MODEL_REVISION,
            "license": TRANSLATION_LICENSE,
            "source_language": "el",
            "target_language": "en",
            "compute_type": TRANSLATION_COMPUTE_TYPE,
            "ctranslate2_version": TRANSLATION_CTRANSLATE2_VERSION,
            "sentencepiece_version": TRANSLATION_SENTENCEPIECE_VERSION,
            "conversion_policy_sha256": TRANSLATION_CONVERSION_POLICY_SHA256,
            "beam_size": 4,
            "inter_threads": 1,
            "intra_threads": 1,
            "max_queued_batches": 1,
            "files": self.files,
            "total_size": sum(int(item["size"]) for item in self.files),
        }
        self.write_manifest()

    def write_manifest(self) -> str:
        raw = canonical_json(self.manifest_document).encode()
        self.manifest_path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def port(self, process, *, timeout_seconds: float = 2):
        async def factory(*args, **kwargs):
            process.spawned += 1
            self.assertIn("-I", args)
            self.assertEqual(kwargs["stderr"], asyncio.subprocess.DEVNULL)
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
            return process

        return OpusCTranslate2TranslationPort(
            model_root=self.model_root,
            manifest_path=self.manifest_path,
            manifest_sha256=self.write_manifest(),
            timeout_seconds=timeout_seconds,
            worker_python=self.worker_python,
            process_factory=factory,
        )

    def test_manifest_verifies_exact_closed_artifact_set(self) -> None:
        manifest = load_translation_manifest(
            self.model_root,
            self.manifest_path,
            self.write_manifest(),
        )
        self.assertEqual(manifest.model_id, TRANSLATION_MODEL_ID)
        self.assertEqual(manifest.total_size, self.manifest_document["total_size"])

        unexpected = self.model_root / "runtime-download.bin"
        unexpected.write_bytes(b"forbidden")
        with self.assertRaisesRegex(ValueError, "artifact set"):
            load_translation_manifest(
                self.model_root,
                self.manifest_path,
                self.write_manifest(),
            )

    def test_manifest_rejects_extra_directories_and_unsorted_files(self) -> None:
        (self.model_root / "empty").mkdir()
        with self.assertRaisesRegex(ValueError, "artifact set"):
            load_translation_manifest(
                self.model_root,
                self.manifest_path,
                self.write_manifest(),
            )
        (self.model_root / "empty").rmdir()

        self.manifest_document["files"] = list(reversed(self.files))
        with self.assertRaisesRegex(ValueError, "not sorted"):
            load_translation_manifest(
                self.model_root,
                self.manifest_path,
                self.write_manifest(),
            )

    def test_manifest_rejects_changed_bytes_and_symlinks(self) -> None:
        (self.model_root / "model.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            load_translation_manifest(
                self.model_root,
                self.manifest_path,
                self.write_manifest(),
            )

        (self.model_root / "model.bin").unlink()
        (self.model_root / "model.bin").symlink_to(self.model_root / "config.json")
        with self.assertRaisesRegex(ValueError, "missing"):
            load_translation_manifest(
                self.model_root,
                self.manifest_path,
                self.write_manifest(),
            )

    def test_manifest_rejects_duplicate_keys_and_boolean_policy_values(self) -> None:
        raw = canonical_json(self.manifest_document).encode()
        duplicated = raw.replace(
            b'"beam_size":4',
            b'"beam_size":4,"beam_size":4',
        )
        self.manifest_path.write_bytes(duplicated)
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            load_translation_manifest(
                self.model_root,
                self.manifest_path,
                hashlib.sha256(duplicated).hexdigest(),
            )

        self.manifest_document["beam_size"] = True
        with self.assertRaisesRegex(ValueError, "policy"):
            load_translation_manifest(
                self.model_root,
                self.manifest_path,
                self.write_manifest(),
            )

    def test_port_preserves_order_and_accepts_only_english_output(self) -> None:
        def responder(raw: bytes) -> bytes:
            request = json.loads(raw)
            response = {
                "version": "jarvis.input-translation-response/1",
                "request_id": request["request_id"],
                "items": [
                    {"id": item["id"], "text": "The system is ready."}
                    for item in request["items"]
                ],
            }
            return (canonical_json(response) + "\n").encode()

        process = FakeProcess(responder)
        port = self.port(process)

        async def exercise():
            await port.start()
            await port.start()
            first = await port.translate(
                task_id="task",
                epoch=1,
                items={"current": "Το σύστημα είναι έτοιμο."},
            )
            second = await port.translate(
                task_id="task",
                epoch=1,
                items={"current": "Το σύστημα είναι έτοιμο."},
            )
            await port.close()
            return first, second

        result, repeated = asyncio.run(exercise())
        self.assertEqual(result, {"current": "The system is ready."})
        self.assertEqual(repeated, result)
        self.assertFalse(contains_greek(result["current"]))
        self.assertEqual(process.spawned, 1)
        self.assertTrue(process.terminated)

    def test_port_rejects_malformed_ready_record_and_stops_worker(self) -> None:
        process = FakeProcess(lambda raw: b"", ready=b"{}\n")
        port = self.port(process)

        with self.assertRaises(ContextAssemblyError) as raised:
            asyncio.run(port.start())

        self.assertEqual(raised.exception.code, "input_translation_unavailable")
        self.assertTrue(process.terminated)

    def test_port_rejects_unterminated_ready_record(self) -> None:
        ready = canonical_json(
            {"version": TRANSLATION_READY_VERSION, "status": "ready"}
        ).encode()
        process = FakeProcess(lambda raw: b"", ready=ready)

        with self.assertRaises(ContextAssemblyError) as raised:
            asyncio.run(self.port(process).start())

        self.assertEqual(raised.exception.code, "input_translation_unavailable")
        self.assertTrue(process.terminated)

    def test_port_maps_readiness_io_failure_and_stops_worker(self) -> None:
        process = FakeProcess(lambda raw: b"", ready=OSError("broken pipe"))
        port = self.port(process)

        with self.assertRaises(ContextAssemblyError) as raised:
            asyncio.run(port.start())

        self.assertEqual(raised.exception.code, "input_translation_unavailable")
        self.assertTrue(process.terminated)

    def test_port_timeout_stops_worker(self) -> None:
        process = FakeProcess(lambda raw: b"", block=True)
        port = self.port(process, timeout_seconds=0.1)

        with self.assertRaises(ContextAssemblyError) as raised:
            asyncio.run(
                port.translate(
                    task_id="task",
                    epoch=1,
                    items={"current": "Ελληνικά"},
                )
            )

        self.assertEqual(raised.exception.code, "input_translation_timeout")
        self.assertTrue(process.terminated)

    def test_port_restarts_cleanly_after_cancellation(self) -> None:
        first = FakeProcess(lambda raw: b"", block=True)

        def responder(raw: bytes) -> bytes:
            request = json.loads(raw)
            return (
                canonical_json(
                    {
                        "version": "jarvis.input-translation-response/1",
                        "request_id": request["request_id"],
                        "items": [{"id": "current", "text": "Ready."}],
                    }
                )
                + "\n"
            ).encode()

        second = FakeProcess(responder)
        processes = iter((first, second))

        async def factory(*args, **kwargs):
            del args, kwargs
            return next(processes)

        port = OpusCTranslate2TranslationPort(
            model_root=self.model_root,
            manifest_path=self.manifest_path,
            manifest_sha256=self.write_manifest(),
            timeout_seconds=2,
            worker_python=self.worker_python,
            process_factory=factory,
        )

        async def exercise() -> dict[str, str]:
            task = asyncio.create_task(
                port.translate(
                    task_id="first",
                    epoch=1,
                    items={"current": "Πρώτο"},
                )
            )
            await first.entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            result = await port.translate(
                task_id="second",
                epoch=2,
                items={"current": "Δεύτερο"},
            )
            await port.close()
            return dict(result)

        result = asyncio.run(exercise())
        self.assertEqual(result, {"current": "Ready."})
        self.assertTrue(first.terminated)
        self.assertTrue(second.terminated)

    def test_port_rejects_unterminated_response_record(self) -> None:
        def responder(raw: bytes) -> bytes:
            request = json.loads(raw)
            return canonical_json(
                {
                    "version": "jarvis.input-translation-response/1",
                    "request_id": request["request_id"],
                    "items": [{"id": "current", "text": "Ready."}],
                }
            ).encode()

        process = FakeProcess(responder)
        with self.assertRaises(ContextAssemblyError) as raised:
            asyncio.run(
                self.port(process).translate(
                    task_id="task",
                    epoch=1,
                    items={"current": "Έτοιμο"},
                )
            )

        self.assertEqual(raised.exception.code, "input_translation_invalid")
        self.assertTrue(process.terminated)

    def test_close_is_idempotent_and_terminal(self) -> None:
        process = FakeProcess(lambda raw: b"")
        port = self.port(process)

        async def exercise() -> None:
            await port.close()
            await port.close()
            with self.assertRaises(ContextAssemblyError) as raised:
                await port.translate(
                    task_id="late",
                    epoch=1,
                    items={"current": "Αργά"},
                )
            self.assertEqual(
                raised.exception.code,
                "input_translation_unavailable",
            )

        asyncio.run(exercise())
        self.assertEqual(process.spawned, 0)

    def test_cancelled_close_keeps_worker_owned_until_reaped(self) -> None:
        process = SlowStopProcess(lambda raw: b"")
        port = self.port(process)

        async def exercise() -> None:
            await port.start()
            closing = asyncio.create_task(port.close())
            await process.wait_entered.wait()
            closing.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await closing
            self.assertIs(port._process, process)
            self.assertTrue(process.killed)
            await port.close()
            self.assertIsNone(port._process)

        asyncio.run(exercise())

    def test_idle_worker_exit_is_reported_by_monitor(self) -> None:
        process = FakeProcess(lambda raw: b"")
        port = self.port(process)

        async def exercise() -> None:
            await port.start()
            with self.assertRaisesRegex(RuntimeError, "worker exited"):
                await port.monitor()
            await port.close()

        asyncio.run(exercise())

    def test_port_rejects_greek_or_unknown_output(self) -> None:
        for bad_text in (
            "Το σύστημα είναι έτοιμο.",
            "The <unk> system",
            "   ",
            "...",
            "Привет",
            "你好",
            "The \ufffd system",
            "The \u2047 system",
        ):
            with self.subTest(bad_text=bad_text):
                def responder(raw: bytes, selected=bad_text) -> bytes:
                    request = json.loads(raw)
                    return (canonical_json({
                        "version": "jarvis.input-translation-response/1",
                        "request_id": request["request_id"],
                        "items": [{"id": "current", "text": selected}],
                    }) + "\n").encode()

                with self.assertRaises(ContextAssemblyError) as raised:
                    asyncio.run(
                        self.port(FakeProcess(responder)).translate(
                            task_id="task",
                            epoch=1,
                            items={"current": "Ελληνικά"},
                        )
                    )
                self.assertEqual(raised.exception.code, "input_translation_invalid")

    def test_cancellation_terminates_owned_worker(self) -> None:
        process = FakeProcess(lambda raw: b"", block=True)
        port = self.port(process)

        async def exercise() -> None:
            task = asyncio.create_task(
                port.translate(
                    task_id="task",
                    epoch=1,
                    items={"current": "Ελληνικά"},
                )
            )
            await process.entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(exercise())
        self.assertTrue(process.terminated)

    def test_worker_uses_exact_marian_eos_and_decode_contract(self) -> None:
        captured: dict[str, object] = {}

        class Processor:
            def __init__(self, *, model_file: str) -> None:
                self.source = model_file.endswith("source.spm")

            def encode(self, text: str, *, out_type):
                self.assertions(text, out_type)
                return ["▁Το", "▁σύστημα"]

            @staticmethod
            def assertions(text: str, out_type) -> None:
                if text != "Το σύστημα" or out_type is not str:
                    raise AssertionError("unexpected tokenizer input")

            def decode_pieces(self, tokens):
                if self.source or tokens != ["▁The", "▁system", "."]:
                    raise AssertionError("unexpected decoder tokens")
                return "The system."

        class Translator:
            def __init__(self, model_root: str, **kwargs) -> None:
                captured["model_root"] = model_root
                captured["constructor"] = kwargs

            def translate_batch(self, tokenized, **kwargs):
                captured["tokenized"] = tokenized
                captured["translate"] = kwargs
                return [
                    types.SimpleNamespace(
                        hypotheses=[["▁The", "▁system", ".", "</s>"]]
                    )
                ]

        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.__version__ = TRANSLATION_CTRANSLATE2_VERSION
        ctranslate2.Translator = Translator
        sentencepiece = types.ModuleType("sentencepiece")
        sentencepiece.__version__ = TRANSLATION_SENTENCEPIECE_VERSION
        sentencepiece.SentencePieceProcessor = Processor
        request = {
            "version": "jarvis.input-translation-request/1",
            "request_id": "request",
            "items": [{"id": "current", "text": "Το σύστημα"}],
        }
        with patch.dict(
            sys.modules,
            {"ctranslate2": ctranslate2, "sentencepiece": sentencepiece},
        ):
            response = translate_document(request, self.model_root)

        self.assertEqual(captured["tokenized"], [["▁Το", "▁σύστημα", "</s>"]])
        self.assertEqual(
            captured["constructor"],
            {
                "device": "cpu",
                "compute_type": "int8",
                "inter_threads": 1,
                "intra_threads": 1,
                "max_queued_batches": 1,
            },
        )
        settings = captured["translate"]
        self.assertEqual(settings["beam_size"], 4)
        self.assertEqual(settings["max_input_length"], 512)
        self.assertEqual(settings["max_decoding_length"], 512)
        self.assertTrue(settings["disable_unk"])
        self.assertTrue(settings["return_end_token"])
        self.assertEqual(response["items"], [{"id": "current", "text": "The system."}])

    def test_worker_resegments_source_piece_missing_from_shared_vocabulary(self) -> None:
        captured: dict[str, object] = {}

        class Processor:
            def __init__(self, *, model_file: str) -> None:
                self.source = model_file.endswith("source.spm")

            def encode(self, text: str, *, out_type):
                del out_type
                if not self.source or text != "Ραντεβού αύριο.":
                    raise AssertionError("unexpected tokenizer input")
                return ["▁Ραντεβού", "▁αύριο", "."]

            def nbest_encode(self, text: str, *, nbest_size: int, out_type):
                del out_type
                if text != "Ραντεβού αύριο." or nbest_size != 128:
                    raise AssertionError("unexpected nbest request")
                return [
                    ["▁Ραντεβού", "▁αύριο", "."],
                    ["▁Ρα", "ντε", "β", "ού", "▁αύριο", "."],
                ]

            def decode_pieces(self, tokens):
                if self.source or tokens != ["▁Appointment", "▁tomorrow", "."]:
                    raise AssertionError("unexpected decoder tokens")
                return "Appointment tomorrow."

        class Translator:
            def __init__(self, model_root: str, **kwargs) -> None:
                del model_root, kwargs

            def translate_batch(self, tokenized, **kwargs):
                del kwargs
                captured["tokenized"] = tokenized
                return [
                    types.SimpleNamespace(
                        hypotheses=[["▁Appointment", "▁tomorrow", ".", "</s>"]]
                    )
                ]

        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.__version__ = TRANSLATION_CTRANSLATE2_VERSION
        ctranslate2.Translator = Translator
        sentencepiece = types.ModuleType("sentencepiece")
        sentencepiece.__version__ = TRANSLATION_SENTENCEPIECE_VERSION
        sentencepiece.SentencePieceProcessor = Processor
        request = {
            "version": "jarvis.input-translation-request/1",
            "request_id": "request",
            "items": [{"id": "current", "text": "Ραντεβού αύριο."}],
        }
        with patch.dict(
            sys.modules,
            {"ctranslate2": ctranslate2, "sentencepiece": sentencepiece},
        ):
            response = translate_document(request, self.model_root)

        self.assertEqual(
            captured["tokenized"],
            [["▁Ρα", "ντε", "β", "ού", "▁αύριο", ".", "</s>"]],
        )
        self.assertEqual(
            response["items"],
            [{"id": "current", "text": "Appointment tomorrow."}],
        )

    def test_worker_main_reuses_one_engine_for_multiple_requests(self) -> None:
        requests = [
            {
                "version": "jarvis.input-translation-request/1",
                "request_id": selected,
                "items": [{"id": "current", "text": "Δοκιμή"}],
            }
            for selected in ("one", "two")
        ]
        incoming = io.BytesIO(
            b"".join((canonical_json(item) + "\n").encode() for item in requests)
        )
        outgoing = io.BytesIO()
        constructed = 0

        class Endpoint:
            def __init__(self, stream: io.BytesIO) -> None:
                self.buffer = stream

        class Engine:
            def __init__(self, model_root: Path) -> None:
                nonlocal constructed
                del model_root
                constructed += 1

            def translate(self, document):
                return {
                    "version": "jarvis.input-translation-response/1",
                    "request_id": document["request_id"],
                    "items": [{"id": "current", "text": "Test"}],
                }

        with (
            patch.object(translation_worker, "TranslationEngine", Engine),
            patch.object(translation_worker, "load_translation_manifest"),
            patch.object(translation_worker.sys, "stdin", Endpoint(incoming)),
            patch.object(translation_worker.sys, "stdout", Endpoint(outgoing)),
        ):
            result = translation_worker.main(
                [
                    "--model-root",
                    str(self.model_root),
                    "--manifest-path",
                    str(self.manifest_path),
                    "--manifest-sha256",
                    "1" * 64,
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(constructed, 1)
        records = [json.loads(line) for line in outgoing.getvalue().splitlines()]
        self.assertEqual(
            records[0],
            {"version": TRANSLATION_READY_VERSION, "status": "ready"},
        )
        self.assertEqual([item["request_id"] for item in records[1:]], ["one", "two"])

    def test_worker_rejects_incomplete_translation_without_eos(self) -> None:
        class Processor:
            def __init__(self, *, model_file: str) -> None:
                self.source = model_file.endswith("source.spm")

            def encode(self, text: str, *, out_type):
                del text, out_type
                return ["▁δοκιμή"]

            def decode_pieces(self, tokens):
                del tokens
                return "should not decode"

        class Translator:
            def __init__(self, model_root: str, **kwargs) -> None:
                del model_root, kwargs

            def translate_batch(self, tokenized, **kwargs):
                del tokenized, kwargs
                return [types.SimpleNamespace(hypotheses=[["▁incomplete"]])]

        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.__version__ = TRANSLATION_CTRANSLATE2_VERSION
        ctranslate2.Translator = Translator
        sentencepiece = types.ModuleType("sentencepiece")
        sentencepiece.__version__ = TRANSLATION_SENTENCEPIECE_VERSION
        sentencepiece.SentencePieceProcessor = Processor
        request = {
            "version": "jarvis.input-translation-request/1",
            "request_id": "request",
            "items": [{"id": "current", "text": "δοκιμή"}],
        }
        with (
            patch.dict(
                sys.modules,
                {"ctranslate2": ctranslate2, "sentencepiece": sentencepiece},
            ),
            self.assertRaisesRegex(RuntimeError, "did not terminate"),
        ):
            translate_document(request, self.model_root)


if __name__ == "__main__":
    unittest.main()
