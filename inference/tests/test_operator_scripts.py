from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from ctypes import wintypes
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jarvis_inference import macos_fixture, windows_asr_corpus, windows_fixture
from jarvis_inference.engine import GenerationEvent
from test_build_windows_manifest import builder as windows_builder
from test_macos_packaging_scripts import builder as macos_builder, load_script

smoke = load_script("smoke_macos_inference")
corpus_cli = load_script("build_windows_asr_audio_corpus")


class OperatorScriptTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires FIFO support")
    def test_operator_hashing_rejects_fifo_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact"
            original_open = os.open

            def substitute(target, flags):
                self.assertTrue(flags & os.O_NONBLOCK)
                path.unlink()
                os.mkfifo(path)
                return original_open(target, flags)

            path.write_bytes(b"original")
            with (
                patch.object(os, "open", side_effect=substitute),
                self.assertRaises(OSError),
            ):
                windows_builder._artifact(path.parent, path)
            path.unlink()
            path.write_bytes(b"original")
            output, errors = io.StringIO(), io.StringIO()
            with (
                patch.object(
                    corpus_cli, "build_corpus_manifest", return_value={"entries": []}
                ),
                patch.object(corpus_cli, "write_corpus_manifest"),
                patch.object(os, "open", side_effect=substitute),
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                status = corpus_cli.main(
                    [
                        "--corpus-root",
                        temporary,
                        "--template",
                        "unused",
                        "--output",
                        str(path),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertFalse(json.loads(errors.getvalue())["written"])

    @unittest.skipUnless(hasattr(os, "fchmod"), "requires POSIX permissions")
    def test_failed_permissions_close_temporary_files(self):
        writers = (
            macos_builder._write,
            windows_builder._write,
            macos_fixture._write_evidence,
            windows_fixture._write_json,
            windows_asr_corpus.write_corpus_manifest,
        )
        for writer in writers:
            with (
                self.subTest(writer=writer.__module__),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary).resolve()
                with patch.object(
                    os, "fchmod", side_effect=OSError("permissions failed")
                ) as chmod:
                    with self.assertRaises((OSError, windows_fixture.FixtureFailure)):
                        writer(root / "output.json", {})
                chmod.assert_called_once()
                descriptor = chmod.call_args.args[0]
                try:
                    with self.assertRaises(OSError):
                        os.fstat(descriptor)
                finally:
                    # Keep the pre-fix reproduction from leaking into later tests.
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                self.assertEqual(list(root.iterdir()), [])

    def test_smoke_reports_one_result_after_closing_engine(self):
        for failure, tokens, expected in (
            (None, 1, True),
            (None, 0, False),
            (None, 25, False),
            ("start", 1, False),
            ("cleanup", 1, False),
            ("close", 1, False),
        ):
            with self.subTest(failure=failure, tokens=tokens):
                output = io.StringIO()
                engine = Mock()
                engine.prepare.return_value = SimpleNamespace(token_ids=(1,))
                engine.generate.return_value = [
                    GenerationEvent("text", text="ναι"),
                    GenerationEvent(
                        "final", finish_reason="stop", completion_tokens=tokens
                    ),
                ]

                def close():
                    self.assertEqual(output.getvalue(), "")
                    if failure == "close":
                        raise OSError("private backend failure")

                engine.close.side_effect = close
                if failure in {"start", "cleanup"}:
                    getattr(engine, failure).side_effect = OSError(
                        "private backend failure"
                    )
                with (
                    patch.object(sys, "argv", ["smoke", "--manifest", "unused.json"]),
                    patch.object(smoke, "load_manifest", return_value=Mock()),
                    patch.object(
                        smoke,
                        "normalize_request",
                        return_value=SimpleNamespace(max_completion_tokens=24),
                    ),
                    patch.object(smoke, "MlxWorkerEngine", return_value=engine),
                    redirect_stdout(output),
                ):
                    status = smoke.main()
                result = json.loads(output.getvalue())
                self.assertEqual(result["ready"], expected)
                self.assertEqual(status, 0 if expected else 2)
                self.assertNotIn("private backend failure", output.getvalue())
                engine.close.assert_called_once()

    def test_dll_hook_preserves_native_error_and_directory_handle(self):
        path = Path(__file__).parents[1] / "packaging/runtime_hook_harden_dll.py"
        source = compile(path.read_text(), str(path), "exec")
        for succeeds in (False, True):
            with (
                self.subTest(succeeds=succeeds),
                tempfile.TemporaryDirectory() as temporary,
            ):
                function = Mock(return_value=succeeds)
                ctypes = SimpleNamespace(
                    WinDLL=Mock(
                        return_value=SimpleNamespace(SetDefaultDllDirectories=function)
                    ),
                    get_last_error=Mock(return_value=5),
                    WinError=Mock(return_value=OSError(5, "native failure")),
                    wintypes=wintypes,
                )
                fake_os = SimpleNamespace(add_dll_directory=Mock())
                modules = {
                    "ctypes": ctypes,
                    "os": fake_os,
                    "sys": SimpleNamespace(
                        platform="win32",
                        _MEIPASS=temporary,
                        executable=sys.executable,
                    ),
                }

                namespace = {}
                with patch.dict(sys.modules, modules):
                    if succeeds:
                        exec(source, namespace)
                        self.assertIs(
                            namespace["_inference_dll_directory_handle"],
                            fake_os.add_dll_directory.return_value,
                        )
                        fake_os.add_dll_directory.assert_called_once_with(
                            str(Path(temporary).resolve())
                        )
                    else:
                        with self.assertRaises(OSError):
                            exec(source, namespace)
                        ctypes.WinError.assert_called_once_with(5)
                        fake_os.add_dll_directory.assert_not_called()
                ctypes.WinDLL.assert_called_once_with("kernel32", use_last_error=True)
                function.assert_called_once_with(0x00001000)
                self.assertEqual(function.argtypes, [wintypes.DWORD])
                self.assertIs(function.restype, wintypes.BOOL)
