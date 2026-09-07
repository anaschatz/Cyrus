from __future__ import annotations

import contextlib
import ctypes
import io
import multiprocessing
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
from jarvis_voice.windows.cuda_bundle import REQUIRED_CUDA_DLLS


class SealedCollectorEntryTests(unittest.TestCase):
    def test_only_an_absolute_frozen_root_can_load_dlls(self):
        source = Path(__file__).resolve().parents[1] / "packaging/windows_asr_collector_entry.py"
        code = compile(source.read_bytes(), str(source), "exec")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name in REQUIRED_CUDA_DLLS:
                (root / name).write_bytes(b"fake DLL; never loaded")
            for location in ("missing", "", ".", None, str(root)):
                with self.subTest(location=location):
                    loader, devices, main = Mock(), Mock(), Mock(return_value=0)
                    output = io.StringIO()
                    with (contextlib.chdir(root), contextlib.redirect_stdout(output), contextlib.redirect_stderr(output),
                          patch.object(sys, "platform", "win32"), patch.object(sys, "frozen", True, create=True),
                          patch.dict(sys.__dict__, {"_MEIPASS": location}), patch.object(sys, "argv", ["collector"]),
                          patch.object(ctypes, "WinDLL", loader, create=True), patch.object(multiprocessing, "freeze_support"),
                          patch.dict(sys.modules, {"ctranslate2": SimpleNamespace(get_cuda_device_count=devices),
                              "jarvis_voice.windows.asr_collector": SimpleNamespace(main=main)})):
                        if location == "missing":
                            del sys._MEIPASS
                        namespace = {"__name__": "sealed_entry_test"}
                        exec(code, namespace)
                        result = namespace["_entrypoint"]()
                    self.assertEqual(output.getvalue(), "")
                    if location == str(root):
                        self.assertEqual(result, 0)
                        self.assertEqual([call.args[0] for call in loader.call_args_list],
                                         [str(root / name) for name in sorted(REQUIRED_CUDA_DLLS)])
                        devices.assert_called_once_with()
                        main.assert_called_once_with()
                    else:
                        self.assertEqual(result, 2)
                        loader.assert_not_called()
                        devices.assert_not_called()
                        main.assert_not_called()

    def test_import_failure_and_extra_arguments_stay_content_free(self):
        source = Path(__file__).resolve().parents[1] / "packaging/windows_asr_collector_entry.py"
        code = compile(source.read_bytes(), str(source), "exec")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name in REQUIRED_CUDA_DLLS:
                (root / name).write_bytes(b"fake DLL; never loaded")
            for failure in (True, False):
                with self.subTest(failure=failure):
                    output = io.StringIO()
                    main = Mock(return_value=0)
                    with (contextlib.redirect_stdout(output), contextlib.redirect_stderr(output),
                          patch.object(sys, "platform", "win32"), patch.object(sys, "frozen", True, create=True),
                          patch.object(sys, "_MEIPASS", str(root), create=True),
                          patch.object(sys, "argv", ["collector"] if failure else ["collector", "unexpected"]),
                          patch.object(ctypes, "WinDLL", side_effect=OSError("private native path") if failure else None, create=True),
                          patch.object(multiprocessing, "freeze_support"),
                          patch.dict(sys.modules, {"ctranslate2": SimpleNamespace(get_cuda_device_count=Mock()),
                              "jarvis_voice.windows.asr_collector": SimpleNamespace(main=main)})):
                        namespace = {"__name__": "sealed_entry_test"}
                        exec(code, namespace)
                        self.assertEqual(namespace["_entrypoint"](), 2)
                    self.assertEqual(output.getvalue(), "")
                    main.assert_not_called()


if __name__ == "__main__":
    unittest.main()
