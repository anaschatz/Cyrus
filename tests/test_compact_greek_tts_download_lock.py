from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
from jarvis_voice.errors import BackendError
from jarvis_voice.macos import compact_greek_tts_acquisition as acquisition
from jarvis_voice.macos.compact_greek_tts_acquisition import _download_artifact
from jarvis_voice.macos.compact_greek_tts_pilot import ArtifactSpec
from test_compact_greek_tts_acquisition import _Response


class CompactDownloadLockTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "artifact.bin"
        self.data = b"pinned fixture"
        self.url = "https://huggingface.co/example/artifact"
        self.spec = ArtifactSpec("artifact.bin", len(self.data), hashlib.sha256(self.data).hexdigest(), self.url)

    def test_competing_resume_and_restart_cannot_publish_corrupt_bytes(self):
        self.path.with_suffix(".bin.part").write_bytes(self.data[:3])
        reading, second_ready, release_second = threading.Event(), threading.Event(), threading.Event()
        class Resume(_Response):
            def read(self, size):
                reading.set()
                if not second_ready.wait(5):
                    raise TimeoutError("second writer did not finish")
                return super().read(size)
        class Restart(_Response):
            def __exit__(self, *args):
                second_ready.set()
                if not release_second.wait(5):
                    raise TimeoutError("first writer did not finish")
                return False
        def second_download():
            try:
                return _download_artifact(self.spec, self.path, urlopen=lambda *a, **k: Restart(self.data, self.url))
            except Exception as exc:
                return exc
            finally:
                second_ready.set()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(_download_artifact, self.spec, self.path, urlopen=lambda *a, **k: Resume(self.data[3:], self.url, status=206, headers={"Content-Range": "bytes 3-13/14"}))
            try:
                self.assertTrue(reading.wait(5))
                second = pool.submit(second_download)
                self.assertEqual(first.result(6)["status"], "downloaded")
            finally:
                release_second.set()
                second_ready.set()
            result = second.result(6)
        self.assertEqual(self.path.read_bytes(), self.data)
        self.assertIsInstance(result, BackendError)
        self.assertEqual(result.code, "tts.f1_compact.download_in_progress")

    def test_lock_excludes_another_process_and_releases_after_interruption(self):
        code = """
import json, sys
from pathlib import Path
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.compact_greek_tts_acquisition import _download_artifact
from jarvis_voice.macos.compact_greek_tts_pilot import ArtifactSpec
from test_compact_greek_tts_acquisition import _Response
spec = ArtifactSpec('artifact.bin', 14, sys.argv[2], sys.argv[3])
try:
    result = _download_artifact(spec, Path(sys.argv[1]), urlopen=lambda *a, **k: _Response(b'pinned fixture', sys.argv[3]))
except BackendError as exc:
    result = {'code': exc.code}
print(json.dumps(result))
"""
        def opener(*args, **kwargs):
            child = subprocess.run([sys.executable, "-c", code, str(self.path), self.spec.sha256, self.url], capture_output=True, text=True, timeout=10, check=True)
            self.assertEqual(json.loads(child.stdout)["code"], "tts.f1_compact.download_in_progress")
            return _Response(self.data, self.url)
        _download_artifact(self.spec, self.path, urlopen=opener)
        self.path.unlink()
        with self.assertRaises(KeyboardInterrupt):
            _download_artifact(self.spec, self.path, urlopen=Mock(side_effect=KeyboardInterrupt))
        _download_artifact(self.spec, self.path, urlopen=lambda *a, **k: _Response(self.data, self.url))
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_windows_lock_branch_closes_handle_on_success_and_failure(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                native = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=0, locking=Mock(side_effect=OSError("busy") if failure else None))
                windows_os = SimpleNamespace(**{**vars(os), "name": "nt"})
                with patch.object(acquisition, "os", windows_os), patch.dict(sys.modules, {"msvcrt": native}):
                    if failure:
                        with self.assertRaisesRegex(BackendError, "download_in_progress"):
                            with acquisition._artifact_lock(self.path):
                                self.fail("busy lock entered")
                    else:
                        with acquisition._artifact_lock(self.path):
                            pass
                descriptor = native.locking.call_args_list[0].args[0]
                self.assertEqual([call.args[1:] for call in native.locking.call_args_list], [(1, 1)] if failure else [(1, 1), (0, 1)])
                with self.assertRaises(OSError):
                    os.fstat(descriptor)

    @unittest.skipIf(os.name == "nt", "POSIX symlink fixture")
    def test_lock_symlink_is_rejected_without_touching_target(self):
        target = self.root / "outside"
        target.write_bytes(b"untouched")
        (self.root / ".artifact.bin.lock").symlink_to(target)
        opener = Mock(return_value=_Response(self.data, self.url))
        with self.assertRaisesRegex(BackendError, "download_lock_unsafe"):
            _download_artifact(self.spec, self.path, urlopen=opener)
        self.assertEqual(target.read_bytes(), b"untouched")
        opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
