from __future__ import annotations

import os
import hashlib
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import test_private_stack as fixtures
from jarvis_orchestrator import stack_config as stack


class StackConfigBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PrivateStackConfigTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.private_file("jarvis.private.json", self.fixture.document())

    def test_stack_json_rejects_deep_and_oversized_documents(self):
        for raw in (
            b"[" * 2000 + b"0" + b"]" * 2000,
            b" " * (4 * 1024 * 1024) + fixtures.canonical(self.fixture.document()),
        ):
            with self.subTest(size=len(raw)):
                self.fixture.stack_path.write_bytes(raw)
                with self.assertRaises(stack.StackConfigError):
                    stack.PrivateStackConfig.load(self.fixture.stack_path)

    def test_component_reports_have_strict_json_and_typed_shape_errors(self):
        current = stack.PrivateStackConfig.load(self.fixture.stack_path)
        for raw in (
            b"[]",
            b"null",
            b"[" * 2000 + b"0" + b"]" * 2000,
            b'{"status":"fail","status":"pass","identity":{"manifest_digest":"'
            + hashlib.sha256(fixtures.canonical(self.fixture.manifest_document))
            .hexdigest()
            .encode()
            + b'"}}',
        ):
            with self.subTest(raw_size=len(raw)):
                self.fixture.report.write_bytes(raw)
                with self.assertRaises(stack.StackConfigError):
                    stack.PrivateStackConfig.load(self.fixture.stack_path)
                with self.assertRaises(stack.StackConfigError):
                    current.expected_inference_runtime_identity()

    def test_unencodable_json_has_a_typed_error_before_publication(self):
        target = self.fixture.root / "unencodable.json"
        with self.assertRaises(stack.StackConfigError):
            stack.write_private_stack_config(target, {"invalid": "\ud800"})
        self.assertFalse(target.exists())
        self.fixture.manifest.write_bytes(
            b'{"alias":"jarvis-local","invalid":"\\ud800"}'
        )
        with self.assertRaises(stack.StackConfigError):
            stack.PrivateStackConfig.load(self.fixture.stack_path)

    def test_runtime_identity_rejects_failed_report_and_non_hex_digest(self):
        current = stack.PrivateStackConfig.load(self.fixture.stack_path)
        for report in (
            {"status": "fail", "identity": {"manifest_digest": "a" * 64}},
            {"status": "pass", "identity": {"manifest_digest": "z" * 64}},
        ):
            with self.subTest(report=report):
                self.fixture.report.write_bytes(fixtures.canonical(report))
                with self.assertRaises(stack.StackConfigError):
                    current.expected_inference_runtime_identity()

    def test_published_cleanup_preserves_replacement_and_removes_own_output(self):
        original_load = stack.PrivateStackConfig.load
        target = self.fixture.root / "new-stack.json"
        for replace in (False, True):
            with self.subTest(replace=replace):

                def fail_final(path):
                    if Path(path) == target:
                        if replace:
                            target.rename(self.fixture.root / "original-published.json")
                            target.write_bytes(b"competing writer")
                        raise stack.StackConfigError("synthetic failure")
                    return original_load(path)

                with patch.object(
                    stack.PrivateStackConfig, "load", side_effect=fail_final
                ):
                    with self.assertRaises(stack.StackConfigError):
                        stack.write_private_stack_config(
                            target, self.fixture.document()
                        )
                if replace:
                    self.assertEqual(target.read_bytes(), b"competing writer")
                else:
                    self.assertFalse(target.exists())
                self.assertEqual(list(self.fixture.root.glob(".new-stack.json.*")), [])

    @unittest.skipUnless(os.name == "posix", "POSIX leaf symlink replacement")
    def test_path_resolution_does_not_hide_a_replaced_leaf(self):
        path = self.fixture.runner
        replacement = self.fixture.root / "other-runner.py"
        replacement.write_bytes(b"other runner")
        real_lstat = Path.lstat
        swapped = False

        def swap_after_stat(selected, *args, **kwargs):
            nonlocal swapped
            before = real_lstat(selected, *args, **kwargs)
            if selected == path and not swapped:
                swapped = True
                path.unlink()
                path.symlink_to(replacement)
            return before

        with patch.object(Path, "lstat", swap_after_stat):
            with self.assertRaises(stack.StackConfigError):
                stack._absolute_file(str(path), "inference_runner_unavailable")

    def test_artifact_growth_stops_at_opened_size_plus_one(self):
        from jarvis_orchestrator import readiness_evidence as evidence
        from jarvis_orchestrator.operations import OperationalError

        path = self.fixture.runner
        initial_size = path.stat().st_size
        original_read = os.read
        received = 0

        def grow_after_read(descriptor, maximum):
            nonlocal received
            chunk = original_read(descriptor, maximum)
            if chunk and received == 0:
                with path.open("ab") as target:
                    target.write(b"growth" * 100)
            received += len(chunk)
            return chunk

        with patch.object(os, "read", side_effect=grow_after_read):
            with self.assertRaises(OperationalError):
                evidence.sha256_regular_artifact(path)
        self.assertLessEqual(received, initial_size + 1)

    def test_artifact_snapshot_handles_windows_synthetic_execute_bits(self):
        from types import SimpleNamespace
        from jarvis_orchestrator.readiness_evidence import _snapshot

        self.assertEqual(
            _snapshot(SimpleNamespace(st_mode=0o100700, st_file_attributes=0)),
            _snapshot(SimpleNamespace(st_mode=0o100600, st_file_attributes=0)),
        )
        self.assertNotEqual(
            _snapshot(SimpleNamespace(st_mode=0o100700)),
            _snapshot(SimpleNamespace(st_mode=0o100600)),
        )

    @unittest.skipUnless(os.name == "posix", "POSIX FIFO boundaries")
    def test_file_readers_reject_fifo_substitution_without_blocking(self):
        script = """import os,sys
from pathlib import Path
from unittest.mock import patch
import test_private_stack as fixtures
from jarvis_orchestrator import stack_config as s
from jarvis_orchestrator import readiness_evidence as r
from jarvis_orchestrator.operations import OperationalError
f=fixtures.PrivateStackConfigTests();f.setUp()
try:
 f.private_file('jarvis.private.json',f.document()); config=s.PrivateStackConfig.load(f.stack_path)
 mode=sys.argv[1]
 if mode=='stack': target=f.stack_path; action=lambda:s.PrivateStackConfig.load(target)
 elif mode=='manifest': target=f.manifest; action=config.validate
 elif mode=='identity': target=f.report; action=config.expected_inference_runtime_identity
 elif mode=='hash': target=f.runner; action=lambda:s._sha256_file(target)
 else: target=f.report; action=lambda:r._read_regular_file(target,maximum_bytes=1024)
 real_os_open=os.open;real_path_open=Path.open;swapped=False
 def swap(path):
  global swapped
  if Path(path).name==target.name and not swapped:
   swapped=True;target.unlink();os.mkfifo(target,0o600)
 def os_open(path,*args,**kwargs):swap(path);return real_os_open(path,*args,**kwargs)
 def path_open(path,*args,**kwargs):swap(path);return real_path_open(path,*args,**kwargs)
 with patch.object(os,'open',os_open),patch.object(Path,'open',path_open):
  try:action()
  except (s.StackConfigError,OperationalError):sys.exit(0)
 sys.exit(1)
finally:f.doCleanups()
"""
        for mode in ("stack", "manifest", "identity", "hash", "readiness"):
            with self.subTest(mode=mode):
                result = subprocess.run(
                    [sys.executable, "-c", script, mode], capture_output=True, timeout=2
                )
                self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
