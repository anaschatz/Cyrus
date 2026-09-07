from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jarvis_inference.durable import durable_replace, read_report
from jarvis_inference.limits import MANIFEST_BYTES
from jarvis_inference.profile_report import _safe_bytes, ProfileQualificationFailure
from jarvis_inference.windows_primary_fixture import _safe_regular_file, WindowsPrimaryFixtureFailure
from jarvis_inference.windows_primary_report import _safe_regular_file as _read_skeleton
from jarvis_inference.windows_asr_corpus import _safe_regular as _read_corpus, WindowsAsrCorpusError
from jarvis_inference.windows_fixture import _safe_file as _read_asset, FixtureFailure


class DurableReportTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX special files")
    def test_qualification_readers_reject_fifo_swap_without_blocking_or_leaking(
        self,
    ) -> None:
        readers = (
            (lambda path: _safe_bytes(path, "profile"), ProfileQualificationFailure),
            (
                lambda path: _safe_regular_file(path, maximum=524_288, name="fixture"),
                WindowsPrimaryFixtureFailure,
            ),
            (_read_skeleton, WindowsPrimaryFixtureFailure),
            (
                lambda path: _read_corpus(path, maximum=524_288, name="corpus"),
                WindowsAsrCorpusError,
            ),
            (
                lambda path: _read_asset(path, maximum=524_288, name="asset"),
                FixtureFailure,
            ),
        )
        original = os.open
        for reader, error in readers:
            with (
                self.subTest(reader=reader),
                tempfile.TemporaryDirectory() as directory,
            ):
                target = Path(directory).resolve() / "report.json"
                target.write_bytes(b"old")
                descriptors = []

                def swap(path, flags):
                    self.assertTrue(flags & os.O_NONBLOCK)
                    target.unlink()
                    os.mkfifo(target)
                    descriptor = original(path, flags)
                    descriptors.append(descriptor)
                    return descriptor

                with patch("jarvis_inference.manifest.os.open", swap):
                    with self.assertRaises(error):
                        reader(target)
                self.assertEqual(len(descriptors), 1)
                with self.assertRaises(OSError):
                    os.fstat(descriptors[0])

    def test_profile_readers_preserve_their_distinct_size_limits(self) -> None:
        readers = (
            (
                lambda path: _safe_bytes(path, "profile"),
                2_097_152,
                ProfileQualificationFailure,
            ),
            (
                lambda path: _safe_regular_file(path, maximum=524_288, name="fixture"),
                524_288,
                WindowsPrimaryFixtureFailure,
            ),
            (_read_skeleton, 65_536, WindowsPrimaryFixtureFailure),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            for reader, maximum, error in readers:
                with self.subTest(maximum=maximum):
                    valid = b"x" * maximum
                    path.write_bytes(valid)
                    self.assertEqual(reader(path), valid)
                    for invalid in (b"", valid + b"x"):
                        path.write_bytes(invalid)
                        with self.assertRaises(error):
                            reader(path)

    def test_profile_readers_reject_a_leaf_replaced_after_inspection(self) -> None:
        readers = (
            (lambda path: _safe_bytes(path, "profile"), ProfileQualificationFailure),
            (
                lambda path: _safe_regular_file(path, maximum=524_288, name="fixture"),
                WindowsPrimaryFixtureFailure,
            ),
            (_read_skeleton, WindowsPrimaryFixtureFailure),
        )
        original = Path.is_symlink
        for reader, error in readers:
            with (
                self.subTest(reader=reader),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                target = root / "report.json"
                other = root / "other.json"
                target.write_bytes(b"old")
                other.write_bytes(b"new")
                probe = root / "probe"
                try:
                    probe.symlink_to(other)
                except (OSError, NotImplementedError):
                    self.skipTest("symlinks are unavailable")
                probe.unlink()

                def swap(path):
                    result = original(path)
                    if path == target and not result:
                        target.unlink()
                        target.symlink_to(other)
                    return result

                with patch.object(Path, "is_symlink", swap):
                    with self.assertRaises(error):
                        reader(target)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX special files")
    def test_nonregular_report_is_rejected_without_blocking_open(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            fifo = Path(root) / "report.json"
            os.mkfifo(fifo)
            with patch.object(Path, "open", side_effect=AssertionError("unsafe buffered open")):
                self.assertIsNone(read_report(fifo))

    def test_invalid_json_does_not_leak_a_temporary_file_descriptor(self) -> None:
        descriptors = []
        original = tempfile.mkstemp

        def capture(*args, **kwargs):
            descriptor, name = original(*args, **kwargs)
            descriptors.append(descriptor)
            return descriptor, name

        try:
            with tempfile.TemporaryDirectory() as root:
                path = Path(root) / "report.json"
                with patch("jarvis_inference.durable.tempfile.mkstemp", side_effect=capture):
                    with self.assertRaises(ValueError):
                        durable_replace(path, {"value": float("nan")})
                self.assertFalse(path.exists())
                for descriptor in descriptors:
                    with self.assertRaises(OSError):
                        os.fstat(descriptor)
        finally:
            for descriptor in descriptors:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def test_deep_or_oversized_report_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "report.json"
            for raw in (b"[" * 2000 + b"0" + b"]" * 2000, b" " * (MANIFEST_BYTES + 1)):
                with self.subTest(size=len(raw)):
                    path.write_bytes(raw)
                    self.assertIsNone(read_report(path))
