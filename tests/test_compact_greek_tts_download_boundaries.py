from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import _bootstrap  # noqa: F401
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.compact_greek_tts_acquisition import _download_artifact
from jarvis_voice.macos.compact_greek_tts_pilot import ArtifactSpec
from test_compact_greek_tts_acquisition import _Response


class CompactDownloadBoundariesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.destination = self.root / "artifact.bin"
        self.partial = self.root / "artifact.bin.part"
        self.data = b"pinned fixture"
        self.source = "https://huggingface.co/example/artifact"
        self.spec = ArtifactSpec("artifact.bin", len(self.data), hashlib.sha256(self.data).hexdigest(), self.source)

    @unittest.skipIf(os.name == "nt", "POSIX symlink fixture")
    def test_dangling_partial_is_rejected_without_writing_its_target(self):
        outside = self.root / "outside"
        self.partial.symlink_to(outside)
        opener = Mock(return_value=_Response(self.data, self.source))
        with self.assertRaisesRegex(BackendError, "partial_unsafe"):
            _download_artifact(self.spec, self.destination, urlopen=opener)
        opener.assert_not_called()
        self.assertFalse(outside.exists())
        self.assertTrue(self.partial.is_symlink())

    def test_complete_partial_is_checked_and_published_without_network(self):
        opener = Mock(side_effect=OSError("network must not be needed"))
        self.partial.write_bytes(b"x" * len(self.data))
        with self.assertRaisesRegex(BackendError, "download_hash_mismatch"):
            _download_artifact(self.spec, self.destination, urlopen=opener)
        self.assertFalse(self.destination.exists())
        self.partial.write_bytes(self.data)
        result = _download_artifact(self.spec, self.destination, urlopen=opener)
        self.assertEqual(result["status"], "downloaded")
        self.assertEqual(self.destination.read_bytes(), self.data)
        self.assertFalse(self.partial.exists())
        opener.assert_not_called()

    def test_existing_partial_supports_range_and_full_response_restart(self):
        for status in (206, 200):
            with self.subTest(status=status):
                self.partial.write_bytes(self.data[:3])
                response = _Response(self.data[3:] if status == 206 else self.data, self.source, status=status, headers={"Content-Range": f"bytes 3-{len(self.data)-1}/{len(self.data)}"})
                def opener(request, timeout):
                    self.assertEqual(request.get_header("Range"), "bytes=3-")
                    return response
                _download_artifact(self.spec, self.destination, urlopen=opener)
                self.assertEqual(self.destination.read_bytes(), self.data)
                self.destination.unlink()

    def test_publication_preserves_destination_created_during_download(self):
        destination = self.destination
        class RacingResponse(_Response):
            def __exit__(self, *args):
                destination.write_bytes(b"another writer")
                return False
        with self.assertRaises(OSError):
            _download_artifact(self.spec, destination, urlopen=lambda *args, **kwargs: RacingResponse(self.data, self.source))
        self.assertEqual(destination.read_bytes(), b"another writer")
        self.assertEqual(self.partial.read_bytes(), self.data)


if __name__ == "__main__":
    unittest.main()
