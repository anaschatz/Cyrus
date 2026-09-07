from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
import urllib.request
import urllib.response
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.compact_greek_tts_acquisition import _download_artifact, _download_url_allowed
from jarvis_voice.macos.compact_greek_tts_pilot import ArtifactSpec


class CompactRedirectTests(unittest.TestCase):
    def test_redirect_hops_are_checked_before_transport_and_responses_close(self):
        source = "https://huggingface.co/example/artifact"
        data = b"pinned fixture"
        spec = ArtifactSpec("artifact.bin", len(data), hashlib.sha256(data).hexdigest(), source)
        for status in (301, 302, 303, 307, 308):
            for allowed in (False, True):
                with self.subTest(status=status, allowed=allowed), tempfile.TemporaryDirectory() as temporary:
                    target = "https://release-assets.githubusercontent.com/file" if allowed else "https://forbidden.invalid/file"
                    requests, responses = [], []
                    def transport(handler, request):
                        requests.append(request.full_url)
                        headers = Message()
                        if len(requests) == 1:
                            headers["Location"] = target
                        response = urllib.response.addinfourl(io.BytesIO(b"" if len(requests) == 1 else data), headers, request.full_url, status if len(requests) == 1 else 200)
                        response.msg = "fixture"
                        responses.append(response)
                        return response
                    destination = Path(temporary) / "artifact.bin"
                    with patch.object(urllib.request.HTTPSHandler, "https_open", transport):
                        if allowed:
                            _download_artifact(spec, destination)
                            self.assertEqual(destination.read_bytes(), data)
                        else:
                            with self.assertRaisesRegex(BackendError, "redirect_forbidden"):
                                _download_artifact(spec, destination)
                            self.assertFalse(destination.exists())
                    self.assertEqual(requests, [source, target] if allowed else [source])
                    self.assertTrue(all(response.closed for response in responses))

    def test_malformed_url_is_not_allowed(self):
        self.assertFalse(_download_url_allowed("https://[invalid"))


if __name__ == "__main__":
    unittest.main()
