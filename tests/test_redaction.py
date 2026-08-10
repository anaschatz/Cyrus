from __future__ import annotations

import json
import unittest

import _bootstrap  # noqa: F401

from jarvis_voice.doctor import doctor_report
from jarvis_voice.errors import ProtocolError


class RedactionTests(unittest.TestCase):
    def test_structured_error_does_not_include_transcript_or_audio(self) -> None:
        transcript = "μυστικό transcript"; audio = "sensitive-audio-bytes"
        error = ProtocolError("protocol.invalid_field", f"{transcript} {audio}", safe_context={"field": "payload.text"})
        record = json.dumps(error.to_log_record(), ensure_ascii=False)
        self.assertNotIn(transcript, record); self.assertNotIn(audio, record)

    def test_doctor_does_not_echo_sensitive_model_path(self) -> None:
        path = "/private/sensitive/user/model"
        report = json.dumps(doctor_report(path))
        self.assertNotIn(path, report)
        self.assertFalse(json.loads(report)["network_used"])


if __name__ == "__main__":
    unittest.main()
