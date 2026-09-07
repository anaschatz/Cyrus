from __future__ import annotations

import io
import os
import threading
import time
import unittest
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jarvis_inference.contracts import Capabilities, normalize_request
from jarvis_inference.engine import EngineFailure, EngineProtocolFailure, PreparedPrompt
from jarvis_inference.http_backend import BackendHTTP
from jarvis_inference.mlx_engine import MlxWorkerEngine


class BackendJsonTests(unittest.TestCase):
    def test_isolated_mlx_worker_disables_bytecode_with_an_interpreter_flag(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "worker.py"
            worker.write_text(
                "import json, sys; print(json.dumps(sys.dont_write_bytecode))"
            )
            (root / "models").mkdir()
            manifest = SimpleNamespace(
                root=root,
                python={"executable": {"path": sys.executable}},
                backend={"worker": {"path": str(worker)}},
                model={"artifacts": [{"path": "models"}]},
            )
            engine = MlxWorkerEngine(manifest)

            with (
                patch("jarvis_inference.mlx_engine.verify_artifacts"),
                patch(
                    "jarvis_inference.mlx_engine.resolve_local", side_effect=lambda _root, value, *args, **kwargs: root / value
                ),
                patch("jarvis_inference.mlx_engine.subprocess.Popen") as launch,
                patch.object(engine, "_exchange", return_value={"event": "ready"}),
            ):
                engine.start()
            argv = launch.call_args.args[0]
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=5,
                env=launch.call_args.kwargs["env"],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "true")

    def test_http_backend_rejects_excessively_nested_json(self) -> None:
        raw = b"[" * 2000 + b"0" + b"]" * 2000
        with patch("jarvis_inference.http_backend.urllib.request.urlopen", return_value=io.BytesIO(raw)):
            with self.assertRaises(EngineProtocolFailure):
                BackendHTTP(12345, "test").request("GET", "/health", None, time.monotonic() + 1)

    def test_mlx_worker_rejects_excessively_nested_json(self) -> None:
        engine = MlxWorkerEngine(SimpleNamespace())
        reader, writer = os.pipe()
        with os.fdopen(reader, "rb") as source, os.fdopen(writer, "wb", buffering=0) as target:
            engine.process = SimpleNamespace(stdout=source)
            target.write(b"[" * 2000 + b"0" + b"]" * 2000 + b"\n")
            with self.assertRaises(EngineProtocolFailure):
                engine._read(time.monotonic() + 1)

    def test_mlx_pipe_reads_buffered_events_and_times_out_partial_lines(self) -> None:
        engine = MlxWorkerEngine(SimpleNamespace())
        reader, writer = os.pipe()
        with os.fdopen(reader, "rb") as source, os.fdopen(writer, "wb", buffering=0) as target:
            engine.process = SimpleNamespace(stdout=source)
            target.write(b'{"event":"ready"}\n{"event":"idle"}\n')
            self.assertEqual(engine._read(time.monotonic() + 1), {"event": "ready"})
            with patch.object(engine, "_terminate") as terminate:
                self.assertEqual(engine._read(time.monotonic() + .1), {"event": "idle"})
                target.write(b'{"event":')
                # Closing the writer eventually also bounds a broken reader,
                # so a regression cannot leave this test hung indefinitely.
                closer = threading.Timer(.5, target.close)
                closer.start()
                try:
                    with self.assertRaisesRegex(EngineFailure, "deadline expired"):
                        engine._read(time.monotonic() + .05)
                    terminate.assert_called_once_with("timeout")
                finally:
                    closer.cancel()
                    closer.join()

    def test_mlx_stop_uses_earliest_match_regardless_of_list_order(self) -> None:
        request = normalize_request(
            {"model": "jarvis-local", "messages": [{"role": "user", "content": "x"}], "stop": ["late", "early"]},
            alias="jarvis-local", capabilities=Capabilities(), context_window=32,
        )
        engine = MlxWorkerEngine(SimpleNamespace())
        events = [
            {"event": "text", "text": "prefix early middle late suffix"},
            {"event": "done", "finish_reason": "stop", "completion_tokens": 3},
        ]
        with patch.object(engine, "_send"), patch.object(engine, "_read", side_effect=events):
            output = list(engine.generate(PreparedPrompt((1,), "x", request), threading.Event(), time.monotonic() + 1))
        self.assertEqual(output[0].text, "prefix ")
