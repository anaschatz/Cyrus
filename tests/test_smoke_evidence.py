from __future__ import annotations

import importlib.util
import sqlite3
import tempfile
import unittest
from collections import deque
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from jarvis_voice.protocol import Envelope, EventFactory, new_uuid4


def smoke_modules():
    root = Path(__file__).resolve().parents[1] / "scripts"
    for name, function in (
        ("smoke_orchestrator_memory", "_task_evidence"),
        ("smoke_orchestrator_text", "_durable_evidence"),
    ):
        spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield name, module, function


class SmokeEvidenceTests(unittest.TestCase):

    def test_missing_evidence_database_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, module, function in smoke_modules():
                with self.subTest(script=name):
                    path = Path(directory) / f"{name}.sqlite3"
                    with self.assertRaises(sqlite3.OperationalError):
                        getattr(module, function)(path, "turn")
                    self.assertFalse(path.exists())

    def test_existing_database_with_uri_characters_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence #?.sqlite3"
            with closing(sqlite3.connect(path)) as database:
                database.executescript("""
                    CREATE TABLE tasks(task_id, state, epoch, conversation_id);
                    CREATE TABLE canonical_turns(task_id, turn_id, status);
                    CREATE TABLE task_events(task_id, event_type, payload_json, event_sequence);
                    CREATE TABLE memory_records(conversation_id);
                    CREATE TABLE reservations(task_id, state, actual_output_tokens, created_at);
                    INSERT INTO tasks VALUES('task', 'COMPLETED', 1, 'conversation');
                    INSERT INTO canonical_turns VALUES('task', 'turn', 'COMPLETED');
                    INSERT INTO memory_records VALUES('conversation');
                    INSERT INTO reservations VALUES('task', 'RECONCILED', 2, 1);
                """)
            before = path.read_bytes()
            for name, module, function in smoke_modules():
                with self.subTest(script=name):
                    result = getattr(module, function)(path, "turn")
                    self.assertEqual(result["task_state"], "COMPLETED")
                    if name.endswith("memory"):
                        self.assertEqual(result["memory_record_count"], 1)
                    else:
                        self.assertTrue(result["usage_reconciled"])
            self.assertEqual(path.read_bytes(), before)


class SmokeProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def run_smoke(self, module, evidence_function, fault=None):
        socket = mock.AsyncMock()
        events = deque()
        server = None

        async def authenticate(_socket, _key, hello_json):
            nonlocal server
            server = EventFactory(session_id=Envelope.from_json(hello_json).session_id)
            ready = server.create("server.ready", {"protocol": "jarvis.voice/1"})
            if fault == "ready_session":
                ready = replace(ready, session_id=new_uuid4())
            events.append(ready.to_json())

        async def send(transcript_json):
            transcript = Envelope.from_json(transcript_json)
            response_id = new_uuid4()
            answer = "μπλε" if module.__name__.endswith("memory") else "έτοιμο"
            if fault == "wrong_answer":
                answer = "Δεν είναι έτοιμο."
            frames = [server.create(kind, payload, turn_id=transcript.turn_id,
                                    response_id=response_id) for kind, payload in (
                ("server.ack", {"acked_event_id": transcript.event_id, "status": "accepted"}),
                ("response.started", {"locale": "el-GR"}),
                ("response.text.delta", {"chunk_index": 0, "text": answer}),
                ("response.text.done", {"chunk_count": 1}),
                ("response.speech.commit", {"unit_index": 1, "locale": "el-GR", "text": answer}),
                ("response.speech.done", {"last_unit_index": 1}),
            )]
            if fault == "ack_event":
                frames[0] = replace(frames[0], payload={"acked_event_id": new_uuid4(), "status": "accepted"})
            elif fault == "ack_turn":
                frames[0] = replace(frames[0], turn_id=new_uuid4())
            elif fault in {"ack_response", "started_response", "delta_response", "terminal_response"}:
                index = {"ack_response": 0, "started_response": 1, "delta_response": 2, "terminal_response": 5}[fault]
                frames[index] = replace(frames[index], response_id=new_uuid4())
            elif fault == "event_session":
                frames[2] = replace(frames[2], session_id=new_uuid4())
            unrelated = replace(frames[2], turn_id=new_uuid4(), response_id=new_uuid4()).to_json()
            replay_count = 4097 if fault == "replay_limit" else 1
            events.extend([unrelated] * replay_count)
            events.extend(frame.to_json() for frame in frames)

        async def recv():
            if not events:
                raise RuntimeError("fixture exhausted without completing the smoke")
            return events.popleft()

        socket.send.side_effect = send
        socket.recv.side_effect = recv
        connection = mock.AsyncMock()
        connection.__aenter__.return_value = socket
        evidence = {"task_state": "COMPLETED", "turn_status": "COMPLETED",
                    "memory_record_count": 1, "recall_event": "memory.recall.completed",
                    "recall_result_count": 1, "reservation_count": 1, "usage_reconciled": True}
        with (
            mock.patch("websockets.asyncio.client.connect", return_value=connection),
            mock.patch.object(module, "load_voice_auth_key", return_value=object()),
            mock.patch.object(module, "authenticate_async_client", side_effect=authenticate),
            mock.patch.object(module, evidence_function, return_value=evidence),
        ):
            return await module._run(SimpleNamespace(
                phase="recall", timeout=5.0, url="ws://127.0.0.1:8765",
                database=Path("unused.sqlite3"), voice_auth_key=Path("unused.json"),
            ))

    async def test_ack_and_response_binding(self):
        for name, module, function in smoke_modules():
            with self.subTest(script=name, fault=None):
                result = await self.run_smoke(module, function)
                self.assertTrue(result["accepted_ack"])
                self.assertEqual(result["terminal"], "response.speech.done")
            for fault in ("ack_event", "ack_turn", "ack_response", "started_response",
                          "delta_response", "terminal_response", "ready_session", "event_session"):
                with self.subTest(script=name, fault=fault), self.assertRaises(RuntimeError):
                    await self.run_smoke(module, function, fault)

    async def test_replay_is_bounded(self):
        for name, module, function in smoke_modules():
            with self.subTest(script=name), self.assertRaisesRegex(RuntimeError, "wire replay exceeded"):
                await self.run_smoke(module, function, "replay_limit")

    async def test_text_smoke_requires_requested_word_only(self):
        for name, module, function in smoke_modules():
            if name.endswith("text"):
                with self.assertRaises(RuntimeError):
                    await self.run_smoke(module, function, "wrong_answer")

    async def test_invalid_timeout_fails_before_io(self):
        for name, module, _function in smoke_modules():
            for timeout in (0.0, -1.0, float("inf"), float("nan")):
                with (
                    self.subTest(script=name, timeout=timeout),
                    mock.patch.object(module, "load_voice_auth_key", side_effect=AssertionError("unexpected I/O")),
                    self.assertRaises(ValueError),
                ):
                    await module._run(SimpleNamespace(phase="recall", timeout=timeout, voice_auth_key=Path("unused")))
