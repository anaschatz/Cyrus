from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator import readiness
from jarvis_orchestrator.operations import OperationalError
from jarvis_voice.fakes import FakeOrchestrator
from jarvis_voice.protocol import Envelope, EventFactory, new_uuid4


class Wire:
    def __init__(self, transcript, *, ready_mutation=None, delay=0):
        self.transcript = transcript
        self.ready_mutation = ready_mutation
        self.delay = delay
        self.queue = deque()
        self.sent = []
        self.closed = False
        self.session = None
        self.server_sequence = 0
        self.sequence_mutation = lambda event: event

    async def send(self, raw):
        event = Envelope.from_json(raw)
        self.sent.append(event)
        if event.type == "client.hello":
            self.session = event.session_id
            ready = EventFactory(session_id=self.session).create(
                "server.ready", {"protocol": "jarvis.voice/1"}
            )
            self.queue.append(
                self.ready_mutation(ready) if self.ready_mutation else ready
            )
        elif event.type == "input.transcript.final":
            self.queue.extend(self.transcript(self, event))

    async def recv(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.queue:
            await asyncio.sleep(10)
        self.server_sequence += 1
        event = replace(self.queue.popleft(), seq=self.server_sequence)
        return self.sequence_mutation(event).to_json()

    async def close(self):
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

    def __await__(self):
        async def connect():
            return self

        return connect().__await__()


def success(wire, transcript):
    fake = FakeOrchestrator(session_id=wire.session)
    response_id = new_uuid4()
    ack = fake.factory.create(
        "server.ack",
        {"acked_event_id": transcript.event_id, "status": "accepted"},
        turn_id=transcript.turn_id,
        response_id=response_id,
    )
    return [
        ack,
        *fake.script_response(
            ("Ready.",),
            locale="en-US",
            turn_id=transcript.turn_id,
            response_id=response_id,
            interleave_speech=True,
        ).events,
    ]


async def authenticate(wire, key, hello):
    await wire.send(hello)


class ReadinessBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def sequential(self, wire, **kwargs):
        with (
            patch("websockets.asyncio.client.connect", return_value=wire),
            patch(
                "jarvis_voice.voice_auth_transport.authenticate_async_client",
                side_effect=authenticate,
            ),
        ):
            return await readiness.run_live_sequential_gate(
                host="127.0.0.1",
                port=8765,
                auth_key=None,
                turns=1,
                timeout_seconds=0.2,
                **kwargs,
            )

    async def test_sequential_rejects_wrong_session_turn_and_duplicate_ack(self):
        def mutate(kind):
            def events(wire, transcript):
                result = success(wire, transcript)
                if kind == "ack_duplicate_status":
                    result[0] = replace(
                        result[0], payload={**result[0].payload, "status": "duplicate"}
                    )
                elif kind == "second_ack":
                    result.insert(1, replace(result[0], event_id=new_uuid4()))
                elif kind == "ack_turn":
                    result[0] = replace(result[0], turn_id=new_uuid4())
                elif kind == "response_turn":
                    result[1] = replace(result[1], turn_id=new_uuid4())
                elif kind == "response_session":
                    result[1] = replace(result[1], session_id=new_uuid4())
                return result

            return events

        for kind in (
            "ack_duplicate_status",
            "second_ack",
            "ack_turn",
            "response_turn",
            "response_session",
        ):
            with self.subTest(kind=kind):
                wire = Wire(mutate(kind))
                with self.assertRaises(OperationalError):
                    await self.sequential(wire)
                self.assertTrue(wire.closed)
        wire = Wire(
            success, ready_mutation=lambda event: replace(event, session_id=new_uuid4())
        )
        with self.assertRaises(OperationalError):
            await self.sequential(wire)
        self.assertTrue(wire.closed)

        for sequence in (1, 4):
            wire = Wire(success)
            wire.sequence_mutation = lambda event: (
                replace(event, seq=sequence) if event.seq == 3 else event
            )
            with self.subTest(sequence=sequence), self.assertRaises(OperationalError):
                await self.sequential(wire)
            self.assertTrue(wire.closed)

    async def test_sequential_timeout_is_for_the_whole_turn(self):
        wire = Wire(success, delay=0.008)
        with (
            patch("websockets.asyncio.client.connect", return_value=wire),
            patch(
                "jarvis_voice.voice_auth_transport.authenticate_async_client",
                side_effect=authenticate,
            ),
        ):
            with self.assertRaises(TimeoutError):
                await readiness.run_live_sequential_gate(
                    host="127.0.0.1",
                    port=8765,
                    auth_key=None,
                    turns=1,
                    timeout_seconds=0.025,
                )
        self.assertTrue(wire.closed)

    async def test_valid_sequential_still_reports_latency_and_playback(self):
        wire = Wire(success)
        report = await self.sequential(wire, include_latency_profile=True)
        self.assertTrue(report["ready"])
        self.assertEqual(
            report["latency_profile_ms"]["transcript_to_speech_done"]["count"], 1
        )
        self.assertEqual(wire.sent[-1].type, "playback.completed")
        self.assertTrue(wire.closed)

    async def restart(
        self,
        mutate=lambda events: events,
        *,
        bad_handshake=False,
        original_mutation=lambda events: events,
        delay=0,
        timeout=0.1,
    ):
        generated = []
        original = []

        def generation(wire, transcript):
            ack = success(wire, transcript)[0]
            generated.append((transcript, ack))
            return [ack]

        def playback(wire, transcript):
            events = original_mutation(success(wire, transcript))
            original.extend(events)
            return events

        first = Wire(
            generation,
            ready_mutation=(
                lambda e: replace(
                    e,
                    type="server.error",
                    payload={
                        "code": "backend_failure",
                        "retryable": False,
                        "terminal": True,
                    },
                )
            )
            if bad_handshake
            else None,
        )
        second = Wire(playback, delay=delay)
        third = Wire(lambda *_: [])
        wires = [first, second, third]

        async def auth(wire, key, hello):
            await authenticate(wire, key, hello)
            if wire is second:
                transcript, ack = generated[0]
                wire.queue.append(
                    EventFactory(session_id=wire.session).create(
                        "response.cancelled",
                        {"reason_code": "service_interrupted", "partial_output": False},
                        turn_id=transcript.turn_id,
                        response_id=ack.response_id,
                    )
                )
            elif wire is third:
                wire.queue.extend(mutate(original.copy()))

        self.wires = wires
        with (
            patch("websockets.asyncio.client.connect", side_effect=wires),
            patch(
                "jarvis_voice.voice_auth_transport.authenticate_async_client",
                side_effect=auth,
            ),
        ):
            return await readiness.run_live_restart_gate(
                host="127.0.0.1",
                port=8765,
                auth_key=None,
                stop_service=lambda: None,
                start_service=lambda: None,
                timeout_seconds=timeout,
            )

    async def test_restart_replay_preserves_order_and_turn_binding(self):
        for kind in ("reorder", "turn", "session"):

            def mutate(events):
                if kind == "reorder":
                    events[2], events[3] = events[3], events[2]
                elif kind == "turn":
                    events[2] = replace(events[2], turn_id=new_uuid4())
                else:
                    events[2] = replace(events[2], session_id=new_uuid4())
                return events

            with self.subTest(kind=kind):
                try:
                    report = await self.restart(mutate)
                except OperationalError:
                    pass
                else:
                    self.assertFalse(report["ready"])
                self.assertTrue(all(w.closed for w in self.wires))
        self.assertTrue((await self.restart())["ready"])
        self.assertTrue(all(w.closed for w in self.wires))

    async def test_restart_closes_connection_on_invalid_handshake(self):
        with self.assertRaises(OperationalError):
            await self.restart(bad_handshake=True)
        self.assertTrue(self.wires[0].closed)

    async def test_failure_gate_observes_queued_output_after_cancellation(self):
        def events(wire, transcript):
            good = success(wire, transcript)
            ack = good[0]
            factory = EventFactory(session_id=wire.session)
            error = factory.create(
                "server.error",
                {"code": "backend_failure", "retryable": False, "terminal": True},
                turn_id=transcript.turn_id,
                response_id=ack.response_id,
            )
            cancelled = factory.create(
                "response.cancelled",
                {"reason_code": "backend_failure", "partial_output": False},
                turn_id=transcript.turn_id,
                response_id=ack.response_id,
            )
            return [ack, error, cancelled] + ([good[1]] if late else [])

        for late in (False, True):
            wire = Wire(events)
            with (
                patch("websockets.asyncio.client.connect", return_value=wire),
                patch(
                    "jarvis_voice.voice_auth_transport.authenticate_async_client",
                    side_effect=authenticate,
                ),
            ):
                report = await readiness.run_live_model_failure_gate(
                    host="127.0.0.1", port=8765, auth_key=None, timeout_seconds=0.2
                )
            self.assertEqual(report["ready"], not late)
            self.assertEqual(report["late_output_count"], int(late))
            self.assertTrue(wire.closed)

    async def test_invalid_numeric_limits_fail_before_io(self):
        base = dict(host="127.0.0.1", port=8765, auth_key=None)
        cases = (
            [
                (readiness.run_live_sequential_gate, {"turns": value})
                for value in (True, 1.5, "3", None)
            ]
            + [
                (function, {"timeout_seconds": value, **extra})
                for function, extra in (
                    (readiness.run_live_sequential_gate, {}),
                    (readiness.run_live_model_failure_gate, {}),
                    (
                        readiness.run_live_restart_gate,
                        {"stop_service": lambda: None, "start_service": lambda: None},
                    ),
                )
                for value in (True, float("nan"), float("inf"), "2", 0, 10**500)
            ]
            + [
                (readiness.run_live_soak_gate, {"database_path": "unused", **invalid})
                for invalid in (
                    {"duration_seconds": float("nan")},
                    {"duration_seconds": 60.5},
                    {"batch_turns": True},
                    {"batch_turns": 10001},
                    {"maximum_database_bytes": float("inf")},
                    {"port": True},
                )
            ]
        )
        cases.append((readiness.run_live_model_failure_gate, {"timeout_seconds": 121}))
        with (
            patch(
                "websockets.asyncio.client.connect",
                side_effect=AssertionError("network reached"),
            ),
            patch.object(
                readiness,
                "operational_paths",
                side_effect=AssertionError("filesystem reached"),
            ),
        ):
            for function, invalid in cases:
                with self.subTest(function=function.__name__, invalid=invalid):
                    with self.assertRaises(ValueError):
                        await function(**{**base, **invalid})
        for value in (True, 1.5, "3", None):
            with self.subTest(synthetic=value), self.assertRaises(ValueError):
                readiness.run_synthetic_lifecycle_gate(turns=value)

    async def test_restart_and_sequential_reject_incomplete_original_response(self):
        mutations = (
            lambda events: [events[0], events[-1]],
            lambda events: [events[0], *events[2:]],
            lambda events: [events[1], *events],
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                with self.assertRaises(OperationalError):
                    await self.sequential(
                        Wire(lambda wire, turn: mutate(success(wire, turn)))
                    )
                with self.assertRaises(OperationalError):
                    await self.restart(original_mutation=mutate)
                self.assertTrue(all(w.closed for w in self.wires if w.sent))

    async def test_restart_phase_deadline_is_not_renewed_per_event(self):
        with self.assertRaises(TimeoutError):
            await self.restart(delay=0.008, timeout=0.025)
        self.assertTrue(all(w.closed for w in self.wires if w.sent))

    async def test_soak_includes_final_database_and_log_growth(self):
        for target in ("database", "log"):
            with (
                self.subTest(target=target),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                clock = SimpleNamespace(value=0.0)
                closed = []

                class Log:
                    def __init__(self, path):
                        self.path = path
                        path.parent.mkdir(exist_ok=True)

                    def emit(self, code, **metrics):
                        self.path.write_bytes(
                            b"x"
                            * (
                                65
                                if code == "soak_completed" and target == "log"
                                else 1
                            )
                        )

                    def close(self):
                        closed.append(True)

                async def batch(**kwargs):
                    clock.value = 61.0
                    return {"ready": True}

                def diagnostics(size):
                    return SimpleNamespace(
                        database_bytes=size,
                        wal_bytes=0,
                        shm_bytes=0,
                        invariant_violation_count=0,
                        ready=True,
                    )

                with (
                    patch.object(
                        readiness,
                        "time",
                        SimpleNamespace(monotonic=lambda: clock.value),
                    ),
                    patch.object(readiness, "ContentFreeOperationalLog", Log),
                    patch.object(
                        readiness, "run_live_sequential_gate", side_effect=batch
                    ),
                    patch.object(
                        readiness,
                        "database_diagnostics",
                        side_effect=[
                            diagnostics(1),
                            diagnostics(101 if target == "database" else 1),
                        ],
                    ),
                    patch.object(readiness, "MAXIMUM_OPERATIONAL_LOG_BYTES", 64),
                ):
                    report = await readiness.run_live_soak_gate(
                        host="127.0.0.1",
                        port=8765,
                        auth_key=None,
                        database_path=root / "state.sqlite3",
                        duration_seconds=60,
                        maximum_database_bytes=100,
                    )
                self.assertFalse(report["ready"])
                self.assertFalse(
                    report[
                        "database_bounded"
                        if target == "database"
                        else "operational_log_bounded"
                    ]
                )
                self.assertEqual(
                    report[
                        "maximum_database_bytes"
                        if target == "database"
                        else "maximum_operational_log_bytes"
                    ],
                    101 if target == "database" else 65,
                )
                self.assertEqual(closed, [True])


class AggregateFreshnessTests(unittest.TestCase):
    def test_parent_expiry_is_rechecked_at_final_time(self):
        import test_readiness_evidence as fixtures
        from jarvis_orchestrator import readiness_evidence as evidence

        fixture = fixtures.ReadinessEvidenceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        paths = {}
        for kind, payload in fixtures._leaf_payloads().items():
            leaf = evidence.build_readiness_evidence(
                evidence_kind=kind,
                payload=payload,
                identity=fixtures._identity(),
                database_diagnostics=fixtures._healthy_diagnostics(),
                qualification=fixtures._qualification(kind),
                signer=fixture.signer,
                captured_at=fixtures.NOW,
                freshness_seconds=3600,
            )
            paths[kind] = evidence.write_readiness_evidence(
                fixture.root / (kind + ".json"),
                leaf,
                verifier=fixture.verifier,
                cohort_state_directory=fixture.cohort_state,
            )
        arguments = dict(
            synthetic_evidence=paths[evidence.SYNTHETIC_KIND],
            live_evidence=paths[evidence.LIVE_KIND],
            restart_evidence=paths[evidence.RESTART_KIND],
            recovery_evidence=paths[evidence.RECOVERY_KIND],
            soak_evidence=paths[evidence.SOAK_KIND],
            native_voice_evidence=paths[evidence.NATIVE_KIND],
            verifier=fixture.verifier,
            expected_identity=fixtures._identity(),
            aggregate_signer=fixture.signer,
            cohort_state_directory=fixture.cohort_state,
            current_state_provider=lambda: (
                fixtures._identity(),
                fixtures._healthy_diagnostics(),
            ),
            now=fixtures.NOW,
        )
        self.assertTrue(
            readiness.evaluate_operational_ready_gate(
                **arguments,
                final_time_provider=lambda: fixtures.NOW + timedelta(seconds=3599),
            ).payload["ready"]
        )
        for final in (None, 1, "bad", fixtures.NOW.replace(tzinfo=None)):
            with (
                self.subTest(final=final),
                self.assertRaisesRegex(
                    OperationalError, "readiness_final_time_invalid"
                ),
            ):
                readiness.evaluate_operational_ready_gate(
                    **arguments, final_time_provider=lambda: final
                )
        with self.assertRaisesRegex(OperationalError, "readiness_evidence_stale"):
            readiness.evaluate_operational_ready_gate(
                **arguments,
                aggregate_freshness_seconds=60,
                final_time_provider=lambda: fixtures.NOW + timedelta(seconds=61),
            )
        with self.assertRaisesRegex(OperationalError, "readiness_evidence_stale"):
            readiness.evaluate_operational_ready_gate(
                **arguments,
                final_time_provider=lambda: fixtures.NOW + timedelta(hours=1),
            )


if __name__ == "__main__":
    unittest.main()
