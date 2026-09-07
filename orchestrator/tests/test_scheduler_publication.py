from __future__ import annotations

import asyncio
import json
import unittest
import uuid
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase
from jarvis_orchestrator.domain import utc_text
from jarvis_orchestrator.gateway import VoiceGateway
from jarvis_orchestrator.scheduler_contract import PackagedTimezoneProvider, compile_proposal
from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher
from jarvis_orchestrator.scheduler_store import SchedulerStore
from jarvis_voice.protocol import EventFactory
from jarvis_voice.scheduler_protocol import SchedulerEventFactory
from test_scheduler_v5 import proposal_input, proposal_output
from test_voice_gateway_auth import AUTH_KEY


class SchedulerPublicationTests(StoreCase, unittest.IsolatedAsyncioTestCase):
    async def test_publisher_rechecks_scheduler_claim_after_waiting_for_lock(self):
        for state in ("live", "revoked", "lease_expired", "reclaimed", "event_expired"):
            with self.subTest(state=state):
                store = self.make_store()
                scheduler = SchedulerStore(store, cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
                                           zones=PackagedTimezoneProvider())
                request_id, session_id = str(uuid.uuid4()), str(uuid.uuid4())
                compiled = compile_proposal(
                    proposal_input(request_id, utc_text(NOW)), proposal_output(request_id),
                    compiler_qualification_sha256="1" * 64, zones=scheduler.zones,
                )
                scheduler.create_proposal(compiled, now=NOW)
                gateway = VoiceGateway(SimpleNamespace(store=store, scheduler=scheduler), auth_key=AUTH_KEY)
                socket = SimpleNamespace(send=AsyncMock(), close=AsyncMock())
                factory = SchedulerEventFactory(session_id)
                claimed, processed = asyncio.Event(), asyncio.Event()
                original_claim = scheduler.claim_outbox

                def claim(owner):
                    if claimed.is_set():
                        processed.set()
                        return []
                    events = original_claim(owner)
                    self.assertEqual(len(events), 1)
                    claimed.set()
                    return events

                lock = gateway._lock(f"scheduler:{session_id}")
                await lock.acquire()
                selected = NOW + timedelta(seconds=598) if state == "event_expired" else NOW
                with patch("jarvis_orchestrator.scheduler_store.utc_now", return_value=selected) as clock, \
                        patch.object(scheduler, "claim_outbox", side_effect=claim):
                    writer = asyncio.create_task(gateway._publish(
                        socket, session_id, EventFactory(), "first", scheduler_factory=factory,
                    ))
                    try:
                        await asyncio.wait_for(claimed.wait(), 0.5)
                        if state == "revoked":
                            scheduler.expire_approvals(now=NOW + timedelta(minutes=10))
                        elif state == "lease_expired":
                            clock.return_value = NOW + timedelta(seconds=5)
                        elif state == "reclaimed":
                            clock.return_value = NOW + timedelta(seconds=5)
                            self.assertEqual(len(original_claim("replacement")), 1)
                        elif state == "event_expired":
                            clock.return_value = NOW + timedelta(minutes=10)
                        lock.release()
                        await asyncio.wait_for(processed.wait(), 0.5)
                        self.assertEqual(socket.send.await_count, int(state == "live"))
                        self.assertEqual(factory.next_seq, 2 if state == "live" else 1)
                        if state == "live":
                            self.assertEqual(json.loads(socket.send.call_args.args[0])["type"],
                                             "scheduler.approval.requested")
                        socket.close.assert_not_awaited()
                    finally:
                        if lock.locked():
                            lock.release()
                        writer.cancel()
                        await asyncio.gather(writer, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
