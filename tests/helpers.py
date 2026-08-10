from __future__ import annotations

import _bootstrap  # noqa: F401

from jarvis_voice.protocol import Envelope, EventFactory, new_uuid4


def factory() -> EventFactory:
    return EventFactory(session_id=new_uuid4())


def speech_commit(event_factory: EventFactory, response_id: str, index: int, text: str = "Βεβαίως.", locale: str = "el-GR") -> Envelope:
    return event_factory.create("response.speech.commit", {"unit_index": index, "locale": locale, "text": text}, response_id=response_id)

