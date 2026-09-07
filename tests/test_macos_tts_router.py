from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_voice.backends import SpeechUnit
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.tts_router import MacLocaleRoutingTtsBackend


class FakeBackend:
    def __init__(self) -> None:
        self.units: list[SpeechUnit] = []
        self.cancelled: list[tuple[int, str]] = []
        self.closed = False
        self.release = threading.Event()
        self._active = False
        self._error: BackendError | None = None

    def enqueue(self, unit: SpeechUnit) -> None:
        self.units.append(unit)
        self._active = True

        def complete() -> None:
            self.release.wait(timeout=1.0)
            self._active = False

        threading.Thread(target=complete, daemon=True).start()

    def cancel(self, new_epoch: int, reason: str) -> None:
        self.cancelled.append((new_epoch, reason))
        self._active = False
        self.release.set()

    @property
    def is_idle(self) -> bool:
        return not self._active

    @property
    def last_error(self) -> BackendError | None:
        return self._error

    def close(self) -> None:
        self.closed = True


def unit(index: int, locale: str, *, epoch: int = 1) -> SpeechUnit:
    return SpeechUnit("response", epoch, index, locale, f"unit {index}")


class MacLocaleRoutingTtsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.greek = FakeBackend()
        self.fallback = FakeBackend()
        self.router = MacLocaleRoutingTtsBackend(
            greek_backend=self.greek,
            fallback_backend=self.fallback,
            fallback_locales={"en-US"},
        )
        self.addCleanup(self.router.close)

    @staticmethod
    def wait_for(predicate, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("condition did not become true")

    def test_routes_locales_and_preserves_global_unit_order(self) -> None:
        self.router.enqueue(unit(1, "el-GR"))
        self.router.enqueue(unit(2, "en-US"))
        self.wait_for(lambda: len(self.greek.units) == 1)
        self.assertEqual(self.fallback.units, [])

        self.greek.release.set()
        self.wait_for(lambda: len(self.fallback.units) == 1)
        self.assertEqual(self.greek.units[0].locale, "el-GR")
        self.assertEqual(self.fallback.units[0].locale, "en-US")

    def test_cancel_reaches_both_backends_and_rejects_stale_queue(self) -> None:
        self.router.enqueue(unit(1, "el-GR", epoch=1))
        self.router.enqueue(unit(2, "en-US", epoch=1))
        self.wait_for(lambda: len(self.greek.units) == 1)

        self.router.cancel(2, "ptt_down")
        self.assertEqual(self.greek.cancelled[-1], (2, "ptt_down"))
        self.assertEqual(self.fallback.cancelled[-1], (2, "ptt_down"))
        time.sleep(0.03)
        self.assertEqual(self.fallback.units, [])
        self.assertTrue(self.router.is_idle)

    def test_unknown_locale_fails_synchronously(self) -> None:
        with self.assertRaisesRegex(BackendError, "tts.locale_unsupported"):
            self.router.enqueue(unit(1, "fr-FR"))


class MacRouterFailureTests(unittest.TestCase):
    def fixture(self):
        greek, fallback = FakeBackend(), FakeBackend()
        router = MacLocaleRoutingTtsBackend(greek_backend=greek, fallback_backend=fallback, fallback_locales={"en-US"})
        self.addCleanup(router.close)
        return router, greek, fallback

    def test_cancel_reaches_fallback_when_greek_cancel_fails(self):
        router, greek, fallback = self.fixture()
        with patch.object(greek, "cancel", side_effect=OSError("cancel failed")):
            with self.assertRaises(OSError):
                router.cancel(2, "ptt_down")
        self.assertEqual(fallback.cancelled, [(2, "ptt_down")])

    def test_close_attempts_every_backend_after_cancel_and_close_errors(self):
        for error in (OSError("cancel failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                router, greek, fallback = self.fixture()
                try:
                    with patch.object(greek, "cancel", side_effect=error), patch.object(greek, "close", side_effect=RuntimeError("close failed")) as close_greek:
                        with self.assertRaises(type(error)) as raised:
                            router.close()
                        self.assertIs(raised.exception, error)
                        close_greek.assert_called_once()
                    self.assertTrue(fallback.closed)
                    self.assertEqual(fallback.cancelled, [(2**63 - 1, "application_close")])
                    self.assertFalse(router._thread.is_alive())
                finally:
                    router._thread.join(timeout=1)

    def test_worker_does_not_dispatch_queued_successor_after_fatal_error(self):
        router, greek, fallback = self.fixture()
        entered, release = threading.Event(), threading.Event()
        error = BackendError("tts.synthetic_failure", "Synthetic failure", subsystem="tts")
        def fail_enqueue(value):
            entered.set()
            if not release.wait(2):
                raise AssertionError("test did not release worker")
            raise error
        with patch.object(greek, "enqueue", side_effect=fail_enqueue):
            try:
                router.enqueue(unit(1, "el-GR"))
                self.assertTrue(entered.wait(1))
                router.enqueue(unit(2, "en-US"))
                release.set()
                router._thread.join(timeout=0.2)
                self.assertEqual(fallback.units, [])
                self.assertIs(router.last_error, error)
                with self.assertRaises(BackendError):
                    router.enqueue(unit(3, "en-US"))
            finally:
                release.set()


if __name__ == "__main__":
    unittest.main()
