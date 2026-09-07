from __future__ import annotations

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from jarvis_orchestrator.memory_key_provider import (
    MEMORY_KEYCHAIN_SERVICE,
    MacOSKeychainSecretClient,
)
from jarvis_orchestrator.scheduler_crypto import (
    _MacOSKeychainKeyProvider,
    ContentBinding,
    SchedulerContentCipher,
    SchedulerKeyNotFound,
    SchedulerKeyUnavailable,
    platform_key_provider,
)


class MacOSSourceRunKeychainModeTests(unittest.TestCase):
    def fake_provider(self):
        secrets = {}
        client = _MacOSKeychainKeyProvider.__new__(_MacOSKeychainKeyProvider)
        client.namespace = "synthetic-test-keys"
        client._string = lambda value: value
        client._data = bytes
        client._constant = lambda name: name
        client._dictionary = dict
        client._release_all = lambda _values: None
        client._cf = SimpleNamespace(CFRelease=lambda _value: None)

        def add(query, _result):
            account = query["kSecAttrAccount"]
            if account in secrets:
                return -25299
            secrets[account] = query["kSecValueData"]
            return 0

        def update(query, values):
            secrets[query["kSecAttrAccount"]] = values["kSecValueData"]
            return 0

        def load(account):
            if account not in secrets:
                raise SchedulerKeyNotFound("synthetic missing item")
            return secrets[account]

        client._security = SimpleNamespace(SecItemAdd=add, SecItemUpdate=update)
        client._load_secret = load
        return client, secrets

    def test_scheduler_invalid_active_marker_cannot_replace_existing_key(self):
        for marker in (None, b"invalid", b"\xff", b"0", b"-1"):
            with self.subTest(marker=marker):
                provider, secrets = self.fake_provider()
                cipher = SchedulerContentCipher(provider)
                binding = ContentBinding("test", "payload", "row", None, 0)
                encrypted = cipher.encrypt(b"retained content", binding)
                if marker is None:
                    del secrets["active-version"]
                else:
                    secrets["active-version"] = marker
                before = dict(secrets)
                with self.assertRaises(SchedulerKeyUnavailable):
                    provider.active_version()
                self.assertEqual(secrets, before)
                self.assertEqual(cipher.decrypt(encrypted, binding), b"retained content")

    def test_scheduler_duplicate_rotation_cannot_overwrite_published_key(self):
        provider, secrets = self.fake_provider()
        cipher = SchedulerContentCipher(provider)
        binding = ContentBinding("test", "payload", "row", None, 0)
        first = cipher.encrypt(b"before rotation", binding)
        self.assertEqual(cipher.rotate_key(), 2)
        second = cipher.encrypt(b"after rotation", binding)
        before = dict(secrets)
        # A second process already read version 1 before this rotation published version 2.
        with patch.object(provider, "_known_versions", return_value=[1]):
            with self.assertRaises(SchedulerKeyUnavailable):
                provider.create_next()
        self.assertEqual(secrets, before)
        self.assertEqual(provider.active_version(), 2)
        self.assertEqual(cipher.decrypt(first, binding), b"before rotation")
        self.assertEqual(cipher.decrypt(second, binding), b"after rotation")

    def test_scheduler_failed_marker_write_does_not_replace_orphan_key_on_retry(self):
        provider, secrets = self.fake_provider()
        self.assertEqual(provider.active_version(), 1)
        first_key = provider.load(1)
        with patch.object(provider._security, "SecItemUpdate", return_value=-1):
            with self.assertRaises(SchedulerKeyUnavailable):
                provider.create_next()
        before = dict(secrets)
        self.assertIn("key-v2", secrets)
        self.assertEqual(provider.active_version(), 1)
        with self.assertRaises(SchedulerKeyUnavailable):
            provider.create_next()
        self.assertEqual(secrets, before)
        self.assertEqual(provider.load(1), first_key)

    def test_scheduler_releases_native_objects_after_each_failure(self):
        for operation in ("read", "write"):
            def run(fail_at):
                calls = 0
                live = set()

                def step():
                    nonlocal calls
                    calls += 1
                    if calls == fail_at:
                        raise SchedulerKeyUnavailable("synthetic native failure")

                def allocate(*_args):
                    step()
                    live.add(calls)
                    return calls

                def release(pointer):
                    self.assertIn(pointer.value, live)
                    live.remove(pointer.value)

                def copy(_query, target):
                    ctypes.cast(target, ctypes.POINTER(ctypes.c_void_p))[0] = allocate()
                    return 0

                provider = _MacOSKeychainKeyProvider.__new__(_MacOSKeychainKeyProvider)
                provider.namespace = "synthetic-resource-test"
                provider._constant = lambda _name: ctypes.c_void_p(1000)
                provider._boolean_true = lambda: ctypes.c_void_p(1001)
                provider._cf = SimpleNamespace(
                    CFStringCreateWithCString=allocate, CFDataCreate=allocate,
                    CFDictionaryCreateMutable=allocate, CFDictionarySetValue=lambda *_: step(),
                    CFRelease=release, CFDataGetLength=lambda _: (step() or 32),
                    CFDataGetBytePtr=lambda _: (step() or b"a" * 32),
                )
                provider._security = SimpleNamespace(SecItemAdd=lambda *_: (step() or 0),
                                                      SecItemCopyMatching=copy)
                try:
                    if operation == "read":
                        self.assertEqual(provider._load_secret("key-v1"), b"a" * 32)
                    else:
                        provider._store_secret("key-v1", b"a" * 32)
                    self.assertEqual(fail_at, 0)
                except SchedulerKeyUnavailable:
                    self.assertGreater(fail_at, 0)
                self.assertEqual(live, set())
                return calls

            steps = run(0)
            for failure in range(1, steps + 1):
                with self.subTest(operation=operation, failure=failure):
                    run(failure)

    def test_memory_client_never_requests_entitlement_only_keychain(self) -> None:
        client = MacOSKeychainSecretClient.__new__(MacOSKeychainSecretClient)
        client.service = MEMORY_KEYCHAIN_SERVICE
        constants: list[str] = []
        dictionaries: list[list[tuple[object, object]]] = []
        client._security = SimpleNamespace(SecItemAdd=lambda *_: 0)
        client._string = lambda value: f"string:{value}"
        client._data = lambda value: ("data", bytes(value))
        client._true = lambda: "true"
        client._constant = lambda name: constants.append(name) or f"constant:{name}"
        client._dictionary = lambda pairs: (
            dictionaries.append(list(pairs)) or f"dictionary:{len(dictionaries)}"
        )
        client._release_all = lambda _values: None

        account = f"install-{'0' * 64}/epoch-v1"
        client._query(account, return_data=True)
        self.assertTrue(client.add(account, b"x" * 32))

        self.assertEqual(
            MEMORY_KEYCHAIN_SERVICE,
            "com.jarvis.memory-records.login-keychain-v1",
        )
        self.assertNotIn("kSecUseDataProtectionKeychain", constants)
        self.assertNotIn("kSecAttrAccessible", constants)
        self.assertNotIn("kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly", constants)
        self.assertGreaterEqual(len(dictionaries), 2)

    def test_scheduler_uses_disjoint_login_keychain_namespace(self) -> None:
        with (
            patch("jarvis_orchestrator.scheduler_crypto.sys.platform", "darwin"),
            patch(
                "jarvis_orchestrator.scheduler_crypto._MacOSKeychainKeyProvider"
            ) as provider,
        ):
            platform_key_provider(store_role="orchestrator", state_root=None)  # type: ignore[arg-type]
        provider.assert_called_once_with(
            "com.jarvis.orchestrator.scheduler-content.login-keychain-v1"
        )

        client = _MacOSKeychainKeyProvider.__new__(_MacOSKeychainKeyProvider)
        client.namespace = "com.jarvis.test.scheduler-content.login-keychain-v1"
        constants: list[str] = []
        client._string = lambda value: f"string:{value}"
        client._boolean_true = lambda: "true"
        client._constant = lambda name: constants.append(name) or f"constant:{name}"
        client._dictionary = lambda _pairs: "dictionary"
        client._query("active-version", return_data=True)

        self.assertNotIn("kSecUseDataProtectionKeychain", constants)


if __name__ == "__main__":
    unittest.main()
