from __future__ import annotations

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
