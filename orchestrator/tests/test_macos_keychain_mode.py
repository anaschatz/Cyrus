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
    platform_key_provider,
)


class MacOSSourceRunKeychainModeTests(unittest.TestCase):
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
