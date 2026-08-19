from __future__ import annotations

import copy
import unittest
from dataclasses import replace

import _bootstrap  # noqa: F401
from jarvis_orchestrator.jcs import jcs_bytes
from jarvis_orchestrator.memory_crypto import (
    MAX_MEMORY_PAYLOAD_BYTES,
    MAX_MEMORY_QUERY_TERMS,
    MEMORY_ENVELOPE_CONTRACT,
    InMemoryMemoryEpochKeyProvider,
    MemoryCryptoError,
    MemoryEnvelopeCipherV1,
    MemoryEnvelopeV1,
    MemoryRecordBindingV1,
)

P1 = "11111111-1111-4111-8111-111111111111"
P2 = "22222222-2222-4222-8222-222222222222"
R1 = "33333333-3333-4333-8333-333333333333"
R2 = "44444444-4444-4444-8444-444444444444"
C1 = "55555555-5555-4555-8555-555555555555"
T1 = "66666666-6666-4666-8666-666666666666"
E1 = "77777777-7777-4777-8777-777777777777"
ROOT_KEY = bytes(range(32))
PAYLOAD = jcs_bytes(
    {
        "content": "Το κρυφό μου χρώμα είναι μπλε",
        "metadata": {"source": "explicit_command"},
    }
)


def binding(*, principal_id: str = P1, record_id: str = R1) -> MemoryRecordBindingV1:
    return MemoryRecordBindingV1(
        principal_id=principal_id,
        record_id=record_id,
        conversation_id=C1,
        task_id=T1,
        source_event_id=E1,
        created_at="2030-01-02T03:04:05.006Z",
        expires_at="2031-01-02T03:04:05.006Z",
        content_bytes=len("Το κρυφό μου χρώμα είναι μπλε".encode()),
        term_count=6,
    )


def flip_first(value: bytes) -> bytes:
    return bytes([value[0] ^ 1]) + value[1:]


class MemoryCryptoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = InMemoryMemoryEpochKeyProvider.initialized(ROOT_KEY)
        self.cipher = MemoryEnvelopeCipherV1(self.provider)
        self.binding = binding()

    def assert_code(self, code: str, function, /, *args, **kwargs) -> None:
        with self.assertRaises(MemoryCryptoError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)

    def test_round_trip_is_randomized_and_closed_document_contains_no_plaintext(
        self,
    ) -> None:
        first = self.cipher.encrypt(PAYLOAD, self.binding)
        second = self.cipher.encrypt(PAYLOAD, self.binding)

        self.assertEqual(self.cipher.decrypt(first, self.binding), PAYLOAD)
        self.assertEqual(self.cipher.decrypt(second, self.binding), PAYLOAD)
        self.assertNotEqual(first.payload_nonce, second.payload_nonce)
        self.assertNotEqual(first.wrap_nonce, second.wrap_nonce)
        self.assertNotEqual(
            first.payload_ciphertext_and_tag, second.payload_ciphertext_and_tag
        )
        self.assertNotEqual(first.wrapped_dek_and_tag, second.wrapped_dek_and_tag)

        document = first.to_document()
        self.assertEqual(set(document), MemoryEnvelopeV1._DOCUMENT_KEYS)
        self.assertEqual(document["contract"], MEMORY_ENVELOPE_CONTRACT)
        serialized = first.to_jcs_bytes()
        self.assertNotIn("κρυφό".encode(), serialized)
        self.assertNotIn(b"explicit_command", serialized)
        self.assertEqual(
            MemoryEnvelopeV1.from_document(document),
            first,
        )

    def test_payload_wrap_nonce_and_binding_tampering_fail_closed(self) -> None:
        envelope = self.cipher.encrypt(PAYLOAD, self.binding)
        tampered = (
            replace(
                envelope,
                payload_ciphertext_and_tag=flip_first(
                    envelope.payload_ciphertext_and_tag
                ),
            ),
            replace(
                envelope,
                wrapped_dek_and_tag=flip_first(envelope.wrapped_dek_and_tag),
            ),
            replace(envelope, payload_nonce=flip_first(envelope.payload_nonce)),
            replace(envelope, wrap_nonce=flip_first(envelope.wrap_nonce)),
            replace(envelope, binding_sha256="0" * 64),
        )
        for candidate in tampered:
            with self.subTest(candidate=candidate):
                self.assert_code(
                    "memory_envelope_invalid",
                    self.cipher.decrypt,
                    candidate,
                    self.binding,
                )

    def test_row_or_principal_swap_cannot_decrypt(self) -> None:
        envelope = self.cipher.encrypt(PAYLOAD, self.binding)
        for swapped in (
            binding(record_id=R2),
            binding(principal_id=P2),
            replace(self.binding, task_id=R2),
            replace(self.binding, content_bytes=self.binding.content_bytes + 1),
        ):
            with self.subTest(swapped=swapped):
                self.assert_code(
                    "memory_envelope_invalid",
                    self.cipher.decrypt,
                    envelope,
                    swapped,
                )

    def test_missing_epoch_key_is_distinct_and_content_free(self) -> None:
        envelope = self.cipher.encrypt(PAYLOAD, self.binding)
        self.cipher.create_epoch()
        self.provider.destroy(1)
        self.assert_code(
            "memory_key_not_found",
            self.cipher.decrypt,
            envelope,
            self.binding,
        )

    def test_rewrap_preserves_payload_ciphertext_then_old_key_crypto_erases_target(
        self,
    ) -> None:
        retained_old = self.cipher.encrypt(PAYLOAD, self.binding)
        target_old = self.cipher.encrypt(b"target-to-erase", binding(record_id=R2))
        new_version = self.cipher.create_epoch()
        self.assertEqual(new_version, 2)

        retained_new = self.cipher.rewrap(
            retained_old, self.binding, new_key_version=new_version
        )
        self.assertEqual(retained_new.payload_nonce, retained_old.payload_nonce)
        self.assertEqual(
            retained_new.payload_ciphertext_and_tag,
            retained_old.payload_ciphertext_and_tag,
        )
        self.assertEqual(retained_new.binding_sha256, retained_old.binding_sha256)
        self.assertNotEqual(retained_new.wrap_nonce, retained_old.wrap_nonce)
        self.assertNotEqual(
            retained_new.wrapped_dek_and_tag, retained_old.wrapped_dek_and_tag
        )
        self.assertEqual(self.cipher.decrypt(retained_new, self.binding), PAYLOAD)

        self.cipher.destroy_epoch(1, live_reference_count=0)
        self.assertFalse(self.provider.exists(1))
        self.assertEqual(self.cipher.decrypt(retained_new, self.binding), PAYLOAD)
        self.assert_code(
            "memory_key_not_found",
            self.cipher.decrypt,
            target_old,
            binding(record_id=R2),
        )

    def test_rewrap_requires_new_existing_epoch(self) -> None:
        envelope = self.cipher.encrypt(PAYLOAD, self.binding)
        self.assert_code(
            "memory_key_version_not_forward",
            self.cipher.rewrap,
            envelope,
            self.binding,
            new_key_version=1,
        )
        self.assert_code(
            "memory_key_target_not_active",
            self.cipher.rewrap,
            envelope,
            self.binding,
            new_key_version=2,
        )

        version2 = self.cipher.create_epoch()
        version3 = self.cipher.create_epoch()
        self.assertEqual((version2, version3), (2, 3))
        self.assert_code(
            "memory_key_target_not_active",
            self.cipher.rewrap,
            envelope,
            self.binding,
            new_key_version=2,
        )
        current = self.cipher.encrypt(PAYLOAD, self.binding)
        self.assert_code(
            "memory_key_version_not_forward",
            self.cipher.rewrap,
            current,
            self.binding,
            new_key_version=1,
        )

    def test_rewrap_authenticates_payload_before_preserving_it(self) -> None:
        envelope = self.cipher.encrypt(PAYLOAD, self.binding)
        new_version = self.cipher.create_epoch()
        tampered = replace(
            envelope,
            payload_ciphertext_and_tag=flip_first(envelope.payload_ciphertext_and_tag),
        )
        self.assert_code(
            "memory_envelope_invalid",
            self.cipher.rewrap,
            tampered,
            self.binding,
            new_key_version=new_version,
        )

    def test_blind_tokens_are_deterministic_domain_and_scope_separated(self) -> None:
        arguments = {
            "principal_id": P1,
            "key_version": 1,
            "content": "Blue",
            "normalized_content": "blue",
            "terms": ("blue", "home"),
        }
        first = self.cipher.blind_tokens(**arguments)
        second = self.cipher.blind_tokens(**arguments)
        self.assertEqual(first, second)
        self.assertNotEqual(first.content_hmac, first.normalized_hmac)
        self.assertNotEqual(first.normalized_hmac, first.term_hmacs[0])
        self.assertNotEqual(first.term_hmacs[0], first.term_hmacs[1])
        self.assertEqual(
            self.cipher.blind_token(
                principal_id=P1,
                key_version=1,
                kind="normalized",
                value="blue",
            ),
            first.normalized_hmac,
        )
        self.assertEqual(
            self.cipher.blind_token(
                principal_id=P1,
                key_version=1,
                kind="term",
                value="home",
            ),
            first.term_hmacs[1],
        )

        other_principal = self.cipher.blind_tokens(**{**arguments, "principal_id": P2})
        self.assertNotEqual(first.content_hmac, other_principal.content_hmac)
        version2 = self.cipher.create_epoch()
        other_epoch = self.cipher.blind_tokens(**{**arguments, "key_version": version2})
        self.assertNotEqual(first.content_hmac, other_epoch.content_hmac)
        self.assertNotEqual(first.normalized_hmac, other_epoch.normalized_hmac)

        serialized = jcs_bytes(first.hex_document())
        self.assertNotIn(b"Blue", serialized)
        self.assertNotIn(b"blue", serialized)
        self.assertNotIn(b"home", serialized)

    def test_blind_token_golden_vectors_freeze_hkdf_and_hmac_contracts(self) -> None:
        tokens = self.cipher.blind_tokens(
            principal_id=P1,
            key_version=1,
            content="Blue",
            normalized_content="blue",
            terms=("blue", "home"),
        )
        self.assertEqual(
            tokens.content_hmac.hex(),
            "b44c092940ea8de7befe189a5bf48ba21a652002e80fa4dc3ef5e606a4dc2a85",
        )
        self.assertEqual(
            tokens.normalized_hmac.hex(),
            "1cb261a27a8245760148d8bf0fe6df78ef7666bdbbbc24839676459a8bc17b75",
        )
        self.assertEqual(
            [item.hex() for item in tokens.term_hmacs],
            [
                "2ae21d077f7d79f97c781a921169c70b62a1371b90e9b8d754f57533b3fd3331",
                "29b05a9c9fa3d5b4dc5d23cee4c4e94a4e0704653f7f4f610c0e3a046604b1e6",
            ],
        )
        self.assertEqual(
            self.binding.binding_sha256(),
            "640682f448881ac1aed62f51e2df755ea1d4f584dd018d3628a58fbaed6e2920",
        )

    def test_blind_token_inputs_are_strict_and_bounded(self) -> None:
        base = {
            "principal_id": P1,
            "key_version": 1,
            "content": "Blue",
            "normalized_content": "blue",
            "terms": ("blue", "home"),
        }
        for override in (
            {"terms": ("home", "blue")},
            {"terms": ("blue", "blue")},
            {"terms": ()},
            {"terms": ["blue"]},
            {"terms": (1,)},
            {"terms": ([],)},
            {"content": ""},
            {"content": "e\u0301"},
            {"content": "\ud800"},
            {"normalized_content": "x" * 4097},
        ):
            with self.subTest(override=override):
                self.assert_code(
                    "memory_index_input_invalid",
                    self.cipher.blind_tokens,
                    **{**base, **override},
                )
        self.assert_code(
            "memory_index_input_invalid",
            self.cipher.blind_token,
            principal_id=P1,
            key_version=1,
            kind="prefix",
            value="blue",
        )

    def test_blind_term_batch_loads_root_once_and_matches_exact_hmacs(self) -> None:
        class CountingProvider(InMemoryMemoryEpochKeyProvider):
            def __init__(self) -> None:
                super().__init__()
                self.load_count = 0

            def load(self, version: int) -> bytes:
                self.load_count += 1
                return super().load(version)

        provider = CountingProvider.initialized(ROOT_KEY)
        cipher = MemoryEnvelopeCipherV1(provider)
        terms = ("blue", "home")
        batch = cipher.blind_term_tokens(
            principal_id=P1,
            key_version=1,
            terms=terms,
        )
        self.assertEqual(provider.load_count, 1)
        expected = tuple(
            self.cipher.blind_token(
                principal_id=P1,
                key_version=1,
                kind="term",
                value=term,
            )
            for term in terms
        )
        self.assertEqual(batch, expected)
        self.assertEqual(len(batch), len(terms))

        for invalid in (
            (),
            ("home", "blue"),
            ("blue", "blue"),
            ["blue"],
            ("e\u0301",),
            (1,),
            tuple(f"term-{index:02d}" for index in range(MAX_MEMORY_QUERY_TERMS + 1)),
        ):
            with self.subTest(invalid=invalid):
                self.assert_code(
                    "memory_index_input_invalid",
                    cipher.blind_term_tokens,
                    principal_id=P1,
                    key_version=1,
                    terms=invalid,
                )

    def test_epoch_session_loads_once_then_has_no_provider_calls(self) -> None:
        class CountingProvider(InMemoryMemoryEpochKeyProvider):
            def __init__(self) -> None:
                super().__init__()
                self.load_count = 0
                self.forbid_load = False

            def load(self, version: int) -> bytes:
                self.load_count += 1
                if self.forbid_load:
                    raise AssertionError("provider load after session open")
                return super().load(version)

        provider = CountingProvider.initialized(ROOT_KEY)
        cipher = MemoryEnvelopeCipherV1(provider)
        session = cipher.open_epoch_session(principal_id=P1, key_version=1)
        self.assertEqual(provider.load_count, 1)
        self.assertEqual(session.install_id, provider.install_id)
        self.assertEqual(session.principal_id, P1)
        self.assertEqual(session.key_version, 1)
        self.assertFalse(hasattr(session, "provider"))
        self.assertFalse(hasattr(session, "root_key"))

        provider.forbid_load = True
        session_envelope = session.encrypt(PAYLOAD, self.binding)
        cipher_envelope = self.cipher.encrypt(PAYLOAD, self.binding)
        self.assertEqual(session.decrypt(session_envelope, self.binding), PAYLOAD)
        self.assertEqual(session.decrypt(cipher_envelope, self.binding), PAYLOAD)
        self.assertEqual(self.cipher.decrypt(session_envelope, self.binding), PAYLOAD)

        session_tokens = session.blind_tokens(
            content="Blue",
            normalized_content="blue",
            terms=("blue", "home"),
        )
        cipher_tokens = self.cipher.blind_tokens(
            principal_id=P1,
            key_version=1,
            content="Blue",
            normalized_content="blue",
            terms=("blue", "home"),
        )
        self.assertEqual(session_tokens, cipher_tokens)
        self.assertEqual(
            session.blind_token(kind="term", value="blue"),
            cipher_tokens.term_hmacs[0],
        )
        self.assertEqual(
            session.blind_term_tokens(terms=("blue", "home")),
            cipher_tokens.term_hmacs,
        )
        self.assertEqual(provider.load_count, 1)

        self.assert_code(
            "memory_binding_invalid",
            session.decrypt,
            session_envelope,
            binding(principal_id=P2),
        )
        self.assert_code(
            "memory_envelope_invalid",
            session.decrypt,
            replace(session_envelope, key_version=2),
            self.binding,
        )
        self.assert_code(
            "memory_envelope_invalid",
            session.decrypt,
            replace(
                session_envelope,
                payload_ciphertext_and_tag=flip_first(
                    session_envelope.payload_ciphertext_and_tag
                ),
            ),
            self.binding,
        )
        self.assertEqual(provider.load_count, 1)

    def test_provider_attestation_is_content_free_and_tracks_high_water(self) -> None:
        attestation = self.provider.attest()
        self.assertEqual(attestation.install_id, self.provider.install_id)
        self.assertEqual(attestation.active_version, 1)
        self.assertEqual(attestation.high_water, 1)
        self.assertEqual(attestation.reconciliation_state, "clean")
        self.assertNotIn(ROOT_KEY.hex(), repr(attestation))

    def test_strict_envelope_parser_rejects_schema_smuggling(self) -> None:
        envelope = self.cipher.encrypt(PAYLOAD, self.binding)
        valid = envelope.to_document()
        invalid_documents = []
        extra = copy.deepcopy(valid)
        extra["extra"] = "ignored?"
        invalid_documents.append(extra)
        missing = copy.deepcopy(valid)
        del missing["binding_sha256"]
        invalid_documents.append(missing)
        wrong_contract = copy.deepcopy(valid)
        wrong_contract["contract"] = "jarvis.memory-envelope/2"
        invalid_documents.append(wrong_contract)
        padded = copy.deepcopy(valid)
        padded["payload_nonce_b64"] = str(padded["payload_nonce_b64"]) + "="
        invalid_documents.append(padded)
        wrong_version_type = copy.deepcopy(valid)
        wrong_version_type["key_version"] = True
        invalid_documents.append(wrong_version_type)
        truncated_wrap = copy.deepcopy(valid)
        truncated_wrap["wrapped_dek_b64"] = str(truncated_wrap["wrapped_dek_b64"])[:-2]
        invalid_documents.append(truncated_wrap)
        oversized_payload = copy.deepcopy(valid)
        oversized_payload["payload_ciphertext_b64"] = "A" * 100_000
        invalid_documents.append(oversized_payload)

        for document in invalid_documents:
            with self.subTest(document=document):
                self.assert_code(
                    "memory_envelope_invalid",
                    MemoryEnvelopeV1.from_document,
                    document,
                )

    def test_payload_budget_and_nonce_uniqueness(self) -> None:
        maximum = b"x" * MAX_MEMORY_PAYLOAD_BYTES
        self.assertEqual(
            self.cipher.decrypt(
                self.cipher.encrypt(maximum, self.binding), self.binding
            ),
            maximum,
        )
        for invalid in (b"", b"x" * (MAX_MEMORY_PAYLOAD_BYTES + 1), bytearray(b"x")):
            with self.subTest(invalid_type=type(invalid), size=len(invalid)):
                self.assert_code(
                    "memory_plaintext_invalid",
                    self.cipher.encrypt,
                    invalid,
                    self.binding,
                )

        envelopes = [self.cipher.encrypt(PAYLOAD, self.binding) for _ in range(64)]
        self.assertEqual(len({item.payload_nonce for item in envelopes}), 64)
        self.assertEqual(len({item.wrap_nonce for item in envelopes}), 64)

    def test_binding_contract_rejects_noncanonical_or_mutable_fields(self) -> None:
        values = {
            "principal_id": P1,
            "record_id": R1,
            "conversation_id": C1,
            "task_id": T1,
            "source_event_id": E1,
            "created_at": "2030-01-02T03:04:05.006Z",
            "expires_at": "2031-01-02T03:04:05.006Z",
            "content_bytes": 10,
            "term_count": 2,
        }
        for override in (
            {"principal_id": "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"},
            {"principal_id": "not-a-uuid"},
            {"created_at": "2030-01-02T03:04:05Z"},
            {"expires_at": values["created_at"]},
            {"content_bytes": True},
            {"content_bytes": 4097},
            {"term_count": 0},
            {"record_schema_version": 9},
            {"normalizer_version": "jarvis.memory-normalizer/2"},
        ):
            with self.subTest(override=override):
                self.assert_code(
                    "memory_binding_invalid",
                    MemoryRecordBindingV1,
                    **{**values, **override},
                )

    def test_destroy_requires_zero_references_and_verifies_provider_deletion(
        self,
    ) -> None:
        self.assert_code(
            "memory_key_still_referenced",
            self.cipher.destroy_epoch,
            1,
            live_reference_count=1,
        )
        self.assertTrue(self.provider.exists(1))
        self.assert_code(
            "memory_key_reference_count_invalid",
            self.cipher.destroy_epoch,
            1,
            live_reference_count=True,
        )
        self.assert_code(
            "memory_key_is_active",
            self.cipher.destroy_epoch,
            1,
            live_reference_count=0,
        )
        self.assertTrue(self.provider.exists(1))
        self.cipher.create_epoch()
        self.assert_code(
            "memory_key_unavailable",
            self.cipher.destroy_epoch,
            99,
            live_reference_count=0,
        )

        class LyingProvider(InMemoryMemoryEpochKeyProvider):
            def destroy(self, version: int) -> None:
                pass

        lying = MemoryEnvelopeCipherV1(LyingProvider.initialized(ROOT_KEY))
        lying.create_epoch()
        self.assert_code(
            "memory_key_destroy_failed",
            lying.destroy_epoch,
            1,
            live_reference_count=0,
        )

    def test_key_contract_rejects_bad_lengths_and_boolean_versions(self) -> None:
        with self.assertRaises(ValueError):
            InMemoryMemoryEpochKeyProvider.initialized(b"short")
        self.assert_code(
            "memory_key_version_invalid",
            self.cipher.encrypt,
            PAYLOAD,
            self.binding,
            key_version=True,
        )

    def test_explicit_bootstrap_and_in_memory_destroy_semantics_match_production(
        self,
    ) -> None:
        provider = InMemoryMemoryEpochKeyProvider()
        cipher = MemoryEnvelopeCipherV1(provider)
        self.assert_code("memory_key_not_initialized", cipher.active_version)
        self.assert_code("memory_key_not_initialized", cipher.create_epoch)
        self.assert_code("memory_key_not_initialized", provider.load, 1)
        self.assert_code("memory_key_not_initialized", provider.exists, 1)
        self.assertEqual(cipher.bootstrap_epoch(), 1)
        self.assertEqual(cipher.bootstrap_epoch(), 1)
        self.assert_code("memory_key_active_destroy_forbidden", provider.destroy, 1)
        cipher.create_epoch()
        provider.destroy(1)
        self.assert_code("memory_key_not_found", provider.destroy, 1)


if __name__ == "__main__":
    unittest.main()
