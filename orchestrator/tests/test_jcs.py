from __future__ import annotations

import hashlib
import unittest

import _bootstrap  # noqa: F401

from jarvis_orchestrator.jcs import CanonicalJsonError, domain_sha256, jcs_bytes, jcs_text


class CanonicalJsonTests(unittest.TestCase):
    def test_frozen_unicode_integer_and_domain_encodings_remain_stable(self):
        value = {"z": [True, None, 2**63 - 1], "a": "e\u0301"}
        expected = '{"a":"é","z":[true,null,9223372036854775807]}'
        self.assertEqual(jcs_text(value), expected)
        self.assertEqual(domain_sha256("fixture/1", value), hashlib.sha256(
            b"JARVIS\x00fixture/1\x00" + expected.encode("utf-8")
        ).hexdigest())
        self.assertNotEqual(domain_sha256("fixture/1", value), domain_sha256("fixture/2", value))

    def test_invalid_unicode_and_recursive_values_raise_the_canonical_error(self):
        cycle = []
        cycle.append(cycle)
        deep = 0
        for _ in range(1500):
            deep = [deep]
        for name, value in (("surrogate", {"a": "\ud800"}), ("cycle", cycle), ("deep", deep)):
            with self.subTest(name=name), self.assertRaises(CanonicalJsonError):
                jcs_bytes(value)

    def test_domain_requires_a_nonempty_ascii_string_without_nul(self):
        for value in (None, 1, True, [], b"fixture", "", "π", "bad\x00prefix"):
            with self.subTest(value=value), self.assertRaises(CanonicalJsonError):
                domain_sha256(value, {})

    def test_floats_non_ascii_keys_and_non_json_values_are_rejected(self):
        for value in ({"a": 0.5}, {"a": float("nan")}, {"a": float("inf")}, {"β": 1}, {"a": b"bytes"}):
            with self.subTest(value=value), self.assertRaises(CanonicalJsonError):
                jcs_bytes(value)


if __name__ == "__main__":
    unittest.main()
