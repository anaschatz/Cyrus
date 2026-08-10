from __future__ import annotations

import math
import unittest
from decimal import Decimal

from jarvis_inference.strict_json import canonical_text, parse_body


class Rfc8785Tests(unittest.TestCase):
    def test_utf16_key_ordering(self) -> None:
        value = {"\ufb33": 1, "😀": 2, "€": 3, "ö": 4, "1": 5, "\r": 6}
        self.assertEqual(
            canonical_text(value),
            '{"\\r":6,"1":5,"ö":4,"€":3,"😀":2,"דּ":1}',
        )

    def test_strings_are_compact_and_minimally_escaped(self) -> None:
        self.assertEqual(
            canonical_text({"s": 'quote" slash/ backslash\\\b\t\n\f\r\u000f'}),
            '{"s":"quote\\\" slash/ backslash\\\\\\b\\t\\n\\f\\r\\u000f"}',
        )

    def test_ecmascript_number_rendering(self) -> None:
        cases = {
            Decimal("-0"): "0",
            42: "42",
            Decimal("4.50"): "4.5",
            Decimal("333333333.33333329"): "333333333.3333333",
            Decimal("0.000001"): "0.000001",
            Decimal("0.0000001"): "1e-7",
            Decimal("100000000000000000000"): "100000000000000000000",
            Decimal("1000000000000000000000"): "1e+21",
            Decimal("1E30"): "1e+30",
            Decimal("2e-3"): "0.002",
            Decimal("1e-27"): "1e-27",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(canonical_text(value), expected)

    def test_nonfinite_and_unsafe_integer_are_outside_supported_domain(self) -> None:
        for value in (math.nan, math.inf, -math.inf, Decimal("NaN"), 2**53):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                canonical_text(value)
        for raw in (b'{"x":9007199254740992}', b'{"x":"\\ud800"}'):
            with self.subTest(raw=raw), self.assertRaises(Exception):
                parse_body(raw)

    def test_input_order_and_whitespace_do_not_change_canonical_form(self) -> None:
        left = parse_body(b'{ "z" : 1.0, "a" : [ true, null ] }')
        right = parse_body(b'{"a":[true,null],"z":1}')
        self.assertEqual(canonical_text(left), canonical_text(right))
        self.assertEqual(canonical_text(left), '{"a":[true,null],"z":1}')


if __name__ == "__main__":
    unittest.main()
