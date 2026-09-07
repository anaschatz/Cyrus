from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from jarvis_orchestrator.cloud_privacy import JarvisPrivacyClassifierV1, PrivacyClass


class CloudPrivacyBoundaryTests(unittest.TestCase):
    def test_greek_inflections_match_the_declared_sensitive_stems(self):
        classifier = JarvisPrivacyClassifierV1()
        examples = {
            PrivacyClass.SENSITIVE_FINANCIAL: ("τραπεζικός λογαριασμός", "πιστωτική κάρτα", "χρεωστική κάρτα"),
            PrivacyClass.SENSITIVE_HEALTH: ("ασθενής", "φάρμακα", "θεραπεία"),
            PrivacyClass.SENSITIVE_LEGAL: ("δικογραφία", "ποινική υπόθεση", "δικηγορικό απόρρητο"),
            PrivacyClass.BIOMETRIC: ("δακτυλικά αποτυπώματα", "βιομετρικά στοιχεία"),
            PrivacyClass.SENSITIVE_INTIMATE_OR_MINOR: ("ανήλικος", "σεξουαλική ζωή"),
            PrivacyClass.SENSITIVE_IDENTITY: ("ταυτότητα", "διαβατήριο"),
        }
        for expected, phrases in examples.items():
            for phrase in phrases:
                with self.subTest(phrase=phrase):
                    result = classifier.classify("Ιδιωτικά στοιχεία: " + phrase)
                    self.assertEqual(result.classification, expected)
                    self.assertFalse(result.cloud_eligible)
        self.assertTrue(classifier.classify("Public product documentation").cloud_eligible)
        self.assertNotEqual(classifier.policy_sha256,
                            "be7aa1f797351285ba3e313610b1458f31315abb63bddbe0e55022d5360efd88")


if __name__ == "__main__":
    unittest.main()
