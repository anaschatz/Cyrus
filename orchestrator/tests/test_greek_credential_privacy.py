from __future__ import annotations

import asyncio
import unittest
import unicodedata
from dataclasses import replace

import _bootstrap  # noqa: F401
from jarvis_orchestrator.cloud_privacy import JarvisPrivacyClassifierV1, PrivacyClass
from jarvis_orchestrator.cloud_routing import cloud_eligible, RouteReason
from test_m1c_cloud_contract import candidate, facts, profile


class GreekCredentialPrivacyTests(unittest.TestCase):
    def test_greek_credential_terms_and_inflections_are_not_cloud_eligible(self):
        classifier = JarvisPrivacyClassifierV1()
        for phrase in (
            "Ο κωδικός μου είναι αβγ",
            "Ο κωδικοσ μου είναι αβγ",
            "Το συνθηματικό μου είναι αβγ",
            "Το ιδιωτικό κλειδί είναι αβγ",
            "Οι κωδικοί μου είναι αβγ",
            "Λίστα κωδικών: αβγ",
            "Η αλλαγή του κωδικού μου",
            "Αλλάζω τους κωδικούς μου",
        ):
            for text in (phrase, phrase.upper(), unicodedata.normalize("NFD", phrase)):
                with self.subTest(text=text):
                    result = classifier.classify(text)
                    self.assertEqual(
                        result.classification, PrivacyClass.SECRET_CREDENTIAL
                    )
                    self.assertFalse(result.cloud_eligible)

    def test_code_and_encoding_terms_remain_public(self):
        for text in (
            "Βρες το σφάλμα στον κώδικά μου",
            "ΚΩΔΙΚΑΣ PYTHON",
            "κωδικοποίηση Unicode",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    JarvisPrivacyClassifierV1().classify(text).classification,
                    PrivacyClass.PUBLIC,
                )

    def test_combined_egress_denies_cloud_even_with_approval(self):
        selected = profile()
        privacy = asyncio.run(
            JarvisPrivacyClassifierV1().classify_bounded(
                ("Public product documentation", "Ο κωδικός μου είναι αβγ")
            )
        )
        routing = replace(
            facts("synthetic-task", selected.sha256, approval="approved"),
            privacy_eligible=privacy.cloud_eligible,
        )
        self.assertEqual(
            cloud_eligible(candidate("deepseek"), selected, routing),
            (False, RouteReason.CLOUD_PRIVACY_DENIED),
        )


if __name__ == "__main__":
    unittest.main()
