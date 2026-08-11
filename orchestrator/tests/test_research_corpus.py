from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from jarvis_orchestrator.research import DateProvenance, SearchFacetV1, SearchPlanV1, SourceRole
from jarvis_research.adapters import FetchResultV1, RetrievalCorpusV1
from jarvis_research.corpus import PSL_SHA256, AuthorityRegistryV1, CorpusBuilderV1, PinnedLanguageDetectorV1, PublicSuffixRegistryV1


def html(title: str, paragraph: str, *, date_meta: str = "", canonical: str = "") -> bytes:
    return (
        "<html><head><title>" + title + "</title>" + date_meta
        + (f"<link rel='canonical' href='{canonical}'>" if canonical else "")
        + "</head><body><main><p>" + paragraph + "</p></main></body></html>"
    ).encode()


class CorpusConstructionTests(unittest.TestCase):
    def plan(self) -> SearchPlanV1:
        return SearchPlanV1(
            "Verified energy report", (SearchFacetV1("energy", "verified energy report output"),),
            ("verified energy report",), "en", "current", "UTC",
        )

    def test_language_boundary_and_psl_are_pinned(self) -> None:
        detector = PinnedLanguageDetectorV1()
        self.assertEqual(detector.detect("English language evidence. " * 20).code, "en")
        self.assertEqual(detector.detect("Ελληνικό τεκμήριο και πληροφορία. " * 20).code, "el")
        psl = PublicSuffixRegistryV1()
        self.assertEqual(psl.registrable_domain("https://news.example.co.uk/a"), "example.co.uk")
        self.assertEqual(psl.sha256, PSL_SHA256)
        self.assertEqual(AuthorityRegistryV1().role("www.gov.gr"), SourceRole.PRIMARY)

    def test_deterministic_corpus_dates_roles_dedupe_and_passages(self) -> None:
        paragraph = (
            "The World Health Organization published this verified energy report output with exact factual context "
            "for deterministic passage ranking and quote first evidence reconstruction. " * 2
        )
        body = html(
            "Verified report", paragraph,
            date_meta="<meta property='article:published_time' content='2026-08-10T10:00:00Z'>",
            canonical="https://www.who.int/canonical",
        )
        duplicate = FetchResultV1("https://mirror.example.com/item", "https://mirror.example.com/item", "text/html", body, 0)
        retrieval = RetrievalCorpusV1(
            (),
            (
                FetchResultV1("https://www.who.int/item", "https://www.who.int/item", "text/html", body, 0),
                duplicate,
                FetchResultV1(
                    "https://unsupported.example.org/item", "https://unsupported.example.org/item", "text/html",
                    html("French", "Ceci est un document en langue francaise qui ne doit pas devenir une preuve citee. " * 5), 0,
                ),
            ),
            (),
        )
        corpus = CorpusBuilderV1(clock=lambda: "2026-08-10T12:00:00.000Z").build(self.plan(), retrieval, network_retry_count=1)
        self.assertEqual(len(corpus.sources), 1)
        source = corpus.sources[0]
        self.assertEqual(source.source_role, SourceRole.PRIMARY)
        self.assertEqual(source.published_at, "2026-08-10T10:00:00.000Z")
        self.assertEqual(source.date_provenance, DateProvenance.ARTICLE_PUBLISHED_TIME)
        self.assertEqual(source.accepted_canonical_url, "https://www.who.int/canonical")
        self.assertTrue(corpus.passages)
        self.assertEqual(corpus.network_retry_count, 1)

    def test_conflicting_metadata_never_invents_date(self) -> None:
        paragraph = "This verified report contains enough English language material for deterministic eligibility and extraction. " * 4
        body = html(
            "Conflict", paragraph,
            date_meta=("<meta property='article:published_time' content='2026-08-10T10:00:00Z'>"
                       "<meta name='datePublished' content='2026-08-09T10:00:00Z'>"),
        )
        corpus = CorpusBuilderV1(clock=lambda: "2026-08-10T12:00:00.000Z").build(
            self.plan(), RetrievalCorpusV1((), (FetchResultV1("https://example.com/a", "https://example.com/a", "text/html", body, 0),), ()), network_retry_count=0,
        )
        self.assertIsNone(corpus.sources[0].published_at)
        self.assertEqual(corpus.sources[0].date_provenance, DateProvenance.CONFLICT)


if __name__ == "__main__":
    unittest.main()
