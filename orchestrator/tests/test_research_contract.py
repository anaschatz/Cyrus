from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from jarvis_orchestrator.research import (
    CORROBORATION_POLICY_VERSION,
    DateProvenance,
    EvidenceBundleV1,
    EvidenceClaimV1,
    EvidenceSourceV1,
    FacetResultV1,
    FacetState,
    NormalizedDocumentV1,
    RankedPassageV1,
    ResearchCorpusV1,
    ResearchStatus,
    SearchFacetV1,
    SearchPlanV1,
    SourceRole,
    SourceFingerprintV1,
    SpanRelation,
    SpanSelectionV1,
    accepted_canonical_url,
    assign_independence_groups,
    build_research_result,
    corroborates,
    coverage_state,
    make_locator,
    reconstruct,
    render_bundle,
    resolve_research_status,
    select_ranked_passages,
    validate_syntactic_citation_coverage,
)


def fixture():
    plan = SearchPlanV1(
        "Τρέχουσα τεκμηριωμένη τιμή",
        (SearchFacetV1("price", "Τρέχουσα τιμή"),),
        ("official current price",),
        "el",
        "current",
        "Europe/Athens",
    )
    document = NormalizedDocumentV1.from_text(
        "Η επίσημη τιμή του Example Product είναι 100 ευρώ στις 10 Αυγούστου 2026."
    )
    passage = RankedPassageV1(
        "p01", "source-1", document.sha256, "price", 0, len(document.text), "Τιμή"
    )
    locator = make_locator(
        SpanSelectionV1("price", "p01", 0, len(document.text), SpanRelation.ANCHOR),
        passage,
        document,
    )
    source = EvidenceSourceV1(
        "source-1",
        "https://example.com/product",
        "https://example.com/product",
        "https://example.com/product",
        "Official [price]",
        "Example Publisher",
        "2026-08-10T00:00:00.000Z",
        DateProvenance.JSON_LD,
        "2026-08-10T09:00:00.000Z",
        SourceRole.PRIMARY,
        "authority-registry/1",
        "example.com",
        "cluster-1",
        document.sha256,
        "el",
        "lid/1",
        0.99,
    )
    claim = EvidenceClaimV1("claim-1", "price", document.text, locator)
    bundle = EvidenceBundleV1(
        plan.plan_hash,
        ResearchStatus.SUFFICIENT,
        (FacetResultV1("price", FacetState.COVERED, ("claim-1",)),),
        (source,),
        (claim,),
    )
    return plan, document, locator, bundle


def corpus_fixture():
    plan, document, _, bundle = fixture()
    passage = RankedPassageV1(
        "p01", "source-1", document.sha256, "price", 0, len(document.text), "Τιμή"
    )
    selection = SpanSelectionV1(
        "price", "p01", 0, len(document.text), SpanRelation.ANCHOR
    )
    corpus = ResearchCorpusV1(
        plan.plan_hash,
        bundle.sources,
        {"source-1": document},
        (passage,),
    )
    return plan, corpus, selection, bundle


class ResearchContractTests(unittest.TestCase):
    def test_two_qualified_secondary_groups_cover_but_one_group_does_not(self) -> None:
        plan = SearchPlanV1(
            "Example Product current price",
            (SearchFacetV1("price", "Example Product current price"),),
            ("Example Product official price",),
            "en",
            "current",
            "Europe/Athens",
        )
        texts = (
            "Example Product official price is 100 euros on 2026-08-10 according to the published list.",
            "According to the published list, Example Product official price is 100 euros on 2026-08-10.",
        )
        documents = {f"s{index}": NormalizedDocumentV1.from_text(text) for index, text in enumerate(texts, 1)}
        sources = tuple(
            EvidenceSourceV1(
                f"s{index}",
                f"https://example{index}.com/a",
                f"https://example{index}.com/a",
                f"https://example{index}.com/a",
                "Example Product official price",
                f"Publisher {index}",
                None,
                DateProvenance.NONE,
                "2026-08-10T09:00:00.000Z",
                SourceRole.SECONDARY,
                "authority-registry/1",
                f"example{index}.com",
                f"cluster-{index}",
                documents[f"s{index}"].sha256,
                "en",
                "lid/1",
                0.99,
            )
            for index in (1, 2)
        )
        passages = tuple(
            RankedPassageV1(f"p{index:02d}", f"s{index}", documents[f"s{index}"].sha256, "price", 0, len(texts[index - 1]))
            for index in (1, 2)
        )
        corpus = ResearchCorpusV1(plan.plan_hash, sources, documents, passages)
        selections = (
            SpanSelectionV1("price", "p01", 0, len(texts[0]), SpanRelation.ANCHOR),
            SpanSelectionV1("price", "p02", 0, len(texts[1]), SpanRelation.SUPPORTS),
        )
        covered = build_research_result(plan, corpus, selections)
        self.assertEqual(covered.bundle.status, ResearchStatus.SUFFICIENT)
        self.assertIsNotNone(covered.bundle.claims[0].corroborating)
        same_group_sources = tuple(
            EvidenceSourceV1(
                **{
                    **{field: getattr(source, field) for field in source.__dataclass_fields__},
                    "independence_group_id": "one-cluster",
                }
            )
            for source in sources
        )
        uncovered = build_research_result(
            plan,
            ResearchCorpusV1(plan.plan_hash, same_group_sources, documents, passages),
            selections,
        )
        self.assertEqual(uncovered.bundle.status, ResearchStatus.INSUFFICIENT)

    def test_model_span_selection_becomes_code_derived_bundle(self) -> None:
        plan, corpus, selection, expected = corpus_fixture()
        result = build_research_result(plan, corpus, (selection,))
        self.assertEqual(result.bundle.status, expected.status)
        self.assertEqual(result.bundle.claims[0].statement, expected.claims[0].statement)
        self.assertEqual(result.bundle.sources, expected.sources)
        self.assertEqual(result.cited_documents["source-1"].sha256, corpus.documents["source-1"].sha256)

    def test_quote_first_locator_reconstructs_exact_normalized_substring(self) -> None:
        _, document, locator, bundle = fixture()
        self.assertEqual(reconstruct(locator, document), document.text)
        self.assertEqual(locator.document_hash, document.sha256)
        validate_syntactic_citation_coverage(bundle, {"source-1": document})
        with self.assertRaisesRegex(ValueError, "outside passage"):
            passage = RankedPassageV1("p01", "source-1", document.sha256, "price", 0, 50)
            make_locator(
                SpanSelectionV1("price", "p01", 0, len(document.text), SpanRelation.ANCHOR),
                passage,
                document,
            )

    def test_status_precedence_and_conflict_override_are_exact(self) -> None:
        self.assertEqual(resolve_research_status([FacetState.COVERED]), ResearchStatus.SUFFICIENT)
        self.assertEqual(
            resolve_research_status([FacetState.COVERED, FacetState.TECHNICAL_FAILURE]),
            ResearchStatus.PARTIAL,
        )
        self.assertEqual(
            resolve_research_status([FacetState.UNCOVERED, FacetState.CONFLICTED]),
            ResearchStatus.INSUFFICIENT,
        )
        self.assertEqual(
            resolve_research_status([FacetState.UNCOVERED, FacetState.TECHNICAL_FAILURE]),
            ResearchStatus.FAILED,
        )
        self.assertEqual(
            coverage_state(conflict=True, primary_anchor=True, corroborating_secondary_groups=2),
            FacetState.CONFLICTED,
        )

    def test_corroboration_is_conservative_qualified_heuristic(self) -> None:
        anchor = "Example Product cost 100 euros on 2026-08-10 according to the official list."
        candidate = "The official list says Example Product cost 100 euros on 2026-08-10."
        common = dict(
            relation=SpanRelation.SUPPORTS,
            anchor_entities=("Example Product",),
            candidate_entities=("example product",),
            different_independence_group=True,
            conflict_candidate_present=False,
        )
        self.assertTrue(corroborates(anchor, candidate, **common))
        self.assertFalse(corroborates(anchor, candidate.replace("100", "101"), **common))
        self.assertFalse(corroborates(anchor, candidate.replace("2026-08-10", "2026-10-08"), **common))
        self.assertFalse(corroborates(anchor.replace("100", "10000"), candidate.replace("100", "20000"), **common))
        self.assertFalse(corroborates(anchor.replace("100", "-100"), candidate, **common))
        self.assertTrue(corroborates(anchor.replace("100", "+100"), candidate, **common))
        self.assertFalse(corroborates(anchor, candidate, **{**common, "relation": SpanRelation.CONFLICTS}))
        self.assertFalse(corroborates(anchor, candidate, **{**common, "conflict_candidate_present": True}))
        self.assertEqual(CORROBORATION_POLICY_VERSION, "jarvis-corroboration/1")

    def test_model_span_offsets_reject_floats_and_boolean_endpoints(self) -> None:
        for start, end in ((0.5, 50), (0, 50.5), (0, True)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                SpanSelectionV1("price", "p01", start, end, SpanRelation.ANCHOR)

    def test_long_paragraphs_produce_bounded_reconstructable_passages(self) -> None:
        document = NormalizedDocumentV1.from_text(
            "The published report contains a verified research finding with explanatory context. " * 16
        )
        passages = select_ranked_passages({"s1": document}, (SearchFacetV1("f1", "verified research finding"),))
        self.assertTrue(passages)
        for passage in passages:
            self.assertLessEqual(passage.end_char - passage.start_char, 1000)
            self.assertEqual(passage.document_hash, document.sha256)
            self.assertTrue(document.text[passage.start_char:passage.end_char])

    def test_global_passage_budget_does_not_starve_later_facets(self) -> None:
        document = NormalizedDocumentV1.from_text("\n\n".join(
            f"The official report number {index} provides a verified research finding with supporting context."
            for index in range(4)
        ))
        facets = tuple(SearchFacetV1(f"f{index}", "verified research finding") for index in range(6))
        passages = select_ranked_passages({"s1": document}, facets)
        self.assertEqual(len(passages), 12)
        self.assertEqual({passage.facet_id for passage in passages}, {facet.facet_id for facet in facets})

    def test_renderer_has_syntactic_citations_and_escapes_hostile_markdown(self) -> None:
        plan, _, _, bundle = fixture()
        rendered = render_bundle(plan, bundle)
        self.assertIn("claim", bundle.claims[0].claim_id)
        self.assertIn("https://example.com/product", rendered)
        self.assertIn(r"Official \[price\]", rendered)
        self.assertNotIn("[price]", rendered)

    def test_canonical_url_is_never_authoritative_without_validation_or_match(self) -> None:
        final = "https://example.com/article"
        self.assertEqual(
            accepted_canonical_url(final, "https://evil.example.net/copy", same_registrable_domain=False, content_matches=False),
            final,
        )
        self.assertEqual(
            accepted_canonical_url(final, "https://example.com/canonical", same_registrable_domain=True, content_matches=False),
            "https://example.com/canonical",
        )
        with self.assertRaises(ValueError):
            accepted_canonical_url("https://user:pass@example.com/", None, same_registrable_domain=True, content_matches=False)
        with self.assertRaises(ValueError):
            accepted_canonical_url("https://example.com/?%74oken=secret", None, same_registrable_domain=True, content_matches=False)
        with self.assertRaises(ValueError):
            accepted_canonical_url("https://example.com/%0Aattack", None, same_registrable_domain=True, content_matches=False)

    def test_syndicated_near_duplicates_are_one_independence_group_across_domains(self) -> None:
        text = "Example Product official price is 100 euros on 2026-08-10. " * 12
        first = NormalizedDocumentV1.from_text(text)
        second = NormalizedDocumentV1.from_text(text.replace("official price", "official listed price", 1))
        groups = assign_independence_groups(
            (
                SourceFingerprintV1("a", "example.com", "Official Example Product price", "Wire", None, first),
                SourceFingerprintV1("b", "mirror.example", "Official Example Product price", "Wire", None, second),
            )
        )
        self.assertEqual(groups["a"], groups["b"])


if __name__ == "__main__":
    unittest.main()
