from __future__ import annotations

import hashlib
import asyncio
import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401
import httpx
from test_research_contract import corpus_fixture

from jarvis_research.online_qualification import (
    DEFAULT_THRESHOLDS,
    BenchmarkV1,
    CaseEvaluationV1,
    RankedResultV1,
    QrelV1,
    RegistryImageV1,
    OnlineRetrievalRunnerV1,
    classification_metrics,
    compute_retrieval_metrics,
    main,
    parse_benchmark,
    parse_image_manifest,
    parse_threshold_manifest,
    performance_metrics,
    query_privacy_result,
    threshold_decisions,
    verify_cancellation_snapshot,
    verify_evidence_digest,
    verify_quote_locator,
    verify_runtime_image,
    _matches,
)


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "research" / "benchmarks" / "retrieval-benchmark-seed-v1.json"


def reviewed_benchmark() -> BenchmarkV1:
    value = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    value["review_state"] = "REVIEWED"
    value["review_provenance"] = "human-review-record/fixture"
    for item in value["cases"]:
        item["review_state"] = "REVIEWED"
        item["review_provenance"] = "human-review-record/fixture"
        item["qrels"] = [{
            "qrel_id": "official",
            "target_kind": "DOMAIN",
            "target": "official.example",
            "relevance": 2,
            "primary": True,
            "facet_ids": ["answer"],
        }]
    return parse_benchmark(value)


class OnlineResearchQualificationTests(unittest.TestCase):
    def test_checked_in_seed_is_strict_balanced_unreviewed_and_digest_stable(self) -> None:
        raw = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        benchmark = parse_benchmark(raw)
        self.assertEqual(benchmark.review_state, "UNREVIEWED")
        self.assertEqual(len(benchmark.cases), 100)
        self.assertEqual(sum(item.language == "el" for item in benchmark.cases), 50)
        self.assertEqual(sum(item.language == "en" for item in benchmark.cases), 50)
        self.assertEqual(
            benchmark.sha256,
            "ecc318c5cd50d1182a85efc63ea1e7e459999de5aaf50d6484f6210218d5196f",
        )
        self.assertEqual(
            benchmark.qrels_sha256,
            "52207d67bb951abe8dc1c5b24010d3203a81354de3676981eb36c680821037d6",
        )

    def test_reviewed_state_requires_qrels_and_provenance_for_every_case(self) -> None:
        value = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        value["review_state"] = "REVIEWED"
        value["review_provenance"] = "review/fixture"
        with self.assertRaisesRegex(ValueError, "review"):
            parse_benchmark(value)
        reviewed_benchmark()

    def test_registry_images_require_repository_manifest_digest_and_runtime_match(self) -> None:
        declared = RegistryImageV1(
            "research-api",
            "registry.example/jarvis/research@sha256:" + "a" * 64,
            "linux/arm64",
            "b" * 64,
        )
        self.assertTrue(verify_runtime_image(
            declared,
            repo_digests=[declared.reference],
            configuration_id="sha256:" + "b" * 64,
            os_name="linux",
            architecture="arm64",
        ))
        self.assertFalse(verify_runtime_image(
            declared,
            repo_digests=[declared.reference],
            configuration_id="sha256:" + "c" * 64,
            os_name="linux",
            architecture="arm64",
        ))
        with self.assertRaisesRegex(ValueError, "digest_pinned"):
            RegistryImageV1("research-api", "registry.example/jarvis/research:latest", "linux/arm64", "b" * 64)

    def test_image_manifest_is_closed_and_has_exactly_three_roles(self) -> None:
        images = [
            {
                "name": name,
                "reference": f"registry.example/jarvis/{name}@sha256:" + str(index) * 64,
                "platform": "linux/arm64",
                "configuration_sha256": "f" * 64,
            }
            for index, name in enumerate(sorted(("research-api", "egress-guard", "searxng")), 1)
        ]
        parsed = parse_image_manifest({"version": "research-image-manifest/1", "images": images})
        self.assertEqual({item.name for item in parsed}, {"research-api", "egress-guard", "searxng"})
        invalid = {"version": "research-image-manifest/1", "images": images, "extra": True}
        with self.assertRaisesRegex(ValueError, "schema"):
            parse_image_manifest(invalid)

    def test_thresholds_are_explicit_versioned_and_closed(self) -> None:
        path = ROOT / "research" / "online-qualification-thresholds-v1.json"
        parsed = parse_threshold_manifest(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(parsed, DEFAULT_THRESHOLDS)
        invalid = {"version": "online-retrieval-thresholds/1", "thresholds": {**parsed, "hidden": 1}}
        with self.assertRaisesRegex(ValueError, "threshold"):
            parse_threshold_manifest(invalid)

    def test_recall_mrr_primary_and_language_aggregates(self) -> None:
        benchmark = reviewed_benchmark()
        result = RankedResultV1("https://official.example/page", "official.example", "official", "PRIMARY")
        evaluations = [
            CaseEvaluationV1(
                item.case_id, item.language, "SUCCESS", (result,), ("answer",),
                1.0, 100, 200, 80, 0,
            )
            for item in benchmark.cases
        ]
        metrics = compute_retrieval_metrics(benchmark, evaluations)
        self.assertEqual(metrics["case_count"], 100)
        self.assertEqual(metrics["overall_macro"]["recall_at_1"], 1.0)
        self.assertEqual(metrics["overall_macro"]["mrr_at_10"], 1.0)
        self.assertEqual(metrics["per_language"]["el"]["recall_at_10"], 1.0)
        self.assertTrue(all(threshold_decisions(metrics, DEFAULT_THRESHOLDS).values()))

    def test_metrics_reject_incomplete_case_set(self) -> None:
        benchmark = reviewed_benchmark()
        with self.assertRaisesRegex(ValueError, "case_set"):
            compute_retrieval_metrics(benchmark, [])

    def test_online_parser_accepts_the_actual_research_corpus_wire_contract(self) -> None:
        _, corpus, _, _ = corpus_fixture()
        wire = json.loads(json.dumps({"corpus": asdict(corpus)}))
        ranked, covered, size, retries = OnlineRetrievalRunnerV1._parse_corpus(wire)
        self.assertEqual(ranked[0].url, corpus.sources[0].accepted_canonical_url)
        self.assertEqual(covered, ("price",))
        self.assertEqual(size, sum(document.byte_length for document in corpus.documents.values()))
        self.assertEqual(retries, 0)

    def test_qrel_urls_normalize_hosts_but_preserve_path_and_query_case(self) -> None:
        qrel = QrelV1("one", "URL", "https://EXAMPLE.com/Article?q=Value", 2, True, ("answer",))
        for url, expected in (
            ("https://example.com:443/Article?q=Value", True),
            ("https://example.com/article?q=Value", False),
            ("https://example.com/Article?q=value", False),
        ):
            with self.subTest(url=url):
                self.assertEqual(_matches(qrel, RankedResultV1(url, "example.com", "s1", "PRIMARY")), expected)

    def test_live_runner_sends_only_benchmark_query_and_persists_sanitized_summaries(self) -> None:
        benchmark = reviewed_benchmark()
        outgoing = []
        document = "An exact normalized source passage that is long enough for the qualification fixture."
        document_hash = hashlib.sha256(document.encode()).hexdigest()

        def handler(request: httpx.Request) -> httpx.Response:
            value = json.loads(request.content)
            outgoing.extend(value["plan"]["subqueries"])
            corpus = {
                "plan_hash": "a" * 64,
                "sources": [{
                    "source_id": "source-1",
                    "requested_url": "https://official.example/page",
                    "final_url": "https://official.example/page",
                    "accepted_canonical_url": "https://official.example/page",
                    "title": "Official publication",
                    "publisher": "Publisher",
                    "published_at": None,
                    "date_provenance": "NONE",
                    "retrieved_at": "2026-09-05T00:00:00.000Z",
                    "authority_registry_version": "authority-registry/1",
                    "registrable_domain": "official.example",
                    "source_role": "PRIMARY",
                    "independence_group_id": "ig-1",
                    "normalized_document_hash": document_hash,
                    "language_code": "en",
                    "language_detector_version": "lid/1",
                    "language_confidence": 0.99,
                }],
                "documents": {"source-1": {
                    "normalization_version": "jarvis-research-text/1",
                    "text": document,
                    "sha256": document_hash,
                    "character_length": len(document),
                    "byte_length": len(document.encode()),
                    "truncated": False,
                }},
                "passages": [{
                    "passage_id": "p01", "facet_id": "answer", "source_id": "source-1",
                    "document_hash": document_hash, "start_char": 0, "end_char": len(document),
                    "section_heading": None,
                }],
                "technical_failure_facets": [],
                "network_retry_count": 0,
            }
            return httpx.Response(200, json={"corpus": corpus})

        async def execute():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await OnlineRetrievalRunnerV1(
                    benchmark,
                    api_base_url="http://127.0.0.1:8770/v1",
                    bearer_token="x" * 32,
                    client=client,
                ).run()

        evaluations, summaries = asyncio.run(execute())
        self.assertEqual(len(evaluations), 100)
        self.assertEqual(len(summaries), 100)
        self.assertEqual(outgoing, [item.canonical_query for item in benchmark.cases])
        serialized = json.dumps(summaries)
        self.assertNotIn("official.example", serialized)
        self.assertNotIn(document, serialized)
        self.assertTrue(all(item["qrel_matches"] == [{"qrel_id": "official", "rank": 1}] for item in summaries))

    def test_quote_locator_requires_document_hash_offsets_and_exact_substring(self) -> None:
        document = "Alpha verified sentence. Beta contextual sentence."
        digest = hashlib.sha256(document.encode()).hexdigest()
        self.assertTrue(verify_quote_locator(document, digest, 0, 24, "Alpha verified sentence."))
        self.assertFalse(verify_quote_locator(document, digest, 0, 23, "Alpha verified sentence."))
        self.assertFalse(verify_quote_locator(document, "0" * 64, 0, 24, "Alpha verified sentence."))

    def test_performance_and_classification_metrics_are_deterministic(self) -> None:
        evaluations = [
            CaseEvaluationV1("case-1", "en", "SUCCESS", (), (), float(index), 10, 20, 30, 0)
            for index in range(1, 101)
        ]
        performance = performance_metrics(evaluations)
        self.assertEqual(performance["p95_latency_seconds"], 95.0)
        self.assertFalse(performance["within_frozen_bounds"])
        classified = classification_metrics([True, True, False, False], [True, False, True, False])
        self.assertEqual(classified["precision"], 0.5)
        self.assertEqual(classified["recall"], 0.5)

    def test_privacy_result_exposes_only_canary_ids_not_values(self) -> None:
        secret = "memory-canary-very-secret-value"
        clean = query_privacy_result(
            approved_queries=["approved public query"],
            outgoing_queries=["approved public query"],
            canaries={"memory-1": secret},
        )
        self.assertEqual(clean["status"], "PASS")
        leaked = query_privacy_result(
            approved_queries=["approved public query"],
            outgoing_queries=["approved public query " + secret],
            canaries={"memory-1": secret},
        )
        self.assertEqual(leaked["status"], "FAIL")
        self.assertNotIn(secret, json.dumps(leaked))

    def test_cancellation_snapshot_requires_tombstone_and_zero_late_effects(self) -> None:
        snapshot = {
            "tombstoned": True,
            "bundle_count": 0,
            "document_count": 0,
            "effect_count": 0,
            "text_output_count": 0,
            "speech_output_count": 0,
        }
        self.assertTrue(verify_cancellation_snapshot(snapshot))
        changed = deepcopy(snapshot)
        changed["bundle_count"] = 1
        self.assertFalse(verify_cancellation_snapshot(changed))

    def test_unreviewed_seed_and_placeholder_images_emit_canonical_rejected_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence.json"
            status = main([
                "--benchmark", str(BENCHMARK_PATH),
                "--images", str(ROOT / "research" / "image-manifest.template.json"),
                "--thresholds", str(ROOT / "research" / "online-qualification-thresholds-v1.json"),
                "--output", str(output),
            ])
            self.assertEqual(status, 1)
            evidence = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(evidence["gate_status"], "REJECTED")
            self.assertIn("benchmark_qrels_unreviewed", evidence["blockers"])
            self.assertFalse(evidence["feature_state"]["research_enabled"])
            self.assertEqual(evidence["feature_state"]["browser_qualification_sha256"], "")
            verify_evidence_digest(evidence)


if __name__ == "__main__":
    unittest.main()
