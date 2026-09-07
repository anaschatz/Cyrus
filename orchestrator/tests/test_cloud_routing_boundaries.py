from dataclasses import asdict, replace
import unittest

import _bootstrap  # noqa: F401
from jarvis_orchestrator.cloud_routing import (
    CandidateFactsV1, CloudContractError, DeterministicModelRouterV1,
    RoutingFactsV1, TaskProfileV1,
)
from jarvis_orchestrator.jcs import domain_sha256
from test_m1c_cloud_contract import candidate, facts, profile


class CloudRoutingBoundaryTests(unittest.TestCase):
    def test_profile_rejects_invalid_shapes_and_direct_enum_values(self):
        selected = profile()
        for field, invalid in (("domains", [{}]), ("domains", [None]),
                               ("task_kind", []), ("context_scope", {}),
                               ("output_contract", "unknown"), ("response_language", [])):
            with self.subTest(field=field, invalid=invalid):
                with self.assertRaises(CloudContractError):
                    TaskProfileV1.from_mapping({**selected.canonical(), field: invalid})
        for field in ("task_kind", "context_scope", "output_contract"):
            with self.subTest(direct=field):
                with self.assertRaises(CloudContractError):
                    replace(selected, **{field: "unknown"}).validate()

    def test_candidate_rejects_non_string_digests_and_non_list_capabilities(self):
        selected = candidate("deepseek")
        for field in ("stable_catalog_sha256", "qualification_sha256"):
            for invalid in (None, False, 0, [], 123):
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaises(CloudContractError):
                        CandidateFactsV1.from_mapping({**asdict(selected), field: invalid})
        for invalid in (None, "text", {"text": True}):
            with self.subTest(capabilities=invalid):
                with self.assertRaises(CloudContractError):
                    replace(selected, capabilities=invalid).validate()

    def test_router_rejects_invalid_facts_before_producing_a_plan(self):
        selected = profile()
        evidence = facts("synthetic-task", selected.sha256)
        for field, invalid in (("epoch", True), ("epoch", 1.5), ("epoch", "1"),
                               ("task_id", None), ("task_id", ""), ("task_id", 123),
                               ("profile_sha256", None), ("approval_status", []),
                               ("flow_kind", {}), ("local_failure", []), ("local", None)):
            with self.subTest(field=field, invalid=invalid):
                with self.assertRaises(CloudContractError):
                    DeterministicModelRouterV1.decide(selected, replace(evidence, **{field: invalid}))
                with self.assertRaises(CloudContractError):
                    RoutingFactsV1.from_mapping({**evidence.canonical(), field: invalid})

    def test_mapping_boundaries_reject_non_objects_and_preserve_valid_hashes(self):
        for contract in (TaskProfileV1, CandidateFactsV1, RoutingFactsV1):
            for invalid in (None, 1, [dict()]):
                with self.subTest(contract=contract.__name__, invalid=invalid):
                    with self.assertRaises(CloudContractError):
                        contract.from_mapping(invalid)
        selected = profile()
        evidence = facts("synthetic-task", selected.sha256)
        restored = RoutingFactsV1.from_mapping(evidence.canonical())
        self.assertEqual(restored, evidence)
        self.assertEqual(restored.sha256, domain_sha256("routing-facts/1", asdict(evidence)))
        self.assertEqual(TaskProfileV1.from_mapping(selected.canonical()), selected)
        self.assertEqual(DeterministicModelRouterV1.decide(selected, evidence),
                         DeterministicModelRouterV1.decide(selected, restored))


if __name__ == "__main__":
    unittest.main()
