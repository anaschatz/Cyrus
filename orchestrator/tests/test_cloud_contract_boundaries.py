from __future__ import annotations

import json
import asyncio
import tempfile
import unittest
import uuid
from dataclasses import asdict, replace
from decimal import Decimal, localcontext
from pathlib import Path
from unittest.mock import Mock

import httpx

import _bootstrap  # noqa: F401

from test_cloud_release_qualification import DEEPSEEK_MODEL, endpoint
from jarvis_orchestrator.cloud_contract import (
    CloudApprovalBindingV1, CloudBindingError, EndpointHealthV1, build_dispatch_body,
    normalize_decimal, reserve_cost_microusd,
)
from jarvis_orchestrator.cloud_provider import OpenRouterCloudAdapter, _decimal_microusd, load_endpoint_registry
from jarvis_orchestrator.domain import OrchestratorError


class CloudContractBoundaryTests(unittest.TestCase):
    def test_reservations_reject_invalid_rates_at_the_shared_boundary(self):
        baseline = dict(prompt_cap=1000, completion_cap=100, prompt_rate_per_m='1',
                        completion_rate_per_m='1', request_fee_usd='0')
        for field in ('prompt_rate_per_m', 'completion_rate_per_m', 'request_fee_usd'):
            for value in ('-1', 'NaN', 'sNaN', 'Infinity', 'invalid', True, None, 0.1, [], {}):
                with self.subTest(field=field, value=value), self.assertRaises(CloudBindingError):
                    reserve_cost_microusd(**(baseline | {field: value}))

    def test_prices_and_reservation_ceiling_do_not_depend_on_decimal_context(self):
        precise = '1.0000000000000000000000000001'
        for precision in (6, 28, 50):
            with self.subTest(precision=precision), localcontext() as context:
                context.prec = precision
                self.assertEqual(normalize_decimal(precise), precise)
                self.assertEqual(reserve_cost_microusd(prompt_cap=10, completion_cap=0,
                    prompt_rate_per_m='0.9090909090909090909090909091', completion_rate_per_m='0'), 11)
                self.assertEqual(reserve_cost_microusd(prompt_cap=60_000, completion_cap=2000,
                    prompt_rate_per_m='2.50', completion_rate_per_m='15'), 198_000)
        for value, expected in (('2.5000', '2.5'), ('1e2', '100'), ('-0.000', '0'), ('0.0000100', '0.00001')):
            self.assertEqual(normalize_decimal(value), expected)

    def test_extreme_decimal_expansion_is_rejected_before_formatting(self):
        for value in ('1e100000', '1e-100000', '9' * 1025, Decimal('NaN'), Decimal('sNaN')):
            with self.subTest(value=str(value)[:20]), self.assertRaises(CloudBindingError):
                normalize_decimal(value)

    def test_endpoint_shapes_are_validated_before_operations(self):
        selected = endpoint(DEEPSEEK_MODEL)
        invalid = [(field, value) for field in ('context_limit', 'output_limit')
                   for value in (True, 0.5, '2048', None)]
        invalid += [(field, value) for field in ('model_canonical_id', 'provider_endpoint_tag',
                    'quantization', 'qualification_version', 'runtime_provider_name', 'qualification_sha256')
                    for value in ([], {}, 42)]
        invalid += [('supported_parameters', ('max_tokens', [])), ('supported_parameters', 'max_tokens')]
        for field, value in invalid:
            with self.subTest(field=field, value=value), self.assertRaises(CloudBindingError):
                replace(selected, **{field: value}).validate()

    def test_registry_does_not_hide_wrong_types_with_string_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'registry.json'
            for field, value in (('provider_endpoint_tag', 42), ('qualification_sha256', int('1' * 64)),
                                 ('prompt_rate_per_m', True), ('context_limit', True)):
                raw = asdict(endpoint(DEEPSEEK_MODEL)) | {field: value}
                path.write_text(json.dumps({'version': 'endpoint-registry/1', 'endpoints': [raw]}))
                with self.subTest(field=field), self.assertRaises((CloudBindingError, RuntimeError)):
                    load_endpoint_registry(path)

    def test_dispatch_message_and_limit_shapes_fail_with_binding_error(self):
        selected = endpoint(DEEPSEEK_MODEL)
        for message in (None, [], {'role': [], 'content': 'text'}, {'role': 'user', 'content': 42}):
            with self.subTest(message=message), self.assertRaises(CloudBindingError):
                build_dispatch_body(selected, messages=(message,), output_token_limit=256)
        for limit in (256.5, '256', None):
            with self.subTest(limit=limit), self.assertRaises(CloudBindingError):
                build_dispatch_body(selected, messages=({'role': 'user', 'content': 'text'},), output_token_limit=limit)

    def test_empty_message_collection_and_malformed_approval_fields_are_rejected(self):
        selected = endpoint(DEEPSEEK_MODEL)
        for messages in (None, (), []):
            with self.subTest(messages=messages), self.assertRaises(CloudBindingError):
                build_dispatch_body(selected, messages=messages, output_token_limit=256)
        binding = CloudApprovalBindingV1(
            conversation_id=str(uuid.uuid4()), task_id=str(uuid.uuid4()), epoch=1,
            expires_at="2026-09-06T20:00:00Z", egress_sha256="a"*64, route_plan_sha256="b"*64,
            dispatch_body_sha256="c"*64, model_canonical_id=DEEPSEEK_MODEL,
            provider_endpoint_tag=selected.provider_endpoint_tag, quantization=selected.quantization,
            parameter_profile_sha256="d"*64, qualification_version="q/1", qualification_sha256="e"*64,
            stable_catalog_sha256="f"*64, privacy_policy_sha256="1"*64, billable_prompt_tokens=100,
            input_token_limit=1024, output_token_limit=256, max_reserved_microusd=1000, included_turn_ids=(),
        )
        self.assertEqual(binding.canonical()["version"], "cloud-approval-binding/1")
        for field, value in (("task_id", b"not-a-uuid"), ("expires_at", None),
                ("included_turn_ids", None), ("egress_sha256", 42), ("provider_endpoint_tag", []),
                ("quantization", 42), ("qualification_version", {})):
            with self.subTest(field=field), self.assertRaises(CloudBindingError):
                replace(binding, **{field: value}).canonical()

    def test_health_requires_known_model_and_valid_integer_metrics(self):
        health = EndpointHealthV1(DEEPSEEK_MODEL, 'provider/full', 0, 9990, 100,
                                  '2026-09-06T00:00:00Z', 0)
        self.assertTrue(health.eligible())
        for field, value in (('model_canonical_id', 'unknown/model'), ('status', False),
                             ('uptime_last_30m_basis_points', 10001), ('age_ms', True),
                             ('p90_latency_ms', 0.5)):
            with self.subTest(field=field, value=value):
                self.assertFalse(replace(health, **{field: value}).eligible())

    def test_provider_microdollar_ceiling_is_exact_and_bounded(self):
        self.assertEqual(_decimal_microusd('0.0000010000000000000000000000000001'), 2)
        with self.assertRaises(OrchestratorError):
            _decimal_microusd('1e1000000')


class CloudLiveCostBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def revalidate(self, *, key_changes=None, live_changes=None):
        selected = endpoint(DEEPSEEK_MODEL)
        live = dict(model_id=selected.model_canonical_id, tag=selected.provider_endpoint_tag,
            provider_name=selected.runtime_provider_name, quantization=selected.quantization,
            supported_parameters=list(selected.supported_parameters), context_length=selected.context_limit,
            max_completion_tokens=selected.output_limit, status=0, uptime_last_30m='99.9',
            latency_last_30m={'p90': '0.1'}, pricing=dict(prompt='0.00000009', completion='0.00000018', request='0'))
        live.update(live_changes or {})
        key = dict(limit='5', limit_remaining='4', usage_monthly='1', limit_reset='monthly',
            is_management_key=False, is_provisioning_key=False, include_byok_in_limit=False)
        key.update(key_changes or {})
        async def control(request):
            return httpx.Response(200, json={'data': [live] if request.url.path.endswith('/zdr') else key})
        adapter = OpenRouterCloudAdapter(Mock(), bearer_token='synthetic-key',
            transport=httpx.MockTransport(control), control_transport=httpx.MockTransport(control))
        try:
            return await adapter.revalidate(selected)
        finally:
            await adapter.aclose()

    async def test_remaining_budget_rounds_down_and_usage_rounds_up(self):
        result = await self.revalidate(key_changes={'limit_remaining': '0.0000001', 'usage_monthly': '0.0000001'})
        self.assertEqual(result.key_remaining_microusd, 0)
        self.assertEqual(result.usage_monthly_microusd, 1)

    async def test_key_limit_cannot_round_into_the_required_policy(self):
        with self.assertRaisesRegex(OrchestratorError, 'cloud_key_policy_mismatch'):
            await self.revalidate(key_changes={'limit': '4.9999999'})

    async def test_live_metrics_do_not_truncate_or_round_into_eligibility(self):
        for field, value in (('context_length', 128000.5), ('max_completion_tokens', 2048.5),
                             ('status', False), ('uptime_last_30m', '98.9999'),
                             ('uptime_last_30m', '100.001'),
                             ('pricing', dict(prompt='0.000000090000000000000000000000000001', completion='0.00000018', request='0'))):
            with self.subTest(field=field, value=value), self.assertRaises(OrchestratorError):
                await self.revalidate(live_changes={field: value})

    async def test_control_failure_drains_sibling_request(self):
        finished = []
        async def control(request):
            if request.url.path.endswith('/zdr'):
                raise httpx.ReadError('synthetic failure', request=request)
            try:
                await asyncio.sleep(0.02)
                return httpx.Response(200, json={})
            finally:
                finished.append(True)
        adapter = OpenRouterCloudAdapter(Mock(), bearer_token='synthetic-key',
            transport=httpx.MockTransport(control), control_transport=httpx.MockTransport(control))
        try:
            with self.assertRaises(OrchestratorError):
                await adapter.revalidate(endpoint(DEEPSEEK_MODEL))
            completed_before_close = list(finished)
        finally:
            await adapter.aclose()
            await asyncio.sleep(0.03)
        self.assertEqual(completed_before_close, [True])


if __name__ == '__main__':
    unittest.main()
