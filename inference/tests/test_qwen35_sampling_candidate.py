from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jarvis_inference.profile_preflight import Fixture
from jarvis_inference.qwen35_sampling_candidate import (
    CHAT_TEMPLATE_KWARGS,
    FROZEN_PREREGISTRATION_SHA256,
    GENERATION_DISPATCHES,
    PUBLIC_MODEL_ALIAS,
    TREATMENT_SAMPLING,
    ArtifactIdentity,
    CandidateFailure,
    OfficialSamplingCandidateClient,
    OneShotMarker,
    RenderedTokenIdentity,
    _component_counts,
    _sse_content_type,
    _read_small_regular,
    build_candidate_report,
    hash_artifact,
    load_preregistration,
    read_private_bearer,
    scorer_code_bundle_identity,
)
from jarvis_inference.strict_json import canonical_bytes, parse_body


ROOT = Path(__file__).parents[1]
PREREGISTRATION = ROOT / "eval/qwen35-9b-official-nonthinking-sampling-v1.prereg.json"


class FakeResponse:
    def __init__(self, lines: list[bytes], *, content_type: str = "text/event-stream", status: int = 200):
        self.lines = iter(lines)
        self.content_type = content_type
        self.status = status

    def getheader(self, name: str):
        return self.content_type if name.casefold() == "content-type" else None

    def readline(self, limit: int) -> bytes:
        return next(self.lines, b"")


class FakeConnection:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.requests: list[tuple[str, str, bytes | None, dict[str, str]]] = []
        self.closed = False

    def request(self, method: str, path: str, body=None, headers=None) -> None:
        self.requests.append((method, path, body, headers))

    def getresponse(self) -> FakeResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


def event(*, content: str | None = None, finish_reason=None, delta_extra=None, choices=True, usage=None) -> bytes:
    value = {
        "id": "chatcmpl-fixed",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": PUBLIC_MODEL_ALIAS,
        "choices": [],
    }
    if choices:
        delta = {} if content is None else {"content": content}
        if delta_extra:
            delta.update(delta_extra)
        value["choices"] = [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
            "logprobs": None,
        }]
    if usage is not None:
        value["usage"] = usage
    return b"data: " + canonical_bytes(value) + b"\n"


def valid_lines(text: str = "yes", *, prompt_tokens: int = 3) -> list[bytes]:
    completion_tokens = 1
    return [
        b": keepalive\n",
        b"\n",
        event(content=text),
        event(finish_reason="stop"),
        event(choices=False, usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }),
        b"data: [DONE]\n",
    ]


class Qwen35SamplingCandidateTests(unittest.TestCase):
    def test_malformed_json_and_stream_fields_return_candidate_failures(self) -> None:
        lines = [valid_lines()[:-1] + [b"data: [DONE]"]]
        for value in (None, [], [1], {"id": "x"}):
            lines.append([b"data: " + canonical_bytes(value) + b"\n"])
        for field, value in (("logprobs", {}), ("finish_reason", []), ("index", False)):
            chunk = parse_body(event(content="yes")[6:])
            chunk["choices"][0][field] = value
            lines.append(
                [b"data: " + canonical_bytes(chunk) + b"\n", *valid_lines()[3:]]
            )
        for field in ("reasoning", "reasoning_content"):
            lines.append([event(delta_extra={field: []}), *valid_lines()])
        for index, stream in enumerate(lines):
            with self.subTest(index=index), self.assertRaises(CandidateFailure):
                self.read(stream)
        for value in (None, [], [1]):
            with self.subTest(json=value):
                response = FakeResponse([], content_type="application/json")
                response.read = lambda maximum: canonical_bytes(value)
                client = self.client()
                with patch.object(
                    client, "_connection", return_value=FakeConnection(response)
                ):
                    with self.assertRaises(CandidateFailure):
                        client.readiness()
                with tempfile.TemporaryDirectory() as directory:
                    config = Path(directory) / "config.json"
                    config.write_bytes(canonical_bytes(value))
                    config.chmod(0o600)
                    with self.assertRaises(CandidateFailure):
                        read_private_bearer(config)

    def test_failed_dispatch_still_consumes_the_one_shot_budget(self) -> None:
        client = self.client()
        self.prime(client)
        client.generation_dispatches = GENERATION_DISPATCHES - 1
        connection = FakeConnection(FakeResponse([]))
        with (
            patch.object(client, "_connection", return_value=connection),
            patch.object(
                connection, "request", side_effect=OSError("dispatch interrupted")
            ) as dispatch,
        ):
            with self.assertRaises(CandidateFailure):
                client.stream(self.fixture())
            self.assertEqual(client.generation_dispatches, GENERATION_DISPATCHES)
            with self.assertRaisesRegex(CandidateFailure, "budget"):
                client.stream(self.fixture())
            dispatch.assert_called_once()
        self.assertTrue(connection.closed)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX special files")
    def test_candidate_inputs_cannot_block_on_a_fifo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input"
            os.mkfifo(path)
            original = os.open

            def checked_open(path, flags):
                self.assertTrue(flags & os.O_NONBLOCK)
                return original(path, flags)

            with patch(
                "jarvis_inference.qwen35_sampling_candidate.os.open", checked_open
            ):
                with self.assertRaises(CandidateFailure):
                    _read_small_regular(path)
                with self.assertRaises(CandidateFailure):
                    hash_artifact(path, expected_size=1, expected_sha256="a" * 64)

    def client(self, before=lambda: None) -> OfficialSamplingCandidateClient:
        return OfficialSamplingCandidateClient(
            "http://127.0.0.1:8878/v1",
            "b" * 48,
            before_first_generation=before,
        )

    def fixture(self) -> Fixture:
        return Fixture("case", "en", "Reply exactly yes", 16, exact="yes", category="instruction")

    def prime(self, client: OfficialSamplingCandidateClient, fixture: Fixture | None = None) -> None:
        fixture = fixture or self.fixture()
        messages = client._messages(fixture)
        client._rendered[fixture.case_id] = RenderedTokenIdentity(
            fixture.case_id,
            hashlib.sha256(canonical_bytes(messages)).hexdigest(),
            "a" * 64,
            "c" * 64,
            3,
        )
        client._rendered_order.append(fixture.case_id)

    def read(self, lines: list[bytes], fixture: Fixture | None = None, *, content_type="text/event-stream"):
        fixture = fixture or self.fixture()
        return self.client()._read_sse(
            FakeResponse(lines, content_type=content_type),
            fixture,
            started=0.0,
            expected_prompt_tokens=3,
        )

    def test_preregistration_is_byte_frozen_and_semantically_closed(self) -> None:
        self.assertEqual(load_preregistration(PREREGISTRATION), FROZEN_PREREGISTRATION_SHA256)
        self.assertEqual(
            scorer_code_bundle_identity()["aggregate_sha256"],
            "ed1682eed182da50cf819bf1bc923a4fc1f728e252009ffb0736c27318f6286c",
        )
        raw = PREREGISTRATION.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "changed.json"
            changed.write_bytes(raw.replace(b'"seed": 101', b'"seed": 102'))
            with self.assertRaises(CandidateFailure):
                load_preregistration(changed)

    def test_endpoint_is_exact_ipv4_loopback_and_bearer_rejects_header_injection(self) -> None:
        for url in (
            "https://127.0.0.1:8878/v1",
            "http://localhost:8878/v1",
            "http://127.0.0.1:8878/v2",
            "http://127.0.0.1:8878/v1///",
            "http://user@127.0.0.1:8878/v1",
            "http://127.0.0.1:8878/v1?x=1",
        ):
            with self.subTest(url=url), self.assertRaises(CandidateFailure):
                OfficialSamplingCandidateClient(url, "b" * 48, before_first_generation=lambda: None)
        with self.assertRaises(CandidateFailure):
            OfficialSamplingCandidateClient(
                "http://127.0.0.1:8878/v1", "b" * 32 + "\r\nInjected: true",
                before_first_generation=lambda: None,
            )

    def test_stream_request_contains_only_the_frozen_treatment_and_authenticated_alias(self) -> None:
        armed: list[bool] = []
        client = self.client(lambda: armed.append(True))
        fixture = self.fixture()
        self.prime(client, fixture)
        connection = FakeConnection(FakeResponse(valid_lines()))
        with patch.object(client, "_connection", return_value=connection):
            result = client.stream(fixture)
        self.assertTrue(result["quality_pass"])
        self.assertEqual(armed, [True])
        self.assertEqual(client.generation_dispatches, 1)
        method, path, raw, headers = connection.requests[0]
        self.assertEqual((method, path), ("POST", "/v1/chat/completions"))
        body = parse_body(raw)
        for key, value in TREATMENT_SAMPLING.items():
            self.assertEqual(float(body[key]), float(value))
        self.assertEqual(body["chat_template_kwargs"], CHAT_TEMPLATE_KWARGS)
        self.assertEqual(body["model"], PUBLIC_MODEL_ALIAS)
        self.assertEqual(body["n"], 1)
        self.assertIs(body["stream"], True)
        self.assertNotIn("tools", body)
        self.assertEqual(headers["Authorization"], "Bearer " + "b" * 48)
        self.assertEqual(headers["Accept"], "text/event-stream")
        self.assertTrue(connection.closed)

    def test_sse_accepts_optional_utf8_charset_blank_lines_and_comments(self) -> None:
        for content_type in ("text/event-stream", "text/event-stream; charset=UTF-8"):
            with self.subTest(content_type=content_type):
                result = self.read(valid_lines(), content_type=content_type)
                self.assertEqual(result["completion_tokens"], 1)
                self.assertTrue(result["quality_pass"])
        self.assertFalse(_sse_content_type("text/event-stream; charset=latin1"))
        self.assertFalse(_sse_content_type("text/event-stream; charset=utf-8; x=1"))

    def test_sse_rejects_reasoning_fields_tags_and_tool_output(self) -> None:
        attacks = (
            [event(content=None, delta_extra={"reasoning_content": "private thought"}), *valid_lines()],
            [event(content="<think>private</think>yes"), event(finish_reason="stop"),
             event(choices=False, usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
             b"data: [DONE]\n"],
            [event(content=None, delta_extra={"tool_calls": [{"id": "x"}]}), *valid_lines()],
            [event(content=None, delta_extra={"function_call": {"name": "x"}}), *valid_lines()],
        )
        for lines in attacks:
            with self.subTest(first=lines[0]), self.assertRaises(CandidateFailure):
                self.read(lines)

    def test_sse_rejects_missing_done_usage_finish_and_prompt_token_mismatch(self) -> None:
        invalid = (
            valid_lines()[:-1],
            [event(content="yes"), event(finish_reason="stop"), b"data: [DONE]\n"],
            [event(content="yes"), event(choices=False, usage={
                "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
            }), b"data: [DONE]\n"],
            [event(content="yes"), event(finish_reason="stop"), event(choices=False, usage={
                "prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5,
            }), b"data: [DONE]\n"],
            [event(content="yes"), event(finish_reason="stop"), event(choices=False, usage={
                "prompt_tokens": 3, "completion_tokens": 17, "total_tokens": 20,
            }), b"data: [DONE]\n"],
        )
        for lines in invalid:
            with self.subTest(lines=len(lines)), self.assertRaises(CandidateFailure):
                self.read(lines)

    def test_sse_rejects_alias_drift_duplicate_usage_unknown_fields_and_bad_content_type(self) -> None:
        alias = parse_body(event(content="yes")[6:].strip())
        alias["model"] = "not-jarvis-local"
        alias_lines = [b"data: " + canonical_bytes(alias) + b"\n", *valid_lines()[3:]]
        duplicate_usage = valid_lines()[:-1]
        duplicate_usage.insert(-1, duplicate_usage[-1])
        unknown = parse_body(event(content="yes")[6:].strip())
        unknown["private_output"] = "must not be accepted"
        unknown_lines = [b"data: " + canonical_bytes(unknown) + b"\n", *valid_lines()[3:]]
        mixed = parse_body(event(content="yes")[6:].strip())
        mixed["usage"] = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        mixed_lines = [b"data: " + canonical_bytes(mixed) + b"\n", *valid_lines()[3:]]
        for lines in (alias_lines, duplicate_usage, unknown_lines, mixed_lines):
            with self.assertRaises(CandidateFailure):
                self.read(lines)
        with self.assertRaises(CandidateFailure):
            self.read(valid_lines(), content_type="application/json")

    def test_template_preflight_requires_explicit_empty_thinking_boundary_and_strict_tokens(self) -> None:
        client = self.client()
        fixture = self.fixture()
        suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        responses = iter([
            {"status": "ok"},
            {"object": "list", "data": [{"id": PUBLIC_MODEL_ALIAS}]},
            {"prompt": "rendered" + suffix},
            {"tokens": [1, 2, 3]},
        ])
        with patch.object(client, "_json_request", side_effect=lambda *args, **kwargs: next(responses)) as request:
            client.prepare_rendered_tokens((fixture,))
        identity = client.rendered_token_identity()
        self.assertEqual(identity["unique_fixture_count"], 1)
        apply_body = request.call_args_list[2].args[2]
        self.assertEqual(apply_body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIs(apply_body["add_generation_prompt"], True)

        bad = self.client()
        responses = iter([
            {"status": "ok"},
            {"object": "list", "data": [{"id": PUBLIC_MODEL_ALIAS}]},
            {"prompt": "<|im_start|>assistant\n<think>\n"},
        ])
        with patch.object(bad, "_json_request", side_effect=lambda *args, **kwargs: next(responses)), self.assertRaises(CandidateFailure):
            bad.prepare_rendered_tokens((fixture,))

    def test_one_shot_marker_is_private_exclusive_and_cannot_be_rearmed_by_new_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.chmod(root, 0o700)
            marker_path = root / "marker.json"
            marker = OneShotMarker(marker_path, FROZEN_PREREGISTRATION_SHA256)
            marker.arm()
            marker.arm()
            self.assertEqual(stat_mode(marker_path), 0o600)
            value = json.loads(marker_path.read_text())
            self.assertTrue(value["generation_dispatch_started"])
            self.assertFalse(value["rerun_permitted"])
            with self.assertRaises(CandidateFailure):
                OneShotMarker(marker_path, FROZEN_PREREGISTRATION_SHA256).arm()

    def test_artifact_hash_rejects_symlink_and_changed_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"pinned")
            digest = hashlib.sha256(b"pinned").hexdigest()
            self.assertEqual(hash_artifact(artifact, expected_size=6, expected_sha256=digest).sha256, digest)
            with self.assertRaises(CandidateFailure):
                hash_artifact(artifact, expected_size=7, expected_sha256=digest)
            link = root / "link.bin"
            link.symlink_to(artifact)
            with self.assertRaises(CandidateFailure):
                hash_artifact(link, expected_size=6, expected_sha256=digest)

    def test_report_contains_only_content_free_components_and_identity_hooks(self) -> None:
        cases = []
        for language in ("el", "en"):
            for index in range(40):
                cases.append({
                    "case_id": f"{language}_case_{index}",
                    "language": language,
                    "category": ("instruction", "extraction", "summary", "reasoning", "knowledge")[index % 5],
                    "protocol_pass": True,
                    "language_ok": True,
                    "exact_ok": True,
                    "required_content_ok": True,
                    "forbidden_content_ok": True,
                    "word_limit_ok": True,
                    "completed_ok": True,
                    "quality_pass": True,
                    "finish_reason": "stop",
                    "ttft_seconds": 0.1,
                    "end_to_end_seconds": 0.2,
                    "completion_tokens": 1,
                    "decode_tokens_per_second": 10.0,
                    "output_utf8_bytes": 3,
                })
        client = self.client()
        client.generation_dispatches = GENERATION_DISPATCHES
        client._rendered["warmup"] = RenderedTokenIdentity("warmup", "a" * 64, "b" * 64, "c" * 64, 3)
        client._rendered_order.append("warmup")
        base = {
            "report_schema_version": 1,
            "suite_version": "jarvis-local-fallback-quality-v1",
            "authoritative_qualification": False,
            "profile_id": "local-fallback",
            "public_model_alias": PUBLIC_MODEL_ALIAS,
            "corpus_sha256": "cc3814b05807908f1235db815f078f5ac2a4034ad3844e9dcc03a85c5d1fdd71",
            "system_prompt_sha256": "05dbfe802df53da6c8660425016b9ea20fef0c745111c38701845d5c422cf4f2",
            "warmups": 2,
            "sample_count": 80,
            "protocol_failures": 0,
            "language_results": {"el": {"passed": 40, "total": 40, "pass_rate": 1.0}, "en": {"passed": 40, "total": 40, "pass_rate": 1.0}},
            "category_results": {},
            "language_pass_rate_gap": 0.0,
            "critical_failures": 0,
            "metrics": {},
            "thresholds": {},
            "cases": cases,
            "status": "deterministic_quality_pass",
        }
        identity = ArtifactIdentity("d" * 64, 1)
        report = build_candidate_report(
            base,
            preregistration_sha256=FROZEN_PREREGISTRATION_SHA256,
            model=identity,
            runner=identity,
            harness=identity,
            scorer_code_bundle={"aggregate_sha256": "e" * 64, "files": []},
            client=client,
        )
        self.assertEqual(report["status"], "official_sampling_candidate_pass")
        self.assertFalse(report["raw_content_persisted"])
        self.assertEqual(report["component_counts"]["overall"]["quality_pass"], {"passed": 80, "total": 80})
        rendered = json.dumps(report, sort_keys=True)
        for forbidden in ("private thought", "Reply exactly yes", "Bearer ", "/Users/"):
            self.assertNotIn(forbidden, rendered)

    def test_component_counts_preserve_failure_dimensions_without_content(self) -> None:
        cases = [{
            "language": "el", "category": "summary", "finish_reason": "length",
            "protocol_pass": True, "language_ok": True, "exact_ok": True,
            "required_content_ok": False, "forbidden_content_ok": True,
            "word_limit_ok": False, "completed_ok": False, "quality_pass": False,
        }]
        counts = _component_counts(cases)
        self.assertEqual(counts["overall"]["required_content_ok"]["passed"], 0)
        self.assertEqual(counts["overall"]["protocol_pass"]["passed"], 1)
        self.assertEqual(counts["finish_reason_counts"], {"length": 1})


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


if __name__ == "__main__":
    unittest.main()
