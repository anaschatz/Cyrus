"""Fresh two-call compatibility contract for the aligned vLLM schema."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

import anamnesis.openmemory_vllm as boundary_module
import anamnesis.openmemory_vllm_v5 as v5
import anamnesis.openmemory_vllm_v5_report as report
from anamnesis.vllm_runtime import api_key_sha256

ROOT = Path(__file__).resolve().parents[1]
API_KEY = "local-v4-loopback-20260809"


def _response(content: str) -> dict[str, object]:
    return {
        "model": "anamnesis-openmemory-v5",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 111, "completion_tokens": 22},
    }


class FakeClient:
    queue: list[dict[str, object]] = []
    requests: list[dict[str, object]] = []

    def __init__(self, *, base_url: str, api_key: str, request_timeout_seconds: float):
        self.base_url = base_url
        self.api_key_sha256 = api_key_sha256(api_key)
        self.request_timeout_seconds = request_timeout_seconds

    async def complete(self, request):
        self.requests.append(dict(request))
        return self.queue.pop(0)

    async def aclose(self) -> None:
        return None


class FakeProbe:
    calls = 0

    def __init__(self, *, declared, **kwargs):
        self.declared = declared

    async def snapshot(self):
        type(self).calls += 1
        return self.declared.model_copy(update={"health_ok": True})


def _patch_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v5, "_verify_source_commit", lambda *args: None)
    monkeypatch.setattr(v5, "verify_vllm_artifact", lambda *args: None)
    monkeypatch.setattr(boundary_module, "verify_vllm_artifact", lambda *args: None)
    monkeypatch.setattr(v5, "OpenAIExternalVllmClient", FakeClient)
    monkeypatch.setattr(v5, "HttpOperatorVllmProbe", FakeProbe)
    FakeClient.requests = []
    FakeProbe.calls = 0


def _emit_action() -> dict[str, object]:
    return {
        "kind": "reminder",
        "action_key": "omv5_event_cobalt_sheet",
        "payload": {"subject": "archive cobalt inspection sheet"},
        "summary": "Archive cobalt inspection sheet",
        "evidence_event_ids": ["omv5_event_cobalt_sheet"],
    }


def test_v5_fixture_and_overlay_pins_are_exact_and_fresh() -> None:
    pin, fixture, runtime = v5._load_inputs()
    v4_text = (ROOT / "eval/openmemory/decision_diagnostic.v4.json").read_text()

    assert hashlib.sha256(v5.FIXTURE_PATH.read_bytes()).hexdigest() == (
        "fe3bdd57dfc51bbd374d81ded6ff6cbd4ba535e08595be39f5ad857bf1e923a0"
    )
    assert hashlib.sha256(v5.PIN_PATH.read_bytes()).hexdigest() == (
        "9de634f8806d295d6b6adad340bf23cdd874ae1c367d10d963144f881cdf6b5a"
    )
    assert tuple(case.expected_mode for case in fixture.cases) == (
        "emit",
        "no_action",
    )
    assert all(case.id not in v4_text for case in fixture.cases)
    assert all(case.event.text not in v4_text for case in fixture.cases)
    assert runtime.served_model == "anamnesis-openmemory-v5"
    assert runtime.base_url == "http://127.0.0.1:18001/v1"
    assert runtime.response_schema_sha256 == pin.response_schema_sha256
    assert runtime.decision_contract_sha256 == pin.decision_contract_sha256


def test_v5_semantic_projection_checks_cardinality_key_evidence_and_subject() -> None:
    _, fixture, _ = v5._load_inputs()
    emit, no_action = fixture.cases
    good = boundary_module.VllmAlignedDecisionWire.model_validate(
        {"mode": "emit", "actions": [_emit_action()]}
    ).to_domain()

    assert v5._semantic_passed(emit, good)
    assert v5._semantic_passed(
        no_action,
        boundary_module.VllmAlignedDecisionWire.model_validate(
            {"mode": "no_action", "actions": []}
        ).to_domain(),
    )
    wrong_key = good.model_copy(
        update={
            "actions": [
                good.actions[0].model_copy(update={"action_key": "wrong_event"})
            ]
        }
    )
    assert not v5._semantic_passed(emit, wrong_key)


def test_v5_runner_executes_exactly_two_aligned_calls(monkeypatch, tmp_path) -> None:
    _patch_runtime(monkeypatch)
    FakeClient.queue = [
        _response(json.dumps({"mode": "emit", "actions": [_emit_action()]})),
        _response('{"mode":"no_action","actions":[]}'),
    ]

    result = asyncio.run(
        v5.run_v5_compatibility(
            artifact_root=tmp_path,
            api_key=API_KEY,
            source_commit="a" * 40,
        )
    )

    assert result.passed
    assert len(result.cases) == 2
    assert len(FakeClient.requests) == 2
    assert FakeProbe.calls == 2
    assert result.usage.input_tokens == 222
    for request in FakeClient.requests:
        schema = request["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["actions"]["maxItems"] == 1

    raw_path = tmp_path / "run.json"
    raw_path.write_text(result.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(report, "_verify_source_commit", lambda *args: None)
    assert report.validate_run(raw_path) == result

    tampered = json.loads(raw_path.read_text(encoding="utf-8"))
    tampered["cases"][0]["audit"]["request_sha256"] = "0" * 64
    raw_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="raw request"):
        report.validate_run(raw_path)


def test_v5_runner_records_schema_rejection_without_retry(
    monkeypatch, tmp_path
) -> None:
    _patch_runtime(monkeypatch)
    repeated = {"mode": "emit", "actions": [_emit_action(), _emit_action()]}
    FakeClient.queue = [
        _response(json.dumps(repeated)),
        _response('{"mode":"no_action","actions":[]}'),
    ]

    result = asyncio.run(
        v5.run_v5_compatibility(
            artifact_root=tmp_path,
            api_key=API_KEY,
            source_commit="a" * 40,
        )
    )

    assert not result.passed
    assert result.cases[0].audit.validation.error_stage == "wire"
    assert len(FakeClient.requests) == 2
    assert FakeProbe.calls == 2


def test_v5_output_is_confined(tmp_path: Path) -> None:
    accepted = ROOT / "results/runs/local/openmemory_vllm_v5_compatibility/run.json"
    assert v5._output_path(accepted) == accepted.resolve()
    with pytest.raises(ValueError, match="frozen run folder"):
        v5._output_path(tmp_path / "run.json")
