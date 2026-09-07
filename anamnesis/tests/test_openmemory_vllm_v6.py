import asyncio
import hashlib
import json

import pytest

import anamnesis.openmemory_vllm as boundary
import anamnesis.openmemory_vllm_v6 as v6
import anamnesis.openmemory_vllm_v6_report as report6
import anamnesis.openmemory_vllm_v7 as v7
import anamnesis.openmemory_vllm_v7_report as report7
from test_openmemory_vllm_v5 import API_KEY, FakeClient, FakeProbe, _response

import numpy as np

from anamnesis.openmemory_recall import OpenMemoryRecallIndex
from anamnesis.openmemory_vllm_v6 import (
    FIXTURE_PATH,
    PIN_PATH,
    LocalVectorMemoryClient,
    RealMemoryFixture,
    _correct,
    _load_inputs,
)
from anamnesis.schema import Decision, ProposedAction


def test_v6_frozen_inputs_and_matrix() -> None:
    pin, fixture, runtime = _load_inputs()
    assert (
        hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == pin.fixture_raw_sha256
    )
    assert len(fixture.cases) == 8
    assert sum(case.helpful_opportunity for case in fixture.cases) == 4
    assert runtime.served_model == "anamnesis-openmemory-v6"
    assert runtime.max_tokens == 256
    assert json.loads(PIN_PATH.read_text())["top_k"] == 1


class _Vectorizer:
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[1.0, 0.0] if "target" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


def test_real_index_add_search_and_scope() -> None:
    async def run() -> None:
        client = LocalVectorMemoryClient(_Vectorizer())  # type: ignore[arg-type]
        index = OpenMemoryRecallIndex(namespace="test", user_id="user", client=client)
        await index.add("irrelevant", metadata={"fixture_id": "memory_b"})
        await index.add("target memory", metadata={"fixture_id": "memory_a"})
        result = await index.search("query", limit=1)
        assert [match.content for match in result.matches] == ["target memory"]
        assert client.last_search_ids == ("memory_a",)
        row = client.rows["memory_a"]
        assert row["user_id"] == "anamnesis::test::user"

    asyncio.run(run())


def test_v6_exact_scoring_rejects_missing_memory_slot() -> None:
    fixture = RealMemoryFixture.model_validate_json(FIXTURE_PATH.read_text())
    case = fixture.cases[0]
    incomplete = Decision(
        actions=[
            ProposedAction(
                kind="reminder",
                action_key=case.event.id,
                payload={"subject": "send observatory calibration packet"},
                summary="Send packet",
                evidence_event_ids=[case.event.id],
            )
        ]
    )
    exact = Decision(
        actions=[
            ProposedAction(
                kind="reminder",
                action_key=case.event.id,
                payload=case.expected.payload,
                summary="Send packet",
                evidence_event_ids=[case.event.id],
            )
        ]
    )
    assert not _correct(case.expected, incomplete)
    assert _correct(case.expected, exact)


def test_v6_fixture_has_no_prior_entities() -> None:
    text = FIXTURE_PATH.read_text().casefold()
    for forbidden in ("meridian archives", "silver gallery", "cobalt inspection"):
        assert forbidden not in text


def _paired_run(monkeypatch, tmp_path, version, outcome="pass"):
    module, report = (v6, report6) if version == 6 else (v7, report7)
    _, fixture, runtime = module._load_inputs()
    targets = {
        memory.content
        for case in fixture.cases
        for memory in case.memories
        if memory.id in case.expected_retrieved_memory_ids
    }

    class Vectorizer(_Vectorizer):
        artifact_sha256 = v6.EMBEDDING_ARTIFACT_SHA256

        def __init__(self, **kwargs):
            pass

        def warmup(self):
            pass

        def embed_documents(self, texts):
            return np.asarray([[1, 0] if text in targets else [0, 1] for text in texts])

    for target in (module, report):
        monkeypatch.setattr(target, "_verify_source_commit", lambda *args: None)
    for target in (module, boundary):
        monkeypatch.setattr(target, "verify_vllm_artifact", lambda *args: None)
    monkeypatch.setattr(module, "FastEmbedVectorizer", Vectorizer)
    monkeypatch.setattr(module, "OpenAIExternalVllmClient", FakeClient)
    monkeypatch.setattr(module, "HttpOperatorVllmProbe", FakeProbe)
    FakeClient.queue, FakeClient.requests, FakeProbe.calls = [], [], 0
    for case in fixture.cases:
        for arm in ("baseline", "recall"):
            actions = []
            if case.expected.mode == "emit" and not (
                case.helpful_opportunity and arm == "baseline" and outcome != "fail"
            ):
                actions = [
                    dict(
                        kind="reminder",
                        action_key=case.expected.action_key,
                        payload=case.expected.payload,
                        summary="Execute expected action",
                        evidence_event_ids=list(case.expected.evidence_event_ids),
                    )
                ]
            content = json.dumps(
                dict(mode="emit" if actions else "no_action", actions=actions)
            )
            if outcome == "rejected" and case == fixture.cases[0] and arm == "recall":
                content = "invalid JSON"
            FakeClient.queue.append(
                _response(content) | {"model": runtime.served_model}
            )
    runner = v6.run_real_memory if version == 6 else v7.run_v7
    result = asyncio.run(
        runner(
            artifact_root=tmp_path,
            embedding_snapshot=tmp_path,
            api_key=API_KEY,
            source_commit="a" * 40,
        )
    )
    assert not FakeClient.queue
    assert len(FakeClient.requests) == FakeProbe.calls == 2 * len(fixture.cases)
    assert result.usage.input_tokens == 111 * len(FakeClient.requests)
    path = tmp_path / "paired.json"
    path.write_text(result.model_dump_json())
    assert report.validate_run(path) == result
    return report, result, path


@pytest.mark.parametrize("version", [6, 7])
@pytest.mark.parametrize("outcome", ["pass", "fail", "rejected"])
def test_paired_report_accepts_actual_runner_outcomes(
    monkeypatch, tmp_path, version, outcome
):
    _, result, _ = _paired_run(monkeypatch, tmp_path, version, outcome)
    assert result.passed == (outcome == "pass")
    assert result.metrics.baseline_correct == (
        len(result.cases) if outcome == "fail" else len(result.cases) // 2
    )
    assert result.metrics.accepted_calls == 2 * len(result.cases) - (
        outcome == "rejected"
    )


@pytest.mark.parametrize("version", [6, 7])
@pytest.mark.parametrize(
    "tamper",
    [
        "raw_no_action",
        "raw_invalid",
        "arm",
        "latency",
        "fixture_sha256",
        "retrieval_correct",
        "baseline_correct",
        "recall_correct",
        "helpful_gain",
        "safety_regressions",
        "accepted_calls",
        "gate_passed",
        "metric_retrieval",
    ],
)
def test_paired_report_rejects_corruption(monkeypatch, tmp_path, version, tamper):
    report, _, path = _paired_run(monkeypatch, tmp_path, version)
    data = json.loads(path.read_text())
    arm = data["cases"][0]["recall"]
    if tamper.startswith("raw_"):
        arm["audit"]["raw_completion"] = (
            '{"mode":"no_action","actions":[]}'
            if tamper == "raw_no_action"
            else "invalid JSON"
        )
    elif tamper == "arm":
        arm["arm"] = "baseline"
    elif tamper == "latency":
        arm["latency_ms"] += 1
    elif tamper == "fixture_sha256":
        data[tamper] = "0" * 64
    elif tamper == "retrieval_correct":
        data["cases"][0][tamper] = False
    elif tamper == "gate_passed":
        data["passed"] = data["metrics"][tamper] = False
    else:
        key = "retrieval_correct" if tamper == "metric_retrieval" else tamper
        data["metrics"][key] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        report.validate_run(path)


@pytest.mark.parametrize(
    "tamper", ["canonicalizer_source_sha256", "canonicalizer_changes"]
)
def test_v7_report_rejects_canonicalizer_metadata_corruption(
    monkeypatch, tmp_path, tamper
):
    report, _, path = _paired_run(monkeypatch, tmp_path, 7)
    data = json.loads(path.read_text())
    if tamper == "canonicalizer_changes":
        data["metrics"][tamper] += 1
    else:
        data[tamper] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        report.validate_run(path)


@pytest.mark.parametrize("version", [6, 7])
def test_paired_report_rejects_promoted_failure(monkeypatch, tmp_path, version):
    report, result, path = _paired_run(monkeypatch, tmp_path, version, "fail")
    assert not result.passed
    data = json.loads(path.read_text())
    data["passed"] = data["metrics"]["gate_passed"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="aggregate metrics"):
        report.validate_run(path)
