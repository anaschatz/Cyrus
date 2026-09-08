"""Frozen pins and stopping behavior for the OpenMemory vLLM v4 run."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import anamnesis.openmemory_vllm as runtime_module
import anamnesis.openmemory_vllm_report as report_module
import anamnesis.openmemory_vllm_run as run_module
from anamnesis.openmemory_vllm import (
    VllmDecisionRuntimePin,
    openmemory_vllm_decision_contract_sha256,
    openmemory_vllm_schema_sha256,
)
from anamnesis.vllm_runtime import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(
    "/private/tmp/anamnesis-vllm-models/"
    "Qwen3.5-4B-MLX-4bit-32f3e8ecf65426fc3306969496342d504bfa13f3"
)
API_KEY = "local-v4-loopback-20260809"


def _response(content: str) -> dict[str, object]:
    return {
        "model": "anamnesis-openmemory-v4",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }


class FakeClient:
    queue: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []

    def __init__(self, *, base_url: str, api_key: str, request_timeout_seconds: float):
        from anamnesis.vllm_runtime import api_key_sha256

        self.base_url = base_url
        self.api_key_sha256 = api_key_sha256(api_key)
        self.request_timeout_seconds = request_timeout_seconds

    async def complete(self, request):
        self.calls.append(dict(request))
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
    monkeypatch.setattr(run_module, "OpenAIExternalVllmClient", FakeClient)
    monkeypatch.setattr(run_module, "HttpOperatorVllmProbe", FakeProbe)
    monkeypatch.setattr(run_module, "verify_vllm_artifact", lambda *args: None)
    monkeypatch.setattr(runtime_module, "verify_vllm_artifact", lambda *args: None)
    monkeypatch.setattr(run_module, "_verify_source_commit", lambda *args: None)
    FakeClient.calls = []
    FakeProbe.calls = 0


def test_runtime_pin_binds_exact_model_server_and_contract() -> None:
    raw = run_module.PIN_PATH.read_bytes()
    pin = VllmDecisionRuntimePin.model_validate_json(raw)

    assert hashlib.sha256(raw).hexdigest() == (
        "9fed6f50f56f4b926d56b3b169692b6b227dc4594f1c204022b5baf120b9fb31"
    )
    assert pin.artifact.revision == "32f3e8ecf65426fc3306969496342d504bfa13f3"
    assert pin.artifact.manifest_sha256 == (
        "1563d753ccd22c5b0e43dd0aa2a452452d04c2b3cdbf5d10b15187926069db7e"
    )
    assert pin.decision_contract_sha256 == openmemory_vllm_decision_contract_sha256()
    assert pin.response_schema_sha256 == openmemory_vllm_schema_sha256()
    assert canonical_json_sha256(run_module._server_config(pin)) == (
        pin.server_config_sha256
    )


def test_preflight_is_fresh_hash_bound_and_hidden_from_v4_cases() -> None:
    pin, preflight = run_module._load_frozen_inputs()
    dataset_text = run_module.DATASET_PATH.read_text(encoding="utf-8")

    assert pin.seed == 101
    assert preflight.expected_mode == "no_action"
    assert preflight.event.id not in dataset_text
    assert preflight.event.text not in dataset_text
    assert preflight.retrospective_recall == ()


def test_failed_canary_stops_before_all_scenario_calls(monkeypatch, tmp_path) -> None:
    _patch_runtime(monkeypatch)
    FakeClient.queue = [
        _response(
            json.dumps(
                {
                    "mode": "emit",
                    "actions": [
                        {
                            "kind": "reminder",
                            "action_key": "omv4_canary_weather_vane",
                            "payload": {"subject": "inspect weather vane"},
                            "summary": "Inspect weather vane",
                            "evidence_event_ids": ["omv4_canary_weather_vane"],
                        }
                    ],
                }
            )
        )
    ]

    result = asyncio.run(
        run_module.run_openmemory_vllm_v4(
            artifact_root=MODEL_ROOT,
            api_key=API_KEY,
            source_commit="a" * 40,
        )
    )

    assert result.status == "preflight_failed"
    assert result.paired_run is None
    assert len(result.audits) == 1
    assert len(FakeClient.calls) == 1
    assert FakeProbe.calls == 1
    assert result.headline_usage.input_tokens == 0
    assert not result.passed

    raw_path = tmp_path / "failed.json"
    raw_path.write_text(result.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        report_module, "_verify_reporting_checkout", lambda *args: "b" * 40
    )
    assert report_module._validate_run(raw_path) == result

    tampered = json.loads(raw_path.read_text(encoding="utf-8"))
    tampered["audits"][0]["request_sha256"] = "0" * 64
    raw_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="raw request"):
        report_module._validate_run(raw_path)


@pytest.mark.parametrize(
    "corruption",
    [
        None,
        "valid_pass",
        "decisions",
        "canary_decision",
        "canary_semantics",
        "usage",
        "case_order",
        "prompt",
        "artifact",
        "accounting",
    ],
)
def test_passing_canary_runs_exactly_one_eight_by_two_matrix(
    monkeypatch, tmp_path, corruption
) -> None:
    _patch_runtime(monkeypatch)
    no_action = _response('{"mode":"no_action","actions":[]}')
    FakeClient.queue = [no_action for _ in range(run_module.EXPECTED_CALLS)]

    artifact = run_module.load_openmemory_diagnostic(run_module.DATASET_PATH)
    cases = {case.id: case for case in artifact.cases}

    def expected_decision(case):
        actions = (
            []
            if case.expected.mode == "no_action"
            else [
                {
                    "kind": "reminder",
                    "action_key": case.expected.action_key,
                    "payload": case.expected.payload,
                    "summary": "Synthetic expected action",
                    "evidence_event_ids": list(case.expected.evidence_event_ids),
                }
            ]
        )
        return run_module.Decision.model_validate({"actions": actions})

    if corruption == "valid_pass":
        FakeClient.queue = [no_action]
        for case in artifact.cases:
            for arm in ("baseline", "recall"):
                decision = (
                    run_module.Decision()
                    if arm == "baseline" and case.helpful_hit_ids
                    else expected_decision(case)
                )
                FakeClient.queue.append(
                    _response(
                        json.dumps(
                            {
                                "mode": "emit" if decision.actions else "no_action",
                                **decision.model_dump(mode="json"),
                            }
                        )
                    )
                )

    result = asyncio.run(
        run_module.run_openmemory_vllm_v4(
            artifact_root=MODEL_ROOT,
            api_key=API_KEY,
            source_commit="a" * 40,
        )
    )

    assert result.status == "complete"
    assert result.paired_run is not None
    assert len(result.paired_run.calls) == 16
    assert len(result.audits) == run_module.EXPECTED_CALLS
    assert len(FakeClient.calls) == run_module.EXPECTED_CALLS
    assert FakeProbe.calls == run_module.EXPECTED_CALLS
    assert result.setup_usage.input_tokens == 100
    assert result.headline_usage.input_tokens == 1600
    assert result.total_usage.input_tokens == 1700
    assert result.passed == (corruption == "valid_pass")

    monkeypatch.setattr(
        report_module, "_verify_reporting_checkout", lambda *args: "b" * 40
    )
    path = tmp_path / "run.json"
    path.write_text(result.model_dump_json())
    assert report_module._validate_run(path) == result
    if corruption in (None, "valid_pass"):
        return
    raw = result.model_dump(mode="json")
    if corruption == "decisions":
        baseline, recall = {}, {}
        for call in raw["paired_run"]["calls"]:
            case = cases[call["case_id"]]
            decision = (
                run_module.Decision()
                if call["arm"] == "baseline" and case.helpful_hit_ids
                else expected_decision(case)
            )
            call["decision"] = decision.model_dump(mode="json")
            (baseline if call["arm"] == "baseline" else recall)[case.id] = decision
        metrics = report_module.score_openmemory_pair(
            artifact, baseline=baseline, recall=recall
        )
        assert metrics.gate_passed
        raw["paired_run"]["metrics"] = metrics.model_dump(mode="json")
        raw["passed"] = True
    elif corruption == "canary_decision":
        raw["canary"]["decision"] = expected_decision(
            next(case for case in artifact.cases if case.expected.mode == "emit")
        ).model_dump(mode="json")
    elif corruption == "canary_semantics":
        raw["canary"]["semantic_passed"] = False
    elif corruption == "usage":
        headline = run_module.Usage(
            input_tokens=1, uncached_input_tokens=1, output_tokens=1
        )
        raw["headline_usage"] = headline.model_dump(mode="json")
        raw["total_usage"] = result.setup_usage.plus(headline).model_dump(mode="json")
    elif corruption == "case_order":
        calls = raw["paired_run"]["calls"]
        calls[:4] = calls[2:4] + calls[:2]
    elif corruption == "prompt":
        raw["paired_run"]["calls"][0]["prompt_sha256"] = "0" * 64
    elif corruption == "artifact":
        raw["paired_run"]["artifact_sha256"] = "0" * 64
    elif corruption == "accounting":
        raw["paired_run"]["calls"][0]["usage_complete"] = False
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        report_module._validate_run(path)


def test_output_is_confined_to_ignored_v4_run_directory(tmp_path: Path) -> None:
    accepted = ROOT / "results/runs/local/openmemory_vllm_v4/result.json"
    assert run_module._contained_output(accepted) == accepted.resolve()
    with pytest.raises(ValueError, match="frozen v4 run folder"):
        run_module._contained_output(ROOT / "results/result.json")
    with pytest.raises(ValueError, match="frozen v4 run folder"):
        run_module._contained_output(tmp_path / "result.json")


def test_report_labels_joint_cell_and_setup_exclusion(monkeypatch) -> None:
    _patch_runtime(monkeypatch)
    FakeClient.queue = [
        _response('{"mode":"no_action","actions":[]}')
        for _ in range(run_module.EXPECTED_CALLS)
    ]
    result = asyncio.run(
        run_module.run_openmemory_vllm_v4(
            artifact_root=MODEL_ROOT,
            api_key=API_KEY,
            source_commit="a" * 40,
        )
    )
    row = report_module._row(result)
    markdown = report_module._markdown_bytes(row).decode()
    csv_bytes = report_module._csv_bytes(row)

    assert b"\r" not in csv_bytes
    assert "joint model-artifact + structured-runtime" in markdown
    assert "not a causal comparison with Ollama" in markdown
    assert "Setup tokens (input/output, excluded)" in markdown
    assert "No retry, repair, alternate artifact" in markdown


@pytest.mark.parametrize("prefix", ["", "anamnesis"])
@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("change", ["report", "input", "outside"])
def test_reporting_checkout_checks_full_repository_paths(
    tmp_path, monkeypatch, prefix, relative, change
):
    repository = tmp_path / "repository"
    project = repository / prefix
    project.mkdir(parents=True)

    def git(*args):
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=Review Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

    git("init", "-q")
    git("config", "diff.relative", str(relative).lower())
    files = {
        "report": project / "README.md",
        "input": project / "src" / "experiment.py",
        "outside": repository / "shared.txt",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("baseline\n")
    git("add", "--", *(str(path.relative_to(repository)) for path in files.values()))
    git("commit", "-qm", "baseline fixture")
    measurement = git("rev-parse", "HEAD")
    files[change].write_text("changed\n")
    git("add", "--", str(files[change].relative_to(repository)))
    git("commit", "-qm", "changed fixture")
    monkeypatch.setattr(report_module, "REPO_ROOT", project)
    if change == "report":
        assert report_module._verify_reporting_checkout(measurement) == git(
            "rev-parse", "HEAD"
        )
        files[change].write_text("uncommitted\n")
        with pytest.raises(ValueError, match="clean worktree"):
            report_module._verify_reporting_checkout(measurement)
    else:
        with pytest.raises(ValueError, match="measurement inputs changed"):
            report_module._verify_reporting_checkout(measurement)
