from __future__ import annotations

import copy
import re
import unittest
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # Root Voice intentionally has no YAML dependency.
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "backend-ci.yml"
ACTION_PIN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}$")
EXPECTED_ACTIONS = {
    "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd",
    "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405",
    "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9",
}
EXECUTION_NAMES = ("test", "build", "install", "smoke", "lint", "format", "validate")


def _walk(value: object):
    yield value
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _assert_contract(document: dict[str, Any], source: str) -> None:
    assert document["env"]["UV_NO_EDITABLE"] == "1"
    assert document["permissions"] == {"contents": "read"}
    concurrency = document["concurrency"]
    assert concurrency["cancel-in-progress"] is True
    assert "github.workflow" in concurrency["group"]
    assert "github.ref" in concurrency["group"]

    serialized = "\n".join(str(item) for item in _walk(document))
    assert "secrets." not in source.lower()
    assert "secrets" not in {str(item).lower() for item in _walk(document)}
    assert "services" not in {str(item).lower() for item in _walk(document)}
    assert "continue-on-error" not in serialized
    assert "actions/upload-artifact" not in source

    jobs = document["jobs"]
    assert set(jobs) == {
        "quality",
        "component-tests",
        "windows-security",
        "packaging",
    }
    for job in jobs.values():
        timeout = job.get("timeout-minutes")
        assert isinstance(timeout, int) and 1 <= timeout <= 30

    actions: set[str] = set()
    for job in jobs.values():
        for step in job["steps"]:
            if "uses" in step:
                action = step["uses"]
                assert isinstance(action, str) and ACTION_PIN.fullmatch(action)
                actions.add(action)
            name = str(step.get("name", "")).lower()
            is_execution = any(
                token in name for token in EXECUTION_NAMES
            ) and not name.startswith(("set up ", "sync "))
            if "run" in step and is_execution:
                environment = step.get("env", {})
                command = str(step["run"])
                assert environment.get("UV_OFFLINE") == "1"
                if "uv " in command:
                    assert "--offline" in command
    assert actions == EXPECTED_ACTIONS

    setup_python_steps = [
        step
        for job in jobs.values()
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/setup-python@")
    ]
    assert setup_python_steps
    assert all(step["with"]["python-version"] == "3.11" for step in setup_python_steps)
    assert all(step["with"]["check-latest"] is False for step in setup_python_steps)

    setup_uv_steps = [
        step
        for job in jobs.values()
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
    ]
    assert setup_uv_steps
    for step in setup_uv_steps:
        assert step["with"]["version"] == "0.9.18"
        assert step["with"]["enable-cache"] is True
        dependency_glob = step["with"]["cache-dependency-glob"]
        assert (
            "uv.lock" in dependency_glob or dependency_glob == "${{ matrix.lockfile }}"
        )

    component_matrix = jobs["component-tests"]["strategy"]["matrix"]["include"]
    assert {entry["lockfile"] for entry in component_matrix} == {
        "uv.lock",
        "anamnesis/uv.lock",
        "inference/uv.lock",
        "orchestrator/uv.lock",
    }

    whitespace_step = next(
        step
        for step in jobs["quality"]["steps"]
        if step.get("name") == "Check committed patch whitespace"
    )
    assert whitespace_step["env"] == {
        "EVENT_NAME": "${{ github.event_name }}",
        "BASE_SHA": "${{ github.event.pull_request.base.sha }}",
        "HEAD_SHA": "${{ github.sha }}",
    }
    whitespace_command = str(whitespace_step["run"])
    assert 'git diff --check "$BASE_SHA...$HEAD_SHA"' in whitespace_command
    assert 'git show --check --format= "$HEAD_SHA"' in whitespace_command
    assert not re.search(r"(?m)^\s*git diff --check\s*$", whitespace_command)


class BackendWorkflowTextContractTests(unittest.TestCase):
    def test_required_release_gates_are_present(self) -> None:
        source = WORKFLOW.read_text(encoding="utf-8")
        for fragment in (
            "uv lock --check --project .",
            "uv lock --check --project anamnesis",
            "uv lock --check --project inference",
            "uv lock --check --project orchestrator",
            "uv lock --check --project research",
            "uv sync --locked --extra dev --no-editable --python 3.11",
            "uv run --locked --extra dev --no-editable --offline",
            "python -m unittest discover -s tests -v",
            "python -m pytest -q",
            "python -m unittest discover -s inference/tests -v",
            "python -m unittest discover -s orchestrator/tests -v",
            "test_config_security.py",
            "test_windows_private_file.py",
            "test_windows_handoff_verifier.py",
            "test_windows_handoff_voice_template.py",
            "test_windows_source_bringup_scripts.py",
            "test_windows_cuda_bundle.py",
            "scripts\\test_windows_source_job.ps1",
            "shell: powershell",
            "Expected Windows PowerShell 5.1.",
            "test_stack_migration.py",
            "test_private_stack.py",
            "test_memory_crypto.py",
            "test_memory_key_provider.py",
            "test_memory_storage_v10.py",
            "test_memory_backup_operations.py",
            "test_memory_backup_cli.py",
            "uv sync --project inference --locked --extra qualification",
            "uv run --project inference --locked --extra qualification --offline python -m unittest discover -s inference/tests -v",
            "uv run --project inference --locked --extra qualification --offline python inference/tests/sdk_conformance.py -v",
            "uv build --offline --no-build-isolation",
            "--python .venv/bin/python --wheel --out-dir \"$RUNNER_TEMP/jarvis-wheels\" research",
            "--python inference/.venv/bin/python --wheel --out-dir \"$RUNNER_TEMP/jarvis-wheels\" inference",
            "uv export --project inference --locked --extra qualification --offline",
            "uv export --project orchestrator --locked --offline",
            "uv pip install --offline --require-hashes",
            "uv pip install --offline --no-deps",
            'smoke-research/bin/python" --requirements "$RUNNER_TEMP/orchestrator-requirements.txt"',
            'smoke-research/bin/python" "$RUNNER_TEMP"/jarvis-wheels/anamnesis_memory_eval-*.whl',
            '"$RUNNER_TEMP/smoke-inference/bin/cyrus-primary-profile-quality" --help',
            "smoke-voice",
            "smoke-anamnesis",
            "smoke-inference",
            "smoke-orchestrator",
            'cyrus-voice" --help',
            'cyrus-inference" --help',
            'cyrus-orchestrator" --help',
            'cyrus-action-host" --help',
            'cyrus-stack" --help',
            'cyrus-research-qualify" --help',
            'jarvis-voice" --help',
            'anamnesis-manifest" --help',
            'jarvis-inference" --help',
            'jarvis-orchestrator" --help',
            'jarvis-research-qualify" --help',
            'git diff --check "$BASE_SHA...$HEAD_SHA"',
            'git show --check --format= "$HEAD_SHA"',
            "ruff format --check",
            "python -m unittest tests.test_module_boundaries -v",
            "research/src",
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("CYRUS_NATIVE_RELEASE_HOST_TEST", source)
        self.assertNotIn(
            "--python research/.venv/bin/python --wheel", source
        )

    def test_raw_action_references_are_full_commit_pins(self) -> None:
        source = WORKFLOW.read_text(encoding="utf-8")
        references = re.findall(r"^\s*uses:\s*([^\s#]+)", source, re.MULTILINE)
        self.assertEqual(set(references), EXPECTED_ACTIONS)
        self.assertTrue(all(ACTION_PIN.fullmatch(item) for item in references))

    def test_no_secret_service_or_artifact_channel_is_declared(self) -> None:
        source = WORKFLOW.read_text(encoding="utf-8").lower()
        self.assertNotIn("secrets.", source)
        self.assertNotRegex(source, r"(?m)^\s*services\s*:")
        self.assertNotIn("actions/upload-artifact", source)
        self.assertNotIn("continue-on-error", source)
        self.assertNotRegex(source, r"(?m)^\s*git diff --check\s*$")


@unittest.skipIf(
    yaml is None, "PyYAML is provided by the locked Anamnesis dev environment"
)
class BackendWorkflowParsedContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = WORKFLOW.read_text(encoding="utf-8")
        parsed = yaml.safe_load(self.source)
        assert isinstance(parsed, dict)
        self.document = parsed

    def test_current_workflow_satisfies_the_closed_contract(self) -> None:
        _assert_contract(self.document, self.source)

    def test_unpinned_action_is_rejected(self) -> None:
        document = copy.deepcopy(self.document)
        document["jobs"]["quality"]["steps"][0]["uses"] = "actions/checkout@v6"
        with self.assertRaises(AssertionError):
            _assert_contract(document, self.source)

    def test_write_permission_is_rejected(self) -> None:
        document = copy.deepcopy(self.document)
        document["permissions"]["contents"] = "write"
        with self.assertRaises(AssertionError):
            _assert_contract(document, self.source)

    def test_secret_and_service_channels_are_rejected(self) -> None:
        for mutation, source in (
            (
                lambda value: value["jobs"]["quality"].update(
                    {"services": {"database": {"image": "sqlite"}}}
                ),
                self.source,
            ),
            (
                lambda value: value["jobs"]["quality"].update(
                    {"env": {"TOKEN": "${{ secrets.CI_TOKEN }}"}}
                ),
                self.source + "\n${{ secrets.CI_TOKEN }}\n",
            ),
        ):
            document = copy.deepcopy(self.document)
            mutation(document)
            with self.assertRaises(AssertionError):
                _assert_contract(document, source)

    def test_missing_timeout_or_concurrency_is_rejected(self) -> None:
        concurrency_document = copy.deepcopy(self.document)
        concurrency_document.pop("concurrency")
        with self.assertRaises(KeyError):
            _assert_contract(concurrency_document, self.source)

        timeout_document = copy.deepcopy(self.document)
        timeout_document["jobs"]["quality"].pop("timeout-minutes")
        with self.assertRaises(AssertionError):
            _assert_contract(timeout_document, self.source)

    def test_non_offline_execution_is_rejected(self) -> None:
        document = copy.deepcopy(self.document)
        step = next(
            item
            for item in document["jobs"]["component-tests"]["steps"]
            if item.get("name") == "Test Voice fully offline"
        )
        step["env"].pop("UV_OFFLINE")
        step["run"] = step["run"].replace(" --offline", "")
        with self.assertRaises(AssertionError):
            _assert_contract(document, self.source)

    def test_bare_worktree_diff_is_rejected(self) -> None:
        document = copy.deepcopy(self.document)
        step = next(
            item
            for item in document["jobs"]["quality"]["steps"]
            if item.get("name") == "Check committed patch whitespace"
        )
        step["run"] = "git diff --check"
        with self.assertRaises(AssertionError):
            _assert_contract(document, self.source)


if __name__ == "__main__":
    unittest.main()
