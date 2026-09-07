"""Report destinations must not damage inputs or each other."""

import hashlib
import json
import os

import pytest

import anamnesis.openmemory_vllm_report as v4
import anamnesis.openmemory_vllm_v5_report as v5
import anamnesis.openmemory_vllm_v6_report as v6
import anamnesis.openmemory_vllm_v7_report as v7
from test_openmemory_vllm_v6 import _paired_run


@pytest.mark.parametrize("version,report", [(4, v4), (5, v5), (6, v6), (7, v7)])
@pytest.mark.parametrize(
    "hazard",
    [
        "shared_output",
        "run_output",
        "outside_results",
        "outside_repo",
        "hardlink_input",
        "hardlink_outputs",
        "escaping_symlink",
        "directory",
        "suffix",
    ],
)
def test_report_rejects_unsafe_destinations_before_validation(
    monkeypatch, tmp_path, version, report, hazard
):
    root = tmp_path / "repo"
    results = root / "results"
    folder = f"openmemory_vllm_v{version}" + ("_compatibility" if version == 5 else "")
    run = results / "runs/local" / folder / "run.json"
    run.parent.mkdir(parents=True)
    run.write_bytes(b"original input")
    csv, md, provenance = [
        results / ("report" + suffix) for suffix in (".csv", ".md", ".json")
    ]
    if hazard == "shared_output":
        md = csv
    elif hazard == "run_output":
        provenance = run
    elif hazard == "outside_results":
        provenance = root / "source.json"
    elif hazard == "outside_repo":
        provenance = tmp_path / "outside.json"
    elif hazard == "hardlink_input":
        os.link(run, csv)
    elif hazard == "hardlink_outputs":
        csv.write_bytes(b"original output")
        os.link(csv, md)
    elif hazard == "escaping_symlink":
        csv.symlink_to(tmp_path / "outside.csv")
    elif hazard == "directory":
        csv.mkdir()
    else:
        csv = results / "report.txt"
    for path in (md, provenance):
        if not path.exists():
            path.write_bytes(b"original output")
    before = {
        path: path.read_bytes() for path in (run, csv, md, provenance) if path.is_file()
    }
    monkeypatch.setattr(report, "REPO_ROOT", root)

    def unexpected_validation(path):
        raise AssertionError("validation reached before rejecting unsafe destinations")

    monkeypatch.setattr(
        report,
        "_validate_run" if version == 4 else "validate_run",
        unexpected_validation,
    )
    with pytest.raises(ValueError):
        report.write_report(
            run_path=run, csv_path=csv, markdown_path=md, provenance_path=provenance
        )
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("version", [6, 7])
def test_paired_report_writes_distinct_resolved_artifacts(
    monkeypatch, tmp_path, version
):
    root = tmp_path.resolve()
    report, result, _ = _paired_run(monkeypatch, root, version)
    for attr in ("PIN_PATH", "FIXTURE_PATH"):
        original = getattr(report, attr)
        target = root / "eval" / original.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
        monkeypatch.setattr(report, attr, target)
    monkeypatch.setattr(report, "REPO_ROOT", root)
    run = root / f"results/runs/local/openmemory_vllm_v{version}/run.json"
    run.parent.mkdir(parents=True)
    original = result.model_dump_json().encode()
    run.write_bytes(original)
    csv = root / "results/csv/report.csv"
    md = root / "results/markdown/report.md"
    provenance = root / "results/provenance/report.json"
    elsewhere = root / "elsewhere"
    elsewhere.mkdir()
    validate = report.validate_run

    def move_after_validation(path):
        value = validate(path)
        monkeypatch.chdir(elsewhere)
        return value

    monkeypatch.setattr(report, "validate_run", move_after_validation)
    monkeypatch.chdir(root)
    assert report.write_report(
        run_path=run.relative_to(root),
        csv_path=csv.relative_to(root),
        markdown_path=md.relative_to(root),
        provenance_path=provenance.relative_to(root),
    )
    assert run.read_bytes() == original
    assert csv.read_bytes().startswith(b"passed,cases,")
    assert md.read_bytes().startswith(b"# ")
    proof = json.loads(provenance.read_text())
    assert proof["inputs"]["run"]["sha256"] == hashlib.sha256(original).hexdigest()
    for name, path in (("csv", csv), ("markdown", md)):
        assert proof["outputs"][name] == dict(
            path=path.relative_to(root).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    assert not list(elsewhere.iterdir())
