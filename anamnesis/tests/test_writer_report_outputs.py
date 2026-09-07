"""All writer entrypoints reject file aliases before opening measured evidence."""

import os

import pytest

from anamnesis import local_report, writer_report, writer_report_w2, writer_report_w3


@pytest.mark.parametrize(
    "module,entrypoint,validator",
    [
        (writer_report, "writer_report_main", "_validate_frozen_writer_manifest"),
        (writer_report_w2, "writer_report_w2_main", "_validate_frozen_w2_manifest"),
        (writer_report_w3, "writer_report_w3_main", "_validate_frozen_w3_manifest"),
    ],
)
@pytest.mark.parametrize(
    "hazard",
    [
        "csv_link",
        "markdown_link",
        "provenance_link",
        "shared_outputs",
        "csv_directory",
        "markdown_directory",
        "provenance_directory",
        "fifo",
        "suffix",
    ],
)
def test_writer_rejects_unsafe_outputs_before_measured_validation(
    monkeypatch, tmp_path, module, entrypoint, validator, hazard
):
    for selected in (local_report, writer_report, module):
        monkeypatch.setattr(selected, "REPO_ROOT", tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"phase":"writer_diagnostic"}')
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text("dataset bytes")
    run = tmp_path / "run.eval"
    run.write_bytes(b"measured evidence")
    results = tmp_path / "results"
    results.mkdir()
    paths = {
        name: results / ("writer" + suffix)
        for name, suffix in (
            ("csv", ".csv"),
            ("markdown", ".md"),
            ("provenance", ".json"),
        )
    }
    if hazard.endswith("_link"):
        os.link(run, paths[hazard.removesuffix("_link")])
    elif hazard.endswith("_directory"):
        paths[hazard.removesuffix("_directory")].mkdir()
    elif hazard == "shared_outputs":
        paths["csv"].write_text("previous result")
        os.link(paths["csv"], paths["markdown"])
    elif hazard == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFOs are unavailable on this platform")
        os.mkfifo(paths["provenance"])
    else:
        paths["csv"] = results / "writer.txt"
    before = {
        path: path.read_bytes()
        for path in (manifest, scenarios, run, *paths.values())
        if path.is_file()
    }

    def unexpected_validation(**kwargs):
        raise AssertionError("unsafe output accepted before measured validation")

    monkeypatch.setattr(module, validator, unexpected_validation)
    args = [
        "--manifest",
        str(manifest),
        "--scenarios",
        str(scenarios),
        "--run",
        str(run),
    ]
    for name, path in paths.items():
        args.extend(["--" + name, str(path)])
    with pytest.raises(ValueError):
        getattr(module, entrypoint)(args)
    assert all(path.read_bytes() == content for path, content in before.items())


def test_regular_existing_outputs_remain_allowed(monkeypatch, tmp_path):
    for module in (local_report, writer_report):
        monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    paths = [results / ("writer" + suffix) for suffix in (".csv", ".md", ".json")]
    for path in paths:
        path.write_text("previous result")
    writer_report._validate_output_locations(
        sources=[], csv_path=paths[0], markdown_path=paths[1], provenance_path=paths[2]
    )
    assert all(path.read_text() == "previous result" for path in paths)
