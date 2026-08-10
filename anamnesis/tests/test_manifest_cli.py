from __future__ import annotations

import pytest

from anamnesis.manifest_cli import manifest_main


def test_manifest_cli_validates_and_fingerprints_template(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert manifest_main([]) == 0

    output = capsys.readouterr().out
    assert "status=draft" in output
    assert "phase=baseline" in output
    assert "scenarios=35" in output
    assert "sha256=" in output


def test_manifest_cli_can_require_frozen() -> None:
    with pytest.raises(ValueError, match="require a frozen"):
        manifest_main(["--require-frozen"])
