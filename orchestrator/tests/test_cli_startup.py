from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_orchestrator.cli import _run_servers
from jarvis_orchestrator.config import OrchestratorConfig


class _CompositionObserved(RuntimeError):
    pass


class CliStartupTests(unittest.TestCase):
    def test_run_servers_uses_its_config_parameter_for_cloud_release_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = OrchestratorConfig.from_mapping(
                {
                    "database_path": str(Path(directory) / "state.sqlite3"),
                    "control_bearer_token": "control-private",
                    "inference_bearer_token": "inference-private",
                }
            )
            captured: dict[str, object] = {}

            def observe_composition(*args: object, **kwargs: object) -> None:
                captured["config"] = args[0]
                captured["cloud_release_verified"] = kwargs[
                    "cloud_release_verified"
                ]
                raise _CompositionObserved

            with patch(
                "jarvis_orchestrator.cli.OrchestratorService",
                side_effect=observe_composition,
            ):
                with self.assertRaises(_CompositionObserved):
                    asyncio.run(
                        _run_servers(
                            config,
                            object(),  # type: ignore[arg-type]
                            model=object(),
                            inference_control=object(),  # type: ignore[arg-type]
                            shadow_memory=None,
                            active_memory=None,
                            research_adapter=None,
                            cloud_adapter=None,
                            cloud_endpoints=(),
                        )
                    )

            self.assertIs(captured["config"], config)
            self.assertIs(captured["cloud_release_verified"], False)


if __name__ == "__main__":
    unittest.main()
