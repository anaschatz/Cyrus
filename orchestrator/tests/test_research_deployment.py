from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ResearchDeploymentTests(unittest.TestCase):
    def test_workload_networks_are_internal_and_only_guard_is_dual_homed(self) -> None:
        compose = (ROOT / "research" / "docker-compose.yml").read_text(encoding="utf-8")
        for network in ("research_searx", "research_proxy", "searx_proxy", "searx_auth"):
            self.assertRegex(compose, rf"(?m)^  {network}:\n    internal: true$")
        self.assertRegex(compose, r"(?m)^  edge_external:\n    internal: false$")
        self.assertEqual(compose.count("      - edge_external"), 1)
        guard = compose.split("  egress-guard:", 1)[1].split("\nnetworks:", 1)[0]
        self.assertIn("      - edge_external", guard)
        searx = compose.split("  searxng:", 1)[1].split("  searx-egress-auth:", 1)[0]
        self.assertNotIn("edge_external", searx)
        research = compose.split("  research-api:", 1)[1].split("  searxng:", 1)[0]
        self.assertNotIn("edge_external", research)

    def test_only_research_api_has_a_loopback_host_port(self) -> None:
        compose = (ROOT / "research" / "docker-compose.yml").read_text(encoding="utf-8")
        published = re.findall(r'^      - "([^"]+:[^"]+)"$', compose, re.MULTILINE)
        self.assertEqual(published, ["127.0.0.1:8770:8770"])
        self.assertNotIn("ports:", compose.split("  searxng:", 1)[1].split("  searx-egress-auth:", 1)[0])
        self.assertNotIn("crawl-worker", compose)

    def test_provider_images_must_be_explicitly_digest_pinned_by_deployment(self) -> None:
        compose = (ROOT / "research" / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertEqual(compose.count("immutable"), 4)
        self.assertIn("qualified image@sha256 digest", compose)

    def test_research_image_is_non_root_read_only_and_locked(self) -> None:
        dockerfile = (ROOT / "research" / "Dockerfile").read_text(encoding="utf-8")
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        self.assertIn("uv sync --frozen --no-dev --no-editable", dockerfile)
        self.assertIn("USER 65532:65532", dockerfile)
        self.assertIn("HEALTHCHECK", dockerfile)
        self.assertTrue(dockerignore.startswith("**\n"))
        self.assertNotIn("!inference/", dockerignore)
        self.assertIn("!research/src/**", dockerignore)
        self.assertIn("!orchestrator/src/jarvis_orchestrator/**", dockerignore)
        runtime = dockerfile.split(" AS runtime", 1)[1]
        self.assertNotIn("pip install", runtime)
        compose = (ROOT / "research" / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("read_only: true", compose)
        self.assertIn('JARVIS_RESEARCH_QUALIFICATION_FILE: "/run/config/research-qualification.json"', compose)
        self.assertIn("browser_qualification_sha256", (ROOT / "research" / "qualification.test.json").read_text())

    def test_qualification_fixtures_are_profile_only_internal_and_unpublished(self) -> None:
        override = (ROOT / "research" / "docker-compose.qualification.yml").read_text(encoding="utf-8")
        self.assertEqual(override.count('profiles: ["qualification"]'), 3)
        self.assertEqual(override.count("disable: true"), 3)
        self.assertNotIn("ports:", override)
        self.assertRegex(override, r"(?m)^  qualification_lab:\n    internal: true$")
        self.assertRegex(override, r"(?m)^  qualification_search:\n    internal: true$")
        self.assertIn('profiles: ["production"]', override)
        self.assertIn("JARVIS_QUALIFICATION_SEARXNG_SETTINGS_SOURCE", override)

    def test_searx_proxy_secret_is_confined_to_repo_owned_bridge(self) -> None:
        compose = (ROOT / "research" / "docker-compose.yml").read_text(encoding="utf-8")
        searx = compose.split("  searxng:", 1)[1].split("  searx-egress-auth:", 1)[0]
        self.assertNotIn("egress_proxy_credentials", searx)
        self.assertIn("http://searx-egress-auth:8081", searx)
        bridge = compose.split("  searx-egress-auth:", 1)[1].split("  crawl-worker:", 1)[0]
        self.assertIn("JARVIS_EGRESS_PROXY_CREDENTIALS_FILE", bridge)
        self.assertNotIn("ports:", bridge)
        self.assertIn("disable: true", bridge)


if __name__ == "__main__":
    unittest.main()
