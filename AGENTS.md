# Repository rules

- Before every commit or push, run the Ponytail review for the exact staged or outgoing diff. In Codex this is `@ponytail-review` (the Codex form of `/ponytail-review`).
- Never bypass the repository's `pre-commit` or `pre-push` hooks with `--no-verify`.
- Address Ponytail findings before continuing, or state explicitly why a finding is intentionally accepted.

