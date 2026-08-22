# CYRUS branding and compatibility boundary

CYRUS is the public name of the assistant and the operator-facing product.
Starting with this rename, the native macOS application is `CyrusUI` and the
preferred executable names use the `cyrus-*` prefix:

- `cyrus-voice`
- `cyrus-orchestrator`
- `cyrus-action-host`
- `cyrus-stack`
- `cyrus-inference`
- `cyrus-research-*`, `cyrus-egress-guard` and
  `cyrus-searx-proxy-bridge`

The original `jarvis-*` executables remain compatibility aliases. Removing them
would break existing launch configurations and operator scripts without adding
any security or correctness benefit.

## Identifiers that deliberately remain unchanged

This is a product rename, not a wire-format or storage migration. The following
identifiers stay frozen until a separately designed, versioned migration exists:

- Python distributions and import namespaces such as `jarvis-voice`,
  `jarvis_voice`, `jarvis-orchestrator`, `jarvis_orchestrator` and
  `jarvis_inference`;
- wire protocols and schemas such as `jarvis.voice/1`, `jarvis.voice-auth/1`,
  `jarvis-private-stack/2` and `jarvis-voice-ui-state/2`;
- model aliases, database schema identifiers, evidence schemas and existing
  qualification digests;
- `JARVIS_*` environment variables used by frozen runtime and qualification
  contracts;
- existing private state roots, including `JARVIS Voice`, and already
  materialized configuration paths.

Changing any of those values in place would make qualified artifacts disagree
with the running system, strand existing private state, or create a silent
downgrade path. CYRUS therefore reads the current legacy state path and speaks
the current protocols without translating or duplicating them.

The source and wheel metadata do change when the new aliases and CYRUS persona
are packaged. Existing qualified binaries remain valid only as the exact old
artifacts they already are; this rename does not rewrite an installed bundle in
place. Any newly built Voice, Orchestrator or Inference artifact needs fresh
digest binding and the qualification required by its release profile before it
can replace a qualified runtime.

## Future removal policy

A future protocol or storage generation may introduce `cyrus.*` identifiers,
but only with explicit schema versions, migration tooling, rollback behavior and
fresh qualification evidence. Until then, `jarvis*` inside a technical
identifier means “legacy compatibility contract,” not the product name.
