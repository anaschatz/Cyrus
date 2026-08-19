# Backend CI and packaging gate

`.github/workflows/backend-ci.yml` is the source-only production gate for the
Voice, Anamnesis, Inference, and Orchestrator Python backends. It never starts a
real model, calls a provider, uses repository secrets, or uploads runtime/test
payloads.

## Gate layout

- `quality` checks the three maintained lockfiles, validates the workflow's
  closed security contract, runs fatal Ruff checks over every backend, applies
  Anamnesis's declared full Ruff/format policy, and rejects committed
  whitespace errors. Pull requests check the immutable base-to-head commit
  range; push and manual runs check the exact GitHub-provided head SHA.
- `component-tests` runs every discoverable unit, security, migration, and
  fault-injection test on Linux with CPython 3.11. Dependency synchronization
  may use the package index; every subsequent test command runs with
  `UV_OFFLINE=1` and `--offline`.
- `windows-security` executes the native Windows configuration, private-state,
  DACL, and stack-migration cases. It uses the Windows branch of the locked
  Orchestrator dependency graph, including its conditional `pywin32` pin.
- `packaging` builds all four wheels without network access, installs each into
  a newly created virtual environment from hash-bearing lock exports and the
  warmed uv cache, imports it with Python isolated mode, and runs its installed
  console entrypoint help path. Build isolation is disabled deliberately: the
  exact root or Anamnesis locked environment supplies the reviewed backend.

Inference deliberately has no independent lockfile because its runtime has no
dependencies. Its tests and setuptools build use the root Voice lock as the
reviewed CPython/build-toolchain boundary. Adding an Inference dependency must
therefore include a dedicated `inference/uv.lock` and a workflow update in the
same change.

No artifact action is present. GitHub's ordinary step logs contain test status;
the workflow does not retain wheels, reports, database files, paths, prompts,
or test payloads after the ephemeral runner is destroyed.

## Supply-chain choices

All reusable actions are pinned to a full 40-character commit SHA. GitHub calls
that the only immutable way to consume an action. The workflow grants only
`contents: read`, disables checkout credential persistence, has bounded job
timeouts and concurrency cancellation, and keys setup-uv caches from the
relevant `uv.lock` files.

The uv executable is fixed at `0.9.18`, matching the repository's reviewed lock
toolchain. `uv lock --check` must succeed before synchronization. A lock-backed
sync is the only network-capable dependency step; tests, builds, fresh-venv
installs, imports, and entrypoint smokes are explicitly offline.

Primary references:

- [GitHub secure-use reference](https://docs.github.com/en/actions/reference/security/secure-use)
- [GitHub workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
- [uv GitHub Actions integration](https://docs.astral.sh/uv/guides/integration/github/)
- [uv command reference](https://docs.astral.sh/uv/reference/cli/)
- [setup-uv cache key contract](https://github.com/astral-sh/setup-uv/blob/main/docs/caching.md)

## Local verification

Use only already-synchronized environments when local policy forbids downloads:

```bash
anamnesis/.venv/bin/python -m unittest tests.test_backend_ci_workflow -v
anamnesis/.venv/bin/ruff check --select E9,F63,F7,F82 \
  src tests anamnesis/src anamnesis/tests anamnesis/eval \
  inference/src inference/tests orchestrator/src orchestrator/tests
git show --check --format= HEAD
```

The CI workflow itself performs the authoritative lock-backed Linux and Windows
runs. A local sandbox that blocks loopback sockets or the operating system
configuration API cannot substitute for those hosted-runner results.
