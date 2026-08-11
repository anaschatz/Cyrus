# JARVIS Research System v1

This directory freezes the deployment boundary for the off-by-default Research
System. It does not make an unqualified provider image production-ready.

`jarvis-research-api` is the model-free executable for `/v1/qualification` and
`/v1/research`. It returns only `ResearchCorpusV1`; span selection remains the
Orchestrator's second model request. `jarvis-research-qualify` runs the live,
fail-closed SearXNG/direct-fetch gate and emits canonical JSON evidence plus its
SHA-256 digest.

The checked-in `qualification.test.json` has fixture digests only. Production
must supply a strict eight-field qualification file and Docker secret files.
No secret is accepted through a command-line argument or returned by the API.

Run the sidecar with `uv run --frozen jarvis-research-api`. The live gate is
`uv run --frozen jarvis-research-qualify --manifest <manifest.json> --output
<evidence.json>`. A manifest contains the loopback `/v1` endpoint, paths to the
bearer secret and Compose directory, all mandatory fixture requests and raw
query/URL markers that must be absent from container logs. The output is
canonical JSON; any missing case, unavailable container or failed invariant
sets `gate_status` to `REJECTED`. Evidence schema v2 has two digests:
`evidence_sha256` is the stable qualification identity digest over image and
artifact hashes plus case decisions (excluding its own field and live timings),
while `record_sha256` covers the complete emitted record. An accepted record
requires the API's `qualification_sha256` to equal `evidence_sha256`.

## Security boundary

- Every workload-facing Docker network is `internal: true`.
- `egress-guard` is the only service attached to `edge_external`.
- SearXNG publishes no host port. The Research API is bound only to
  `127.0.0.1:8770` and requires a bounded bearer token.
- Provider and guard images must be supplied as immutable `image@sha256:...`
  values. Tags are intentionally rejected by the deployment procedure.
- Secrets are local files under `research/secrets/` and are ignored by Git.
- Browser/Crawl4AI services are absent from this milestone. Their qualification
  digest remains empty.

The proxy configuration is not the security boundary: route absence from the
internal networks is. Qualification must additionally prove that the selected
egress-guard image independently resolves A/AAAA records, rejects the complete
answer when any address is non-public, pins the validated address, repeats the
policy on redirects, permits only ports 80/443, and cannot inherit an upstream
proxy from its environment.

## Live qualification lab

`docker-compose.qualification.yml` adds only profile-scoped, offline HTTP,
search and authoritative DNS fixtures. The DNS fixture returns deterministic
public, private, mixed and rebinding answers; the HTTP fixture covers robots,
redirect, MIME, compression, language and date cases. It publishes no host
ports and both qualification networks are `internal: true`.

The repo-owned `jarvis-egress-guard` executable performs authenticated
absolute-form HTTP and CONNECT proxying, resolves A/AAAA itself, rejects the
complete answer if any address is non-global or configured container-internal,
and connects to a selected pinned IP. `jarvis-searx-proxy-bridge` is isolated
between SearXNG and the guard and reads the proxy credential from a Docker
secret; SearXNG receives neither the credential nor an authenticated proxy URL.

Run the fixture lab with both Compose files and `--profile qualification`.
The checked-in `qualification.manifest.template.json` is strict and complete.
It is not production evidence: only `jarvis-research-qualify` output from live
containers and immutable images can produce an accepted gate.

## Gates

Enable `research_enabled` only after the structured router digest, language
detector digest, authority registry, egress image and provider images match an
approved qualification record. This deployment qualifies only SearXNG and
direct fetch; browser support requires a later, separately reviewed milestone.

The Orchestrator owns durable plans, approvals, epochs, cited snapshots and
publication. The sidecar owns public search/fetch/extraction and returns the
closed `ResearchCorpusV1` contract. Firecrawl is not part of v1.

## Online retrieval quality gate

The next, separate gate is `online-retrieval-quality/1`.  Its checked-in seed
is `benchmarks/retrieval-benchmark-seed-v1.json`: exactly 50 Greek and 50
English cases.  It is intentionally marked `UNREVIEWED`; empty qrels are not
qualification evidence and force a rejected result.  A reviewer can validate
the artifact and export a JSONL worksheet without invoking a model:

```console
uv run --frozen jarvis-research-benchmark \
  --benchmark benchmarks/retrieval-benchmark-seed-v1.json \
  --review-jsonl /secure/review/retrieval-qrels-v1.jsonl
```

Production image identity is supplied in the closed
`research-image-manifest/1` format.  `image-manifest.template.json` contains
deliberately invalid placeholders: replace them with registry
`repository@sha256:<manifest-digest>` references plus the matching image
configuration digests.  Local image IDs and mutable tags are rejected.

The metric thresholds are explicit in
`online-qualification-thresholds-v1.json`; the runner has no permissive hidden
fallback.  A fail-closed evidence record can be produced with:

```console
uv run --frozen jarvis-research-online-qualify \
  --benchmark benchmarks/retrieval-benchmark-seed-v1.json \
  --images image-manifest.production.json \
  --thresholds online-qualification-thresholds-v1.json \
  --output results/online-retrieval-quality-v1.json
```

`--execute-live --bearer-token-file <secret>` is accepted only after every
case is human-reviewed and the three registry images match the runtime image
configuration and platform.  Case evidence contains IDs, qrel ranks, counts,
durations and hashes only—never queries, result URLs, titles or documents.
The live retrieval metrics alone cannot accept the release: quote integrity,
freshness/conflict/independence, bandwidth, real Orchestrator cancellation and
privacy interception remain mandatory independent decisions.  Until those
evidence producers are connected, the runner reports their typed blockers and
keeps `research_enabled` false.
