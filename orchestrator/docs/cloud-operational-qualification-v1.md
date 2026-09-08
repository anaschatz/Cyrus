# Cloud operational qualification v1

Cloud remains off by default. `cloud_enabled:true` is only an operator request;
it is not authority to construct an OpenRouter adapter. Production composition
first disables every persisted endpoint row and then requires all static and
live gates below. Any failure leaves the local assistant running with zero
active cloud endpoints.

## Dedicated key

Create a separate non-management OpenRouter key with an exact `$5` limit and
monthly reset. Do not use a management/provisioning key and do not include BYOK
usage in the limit. Store the plaintext once in an owner-only file outside the
repository. The release evidence binds a domain-separated fingerprint of that
exact file; the token is never written to evidence or logs.

The live gate verifies `/api/v1/key` immediately before activation and again
during every dispatch revalidation. It requires:

- `limit == 5`, `limit_reset == "monthly"`;
- `is_management_key == false`;
- `is_provisioning_key == false`;
- `include_byok_in_limit == false`.

See the official [current-key contract](https://openrouter.ai/docs/api/api-reference/api-keys/get-current-key).

## Exact endpoint unit

Start from `cloud-qualification.example.json`. Every active row is qualified as
the exact tuple:

```text
model canonical ID
+ full provider endpoint tag
+ quantization
+ parameter profile SHA-256
+ qualification version
```

DeepSeek is mandatory when cloud is requested. Terra is excluded unless
`cloud_terra_enabled:true`; when enabled it requires its own exact tuple and
evidence. The live `/endpoints/zdr` result must match context/output limits,
all supported parameters, pricing including request fee, quantization and the
full endpoint tag. Health never selects a replacement endpoint. OpenRouter's
official [ZDR endpoint response](https://openrouter.ai/docs/api/api-reference/endpoints/list-endpoints-zdr)
is the live oracle, not automatic qualification.

Every captured request must contain one model, one endpoint in `only` and
`order`, `allow_fallbacks:false`, `require_parameters:true`, `zdr:true` and the
exact quantization. The generation receipt must prove the approved model and
provider identity. The provider-routing contract documents why explicit
ordering and disabled fallbacks are required: [provider selection](https://openrouter.ai/docs/guides/routing/provider-selection).

## Evidence required before activation

Copy `cloud-release-qualification.template.json` outside the repository and
replace it only with evidence produced by real qualification runs:

- at least one real dispatch and verified receipt for every active endpoint;
- cost ledger reconciliation against provider prompt/completion/reasoning usage;
- one applied F10 and one denied F11 from the real voice runtime;
- zero duplicate dispatches;
- passing Greek, English and mixed combined-egress privacy fixtures;
- passing timeout, disconnect, stale-catalog and missing-usage injections;
- explicit proof of no model fallback and no provider fallback;
- at least 30 continuous days of shadow observations, no external dispatch,
  no fallback, no budget violation, and projected/observed ledger at or below
  `5_000_000` micro-USD.

The bundle is JCS/domain-hashed as `cloud-release-qualification/1`, configured
through `cloud_release_qualification_sha256`, and may be valid for no more than
31 days. Editing any evidence invalidates the configured digest.

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations cloud-release-digest /absolute/path/evidence/cloud-release.json
```

The local privacy gate is immediately runnable and content-free:

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations cloud-privacy-gate \
  --output /absolute/path/evidence/cloud-privacy.json
```

Release JSON is read once from a bounded regular file (maximum 1 MiB). Duplicate
keys and noncanonical value types are rejected before computing the release
hash. Qualification validity ends at `valid_until`; that instant is excluded.
The privacy evidence must cover at least all built-in English, Greek and mixed
fixture cases, and its policy hash must match the running classifier.

The Greek inflection correction changes the privacy policy digest to
`bfbdbc1aec2e49e3213671f04fcd722f8514772c5717aa1afe43565657d0ac1f`.
Old privacy evidence cannot activate this revision. Rerun qualification and
produce a new release bundle; do not rewrite historical evidence to match.
These fixture results are limited tests of a keyword classifier. Unmatched
nonempty text currently defaults to PUBLIC, so they do not establish semantic
privacy detection for arbitrary user text. The codebase review tracks known
remaining false negatives and provider boundary issues.

After the real evidence bundle, endpoint registry and owner-only key file are
present, run the static plus live activation gate:

```bash
python -m jarvis_orchestrator --config /absolute/path/config.json \
  operations cloud-gate \
  --output /absolute/path/evidence/cloud-activation.json
```

Only `"cloud_ready":true` permits the same endpoint set to be composed at
service startup. A missing Terra qualification leaves Terra disabled; a
missing DeepSeek qualification, wrong key, stale/invalid release bundle,
failed live ZDR/key probe, or any missing evidence leaves all cloud routing
disabled. It does not make local `/readyz` unavailable.

Both activation paths wait for all concurrent endpoint checks to finish before
closing the shared clients or store. Cancellation is propagated after child
checks finish cancellation. A cleanup failure still attempts the remaining
owned cleanup operations.

Before publishing a model response, the adapter requires nonnegative integer
usage and native token counts, rejects contradictory router metadata, and
matches the generation receipt's `id` to the generation returned by the POST.
Missing or invalid native prompt/completion counts cannot become zero by
default. A mismatching generation is quarantined and conservatively accounted
using the existing reservation policy. See the official
[generation receipt fields](https://openrouter.ai/docs/api/api-reference/generations/get-generation)
and [usage accounting fields](https://openrouter.ai/docs/cookbook/administration/usage-accounting).
These are additional checks; live provider qualification is still required.

## Exact numeric admission

Endpoint limits and health counters must have their declared integer types;
boolean/fractional values cannot become integers through truncation. Prices
must be finite and nonnegative. Decimal text is limited to 1,024 characters,
and coefficient length plus absolute exponent is limited to 1,024 before
expanding exponent notation. This bounds both canonical formatting and exact
rational arithmetic. Zero normalizes directly without exponent expansion.

Reservation calculations use exact rational arithmetic, preserve the existing
10% buffer and apply one upward micro-USD rounding step. Provider cost and
usage round upward; available budget rounds downward. The dedicated key limit
must equal the configured five-dollar policy before rounding. Uptime cannot
round into the 99% threshold, and percentages above 100 are invalid. None of
these operations depend on the process-wide Decimal context.

Canonical price text now preserves significant digits beyond Decimal's
ambient precision. Ordinary existing price strings and request snapshots are
unchanged. Previously rounded high-precision prices can produce a different
stable catalog hash; obtain fresh qualification for affected artifacts instead
of rewriting persisted approvals, registry identities or old evidence.

## Provider input bounds

The secret reader requires an absolute path to a private regular file and
rejects leaf symlinks and changed file snapshots. It reuses the configuration
reader's 256 KiB file limit and platform ownership/permission checks. After
trimming surrounding whitespace, the token must contain 1–512 printable ASCII
characters without spaces or control characters. The endpoint registry is a
regular file of at most 1 MiB and uses the existing strict JSON parser.

Model responses, generation receipts, catalog and key responses are limited
to 4 MiB each, counting both encoded and decoded bytes. The adapter accepts
JSON with identity, gzip or zlib-wrapped deflate encoding, bounds decompression
before assembling the document, and rejects incomplete/trailing compressed
data and duplicate JSON keys. Exceeding a limit fails the request; it does not
truncate a catalog or model response. This operational ceiling still requires
live qualification against the current provider catalog.

Each complete POST is covered by the configured timeout, including its body;
control requests use two seconds. The service retains its existing overall
dispatch deadline, and revalidation retains its combined two-second deadline.
Timeouts close the response and preserve conservative reservation accounting.
The physical dispatch permit authorizes only the exact HTTPS OpenRouter chat
endpoint, before any request is sent. No retry or fallback is added.

## Current status

The repository template is deliberately unready: counters are zero, fault
flags are false and endpoint evidence is empty. No real OpenRouter key or live
qualification artifact is stored in the repository. Therefore cloud must
remain disabled until the month-long evidence collection is complete.
