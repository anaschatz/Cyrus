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

## Current status

The repository template is deliberately unready: counters are zero, fault
flags are false and endpoint evidence is empty. No real OpenRouter key or live
qualification artifact is stored in the repository. Therefore cloud must
remain disabled until the month-long evidence collection is complete.
