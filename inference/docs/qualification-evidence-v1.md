# Qualification evidence v1

The evidence file is created only by a native fixture runner launched by the
qualifier during the same exclusive session. `qualify` does not accept an
evidence path. It creates a fresh output directory and exposes only:

```text
JARVIS_QUAL_CHALLENGE=<64 lowercase hex>
JARVIS_QUAL_IDENTITY_DIGEST=<64 lowercase hex>
JARVIS_QUAL_OUTPUT=<fresh non-existing output path>
JARVIS_QUAL_MANIFEST=<absolute manifest path>
JARVIS_QUAL_BASE_URL=<ephemeral http://127.0.0.1:port/v1>
JARVIS_QUAL_API_KEY=<boot-scoped bearer secret>
```

The SDK bootstrap and native fixture runner use this same managed engine and
ephemeral gateway. The bearer secret exists only in the child environment and
is never placed in command-line arguments or the final report.

The runner writes exactly one strict JSON object to `JARVIS_QUAL_OUTPUT` and
exits zero. Its top-level object includes the numerical fields below plus:

```json
{
  "suite_version": "jarvis-inference-qualification-v1",
  "fixture_digest": "<64 lowercase hex>",
  "deterministic_seeds": [1],
  "identity": {},
  "provenance": {
    "challenge": "<JARVIS_QUAL_CHALLENGE>",
    "identity_digest": "<JARVIS_QUAL_IDENTITY_DIGEST>",
    "fixture_digest": "<same top-level fixture_digest>",
    "started_unix_ns": "1",
    "completed_unix_ns": "2",
    "monotonic_elapsed_ns": 1
  }
}
```

The two Unix-nanosecond wall timestamps are decimal digit strings so their
full precision remains inside the appliance's strict I-JSON/JCS numeric domain.
They must fall inside the qualifier-observed subprocess
boundary and the monotonic duration cannot exceed that boundary. The runner is
hashed before and after execution and its identity is recorded in the pass
report. This prevents accidental stale/static evidence reuse; resistance to a
malicious same-user administrator remains outside the threat model.

The qualifier compares the entire `identity` object to the identity captured
immediately before engine spawn, rehashes all artifacts, runs the official
sync/async OpenAI SDK bootstrap, and then applies these exact gates:

- 20 Greek and 20 English streaming successes.
- When enabled, 10 named plus 10 required tool calls and 20 strict schemas.
- 5 prefill and 5 mid-stream cancellations.
- 5 isolated crash/restart cases and 50 isolation pairs.
- 100% protocol conformance, zero privacy leaks, unexpected OOMs, crashes, and
  hangs.
- Windows: the exact 100-utterance ASR fixture, at least 80 overlaps, no lost
  final transcript/throttle/OOM/crash, and p95 degradation no greater than 15%.
- Mac: 20 warmups and 200 normal requests, final median drift and Theil-Sen
  projected growth each no greater than 5%, and no OOM/crash/stale state.

The authoritative report is overwritten durably with `in_progress` before any
native work. Any failure durably replaces it with `failed`; only all successful
gates replace it with `pass`. Every state has `report_schema_version:1`.
Production accepts only a matching v1 pass and rehashes the recorded fixture
runner; a changed/missing runner invalidates readiness.

## Exact top-level result fields

Unknown fields are rejected. In addition to `suite_version`, `fixture_digest`,
`deterministic_seeds`, `identity`, and `provenance`, the runner supplies exactly:

```json
{
  "streaming": {"greek_success": 20, "english_success": 20},
  "tools": {"named_success": 0, "required_success": 0},
  "json_schema": {"strict_success": 0},
  "cancellation": {"prefill_success": 5, "mid_stream_success": 5},
  "crash_restart": {"isolated_success": 5},
  "isolation": {"pairs_success": 50, "leaks": 0},
  "protocol_conformance": 1,
  "privacy_leaks": 0,
  "unexpected_oom": 0,
  "unexpected_crashes": 0,
  "unexpected_hangs": 0,
  "platform_gate": {}
}
```

When `tools` or `json_schema` capability is true, its zero values are exactly
`10`/`10` or `20`; otherwise they must remain zero.

Windows `platform_gate` is exactly:

```json
{
  "kind": "windows_asr_coexistence",
  "fixture": {
    "rendered_prompt_token_lengths": [512, 2048, 4096],
    "max_completion_tokens": 512,
    "request_count": 100,
    "concurrency": 1,
    "schedule": "serial-continuous-from-audio-start",
    "minimum_overlapping_utterances": 80,
    "sampling": {"temperature": 0, "top_p": 1},
    "power_profile": {
      "windows_power_scheme_guid": "12345678-1234-1234-1234-123456789abc",
      "ac_state": "ac",
      "gpu_power_limit_watts": 360,
      "gpu_clock_policy": "exact-policy",
      "driver_version": "exact-version",
      "compute_runtime": "exact-version",
      "require_zero_thermal_power_throttle_reasons": true
    }
  },
  "audio_corpus_digest": "<64 lowercase hex>",
  "greek_utterances": 50,
  "english_utterances": 50,
  "baseline_power_identity_digest": "<SHA-256 of RFC8785 power_profile>",
  "coexistence_power_identity_digest": "<same SHA-256>",
  "overlapping_utterances": 80,
  "baseline_p95": 1,
  "coexistence_p95": 1,
  "lost_final_transcripts": 0,
  "oom": 0,
  "crashes": 0,
  "throttle_reasons": 0
}
```

The hashed fixture corpus contains exactly 100 utterances (50 Greek and 50
English), the two-second inference lead-in and the overlap schedule. The exact
power object is checked and its RFC 8785 digest must be identical for baseline
and coexistence.

Mac `platform_gate` is exactly:

```json
{
  "kind": "mac_unified_memory",
  "warmup_requests": 20,
  "normal_requests": 200,
  "final_drift": 0,
  "theil_sen_projected_growth": 0,
  "oom": 0,
  "crashes": 0,
  "stale_state_references": 0
}
```

All decimal metrics must be finite. Boolean values never satisfy integer or
numeric gates.
