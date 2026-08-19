# Qwen3.5 9B official non-thinking sampling — one-shot preregistration

This is a candidate-only diagnostic. It does not modify or qualify the
production inference appliance, its model profiles, or its public contract.
The model must already be running in the separately controlled offline
`llama-server`; this harness never starts or stops a model.

## Frozen question

The prior screening forced greedy decoding (`temperature=0`, `top_p=1`). This
single treatment asks whether the official Qwen3.5 non-thinking sampling
settings materially improve the same pinned 9B artifact without changing the
frozen bilingual corpus, system prompt, scorer, thresholds, template mode, or
runtime identity.

The exact preregistration is
[`../eval/qwen35-9b-official-nonthinking-sampling-v1.prereg.json`](../eval/qwen35-9b-official-nonthinking-sampling-v1.prereg.json).
Its frozen SHA-256 is
`4c28f5527ff5b3a9c968667792048de798fc5232bb907ea3eab6177797836d88`.
The harness rejects any byte change to that file. The pinned scorer-code bundle
SHA-256 is
`ed1682eed182da50cf819bf1bc923a4fc1f728e252009ffb0736c27318f6286c`.

## Single permitted treatment

- temperature: `0.7`
- top-p: `0.8`
- top-k: `20`
- presence penalty: `1.5`
- seed: `101`
- `chat_template_kwargs.enable_thinking`: `false`

Everything else remains frozen: the exact model and runner bytes, 8192-token
context, one slot, native chat template, two warmups, public alias, bearer
authentication, 40 Greek and 40 English cases, original deterministic scorer,
and original acceptance thresholds.

The harness performs authenticated loopback readiness and alias checks, hashes
the supplied model and runner artifacts, applies and tokenizes the native
template for every unique public fixture, and records only hashes and token
counts. The preregistration also pins the exact scorer, corpus loader, strict
JSON, and limit-source bundle; those source identities must be reviewed before
the one permitted run and are repeated in its report. It rejects reasoning
fields, thinking tags, tool output, malformed or
incomplete SSE, missing usage, alias changes, and nonterminal streams. Prompts,
completions, bearer values, and local paths are never written to evidence.

## One-shot rule

A private exclusive marker is created immediately before the first generation
dispatch. Once that marker exists, no second run is allowed, even if the first
dispatch or a later case fails. Readiness, artifact, preregistration, template,
and tokenizer checks happen before the marker so an infrastructure-only
preflight failure does not consume the experiment.

There are no retries, repairs, prompt edits, scorer edits, threshold edits, or
post-result tuning. A pass requires all original gates simultaneously:

- at least 34/40 Greek cases;
- at least 34/40 English cases;
- at most a 10-percentage-point language gap;
- 80/80 complete protocol exchanges;
- zero critical failures.

A failure rejects this exact sampling profile. A pass is still only challenger
evidence and authorizes a later sealed-renderer qualification; it does not
promote the model by itself.

## Operator invocation

After separately starting and verifying the frozen raw server, run from the
repository environment with private, previously nonexistent marker and output
paths:

```text
python inference/scripts/run_qwen35_official_sampling_candidate.py \
  --base-url http://127.0.0.1:<port>/v1 \
  --orchestrator-config <private-config> \
  --corpus inference/fixtures/local-fallback-quality-v1.json \
  --preregistration inference/eval/qwen35-9b-official-nonthinking-sampling-v1.prereg.json \
  --runner-artifact <pinned-llama-server> \
  --model-artifact <pinned-gguf> \
  --one-shot-state <new-private-marker.json> \
  --output <new-private-report.json>
```

The 80 cases are primarily exact-output and short extractive-conformance
checks. Therefore even a passing result is not a general conversational or
multi-turn quality measurement.
