# Hosted model selection gate

## Preregistered candidate record

Candidate selection is compatibility-first and occurred without inspecting any
scenario output:

1. [`openai/gpt-5.4-mini-2026-03-17`](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
   was the first candidate. It was rejected before any API call because the
   installed Inspect AI 0.3.252 Responses path does not forward the requested
   seed and, with the model's default reasoning configuration, cannot preserve
   temperature zero. That fails criterion 5 below; it is not evidence about the
   model's task quality.
2. [`openai/gpt-4.1-mini-2025-04-14`](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
   is the preregistered next candidate. The official dated snapshot supports
   Chat Completions and Structured Outputs. Every compatibility and measured
   call must force `responses_api=false`, use the standard OpenAI endpoint with
   no `base_url` override, and preserve temperature zero and the declared seed.

The tracked `eval/model_costs.json` uses the official standard rates per million
tokens: USD 0.40 input, USD 0.10 cached input, and USD 1.60 output. Inspect's
schema also requires a cache-write price; USD 0.40 conservatively treats a
reported cache write as ordinary input. The exact pricing bytes and SHA-256 are
frozen in the experiment manifest.

The GPT-4.1 mini candidate has not yet passed the live preflight and is not a
frozen experiment model. An `OPENAI_API_KEY` is still required operationally.
No development or sealed scenario may be run merely to decide whether this
candidate is acceptable.

The primary v0 model must be a hosted, immutable snapshot addressable by an
exact Inspect model identifier. Before any development baseline result is
recorded, run a compatibility check on a synthetic case that is not one of the
50 evaluation scenarios. The candidate passes only when it:

1. accepts strict JSON schemas for both `Decision` and `MemoryDelta`;
2. returns valid structured output without a repair call;
3. reports input, cache, and output tokens for every request;
4. has a pinned pricing configuration from which USD cost is reproducible;
5. supports temperature zero and the preregistered seeds; and
6. can be identified by an immutable snapshot rather than a moving alias.

Run the check with Inspect response caching disabled, `max_samples=1`,
`max_tasks=1`, and `max_connections=1`; use those same settings in every
measured task. Strict reports read the resulting `.eval` metadata rather than
trusting a declared command line.

The successful preflight `.eval` is itself a required manifest artifact. Its
byte hash is pinned, and measured task construction revalidates the exact two
model calls, current prompts and wire schemas, no-cache/no-retry policy, token
usage, and cost recomputed from the pinned pricing entry. A boolean declaration
of structured-output support is not sufficient.

This is a compatibility gate, not a model leaderboard. Select the first
preregistered candidate that passes; do not compare candidate F1 on the
development or sealed scenarios. Then fill and freeze a copy of
`experiment_manifest.template.json` after committing the source tree. Store the
generated baseline/final manifest at its ignored local path, set `git_commit` to
the clean source `HEAD`, and treat it as immutable once the first measured task
starts. Archive and publish those exact frozen bytes, rather than a regenerated
copy, with the `.eval` logs. It must not be committed into the revision it
identifies because that would make the Git hash self-referential. The same
snapshot must serve as both the Anamnesis memory compiler and the final decision
model, and it must be used for all four systems.

Changing the model, pricing configuration, generation settings, or schema
after a measured development run creates a new experiment and requires a new
manifest. The 15 sealed scenarios must never be used for compatibility testing.

The vector baseline has a separate immutable-input gate. Its model is
`BAAI/bge-small-en-v1.5`, repository
`qdrant/bge-small-en-v1.5-onnx-q`, revision
`52398278842ec682c6f32300af41344b1c0b0bb2`, canonical artifact-tree SHA-256
`d435d05b3411502ad9a280cc9ac0157f7bcd9f176df2fdc8971f788a121a02d7`,
and `top_k=5`. The snapshot contains nine regular files, no symlinks, and
produces 384-dimensional vectors in a local-only warm-up. Before any baseline
run:

- pass its explicit local directory as
  `embedding_snapshot_path`; measured runs must not rely on implicit cache or
  network resolution;
- recheck that directory against the pinned canonical SHA-256; and
- pass the same repository, revision, `top_k`, and local snapshot directory to
  every vector-RAG task.

The local absolute path is transport configuration and may differ between
machines; the repository, commit and verified tree hash define artifact
identity. The artifact identity is now preregistered; only a valid local path
and the hosted-model preflight remain before freezing the development manifest.
