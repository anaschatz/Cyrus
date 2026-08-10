# Local OpenMemory decision diagnostic execution

This protocol was added only after the v1 diagnostic artifact was frozen in
commit `b32335f`. It does not change the eight cases, recall hits, expected
decisions, paired gate, or the already committed OpenMemory recall prompt.

## Frozen local matrix

- model: `ollama/qwen3.5:9b-q4_K_M` with the existing exact local artifact pin;
- decision schema and local decision instructions: unchanged;
- seed `101`, temperature `0`, cache disabled, retries `0`, repair calls `0`;
- case order: exact artifact order;
- call order inside every case: baseline first, recall second;
- total measured calls: exactly 16 (8 baseline + 8 recall);
- one attempt only, with no selected duplicate run;
- no live OpenMemory writes: the measured input is the frozen deterministic
  search-only snapshot;
- the baseline and recall prompts differ only by the labelled retrospective
  recall section;
- all raw completions, parse status, tokens, latency and zero provider API cost
  must be retained;
- OpenMemory retrieval usage remains incomplete and is not a token-efficiency
  claim.

The strict paired gate is the one frozen in
`eval/OPENMEMORY_DIAGNOSTIC.md`: one helpful gain, no safety or no-hit
regression, no recall-induced false action, and no evidence contamination. A
valid failed result is preserved. No second prompt may be tuned or run on these
same v1 cases.

Before a live run, the task must prove the exact local response schema, model,
loopback endpoint, seed/config, complete model usage and clean source commit.
If any preflight or raw-log integrity check fails, no metric is interpreted.

The original draft named the older 4B artifact. Before any diagnostic call, a
read-only local inventory showed that artifact was no longer installed while
the independently byte-pinned 9B artifact was already present. The model choice
was therefore changed and committed before execution, with no download and no
case/result inspection. This remains a paired within-model recall comparison;
it is not comparable to earlier 4B result levels.
