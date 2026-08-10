# Local OpenMemory decision diagnostic v2 execution

The fresh v2 case artifact was frozen in source commit `4fda3935d00548a8e50e35abb286c6dfe6256cac` before this transport task was authored. Its text, IDs, entities and expected decisions are disjoint from v1.

## Frozen cell

- artifact: `eval/openmemory/decision_diagnostic.v2.json`;
- model: the existing byte-pinned `ollama/qwen3.5:9b-q4_K_M`;
- transport intervention: `GenerateConfig.extra_body={"reasoning_effort":"none"}`;
- seed `101`, temperature `0`, cache disabled, retries `0`, repair calls `0`;
- exact order: eight cases, baseline then recall for each, 16 calls total;
- one measured attempt only; no retry, duplicate selection or prompt repair;
- no live OpenMemory writes: recall hits come from the frozen deterministic snapshot;
- the two prompts in a pair differ only by the labelled retrospective recall section;
- provider API cost is exactly zero; electricity and hardware cost are unmeasured;
- retrieval token/cost accounting remains incomplete and is not an efficiency claim.

The no-thinking field is a transport compatibility correction motivated by the terminal v1 context-exhaustion failure. It was selected before any v2 model call. Because v2 uses fresh cases, this is not a rerun or tuning pass on v1.

## Stopping rule

Run the task once from its clean source commit. Preserve a valid pass or fail. If execution reaches a model call, do not rerun v2, tune the prompt, change output limits, or select a duplicate. A further transport or prompt change requires fresh v3 cases.

Interpret only the frozen paired gate: at least one helpful recall gain, no safety/no-hit regression, no recall-induced false action, and no evidence contamination. This remains a development diagnostic, not a hypothesis test or general OpenMemory benchmark.
