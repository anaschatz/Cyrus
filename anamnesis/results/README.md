# Results

No preregistered benchmark result is claimed yet. A strict 10-scenario
zero-provider-cost [local smoke diagnostic](local_smoke.md) is available with
its [CSV](local_smoke.csv), [failure analysis](local_smoke_analysis.md), and
[provenance sidecar](local_smoke.provenance.json). It is explicitly not a
hypothesis test and it rejected the current Qwen 4B compiler/prompt
configuration before expansion to the development set.

A separate gold-assisted [oracle-compiler ceiling](local_oracle_smoke.md), its
[CSV](local_oracle_smoke.csv),
[forensic analysis](local_oracle_smoke_analysis.md), and
[provenance sidecar](local_oracle_smoke.provenance.json) isolate the writer from
the deterministic memory and shared decision path. It achieved F1 87.5% (7 TP,
1 FP, 1 FN) with zero obsolete-memory errors. It is a diagnostic, reports a
decision-only token lower bound, and is neither a baseline nor a hypothesis
test.

The post-hoc, preregistered [D1 shared decision-prompt ablation](local_smoke_d1.md)
is published with its [CSV](local_smoke_d1.csv),
[analysis](local_smoke_d1_analysis.md), and
[provenance sidecar](local_smoke_d1.provenance.json). All four systems still
scored F1 0. The simple baselines produced no false alarms, but Anamnesis kept 7
false reminders and 2 invalid compiler outputs and failed the frozen promotion
gate. D1 is rejected; it is not a hypothesis test and does not authorize a
35-scenario run.

The subsequent [W1 writer preflight record](local_writer_w1_preflight_failure.md)
documents a stricter compiler prompt tested against the same pinned local model
only on a synthetic compatibility request. Its compiler output failed domain
validation because unused optional payload fields were emitted as empty values,
including `date: ""`. The frozen stopping rule therefore prevented all 10 fresh
writer scenarios from running. The accompanying
[provenance sidecar](local_writer_w1_preflight_failure.provenance.json) binds the
source commit and exact ignored `.eval` bytes. This is a preflight rejection,
not a scenario result or a hypothesis test.

The follow-up [W2 writer diagnostic](local_writer_w2.md) is published with its
[CSV](local_writer_w2.csv), [forensic analysis](local_writer_w2_analysis.md),
and [provenance sidecar](local_writer_w2.provenance.json). W2 changed only the
serialization rule for unused optional payload slots and used a new dataset
frozen before the prompt. Its four-call semantic preflight passed, but its one
authorized 10-scenario run failed the frozen writer gate: 4 parse/domain
invalid deltas, 1 semantic/store-invalid delta, 41/46 accepted deltas, and
candidate TP=0, FP=3, FN=8. W2 is development-only, is not hypothesis-test
evidence, and does not authorize a run on the 35-scenario development set or a
repaired rerun on the same cases.

The bundled [W3 semantic preflight](local_writer_w3_preflight_failure.md) was
then run exactly once against a fourth blind writer set and stopped before all
scenario evaluation. Only C5 (stable-ID trigger update), C7 (complete sparse
payload), and D1 (no action) passed. C2/C4 failed domain conversion and
C1/C3/C6/C8 were schema-valid but semantically wrong. The accompanying
[provenance sidecar](local_writer_w3_preflight_failure.provenance.json) binds
the 9-call raw `.eval`, contracts, usage, and clean source commit. W3 is
rejected at preflight; no W3 manifest or measured result exists, and no W4 may
be tuned on the v4 cases.

The follow-up [W3-M2 model-only preflight](local_writer_w3_m2_preflight_failure.md)
kept the W3 prompt, schemas, fixture, and execution policy fixed and changed
only the pinned local model to Qwen35 9.7B Q4_K_M. The artifact loaded and ran
locally, but all eight compiler calls ended at `max_tokens` with no parseable
completion; only D1 no-action passed. The accompanying
[provenance sidecar](local_writer_w3_m2_preflight_failure.provenance.json)
binds the clean source commit and exact nine-call `.eval`. W3-M2 is rejected,
and no scenario dataset, manifest, or measured scenario log exists for it.

The transport-only
[W3-M2-T1 preflight](local_writer_w3_m2_t1_preflight_failure.md) then kept the
model, W3 prompt/schema/fixture, context budget, and execution settings fixed
while forwarding `reasoning_effort: "none"` to Ollama. This eliminated the
hidden-reasoning output exhaustion: all compiler calls returned final,
non-truncated content and total setup time fell to about 8.14 minutes. The gate
still failed because all eight compiler outputs used structures outside the
frozen `LocalMemoryDeltaWire`; D1 no-action passed. Its
[provenance sidecar](local_writer_w3_m2_t1_preflight_failure.provenance.json)
binds the exact clean source and raw nine-call log. T1 is rejected with zero
scenario calls and no authorized rerun.

For this W2 publication, the exact frozen manifest, standalone preflight
`.eval`, and measured `.eval` are deliberately tracked under
`results/runs/local/writer_diagnostic_w2/` despite the directory's normal
ignore policy. Their byte hashes are pinned in the provenance sidecar. The raw
logs contain local absolute filesystem paths but no credentials or API secrets.

The exact W3 preflight `.eval` is likewise tracked under
`results/runs/local/writer_diagnostic_w3/`. It contains local absolute paths
but no credentials or API secrets. There is deliberately no W3 experiment
manifest or scenario log because the semantic gate failed.

The exact W3-M2 preflight `.eval` is tracked under
`results/runs/local/writer_w3_m2/`. It contains local absolute paths but no
credentials or API secrets. There is deliberately no W3-M2 scenario artifact.

The exact W3-M2-T1 preflight `.eval` is tracked under
`results/runs/local/writer_w3_m2_t1/`. It likewise contains local absolute
paths but no credentials or API secrets. There is deliberately no T1 scenario
artifact.

The first complete [OpenMemory-style paired recall diagnostic
v2](local_openmemory_diagnostic_v2.md) used eight fresh cases frozen before its
no-thinking transport task. All 16 local model calls completed with no retries,
parse errors, or provider API cost. Baseline and recall each scored 4/8;
helpful gain was zero, while all recall safety and evidence-contamination gates
passed. The overall frozen gate therefore failed. Its
[provenance sidecar](local_openmemory_diagnostic_v2.provenance.json) pins the
exact local raw log, which is not published because it includes full prompts
and local absolute filesystem metadata.

The follow-up [OpenMemory immediate-action diagnostic
v3](local_openmemory_diagnostic_v3.md) corrected the v2 temporal-reminder prompt
mismatch on eight new cases frozen before treatment. All positive raw outputs
now attempted `emit`, and helpful recall recovered the intended recipient and
address, but 12/16 outputs violated the closed response schema. Only 2/8 calls
per arm were both parse-valid and correct. The frozen gate failed with no
recall safety or evidence regression. Its
[provenance sidecar](local_openmemory_diagnostic_v3.provenance.json) pins the
unpublished raw log and exact source contract.

The fresh [OpenMemory + vLLM v4 diagnostic](local_openmemory_vllm_v4.md)
then used an immutable Qwen3.5 4B MLX artifact and an explicitly pinned
xgrammar backend. The neutral canary passed, all no-action safety cases were
correct, and recall caused no safety or evidence contamination regression.
The overall gate nevertheless failed: 6/16 measured calls repeated an allowed
action object until the 256-token limit, and one additional call passed JSON
and wire validation but failed the stricter domain subject invariant. Baseline
and recall each scored 4/8 with zero helpful gain. The
[forensic analysis](local_openmemory_vllm_v4_analysis.md),
[CSV](local_openmemory_vllm_v4.csv), and
[provenance sidecar](local_openmemory_vllm_v4.provenance.json) bind the exact
tracked 17-call raw artifact. V4 identifies a JSON-Schema-to-domain alignment
gap; it is not a causal comparison with Ollama and cannot be rerun or repaired
on the same cases.

The additive [v5 schema-compatibility gate](local_openmemory_vllm_v5.md) then
tested the corrected closed schema on two fresh cases frozen before execution.
Both calls finished with `stop` and passed JSON, wire, domain, usage, and
semantic validation: one exact `emit` and one exact `no_action`. Usage was 913
input and 132 output tokens at `$0.00` provider API cost. Its
[CSV](local_openmemory_vllm_v5.csv) and
[provenance sidecar](local_openmemory_vllm_v5.provenance.json) bind the exact
tracked raw run. This validates the schema correction only; it is not a v4
rerun, a recall-quality measurement, or a hypothesis test.

The [v6 real indexed-memory diagnostic](local_openmemory_vllm_v6.md) then
stored and searched memory records at runtime rather than passing preselected
hits directly to the model. Retrieval was correct in 8/8 cases, all 16
structured calls were accepted, exact accuracy improved from 3/8 to 4/8, and
there were zero safety regressions. The [analysis](local_openmemory_vllm_v6_analysis.md)
shows that recall supplied the intended missing value in all four helpful
cases, but three failed strict canonical payload normalization. Its
[CSV](local_openmemory_vllm_v6.csv) and
[provenance sidecar](local_openmemory_vllm_v6.provenance.json) bind the tracked
raw run. The backend is the OpenMemory-compatible Anamnesis boundary over the
pinned local FastEmbed index, not the upstream Cavira SDK.

The fresh [v7 prospective canonicalizer diagnostic](local_openmemory_vllm_v7.md)
validated the post-v6 architecture revision on six new cases. Retrieval was
6/6, exact accuracy improved from 2/6 to 5/6, all three helpful opportunities
became exact actions, and there were zero safety regressions across 12/12
accepted structured calls. The [analysis](local_openmemory_vllm_v7_analysis.md)
documents the single common no-hit failure, which was an article-normalization
error in both arms rather than a memory regression. The
[CSV](local_openmemory_vllm_v7.csv) and
[provenance sidecar](local_openmemory_vllm_v7.provenance.json) bind the tracked
raw run.

Raw Inspect logs belong in the ignored `results/runs/` directory. A strict
development report requires one complete 3-system × 35-scenario matrix from a
frozen baseline manifest and is titled “Development baseline — not a final
hypothesis test.” A strict final report requires four systems × 50 scenarios ×
three repetitions from an eligible post-freeze dataset.

Incomplete or exploratory inputs are accepted only with `--allow-incomplete`.
Those reports are visibly labeled diagnostic and never print the preregistered
success gate. Provider-neutral JSONL remains useful for such diagnostics;
headline reports use Inspect `.eval` logs so effective execution settings can
be audited.

The first hosted experiment artifact to place here is `development.md`, reporting the
full-context baseline's recall, precision, tokens, cost, and latency alongside
the other two simple baselines. Until a hosted model snapshot, pricing file,
pinned successful preflight log, embedding commit/artifact, and clean experiment
manifest are frozen, any table would be illustrative rather than measured
evidence.

The frozen experiment manifest is a run artifact. Publish its exact bytes next
to the `.eval` logs and tables, even though the local baseline/final manifest
paths are Git-ignored to let `git_commit` identify an already-clean source
revision without a self-referential commit hash.
