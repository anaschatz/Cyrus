# OpenMemory + vLLM v5 schema-alignment compatibility

V5 is a two-call post-fix engineering regression test. It validates that the
new constrained schema produces at most one action and rejects a subject that
cannot satisfy the domain's verb-plus-object shape. It is not a blind recall
benchmark, does not reuse v4 cases, and is not hypothesis-test evidence.

## Frozen inputs

- Parent architecture-fix commit: `b7c88aec9e4587fe3e0858985e5653bb6ac50a08`.
- Fixture bytes: `fe3bdd57dfc51bbd374d81ded6ff6cbd4ba535e08595be39f5ad857bf1e923a0`.
- Overlay pin bytes: `9de634f8806d295d6b6adad340bf23cdd874ae1c367d10d963144f881cdf6b5a`.
- Aligned contract: `133b34adb292381f72b08e09a783f5b4103613e60c001c66968545da8ddc1999`.
- Aligned schema: `aa1bae78fa12e24d85028e3c0f505ddbe6901ece67baf2a4bc60554dcb259c1b`.
- Exact v4 runtime/model artifact pin is inherited by byte hash; only the
  loopback port (`18001`), served alias and aligned contract/schema change.
- Exactly two ordered calls: one immediate `emit`, then one unrelated
  `no_action`; seed 101, temperature 0, max output 256, no retry or repair.

Start the separately pinned local server:

```bash
VLLM_PLUGINS=metal \
VLLM_METAL_USE_PAGED_ATTENTION=1 \
VLLM_METAL_MULTIMODAL_MODE=text-only-compat \
/private/tmp/vllm-metal-v0.2.0-20260604/.venv-vllm-metal/bin/vllm serve \
  /private/tmp/anamnesis-vllm-models/Qwen3.5-4B-MLX-4bit-32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --host 127.0.0.1 --port 18001 \
  --served-model-name anamnesis-openmemory-v5 \
  --api-key local-v4-loopback-20260809 \
  --generation-config vllm --max-model-len 4096 --max-num-seqs 1 \
  --structured-outputs-config '{"backend":"xgrammar"}' \
  --no-enable-log-outputs --no-enable-log-deltas
```

From the clean source commit, run once:

```bash
PYTHONPATH=src /private/tmp/anamnesis-test-venv/bin/python \
  -m anamnesis.openmemory_vllm_v5 \
  --artifact-root /private/tmp/anamnesis-vllm-models/Qwen3.5-4B-MLX-4bit-32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --api-key local-v4-loopback-20260809 \
  --source-commit "$(git rev-parse HEAD)" \
  --output results/runs/local/openmemory_vllm_v5_compatibility/run.json
```

Then generate the strict report from the same clean commit:

```bash
PYTHONPATH=src /private/tmp/anamnesis-test-venv/bin/python \
  -m anamnesis.openmemory_vllm_v5_report \
  --run results/runs/local/openmemory_vllm_v5_compatibility/run.json \
  --csv results/local_openmemory_vllm_v5.csv \
  --markdown results/local_openmemory_vllm_v5.md \
  --provenance results/local_openmemory_vllm_v5.provenance.json
```

Exit 0 means both structured and semantic projections passed. Exit 2 is a
valid negative result. No second attempt, schema adjustment or v4-case run is
authorized by this compatibility cell.
