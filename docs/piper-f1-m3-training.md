# STOMA F1 Piper training on the M3 laptop

The selected F1 voice may be fine-tuned locally on the 16 GB Apple Silicon
laptop through PyTorch MPS. This replaces the WSL/CUDA host for the personal
pilot only; it does not change the active JARVIS TTS backend.

PyTorch and Lightning still describe MPS as experimental. The profile therefore
uses one device, batch size 1, `32-true` precision, zero data-loader workers and
CPU fallback for unsupported operators. The main LLM and other large local
models must be stopped during the smoke and training run.

## Passive readiness

```sh
python3 scripts/run_piper_mps_training.py readiness \
  --data-dir "$PWD/artifacts/compact-greek-tts-f1/f1-training-data" \
  --artifact-root "$PWD/artifacts/compact-greek-tts-f1" \
  --work-dir "$PWD/artifacts/compact-greek-tts-f1/mps-training" \
  --source-root "$PWD/artifacts/compact-greek-tts-f1/piper-source"
```

The check performs no installation, download, checkpoint load or training. It
rehashes all F1 inputs and requires native arm64 CPython 3.11, 16 GiB unified
memory, Clang, the pinned Piper source revision, exact training dependencies,
an available MPS backend and enough disk.

Eight GiB free is the minimum for environment materialization and the one-batch
smoke. Full training is blocked below 20 GiB free because the Piper cache,
environment and several large Lightning checkpoints coexist. Do not disable the
MPS allocator watermark to bypass an OOM; this can destabilize the whole system.

## Mandatory first run

After a separate, locked environment is materialized, generate the exact argv
profile with `plan`. Run only `smoke_argv` first. It enables Lightning
`fast_dev_run`, so exactly one training and validation batch exercise checkpoint
loading, phonemization, cache construction, forward/backward passes and MPS/CPU
fallback. A successful process must also show that MPS—not CPU—is the selected
accelerator and must leave the system without memory pressure.

Only after that gate passes and at least 20 GiB is free may the bounded initial
25-epoch run start. Listen to an exported checkpoint before extending the run.
The production voice remains the macOS system backend until ONNX size, runtime
RSS, first-audio, cancellation, pronunciation and explicit listening approval
all pass.

The environment lock is resolved only for CPython 3.11/macOS arm64 from
`requirements/piper-f1-mps.in`, with binary wheels and SHA-256 hashes. Piper
itself is installed from the already pinned local `piper-tts==1.6.0` wheel. The
monotonic-alignment extension is compiled with the installed Cython/NumPy and
Clang from the exact pinned source checkout; the incomplete training copy in the
runtime wheel is never treated as sufficient.

After syncing the hashed lock into a dedicated virtual environment, install the
pinned local Piper wheel without dependencies and run:

```text
python scripts/materialize_piper_mps_extension.py \
  --python /absolute/path/to/.venv-mps/bin/python \
  --source-root /absolute/path/to/piper-source
```

The command rejects every source revision except the release-pinned commit,
builds in a temporary directory, writes a source/binary provenance manifest,
refuses to overwrite an unmanaged or different extension, and performs an
import probe. Repeated runs validate the manifest without rebuilding. It does
not use the network or start training.

The training command runs through `scripts/run_piper_mps_fit.py` and uses Piper
1.6's full non-strict `model.warmstart_ckpt` path, starting a fresh optimizer
instead of trying to resume the legacy Lightning configuration. Before Piper
loads the warm-start checkpoint, the wrapper verifies its pinned size and
SHA-256 and statically requires its complete weights-only unsafe-global set to
be exactly `pathlib.PosixPath`. That single type is allow-listed only for the
scoped Piper invocation, and the wrapper forces `weights_only=True` only for
that exact verified checkpoint even though Piper requests otherwise.

The bounded M3 profile sets `model.mos_metric=none`. Piper's optional UTMOS
validator uses `torch.hub` and otherwise attempts a network download during the
first validation batch; perceptual scoring belongs in the later offline
qualification stage, not in training or its smoke gate.

## Native M3 evidence — 2026-08-12

The isolated environment was materialized on the target 16 GiB Apple M3 with
CPython 3.11.14, `torch==2.12.1`, `lightning==2.6.1` and `piper-tts==1.6.0`.
The Metal tensor probe reported `mps:0`, and the final offline
`trainer.fast_dev_run=true` completed successfully:

- 1,301 F1 utterances were accepted from the local cache;
- all 784 matching parameters were copied from the hash-pinned Greek checkpoint
  and none were skipped;
- the 70.4M-parameter model used the MPS accelerator with batch size 1;
- one training batch completed in about 2.4 seconds and one validation batch in
  about 0.35 seconds; and
- Lightning stopped at exactly `max_steps=1`, without invoking UTMOS or another
  network-backed metric.

The bounded 25-epoch run has not started. The post-smoke readiness report had
13,636,022,272 free bytes, below the fail-closed 21,474,836,480-byte full-run
threshold. The generated training cache is retained because it is required by
the full run.

When `model.mos_metric=none`, the safe training wrapper also removes exactly
the pinned Piper `val_mos` checkpoint callback while retaining the `val_mel`
and `last.ckpt` callback. A resume is accepted only from a regular, local
`lightning_logs/version_N/checkpoints/last.ckpt` with bounded size and an empty
weights-only unsafe-global set; warm-start and resume arguments are mutually
exclusive. Resume also passes `model.warmstart_ckpt=null` explicitly so
LightningCLI cannot restore the original warm-start path from checkpoint
hyperparameters and overwrite the resumed model weights.
