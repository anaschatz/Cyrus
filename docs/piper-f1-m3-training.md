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
builds in a temporary directory, refuses to overwrite a different extension,
and performs an import probe. It does not use the network or start training.
