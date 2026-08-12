# STOMA F1 Piper training — Windows/WSL2 readiness

This milestone is a passive gate. It does not install PyTorch, Piper, a CUDA
toolkit, or a driver, and it never starts training.

## Why WSL2

The pinned Piper training instructions target a Linux toolchain
(`build-essential`, CMake, Ninja, and a Cython extension). CUDA in WSL2 is
provided by the NVIDIA driver installed on Windows. Do not install a Linux
NVIDIA display driver inside WSL and do not use the `cuda`, `cuda-12-x`, or
`cuda-drivers` meta-packages.

The candidate dependency profile is:

- CPython 3.11, Linux x86_64 under WSL2;
- Piper `v1.6.0` at commit
  `f04d52c5528ac7cf2d73757f57990ff490f75005`;
- PyTorch `2.12.1`, official `cu126` wheel index;
- Windows NVIDIA driver at least `560.76`;
- at least 8,000 MiB total and 6,500 MiB currently free VRAM;
- at least 16 GiB system RAM and 25 GiB free in the WSL ext4 work directory.

The 25 GiB disk gate includes the environment, 565 MiB F1 dataset, Piper cache,
the 846 MiB warm-start, and multiple large Lightning checkpoints. Training is a
one-time workload: stop Ollama and other GPU workloads first. No GPU process list
is queried; only aggregate total/free memory is read.

## Passive probe

Copy the repository or this probe into the WSL filesystem, not `/mnt/c`, then
run from WSL:

```sh
python3.11 scripts/run_piper_training_readiness.py \
  --work-root "$HOME/jarvis-f1-training" \
  --device-index 0
```

The work root must already exist. A passing report has
`code: tts.f1_training.readiness_passed` and
`lock_profile.materialization_allowed: true`. The probe calls `nvidia-smi` with
only index, name, driver version, total VRAM, and free VRAM. It never queries
UUIDs or processes and performs no network access.

## Minimal transfer from the Mac

Move only these inputs into the WSL ext4 filesystem:

- `artifacts/compact-greek-tts-f1/f1-training-data/` (about 565 MiB);
- `artifacts/compact-greek-tts-f1/checkpoint/el/el_GR/rapunzelina/medium/`
  (845,906,140 bytes across the checkpoint, config, and model card);
- the repository source, `requirements/piper-f1-*.in`, and this documentation.

Do not transfer the original Parquet shards, the macOS `.venv-data-prep`, or the
macOS arm64 Piper wheel. Keep training data under the WSL ext4 filesystem rather
than `/mnt/c` to avoid cross-filesystem training I/O overhead. After copying,
re-run the existing bundle-manifest hashes and the pinned checkpoint hash before
materializing any environment.

Build the deterministic transfer manifest on the Mac. This command reads and
hashes every one of the 1,301 F1 WAV files, the metadata, extraction manifest,
and all three warm-start files. It performs no installation or network access:

```sh
python3 scripts/run_piper_training_handoff.py build \
  --data-dir "$PWD/artifacts/compact-greek-tts-f1/f1-training-data" \
  --artifact-root "$PWD/artifacts/compact-greek-tts-f1" \
  --output "$PWD/artifacts/compact-greek-tts-f1/f1-wsl-handoff.json"
```

Copy `f1-training-data/`, `checkpoint/`, and `f1-wsl-handoff.json` into one
private directory in the WSL ext4 filesystem. Verify the copy before running the
readiness probe or installing anything:

```sh
python3.11 scripts/run_piper_training_handoff.py verify \
  --transfer-root "$HOME/jarvis-f1-inputs" \
  --manifest "$HOME/jarvis-f1-inputs/f1-wsl-handoff.json"
```

The verifier rejects symlinks, missing or extra source audio, unsafe relative
paths, changed bytes, incompatible dataset/Piper revisions, and incorrect
aggregate totals. It does not write to the transferred inputs.

## Lock gate

`requirements/piper-f1-torch.in` resolves PyTorch exclusively from the official
CUDA 12.6 index. `requirements/piper-f1-training.in` separately mirrors the
dependency constraints in the pinned Piper source using PyPI. They are
intentionally not installable locks. Resolve both to fully hashed Linux x86_64
locks only after the passive probe passes, then verify that the selected torch
build reports CUDA 12.6. Before any training, the resulting environment must
also pass:

1. exact package/hash audit;
2. `torch.cuda.is_available()` and selected-device allocation/release smoke;
3. CPU-only load of the pinned Greek checkpoint;
4. one-batch `fast_dev_run` with network disabled;
5. a short generated Greek sample for listening.

Failure at any stage leaves the current voice backend unchanged.
