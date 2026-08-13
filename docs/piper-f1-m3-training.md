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

The initial full-run attempt remained blocked while free disk was below the
fail-closed 21,474,836,480-byte threshold. After local space was recovered, the
bounded run completed all 25 epochs on the same M3. The best retained checkpoint
was epoch 22 with `val_mel=0.4752`, 845,857,554 bytes and SHA-256
`88015f28184de746d9a4e2ff3a069657bfeb996584557555bf5a8324764452e4`.

That checkpoint was exported offline through the repository's fail-closed
legacy ONNX wrapper. The wrapper requires an empty weights-only unsafe-global
set, exports to a same-directory partial, validates the graph with ONNX, adds
canonical Greek discovery metadata to Piper's generated config, and publishes
the model atomically. No package was installed for export. The resulting model
is 63,516,051 bytes with SHA-256
`d1fd62b6ee8c2018abc0a030c786de8cd35a2520f7d9304fa893a3c7f0503f11`;
its 5,213-byte config has SHA-256
`08c0eadfc6f6eacb66f1effc2ace3447ef39d5e83994e05566899b174cf83cec`.

The first offline epoch-22 fixture run generated all four required WAV files.
Its maximum warm first-audio latency was 125.081 ms and peak worker RSS was
312,229,888 bytes, so the model-size, memory, latency, fixture and offline gates
passed. Real-output cancellation and human voice approval were deliberately not
claimed by that run. The production TTS backend remains unchanged until those
native gates and the wider qualification profile pass.

Human listening subsequently rejected epoch 22 because of intermittent
metallic/robotic timbre failures inside otherwise intelligible speech. This was
not a playback underrun: every affected utterance was emitted as one complete
Piper chunk, the glitches remained in the saved WAV, and changing duration or
acoustic-noise inference settings did not remove them. A same-text diagnostic
also reproduced the defect on text present verbatim in the F1 training set.

Five repeated renders per checkpoint showed why `val_mel` must not select the
voice by itself. For the same seen sentence, epoch 22 had mean p95/p99
frame-to-frame log-spectral flux of 0.3888/0.5180, compared with 0.3405/0.4404
for epoch 18 and 0.3472/0.4546 for epoch 24. The source recordings are valid
44.1-kHz mono PCM, but contain only 6,418 seconds (1.78 hours) of F1 speech;
the 25-epoch run completed about 29,000 fine-tuning steps. Epoch 22 is therefore
disqualified from promotion despite its best `val_mel`. Epochs 18 and 24 must
be compared by listening before deciding whether either is usable or training
must continue with periodic perceptual checkpoints.

The first epoch-18/24 listening comparison also exposed a separate pacing
failure. Across 40 deterministically selected, seen F1 sentences synthesized
with duration noise disabled, epoch 18 produced 177.45 seconds for 200.26
seconds of source speech (source/model ratio 1.129), while epoch 24 produced
169.98 seconds (ratio 1.178). Per-utterance ratio coefficients of variation
were 6.41% and 6.73%, respectively. The original comparison at
`length_scale=1.0` was therefore too fast, particularly for epoch 24; earlier
experimental `length_scale=0.95` renders moved in the wrong direction because
Piper values below one accelerate phoneme duration. Any further epoch-18
listening gate must disable or sharply bound duration noise and test calibrated
`length_scale` values near 1.08--1.13. This calibration is diagnostic only and
does not restore promotion eligibility while timbre and pacing remain under
human review.

An offline articulation-repeatability diagnostic then separated pronunciation
realization from pacing. Three epoch-18 renders with both acoustic and duration
noise set to zero were byte-identical. The existing pinned local
`faster-whisper-base` transcribed the same stable corruptions in all three,
including splits or losses around `επεξεργάζεται`, `ελληνική` and `σταθερή`.
With duration noise still disabled but acoustic noise restored to 0.667, three
renders produced different and substantially broader word corruptions. Whisper
round trips are diagnostic evidence rather than a perceptual acceptance metric,
but this controlled result supports two concurrent failures: learned acoustic
articulation errors in the checkpoint and stochastic amplification during
sampling. Neither pacing calibration nor inference-noise tuning makes epoch 18
eligible for promotion.

The next bounded experiment is a resume from the validated local `last.ckpt` to
epoch 30 only. Epoch 30 must be exported and pass the same deterministic and
stochastic articulation set plus human listening before any later epoch budget
is authorized. Training must not run blindly to epoch 50 and the active JARVIS
TTS backend remains unchanged.

The bounded resume completed normally at epoch 29 (`max_epochs=30`). Its
845,857,618-byte checkpoint has SHA-256
`9fbaf507293c38c67145f351c8e7681b1e723712ac873e473a802f1224903f26`
and an empty weights-only unsafe-global set. The exported 63,516,051-byte ONNX
has SHA-256
`0ef687f3255bc4544230e3a3ad9bac8a14ad706467e2ec12cdbae6f9f464af4e`
and passed the structural Piper preflight, but it failed articulation.

The same offline Whisper diagnostic was run against every retained resumed
checkpoint. Deterministic WER on the fixed 15-word articulation sentence was
26.7% at epoch 25, 46.7% at epochs 26 and 27, 33.3% at epoch 28, and 53.3% at
epoch 29. Mean normal-sampling WER was respectively 53.3%, 48.9%, 82.2%, 53.3%
and 51.1%. Epoch 25 was the least-bad deterministic result, but still failed the
gate and remained sampling-sensitive.

A final same-text control ruled out the local ASR as the source of the finding.
For a real F1 recording and its verbatim transcript, the pinned local
`faster-whisper-base` produced 0/11 word errors. Epoch 25 synthesis of that same
text produced 2/11 deterministic errors and 1/11 errors in the sampled render,
including phoneme-level corruptions such as `Ελπίζουμε` to `Ειπίζουμε` and
`επόμενη` to `ριεπόμενη`. The 30-epoch run is rejected for production and must
not be resumed further. Any future F1 attempt requires a new, explicitly
approved training recipe—starting again from the pinned Greek warm-start with a
substantially lower learning rate and perceptual/articulation gates from the
first checkpoints—not additional epochs on this optimizer trajectory.

Human listening initially selected `1.15x` against the real F1 source, but a
later model listening check found that pace excessively fast and superseded it
with `1.00x`. Pacing remains an inference/output concern only: training
recordings must remain at their original timing and sample rate so that tempo
processing cannot contaminate articulation or speaker identity. A future
accepted F1 model must preserve the source voice and use its native `1.00x`
output timing unless a later listening gate explicitly changes it.

## Low-learning-rate recovery experiment

The next approved experiment starts a new optimizer trajectory from the exact
hash-pinned Greek `rapunzelina` warm-start. It must never consume a checkpoint
from the rejected 30-epoch run. Generator/discriminator learning rates are
reduced tenfold to `2e-5` and `1e-5`; all other bounded M3 parameters remain
unchanged. The experiment writes under `recovery-low-lr`, stops after five
epochs, and requires export plus the deterministic/stochastic articulation
control before any extension. The then-current `1.15x` listening preference
remained output-only and was not applied during this training experiment. It
has since been superseded by `1.00x`.

The recovery run completed all five epochs normally. Validation mel loss fell
monotonically from `0.5259` at epoch 0 to `0.4863` at epoch 4. The final
845,857,682-byte checkpoint has SHA-256
`21ee1595fa3005e19cf516e6d10d8b6daeff8a2721050eb37c3caae38b8ad841`.
Its offline export passed ONNX and structural Piper validation; the
63,516,051-byte model has SHA-256
`835a2e997b8e7cda669b6debe1e27fe201c175c82393d38c6b769a0445dc0e4d`.
The four fixed qualification fixtures also passed model-size, offline,
peak-RSS and warm-first-audio gates, with 310,558,720 bytes peak RSS and
122.538 ms maximum warm first audio. Cancellation and human approval were not
claimed by that non-playback run.

All five checkpoints were then exported and evaluated with the same local,
manifest-pinned `faster-whisper-base` articulation sentence. Phoneme-width
noise was disabled throughout. Each deterministic result used zero acoustic
noise; three normal samples per checkpoint used acoustic noise `0.667`.

| Recovery epoch | Deterministic WER | Mean normal WER | Normal range |
| --- | ---: | ---: | ---: |
| 0 | 60.0% | 68.9% | 40.0%--93.3% |
| 1 | 73.3% | 64.4% | 46.7%--93.3% |
| 2 | 60.0% | 64.4% | 53.3%--86.7% |
| 3 | 40.0% | 68.9% | 60.0%--73.3% |
| 4 | 46.7% | 68.9% | 60.0%--80.0% |

The three repeated deterministic epoch-4 renders were byte-identical, so the
failure is not a playback underrun or run-to-run timing issue. Even the
least-bad deterministic checkpoint, epoch 3, retained large phoneme-level
corruptions, and every normal-sampling mean was above 60%. The low-learning-rate
recovery experiment therefore fails the articulation gate and is rejected for
extension or promotion. The pitch-preserving `1.15x` listening derivative was
also rejected as excessively fast; the current listening target is the native
`1.00x`, although pacing cannot change the articulation result.

## Zero-training warm-start baseline

The original hash-pinned `rapunzelina` checkpoint was evaluated before any F1
fine-tuning to separate a base-model limitation from damage introduced by the
training trajectory. The existing 63,516,051-byte zero-training ONNX passed the
offline structural checker and has SHA-256
`bec65b49fdcf1629ef66c9e5244e64b95e4f6b83df9cd7c52ff665c1bc770f14`.
Three zero-noise renders of the fixed 15-word articulation sentence were
byte-identical but had 73.3% WER. Three normal-acoustic-noise renders averaged
62.2% WER. A realistic three-unit system preview at the model's default
acoustic and phoneme-width noise was also substantially corrupted.

A same-text source control then provided the decisive separation. The real F1
recording of `Greek-F1-B2-03-14.wav` produced 0/11 word errors. On its verbatim
transcript, the warm-start produced 1/11 errors with all inference noise
disabled and 5/11 with normal acoustic noise. Recovery epoch 3 produced 5/11
deterministically and 7/11 with normal acoustic noise. Phoneme-width noise was
disabled for all four synthesized controls.

The evidence identifies two independent problems. First, normal stochastic
sampling can severely amplify errors even before fine-tuning, so a future
candidate must qualify with zero or tightly bounded inference noise. Second,
the recovery trajectory damages an utterance that the deterministic warm-start
can render almost correctly, so additional epochs or another global low-rate
fine-tune are not justified. Any new Piper experiment must preserve the
linguistic/text and duration stack, use step-level checkpoints from the first
hundreds of updates, and gate every checkpoint against the source-controlled
articulation set. The existing recovery checkpoints remain rejected.

When `model.mos_metric=none`, the safe training wrapper also removes exactly
the pinned Piper `val_mos` checkpoint callback while retaining the `val_mel`
and `last.ckpt` callback. A resume is accepted only from a regular, local
`lightning_logs/version_N/checkpoints/last.ckpt` with bounded size and an empty
weights-only unsafe-global set; warm-start and resume arguments are mutually
exclusive. Resume also passes `model.warmstart_ckpt=null` explicitly so
LightningCLI cannot restore the original warm-start path from checkpoint
hyperparameters and overwrite the resumed model weights. Because Lightning may
still hydrate those private fields from checkpoint metadata, the scoped resume
hook clears both `_warmstart_ckpt` and `_vocoder_warmstart_ckpt` immediately
before Piper's original `on_fit_start`; the original method is restored after
the training invocation.
