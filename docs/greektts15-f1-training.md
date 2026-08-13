# GreekTTS-1.5 F1 continued-training plan

The selected path is continued LoRA training of the pinned GreekTTS-1.5
rank-64 adapter on the native-Greek STOMA F1 speaker. The 4-bit Orpheus base
remains frozen and the trained adapter remains separate; fusing a small LoRA
delta into a quantized base can erase useful updates through rounding.

The first stage is intentionally dependency-free. It verifies all 1,301 WAV
hashes and formats, repairs only known sentence-initial Latin/Greek
confusables in a derived manifest, and produces deterministic train,
validation, and untouched control splits independently for B2, C1, C2, and
Harvard. The source metadata and recordings are never modified.

Run the preflight with the repository on `PYTHONPATH`:

```bash
PYTHONPATH=src python scripts/run_greektts15_training.py \
  "$PWD/artifacts/compact-greek-tts-f1/f1-training-data" \
  "$PWD/artifacts/greektts15-training/f1-plan.json"
```

Training will begin with two short branches from the same pinned adapter at
learning rates `5e-6` and `1e-5`. Checkpoints at 25, 50, and 100 optimizer
steps must pass validation loss, fixed Greek pronunciation fixtures, memory,
and human listening gates. A longer run is allowed only after one branch
clearly improves the untouched baseline. No result is called production-ready
from training loss or ASR alone.

STOMA audio is CC-BY-4.0. GreekTTS-1.5 publishes its adapter under Apache-2.0,
while the Orpheus base inherits its own upstream model terms. Packaging or
redistribution therefore remains a separate license review gate.
