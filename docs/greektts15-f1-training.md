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

The upstream MLX-Tune Orpheus collator is not used directly. At its pinned
revision it labels `<custom_token_3>` as an audio-start token even though the
GreekTTS runtime uses it as start-of-human, omits the explicit speech boundary,
and offsets SNAC codes by codebook level instead of by each position in the
seven-token Orpheus frame. JARVIS owns the small compatibility collator and
tests its SNAC output by round-tripping it through the production decoder.

Audio tokenization is a separate resumable stage. It uses the pinned local
SNAC checkpoint offline on CPU, resamples the untouched 44.1 kHz F1 recordings
to 24 kHz with SoXR VHQ, and writes one content-bound cache entry per clip.
Every entry records the source-audio hash, codec hash, revision, resampler, and
exact Orpheus tokens. A retry reuses an entry only when that full signature
still matches.

An audit of all 2,602 recording edges found the first/last 50 ms to be silence
or near-silence (median about -64 dBFS; only one edge above -35 dBFS). The cache
therefore removes exactly 50 ms from each edge before resampling. This retains
roughly half of STOMA's apparent 100 ms speech margin and avoids an aggressive
VAD cut that could erase Greek unvoiced consonants.

The first real M3 probe encoded one F1 clip into 798 valid Orpheus tokens and a
second run reused the content-bound entry. Full caching must use the final
50 ms trim signature, so the earlier untrimmed probe is intentionally invalidated.

The complete cache contains all 1,301 clips and 541,758 audio tokens. A second
full pass encoded nothing and reused all 1,301 entries, proving resumability
against the final encoder signature.

Before any optimizer state or weight update is allowed, the M3 training probe
loads the frozen 4-bit base, attaches the existing rank-64 Greek adapter, proves
that exactly its 392 LoRA tensors are trainable, and computes one real
completion-only forward/backward pass. It records MLX and process peak memory
and deliberately creates no optimizer and saves no weights.

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
