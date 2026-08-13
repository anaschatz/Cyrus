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
on the longest training sequence and deliberately creates no optimizer and
saves no weights.

The initial plumbing probe used a 165-token sample and measured 3.659 GB peak
MLX memory. It is not the training memory gate. The authoritative gate must be
rerun on the longest planned sequence and stay below 6 GiB. It must still find
exactly 97,255,424 trainable parameters in 392 LoRA tensors with a matching
392-group gradient tree. Only then may an optimizer checkpoint be promoted.

The first longest-sequence probe without recomputation measured 13.480 GB MLX
for 1,331 tokens and is rejected. The next probe uses MLX-LM gradient
checkpointing on the identical sample; it is accepted only if the same strict
6 GiB gate passes.

The checkpointed longest-sequence gate passed at 4.366 GB MLX peak memory with
the same loss and gradient surface. The first quality experiment is therefore
two independent 25-update branches from the immutable Greek adapter: AdamW at
`5e-6` and `1e-5`, both with zero weight decay, batch size one, seed 10,001,
and gradient checkpointing. Each branch measures the same deterministic eight
validation clips before and after training and saves to its own directory.

Both initial branches reduced validation loss (`5e-6`: 5.1987 to 4.6492;
`1e-5`: 5.1987 to 4.4983), but both failed to emit end-of-speech on the second
deterministic quality fixture within 1,200 tokens. They are rejected despite
the lower loss. The rescue experiment keeps the same 25 samples and gates but
uses `5e-7` and `1e-6`, reflecting that this is continuation of a mature
rank-64 language/voice adapter rather than training a fresh small LoRA.

Both rescue branches passed all four deterministic end-of-speech fixtures.
At `1e-6`, validation loss moved from 5.1987 to 4.9780 and render RTF was 7.14.
At `5e-7`, validation loss moved to 5.0724 and render RTF was 5.17. The baseline
render RTF was 3.19. These timings are standalone non-streaming diagnostics,
not production latency approval. The `5e-7` branch is the conservative
listening candidate; `1e-6` remains the stronger adaptation candidate. Human
comparison of identical fixtures is required before either branch can continue
to 50 updates.

The listening set is also transcribed offline by the already-local pinned
`Systran/faster-whisper-base` CPU model. Identical forced-Greek beam-search
settings produce a per-fixture and micro word-error comparison. Whisper is an
independent intelligibility signal only: it cannot approve timbre, pacing,
speaker identity, or robotic artifacts, so human listening remains mandatory.

Because a single fixed seed at temperature 0.6 can amplify small checkpoint
differences into a different audio-token path, candidate decoding is qualified
separately at bounded temperatures 0.0, 0.2, 0.4, and 0.6. Any comparison must
record the exact temperature and use identical settings for the baseline.

Decoding qualification selected temperature 0.4 for the `1e-6`, 25-step
candidate. It completed all four fixtures at mean RTF 4.23 and scored 36.36%
micro WER with no English collapse. The same candidate scored 45.45% at
temperature 0.2 and 39.39% at 0.6; 0.4 is therefore the best measured balance.
The baseline failed end-of-speech at both 0.2 and 0.4, while its completed 0.6
render scored 63.64% WER. The candidate is the objective listening winner but
cannot proceed to 50 steps until human evaluation confirms timbre, pacing,
native pronunciation, and absence of robotic artifacts.

Human evaluation then found the smoother candidate free of the earlier
robotic failure but less natively Greek, described as sounding foreign. That
checkpoint is therefore not extended. Pronunciation qualification now uses
four phonetically rich, untouched Harvard sentences from the F1 control split.
Each synthesized sentence is paired with the exact same text spoken by the
native F1 recording, whose SHA-256 is pinned. This separates accent and rhythm
judgment from speaker preference and from Whisper's imperfect Greek WER. No
checkpoint can continue until it wins this same-text native-reference gate.

The first native-reference run rejected the `1e-6`, 25-step checkpoint: at
temperature 0.4 it failed to emit end-of-speech on the pronunciation suite.
The immutable GreekTTS-1.5 baseline completed all four reference texts at its
qualified temperature 0.6. Further adaptation therefore restarts from the
immutable adapter; the rejected checkpoint is retained only as diagnostic
evidence and is never used as a continuation source.

Before another optimizer run, bounded parameter interpolation tests whether a
small part of the smoother checkpoint can remove the robotic artifact without
carrying its accent drift. The bounded 25% and 50% endpoints, plus a single
37.5% binary-search midpoint, are permitted and always measured from the
immutable adapter. The candidate manifest must prove
that it descended directly from that adapter, all 392 tensors must match in
name, shape, and dtype, and every blend records both source hashes. A blend is
a listening experiment, never a training continuation source or production
promotion by itself.

The same native-reference suite at temperature 0.6 accepted the 25% blend: it
completed all four texts and reduced mean standalone render RTF from about
4.28 for the baseline to about 3.32. The 50% blend failed to emit
end-of-speech and is rejected. This bounds the useful interpolation region
close to the immutable adapter. Only the 25% blend advances to paired human
listening; neither blend may seed subsequent training.

Offline Whisper calibration on the same four texts scored the native F1 at
36.36% WER, the baseline at 69.70%, and the 25% blend at 60.61%. The ASR is too
weak to judge accent, but the nine-point relative improvement is enough to
retain the blend. A final 37.5% midpoint is permitted solely to binary-search
the interval between the accepted 25% and rejected 50% endpoints.

The 37.5% midpoint completed all four native-reference fixtures and scored
54.55% WER, improving on both the 25% blend (60.61%) and baseline (69.70%). It
is the objective listening candidate. No higher fraction is attempted before
human comparison because the 50% endpoint already loses termination and the
full checkpoint was judged to have foreign accent drift.

The discontinuity diagnostic reverses the ASR-only preference. Against the
same four renders, the 25% blend has lower mean pitch-jump rate, spectral-flux
tail, and normalized sample-step tail than the 37.5% blend. The midpoint also
doubles mean internal silence (4.76% versus 2.37%); on the difficult clusters
fixture it reaches 17.85% versus 7.91%. Because the reported user defect is
robotic instability rather than recognition alone, 25% is the conservative
listening candidate. The 37.5% result remains an intelligibility boundary, not
the preferred checkpoint. Acoustic metrics are diagnostic and cannot replace
human listening approval.

The stable 25% interpolation motivates one new optimizer experiment rather
than promotion of a blended checkpoint. A clean branch restarts from the
immutable adapter at `2.5e-7` for the same bounded 25 updates. This tests
whether a genuinely smaller training trajectory can preserve native accent
and continuity without parameter interpolation. It must pass the full EOS,
native-reference, offline-ASR, and acoustic-discontinuity gates; it cannot
continue beyond 25 updates from loss alone.

The clean `2.5e-7` branch completed 25 updates in 172 seconds and reduced the
fixed validation loss from 5.1987 to 5.1307 while keeping peak MLX memory near
5.01 GB. It emitted EOS on all four native-reference fixtures, but scored only
66.67% WER versus 60.61% for the 25% blend. Its acoustic means also had higher
pitch-jump rate, spectral-flux tail, and normalized sample-step tail than the
25% blend; only internal silence was slightly lower. The branch is rejected
as a listening winner. This proves that merely shrinking the learning rate
does not reproduce the useful interpolated parameter region, so no still
smaller LR branch is justified from these results.

Delta analysis of both the `1e-6` and `2.5e-7` trajectories shows the same
concentration: layers 26 and 27 absorb about 11.1–11.5% of update energy even
though two of 28 uniform layers would account for 7.1%, and the six largest
tensors are all in those two output-near layers. Module and LoRA A/B shares
remain stable across rates, so there is no evidence for arbitrarily removing
all attention or MLP updates. The first targeted branch therefore restarts
from the immutable adapter at `1e-6` for 25 updates while freezing every LoRA
tensor in layers 26–27. Exactly 364 groups may train, but all 392 groups must
remain present in the saved standalone adapter. The targeted profile receives
no relaxed quality gate.

Candidate rendering accepts a separate adapter directory only when it contains
the exact native rank-64 configuration and a regular `adapters.safetensors`
file. Reports record the checkpoint hash, so baseline and both pilot branches
can be rendered with identical fixtures, seed, and sampling parameters without
ever replacing the qualified baseline adapter.

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
