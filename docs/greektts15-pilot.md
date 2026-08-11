# GreekTTS-1.5 MLX pilot

This is an isolated Apple Silicon pronunciation experiment. It does not replace
the current `MacSystemTtsBackend`, change `jarvis.voice/1`, or alter the active
voice configuration.

## Pinned runtime

- Greek adapter: `moiralabs/GreekTTS-1.5` at
  `0da4d9fff8b974fb1127f3c829f793dc0439dfd1` (Apache-2.0)
- quantized base: `mlx-community/orpheus-3b-0.1-ft-4bit` at
  `b30aed4ecfa03c0177a5419b51919d14a105f183` (Apache-2.0)
- codec: `hubertsiuzdak/snac_24khz` at
  `d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec` (MIT)
- runtime: `mlx==0.32.0`, `mlx-lm==0.31.3`, `torch==2.6.0`, `snac==1.2.1`

The official GreekTTS instructions target CUDA and Unsloth. The Mac pilot uses
the 1.86 GB four-bit MLX conversion of the same Orpheus base, then converts the
389 MB PEFT LoRA adapter to MLX's native parameter layout. The conversion is a
small audited implementation of the mapping proposed in official MLX-LM PR
`#1120` at commit `a224d34`: strip the PEFT prefix, rename `lora_A/B`, transpose
both matrices and use `scale = lora_alpha / r = 1`.

The released MLX-LM package does not yet perform this conversion itself. The
pilot therefore requires all 392 expected tensors (28 layers × 7 targets × 2
matrices), rejects unknown/missing LoRA keys, and records a hash manifest for
the converted file. This remains a qualification path, not a production
dependency decision.

## Stages

Use a dedicated CPython 3.11 environment and private pilot root:

```sh
python scripts/run_greektts15_pilot.py preflight \
  --storage-root "/absolute/private/root" \
  --model-root "/absolute/private/root/models"
```

The read-only preflight performs no installation, download, capture or model
start. Acquisition pins exact revisions, allows only the ten runtime artifacts,
uses one download worker, and validates size plus SHA-256 for every file:

```sh
python scripts/run_greektts15_pilot.py acquire \
  --storage-root "/absolute/private/root" \
  --model-root "/absolute/private/root/models" \
  --report "/absolute/private/root/evidence/acquisition.json"

python scripts/run_greektts15_pilot.py convert-adapter \
  --model-root "/absolute/private/root/models" \
  --report "/absolute/private/root/evidence/conversion.json"
```

The offline run generates one official pronunciation fixture at mono 24 kHz:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_greektts15_pilot.py run \
  --model-root "/absolute/private/root/models" \
  --output-dir "/absolute/private/root/output" \
  --report "/absolute/private/root/evidence/pilot.json"
```

The run blocks Python network connections, validates the attached 196 LoRA
modules, uses the official sampling and SNAC redistribution settings, and writes
content-free latency/memory evidence. Pronunciation approval is required before
any production integration or streaming/cancellation work.

## Observed M3 pilot (2026-08-10)

- converted adapter: 392 tensors, 389,065,055 bytes, SHA-256
  `e2e5b40415fede35efd33c6a0946bf77104b95246d79b4f196841b7e81117a5d`
- MLX model/adapter load: 1.33 s; SNAC load: 0.08 s
- first speech token: 0.22 s after generation begins
- complete generation: 6.87 s; SNAC decode: 0.23 s
- output: 2.39 s mono 24 kHz PCM16 WAV, peak 0.539, no normalization needed
- warm audio-ready time: 7.10 s; real-time factor: 2.97
- peak MLX memory: 2.38 GB; peak resident memory: 3.07 GB

The first-speech-token figure is not time-to-audio: this pilot decodes only after
the complete utterance. It has reached the pronunciation-listening gate, but it
has not passed the conversational latency or streaming gates.

## Matched 8-bit A/B

The optional local-only comparison reuses the exact Greek adapter, SNAC codec,
prompt, seed and runtime. It changes only the Orpheus base to
`mlx-community/orpheus-3b-0.1-ft-8bit` at
`c8521bf4ea42f8dd96df099219d7871fc9873142` (3,507,408,624-byte weights).

```sh
python scripts/run_greektts15_pilot.py preflight \
  --storage-root "/absolute/private/root" \
  --model-root "/absolute/private/root/models" \
  --base-variant 8bit

python scripts/run_greektts15_pilot.py acquire \
  --storage-root "/absolute/private/root" \
  --model-root "/absolute/private/root/models" \
  --base-variant 8bit \
  --report "/absolute/private/root/evidence/acquisition-8bit.json"

python scripts/run_greektts15_pilot.py run \
  --model-root "/absolute/private/root/models" \
  --base-variant 8bit \
  --output-dir "/absolute/private/root/output" \
  --report "/absolute/private/root/evidence/pilot-8bit.json"
```

The 8-bit sample has its own filename and evidence. It never overwrites the
4-bit comparison or changes the production backend.

### Observed 8-bit A/B (2026-08-10)

- the pinned 8-bit runtime pack passed the offline size/SHA-256 manifest check;
  all ten shared runtime artifacts were present and no network was used by the
  preflight or inference runs
- the first 8-bit process produced 2.22 s of mono 24 kHz PCM16 audio in 44.49 s;
  a second process, after the MLX compilation cache was populated, took 21.63 s
- in fresh matched processes, 4-bit audio-ready time was 14.67 s (RTF 6.14)
  versus 21.63 s (RTF 9.75) for 8-bit
- first speech token was 0.22 s for 4-bit versus 0.96 s for 8-bit; because this
  pilot decodes only after the utterance, neither number is streaming audio
  latency
- peak MLX memory was 2.42 GB for 4-bit versus 4.07 GB for 8-bit
- the outputs contained 28 versus 26 SNAC frames respectively; a pronunciation
  decision must therefore be made by listening, not inferred from quantization
  bit depth alone

The 8-bit path is a local pronunciation comparison only. Its measured latency
and memory prevent automatic promotion to the active TTS backend even if the
listener prefers its voice.

### Experimental speaker prompts

The upstream Orpheus base recognizes named speaker prefixes, while the official
GreekTTS-1.5 recipe deliberately uses `chosen_voice = None` and describes the
Greek adapter as single-speaker. The pilot can therefore test a named base
speaker only as a listening experiment; it must not claim that the Greek model
officially supports multiple voices:

```sh
python scripts/run_greektts15_pilot.py run \
  --model-root "/absolute/private/root/models" \
  --base-variant 8bit \
  --speaker leo \
  --output-dir "/absolute/private/root/output" \
  --report "/absolute/private/root/evidence/pilot-8bit-leo.json"
```

Each speaker gets a distinct output filename and evidence record. Promotion is
forbidden unless listening confirms that both Greek pronunciation and speaker
identity remain acceptable.
