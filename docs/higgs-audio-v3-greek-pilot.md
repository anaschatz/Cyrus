# Higgs Audio v3 4-bit MLX Greek pilot

Date: 2026-08-13  
Platform: MacBook M3, 16 GB unified memory  
Outcome: rejected before inference because the downloaded 4-bit artifact is not a complete model

## Candidate

- Main artifact: `Reza2kn/Higgs-Audio-v3-TTS-4bit-MLX`
- Pinned revision: `013dc93470aa01922fe45960e8e8b7ea8935a515`
- Upstream architecture: Higgs Audio v3 TTS 4B (`higgs_multimodal_qwen3`)
- Codec: `bosonai/higgs-audio-v2-tokenizer`
- Codec revision: `403fbacf2f60caaa102f893fdfabb694619b2417`
- Runtime probe: `Blaizzy/mlx-audio` revision `76f8363d64a9e201cb3c11c4a3fa04743dd0922e`
- Qualification dependency: `mlx-lm==0.31.3`, installed only in the isolated qualification virtual environment

This candidate was selected because Higgs Audio v3 declares multilingual support including Greek, supports zero-shot voice cloning, and the apparent download size was compatible with the M3 disk and memory envelope. The model and codec use Boson research/non-commercial licenses, so a passing technical pilot would still require a separate production-license decision.

## Downloaded artifacts and integrity

| File | Bytes | SHA-256 |
|---|---:|---|
| `tokenizer.json` | 11,433,924 | `eb883de2de5adc5113f1f02b54830a0ea7cd6ef191cde65c41aceb3737d4d1c1` |
| `quantized.safetensors` | 2,043,828,720 | `ba9742db63a2c2dd18d26f765baf47b4719aa300f5b4fb61ee74afa2540d8a74` |
| `audio_tokenizer/model.safetensors` | 805,665,628 | `fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d` |

The metadata and blobs were fetched from the pinned revisions. The three large runtime files total 2,860,928,272 bytes. The qualification runtime imported successfully on the physical M3 Metal device, and no hosted inference or API key was used.

## Structural gate

The gate inspected the safetensors header without materializing the model. It found exactly 756 arrays, all below `body.layers.*`:

- 252 `weight.qweight` arrays
- 252 `weight.scales` arrays
- 252 `weight.biases` arrays

There are no text embeddings, modality embeddings, layer norms, final norm, output head, or other preserved non-quantized tensors in `quantized.safetensors`. The included `model.safetensors.index.json` describes the original approximately 8.49 GB checkpoint and maps those missing tensors to `model.safetensors`, but that file is not included in the 4-bit repository. The model card also calls the artifact “not a complete drop-in runtime yet” and says custom runtime integration is required.

The packed quantization format is also not the native `mlx.nn.QuantizedLinear` serialization expected by the current `mlx-audio` loader. A custom repacker could adapt the 252 quantized linear layers, but it cannot reconstruct the absent tensors. Completing the artifact would therefore require downloading the original multi-gigabyte checkpoint and extracting the preserved tensors.

## Decision

Reject this artifact without synthesis. Producing a WAV from an incomplete checkpoint would not be a valid Greek-pronunciation test. Downloading the original approximately 8.49 GB checkpoint would exceed the current free-disk envelope and would not satisfy the requirement to coexist with the main local LLM on a 16 GB unified-memory machine.

No voice sample from this candidate should be presented to the user, and it must not be integrated into the voice layer. The next candidate must be a complete, runnable quantized package whose model, codec, and runtime all fit the M3 coexistence budget.
