# Windows Whisper large-v3-turbo profile

This profile upgrades the independent Windows Voice client from the pinned
`faster-whisper-base` pack to multilingual Whisper large-v3-turbo. It remains
local-only: the runtime accepts an explicit directory, validates every runtime
artifact, and passes `local_files_only=True`. It never resolves a model alias or
downloads a tokenizer at runtime.

## Frozen model contract

- CTranslate2 repository: `dropbox-dash/faster-whisper-large-v3-turbo`
- Revision: `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`
- Source model: `openai/whisper-large-v3-turbo`
- License: MIT
- Runtime files: `config.json`, `model.bin`, `preprocessor_config.json`,
  `tokenizer.json`, and `vocabulary.json`
- Exact total size: `1621665983` bytes
- Stored weights: FP16; selected Windows runtime compute type: `int8_float16`

The exact sizes and SHA-256 values are frozen in
`jarvis_voice.manifest.LARGE_V3_TURBO_MODEL_MANIFEST`. The model directory must
contain exactly those five files. Documentation, cache metadata, partial
downloads, and any extra runtime file are rejected.

## RTX 5080 16-GB qualification profile

Use this overlay only after the exact pack is already present on the Windows
machine:

```json
{
  "whisper_model_directory": "C:\\ProgramData\\JARVIS\\models\\faster-whisper-large-v3-turbo-0a363e9",
  "asr": {
    "model_profile": "large-v3-turbo",
    "execution_profile": "cuda",
    "cpu_compute_type": "int8",
    "cuda": {
      "device_index": 0,
      "compute_type": "int8_float16",
      "residency": "turn_scoped",
      "minimum_free_vram_mb": 2048,
      "reserve_for_other_workloads_mb": 2048,
      "fallback_to_cpu": true,
      "worker_start_timeout_seconds": 30.0,
      "transcription_timeout_seconds": 120.0,
      "warm_restart_limit": 1,
      "vram_poll_interval_ms": 100
    }
  }
}
```

`turn_scoped` is the first native qualification mode. A warm resident worker is
allowed only after the representative main LLM remains within the coexistence
gates in `cuda-coexistence.md`. The admission policy requires at least 4 GiB
free at the sample point: 2 GiB available to Voice plus 2 GiB retained for
display, KV-cache growth, and workload spikes. This is a point-in-time safety
check, not a VRAM reservation.

CPU fallback uses the same validated pack with `int8`; it does not load a
second Whisper model. Isolated PTT turns use beam size 5, disable previous-text
conditioning, and apply the same low-confidence Greek/English log-probability
tie-break on both CPU and CUDA.

## Activation gate

Do not enable this profile based on source tests from macOS. On the actual
Windows 11 host:

1. Validate the five downloaded/copied blobs against the built-in manifest.
2. Run passive Doctor, then the explicit active CUDA probe.
3. Compare Base and Turbo on the same Greek/English audio set, including
   `ουρανός` versus `βουνά`, names, numbers, paths, and short corrections.
4. Run the representative main LLM concurrently and exercise barge-in while a
   prior generation still occupies VRAM.
5. Require zero OOM, duplicate transcript, lost turn, stale output, and stuck
   worker; retain CPU fallback evidence.

No model, CUDA runtime, driver, or package is installed by this integration
commit. The approved CUDA/cuDNN DLL closure is supplied privately, recorded in
`cyrus.windows-cuda-runtime-bundle/1`, and embedded into the independently
sealed ASR qualification collector without `PATH` or download fallback.
Packaging and machine activation remain separate native-Windows steps.
The configured pack must remain on a local drive-letter path; UNC, device-path
and Win32 `DRIVE_REMOTE` roots fail before any model artifact is read.
