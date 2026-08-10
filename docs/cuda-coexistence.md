# Native Windows CUDA coexistence verification

Milestone 2 requires native Windows 11 x64 with an NVIDIA GPU and the representative local LLM workload. Source implementation and fake tests on another host are not hardware approval.

## Installation gate

Before installing anything, record and present for approval:

- target GPU and driver version;
- installed or proposed `faster-whisper==1.2.1`;
- exact CTranslate2 version resolved for that faster-whisper build;
- CTranslate2's required CUDA and cuDNN major versions;
- exact Python/NVIDIA packages, sizes, licenses, and rollback steps;
- whether compatible CUDA/cuDNN DLLs are already present.

Do not download a model, driver, CUDA toolkit, cuDNN, or package; do not modify machine-wide `PATH`; do not proceed until the user explicitly approves the exact matrix.

## Automated checks after approval

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
py -3.11 -m unittest discover -s tests -v
py -3.11 -m compileall src tests
py -3.11 -m jarvis_voice doctor --config C:\path\to\voice.local.json
py -3.11 -m jarvis_voice doctor --config C:\path\to\voice.local.json --active-cuda-probe
```

Confirm exact local Whisper manifest validation, explicit GPU index, admission headroom, CPU fallback readiness, spawn support, driver/runtime compatibility, `network_used: false`, and no unexpected model files or tokenizer fallback.

The passive doctor must allocate no model VRAM. The active probe must not print its transcript and must terminate the worker in all success/failure paths.

## Profiles and sample counts

Run separately:

1. CPU baseline.
2. CUDA `turn_scoped`.
3. CUDA `warm`, only with measured headroom.
4. Representative local LLM alone.
5. The same LLM workload with CUDA ASR.
6. Barge-in while a previous LLM turn still occupies GPU resources.

For each approved profile collect at least:

- 20 Greek utterances;
- 20 English utterances;
- 5 no-speech turns;
- 10 interruptions;
- 5 forced CPU fallbacks;
- 5 worker crash/restart simulations.

Record failures and p50/p95 for PTT_UP → transcript, CUDA inference, worker spawn/load/shutdown, and CPU fallback. Record VRAM before worker, after model load, minimum free VRAM observed during polling, and after worker exit. The polling interval must accompany the data; the observation is not an absolute hardware peak.

## Coexistence acceptance

Approval requires:

- zero voice or LLM OOM/crashes;
- zero duplicate final transcripts;
- zero lost accepted turns;
- all late cancelled results rejected;
- turn-scoped worker exit before transcript submission;
- observed VRAM return after turn-scoped exit;
- LLM p95 latency/throughput regression no greater than 15%, unless another threshold is explicitly accepted.

The resource check is not a reservation guarantee. Normal ASR precedes the new LLM generation, but barge-in can overlap with a previous LLM turn; that race must be measured explicitly.
