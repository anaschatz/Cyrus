# Native Windows CUDA coexistence verification

Milestone 2 requires native Windows 11 x64 with an NVIDIA GPU and the representative local LLM workload. Source implementation and fake tests on another host are not hardware approval.

The pinned `large-v3-turbo` model contract and the initial RTX 5080 16-GB
settings are defined in `windows-whisper-large-v3-turbo.md`. The Base and Turbo
profiles share this coexistence gate; selecting Turbo does not authorize a
download, CUDA installation, or warm residency.

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

The authoritative clean qualification population is:

- 100 Greek utterances with matching private reference sidecars;
- 100 English utterances with matching private reference sidecars;
- at least 95 correct language IDs and at most 20% aggregate WER independently
  for each locale;
- exact cases `001..050` per locale for the 100-case coexistence subset, with
  at least 80 measured real LLM/ASR overlaps;
- 5 no-speech turns;
- 10 interruptions;
- exactly 5 bilingual CPU routes through production `FallbackAsrBackend`,
  selected by an explicit collector-local injected CUDA admission failure;
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
- ASR p95 regression no greater than 15% on the same coexistence population;
- at least 2,048 MiB minimum observed free VRAM during bounded 100 ms polling;
- zero measured NVIDIA thermal/power throttle reasons.

The injected five-case CPU route proves the pinned CPU `int8` model loads and
the production fallback wrapper routes correctly. It does not simulate or
claim a real VRAM-exhaustion event. Short, ambiguous and mixed-language clips
remain in a separate non-authoritative report and cannot count toward clean
language-ID or WER acceptance.

The resource check is not a reservation guarantee. Normal ASR precedes the new LLM generation, but barge-in can overlap with a previous LLM turn; that race must be measured explicitly.
