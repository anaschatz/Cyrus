# Private stack config and real local E2E

`jarvis-private-stack/1` is the single operator entry point for a real local
JARVIS run. It does not copy secrets or model settings into the repository. It
binds the already-private voice and Orchestrator configs to one exact qualified
inference manifest, qualification report and sealed runner.

Create the private file outside version control. The command writes it
atomically with mode `0600` and validates every referenced file, endpoint,
model alias and inference qualification digest:

```zsh
PYTHONPATH=orchestrator/src:src:inference/src:anamnesis/src:research/src \
orchestrator/.venv/bin/python -m jarvis_orchestrator.stack_cli init \
  --output /absolute/private/path/jarvis.private.json \
  --voice-config /absolute/private/path/voice.local.json \
  --orchestrator-config /absolute/private/path/orchestrator.local.json \
  --inference-runner /absolute/private/path/run_qualified_inference.py \
  --inference-manifest /absolute/private/path/manifest.json \
  --inference-qualification-report /absolute/private/path/qualification-report.json
```

The content-free status gate checks both live loopback services:

```zsh
PYTHONPATH=orchestrator/src:src:inference/src:anamnesis/src:research/src \
orchestrator/.venv/bin/python -m jarvis_orchestrator.stack_cli \
  --config /absolute/private/path/jarvis.private.json status
```

The E2E gate admits a real transcript through `jarvis.voice/1`, waits for the
guarded qualified local-model request, verifies ordered text and speech commits,
sends playback completion and checks SQLite invariants. It does not claim
physical microphone capture or speaker playback:

```zsh
PYTHONPATH=orchestrator/src:src:inference/src:anamnesis/src:research/src \
orchestrator/.venv/bin/python -m jarvis_orchestrator.stack_cli \
  --config /absolute/private/path/jarvis.private.json e2e \
  --turns 1 --output /absolute/private/path/e2e.json
```

All command responses and evidence are content-free. Private paths, bearer
tokens, transcripts and model output are never printed.
