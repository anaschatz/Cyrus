# Local OpenMemory decision diagnostic — transport failure

The frozen v1 paired diagnostic did not produce a comparable result. It is a
transport/preflight failure, not evidence for or against OpenMemory recall.

## What ran

- Source commit: `d1732af873193ca9f59d3fcd339abc4a590e9adf`
- Model: `ollama/qwen3.5:9b-q4_K_M`, already installed and byte-pinned
- Runtime: Ollama `0.31.1`, loopback `127.0.0.1:11434`, cloud disabled,
  context length 4096, one parallel request, one loaded model, Metal on Apple M3
- Frozen config: seed 101, temperature 0, cache off, retries 0, repair calls 0
- Intended matrix: 8 baseline + 8 recall decision calls
- Completed matrix: 0/8 pairs; no diagnostic metric is defined

The first baseline call consumed 707 input and 3,389 output tokens, filled the
entire 4,096-token context, and ended with `stop_reason=max_tokens`. Ollama
reported the response as truncated and Inspect retained an empty completion, so
there was no parseable decision. The call took approximately 6 minutes 17
seconds. The second, recall-side call for the same case began but was cancelled
immediately after the first failure was confirmed; it produced no retained
completion or usage record. The remaining fourteen calls did not run.

Inspect therefore records status `cancelled`, one incomplete sample, two model
events, aggregate measured usage of 707 input and 3,389 output tokens, and
provider API cost `$0.00`. Electricity, hardware, and operator time remain
unmeasured.

## Interpretation and stopping decision

The installed 9B transport entered an unbounded reasoning path instead of
producing the constrained decision JSON within the pinned context. Continuing
would have repeated a roughly six-minute, context-filling failure for the
remaining calls. The run was stopped to preserve the exact failure and avoid
another fifteen uninformative generations.

Because the second call received the first case's recall prompt before
cancellation, v1 is considered opened. It will not be rerun or tuned. Any
no-thinking transport experiment must use a newly frozen v2 diagnostic artifact
and a separately committed protocol. No OpenMemory usefulness, safety,
accuracy, latency, or token-efficiency claim is made from this run.

The exact raw `.eval` log is retained locally and hash-pinned in provenance,
but is not published because it contains full prompts and local absolute
filesystem metadata. A sanitized machine-readable event/usage summary is
committed instead. The local log contains no credentials or remote API calls.
