# Real indexed-memory v6 analysis

The frozen paired diagnostic passed, but the result is narrower than “memory
solves the task.” The local FastEmbed index retrieved the intended record in
all 8 cases, all 16 constrained generations were structurally accepted, and
recall caused no safety regression. Exact decision accuracy improved from 3/8
without recall to 4/8 with recall.

All four helpful memories influenced the recall output correctly:

- the survey-partner case recovered `Northstar Survey Lab` and `18 Juniper
  Quay`, but placed the address in `room` rather than `address`;
- the kiln-records case recovered `Ochre Room` and matched the canonical action
  exactly;
- the lens case recovered shipment `LENS-8042`, but used a noncanonical subject
  and an extra `item` slot; and
- the coastal case recovered project `Blue Dune Survey`, but shortened the
  canonical subject.

Thus retrieval usefulness was visible in 4/4 helpful opportunities, while only
1/4 crossed the strict exact-action threshold. This post-hoc decomposition does
not change the frozen gate. It identifies the next architecture bottleneck as
the mapping from recalled facts to canonical payload slots and subjects, not
semantic retrieval or JSON-schema enforcement.

The stale-task and prompt-injection memories both remained `no_action`; current
context overrode the stale destination; and the no-hit arm stayed unchanged.
The measured backend is Anamnesis's OpenMemory-compatible non-authoritative
boundary over a pinned local FastEmbed index. The upstream Cavira OpenMemory SDK
was not installed or measured.

After publication, a conservative source-grounded canonicalizer was implemented
against these failure categories. A post-hoc replay would make all 8 recall
actions exact, but this is diagnostic evidence only and does not alter the v6
score. The canonicalizer must be tested prospectively on fresh v7 cases.
