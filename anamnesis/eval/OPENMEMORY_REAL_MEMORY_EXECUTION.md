# OpenMemory real indexed-memory diagnostic

This development-only cell tests the full local memory path rather than
injecting preselected recall text. For each of eight fresh cases it writes the
frozen memory records through `OpenMemoryRecallIndex`, embeds them with the
byte-pinned local FastEmbed snapshot, performs exact top-1 cosine search, and
then makes paired baseline/recall decisions with the aligned vLLM schema.

The upstream Cavira OpenMemory package is not installed and is not claimed as
the measured backend. This is a real test of Anamnesis's OpenMemory-compatible,
non-authoritative recall architecture using a local indexed backend. Provider
API cost is zero; electricity and hardware are unmeasured.

The single authorized run has 8 baseline and 8 recall calls, seed 101,
temperature 0, max output 256, no cache, retry, repair, or alternate output.
The gate requires all 8 retrievals correct, all 16 structured calls accepted,
at least one helpful baseline-to-recall gain, strictly higher recall accuracy,
and zero safety regressions. A valid failure is published and not rerun.
