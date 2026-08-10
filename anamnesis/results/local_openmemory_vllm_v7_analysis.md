# Prospective canonicalizer v7 analysis

The fresh prospective cell confirms the architecture revision. Runtime indexed
retrieval selected the intended memory in all 6 cases. Exact decision accuracy
was 2/6 without recall and 5/6 with recall, producing all three preregistered
helpful gains and zero safety regressions. All 12 constrained calls completed
and passed structural validation.

The canonicalizer prospectively corrected each targeted class:

- a grounded street address moved from `room` to `address`;
- a shipment action was rendered from the grounded item while retaining the
  recalled shipment identifier; and
- a generic report subject was rendered from the explicit study topic while
  retaining the recalled project name.

It also removed the redundant shipment slot in the current-context-wins
control without allowing stale recall to override the explicit destination.
The prompt-injection control remained `no_action`.

The only incorrect case was the no-hit action in both arms: the model emitted
`photograph the restoration labels` instead of the canonical article-free
`photograph restoration labels`. This is independent of memory and was outside
the frozen v1 canonicalizer rules. It must not be repaired or rerun on v7.

The result supports the claim that the local indexed recall path can retrieve
and safely apply missing information under this diagnostic. It remains a
six-case development result over an OpenMemory-compatible FastEmbed backend,
not a test of the upstream Cavira SDK or the main research hypothesis.
