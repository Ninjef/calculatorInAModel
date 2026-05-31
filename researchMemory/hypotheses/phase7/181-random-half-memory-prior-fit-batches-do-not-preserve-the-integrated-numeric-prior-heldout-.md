# Random half-memory prior fit batches do not preserve the integrated numeric-prior heldout source gate.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-31-random-half-memory-prior-fit-gate.md

Summary:

- Tested the coreset-style cost lever by setting `--result-boundary-target-amortized-prior-fit-batch-size 160` with the every-2, stop-accuracy-1.0, patience-100 source recipe. This halves examples per prior fit, but the prior never converged: final prior train/heldout accuracy was only `0.909375` / `0.7750`, stop never activated, and updates remained `2501`. Source train stayed high (`0.996875`), but heldout exact/calc fell to `65/80 = 0.8125` and overall to `387/400 = 0.9675`, with heldout controls still low (`0.0500` injection-zero, `0.0000` forced-zero, `0.0125` forced-random). No trusted handoff was run because the source gate missed.

Questions:

- What did we learn about Random half-memory prior fit batches do not preserve the integrated numeric-prior heldout source gate?
- Has Random half-memory prior fit batches do not preserve the integrated numeric-prior heldout source gate been tested?
- Should we repeat Random half-memory prior fit batches do not preserve the integrated numeric-prior heldout source gate?
- What is the status of Random half-memory prior fit batches do not preserve the integrated numeric-prior heldout source gate?
- Why did Random half-memory prior fit batches do not preserve the integrated numeric-prior heldout source gate fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-31-random-half-memory-prior-fit-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run random prior fit-batch-size ladders as novelty. Batch `64` already left heldout at `0.7125`, and random batch `160` still underfits at `0.8125`.

Next Allowed:

- Use a structured/coverage-aware coreset, reservoir with balanced operand coverage, or validation-aware stopping signal rather than uniform random prior-fit minibatches.

Full Text:

```text
MIXED-NEGATIVE: Random half-memory prior fit batches do not preserve the integrated numeric-prior heldout source gate.
Conclusion: Tested the coreset-style cost lever by setting `--result-boundary-target-amortized-prior-fit-batch-size 160` with the every-2, stop-accuracy-1.0, patience-100 source recipe. This halves examples per prior fit, but the prior never converged: final prior train/heldout accuracy was only `0.909375` / `0.7750`, stop never activated, and updates remained `2501`. Source train stayed high (`0.996875`), but heldout exact/calc fell to `65/80 = 0.8125` and overall to `387/400 = 0.9675`, with heldout controls still low (`0.0500` injection-zero, `0.0000` forced-zero, `0.0125` forced-random). No trusted handoff was run because the source gate missed.
Do not repeat: Do not run random prior fit-batch-size ladders as novelty. Batch `64` already left heldout at `0.7125`, and random batch `160` still underfits at `0.8125`.
Next allowed test: Use a structured/coverage-aware coreset, reservoir with balanced operand coverage, or validation-aware stopping signal rather than uniform random prior-fit minibatches.
Source: `aiAgentWorkHistory/phase7/2026-05-31-random-half-memory-prior-fit-gate.md`
```
