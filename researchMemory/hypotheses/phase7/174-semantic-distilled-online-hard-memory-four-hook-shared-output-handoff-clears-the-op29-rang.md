# Semantic-distilled online-hard-memory four-hook shared-output handoff clears the op29 range stress.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-routed-shared-output-op29.md

Summary:

- Ran the four-hook `left_operand_mod` routed shared-output online-hard-memory plus additive-semantic-distillation recipe at `operand_max=29` with a wider product decoder (`n_embd=32`, `n_head=2`), `operand_spans` readout, and shallow result heads. The sparse source reached final/calc `1.0000` on the `900`-prompt grid; online memory filled/froze by step `50`, with cumulative forced-result evals capped at `367,200`. The trusted 600-step frozen-policy additive handoff reached `900/900 = 1.0000` final / step-600 normal, calculator-result accuracy `1.0000`, and low controls (`0.0133` injection-zero, `0.0022` forced-zero, `0.0156` forced-random at step 600). All four routed hooks reached calculator-result accuracy `1.0000`.

Questions:

- What did we learn about Semantic-distilled online-hard-memory four-hook shared-output handoff clears the op29 range stress?
- Has Semantic-distilled online-hard-memory four-hook shared-output handoff clears the op29 range stress been tested?
- Should we repeat Semantic-distilled online-hard-memory four-hook shared-output handoff clears the op29 range stress?
- What is the status of Semantic-distilled online-hard-memory four-hook shared-output handoff clears the op29 range stress?
- What follow-up is allowed for Semantic-distilled online-hard-memory four-hook shared-output handoff clears the op29 range stress?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-routed-shared-output-op29.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- More fixed-grid op19/op29 four-hook shared-output seed/range repeats as novelty. Also do not convert this into local semantic-distill weight/sample tuning.

Next Allowed:

- Streaming/fresh-prompt online memory, or a materially different many-calculator scaling gate where per-prompt memory cannot simply store the fixed grid.

Full Text:

```text
POSITIVE: Semantic-distilled online-hard-memory four-hook shared-output handoff clears the op29 range stress.
Conclusion: Ran the four-hook `left_operand_mod` routed shared-output online-hard-memory plus additive-semantic-distillation recipe at `operand_max=29` with a wider product decoder (`n_embd=32`, `n_head=2`), `operand_spans` readout, and shallow result heads. The sparse source reached final/calc `1.0000` on the `900`-prompt grid; online memory filled/froze by step `50`, with cumulative forced-result evals capped at `367,200`. The trusted 600-step frozen-policy additive handoff reached `900/900 = 1.0000` final / step-600 normal, calculator-result accuracy `1.0000`, and low controls (`0.0133` injection-zero, `0.0022` forced-zero, `0.0156` forced-random at step 600). All four routed hooks reached calculator-result accuracy `1.0000`.
Do not repeat: More fixed-grid op19/op29 four-hook shared-output seed/range repeats as novelty. Also do not convert this into local semantic-distill weight/sample tuning.
Next allowed test: Streaming/fresh-prompt online memory, or a materially different many-calculator scaling gate where per-prompt memory cannot simply store the fixed grid.
Source: `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-routed-shared-output-op29.md`
```
