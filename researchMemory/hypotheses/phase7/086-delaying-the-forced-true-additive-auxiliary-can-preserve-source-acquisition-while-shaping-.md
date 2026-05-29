# Delaying the forced-true additive auxiliary can preserve source acquisition while shaping additive geometry.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-schedule-gate.md

Summary:

- On the same reduced `operand_max=9`, 100-step seed-13 gate, turning on `--additive-forced-true-loss-weight 0.5` only after step `50` beat both baseline and always-on aux on source acquisition (`0.3900` source calc and `0.4000` final eval, vs baseline `0.3500`/`0.3800` and always-on `0.2800`/`0.2800`) while keeping a large additive geometry gain (`forced_best_true=0.5100`, `top3=0.5600`, 50-step slope final loss `0.7979` vs baseline `0.0000`/`0.0000`/`1.5305`).

Questions:

- What did we learn about Delaying the forced-true additive auxiliary can preserve source acquisition while shaping additive geometry?
- Has Delaying the forced-true additive auxiliary can preserve source acquisition while shaping additive geometry been tested?
- Should we repeat Delaying the forced-true additive auxiliary can preserve source acquisition while shaping additive geometry?
- What is the status of Delaying the forced-true additive auxiliary can preserve source acquisition while shaping additive geometry?
- What follow-up is allowed for Delaying the forced-true additive auxiliary can preserve source acquisition while shaping additive geometry?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-schedule-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same small `operand_max=9`, seed-13, 100-step, start-step-50 schedule gate as novelty.

Next Allowed:

- Scale to `operand_max=19` with source-only checkpointing first, then verify promising scheduled-aux checkpoints with targeted standalone 600-step additive handoff.

Full Text:

```text
POSITIVE: Delaying the forced-true additive auxiliary can preserve source acquisition while shaping additive geometry.
Conclusion: On the same reduced `operand_max=9`, 100-step seed-13 gate, turning on `--additive-forced-true-loss-weight 0.5` only after step `50` beat both baseline and always-on aux on source acquisition (`0.3900` source calc and `0.4000` final eval, vs baseline `0.3500`/`0.3800` and always-on `0.2800`/`0.2800`) while keeping a large additive geometry gain (`forced_best_true=0.5100`, `top3=0.5600`, 50-step slope final loss `0.7979` vs baseline `0.0000`/`0.0000`/`1.5305`).
Do not repeat: The same small `operand_max=9`, seed-13, 100-step, start-step-50 schedule gate as novelty.
Next allowed test: Scale to `operand_max=19` with source-only checkpointing first, then verify promising scheduled-aux checkpoints with targeted standalone 600-step additive handoff.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-schedule-gate.md`
```
