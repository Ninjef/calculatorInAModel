# A quality-gated update cap preserves op29 source/handoff while cutting proportional prior-fit cost.

Kind: hypothesis_memory
Status: POSITIVE-WITH-CAVEAT
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-02-op29-quality-gated-prior-cap.md

Summary:

- Added `--result-boundary-target-amortized-prior-quality-gate-update-cap`, which freezes amortized-prior fitting after prompt memory is full once the configured stop metric and train-accuracy requirement are met at or beyond a specified update count. With the proportional op29 h128 recipe, cap `2000` froze the prior at `2000` updates / `1,254,817` fit examples / `1,080,000` full-fit examples. The source reached overall exact/calc `0.9956`, train `1.0000`, heldout `0.9611`, prior train/validation `0.9861`/`0.9927`, and low heldout controls (`0.0222` injection-zero, `0.0000` forced-zero, `0.0056` forced-random). Trusted 600-step additive handoff reached `900/900 = 1.0000`, diagnostic exact/calc `1.0000`/`0.9922`, and low final snapshot controls (`0.0000` injection-zero, `0.0000` forced-zero, `0.0078` forced-random).

Questions:

- What did we learn about A quality-gated update cap preserves op29 source/handoff while cutting proportional prior-fit cost?
- Has A quality-gated update cap preserves op29 source/handoff while cutting proportional prior-fit cost been tested?
- Should we repeat A quality-gated update cap preserves op29 source/handoff while cutting proportional prior-fit cost?
- What is the status of A quality-gated update cap preserves op29 source/handoff while cutting proportional prior-fit cost?
- What follow-up is allowed for A quality-gated update cap preserves op29 source/handoff while cutting proportional prior-fit cost?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-02-op29-quality-gated-prior-cap.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run cap-value, refresh-window, or proportional-fraction ladders as novelty. The single cap improves cost materially but is still one op29 seed and still depends on answer-derived sparse scoring plus staged handoff.

Next Allowed:

- Validate this capped recipe on a fresh seed or many-calculator cost axis, or move to a less-prescriptive/non-enumerative credit mechanism.

Full Text:

```text
POSITIVE-WITH-CAVEAT: A quality-gated update cap preserves op29 source/handoff while cutting proportional prior-fit cost.
Conclusion: Added `--result-boundary-target-amortized-prior-quality-gate-update-cap`, which freezes amortized-prior fitting after prompt memory is full once the configured stop metric and train-accuracy requirement are met at or beyond a specified update count. With the proportional op29 h128 recipe, cap `2000` froze the prior at `2000` updates / `1,254,817` fit examples / `1,080,000` full-fit examples. The source reached overall exact/calc `0.9956`, train `1.0000`, heldout `0.9611`, prior train/validation `0.9861`/`0.9927`, and low heldout controls (`0.0222` injection-zero, `0.0000` forced-zero, `0.0056` forced-random). Trusted 600-step additive handoff reached `900/900 = 1.0000`, diagnostic exact/calc `1.0000`/`0.9922`, and low final snapshot controls (`0.0000` injection-zero, `0.0000` forced-zero, `0.0078` forced-random).
Do not repeat: Do not run cap-value, refresh-window, or proportional-fraction ladders as novelty. The single cap improves cost materially but is still one op29 seed and still depends on answer-derived sparse scoring plus staged handoff.
Next allowed test: Validate this capped recipe on a fresh seed or many-calculator cost axis, or move to a less-prescriptive/non-enumerative credit mechanism.
Source: `aiAgentWorkHistory/phase7/2026-06-02-op29-quality-gated-prior-cap.md`
```
