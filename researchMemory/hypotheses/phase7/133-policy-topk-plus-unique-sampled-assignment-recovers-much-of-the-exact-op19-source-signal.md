# Policy-topk plus unique sampled assignment recovers much of the exact op19 source signal.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-policy-topk-assignment-proposal-gate.md

Summary:

- Added `--result-policy-improvement-assignment-policy-topk-count` so sampled hard-assignment candidates reserve slots for the model's current result-policy top-k classes, then fill the rest with unique random candidates. On the op19 `rhead64` 200-step source gate, topk8+unique16 scored only `16/39` result classes but reached step-200 true coverage `1.0000`, target accuracy `0.9333`, best snapshot `0.6850`, and final `269/400 = 0.6725`, far above unique16 (`0.4050` final). Topk8+unique24 reached true coverage/target accuracy `1.0000`, best snapshot `0.7725`, and final `300/400 = 0.7500`, slightly above the exact branch final `0.7350` while scoring `24/39` classes. Topk8+unique32 reached final `344/400 = 0.8600`, above exact final and near exact best snapshot (`0.8625`), while scoring `32/39`. This is the first assignment-cost proposal that preserves most of the source signal at materially lower scorer count, but it remains a source gate only and needs validation beyond op19/seed43 before being treated as scalable.

Questions:

- What did we learn about Policy-topk plus unique sampled assignment recovers much of the exact op19 source signal?
- Has Policy-topk plus unique sampled assignment recovers much of the exact op19 source signal been tested?
- Should we repeat Policy-topk plus unique sampled assignment recovers much of the exact op19 source signal?
- What is the status of Policy-topk plus unique sampled assignment recovers much of the exact op19 source signal?
- What follow-up is allowed for Policy-topk plus unique sampled assignment recovers much of the exact op19 source signal?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-assignment-proposal-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run more topk8 unique count ladders on the same op19 `rhead64` 200-step gate as novelty; the useful threshold is already mapped at 16/24/32.

Next Allowed:

- Validate policy-aware proposals where it matters: longer source plus trusted handoff, fresh seed, larger range, or many-calculator cost accounting. Compare against exact assignment and keep coverage/target-quality diagnostics.

Full Text:

```text
MIXED-POSITIVE: Policy-topk plus unique sampled assignment recovers much of the exact op19 source signal.
Conclusion: Added `--result-policy-improvement-assignment-policy-topk-count` so sampled hard-assignment candidates reserve slots for the model's current result-policy top-k classes, then fill the rest with unique random candidates. On the op19 `rhead64` 200-step source gate, topk8+unique16 scored only `16/39` result classes but reached step-200 true coverage `1.0000`, target accuracy `0.9333`, best snapshot `0.6850`, and final `269/400 = 0.6725`, far above unique16 (`0.4050` final). Topk8+unique24 reached true coverage/target accuracy `1.0000`, best snapshot `0.7725`, and final `300/400 = 0.7500`, slightly above the exact branch final `0.7350` while scoring `24/39` classes. Topk8+unique32 reached final `344/400 = 0.8600`, above exact final and near exact best snapshot (`0.8625`), while scoring `32/39`. This is the first assignment-cost proposal that preserves most of the source signal at materially lower scorer count, but it remains a source gate only and needs validation beyond op19/seed43 before being treated as scalable.
Do not repeat: Do not run more topk8 unique count ladders on the same op19 `rhead64` 200-step gate as novelty; the useful threshold is already mapped at 16/24/32.
Next allowed test: Validate policy-aware proposals where it matters: longer source plus trusted handoff, fresh seed, larger range, or many-calculator cost accounting. Compare against exact assignment and keep coverage/target-quality diagnostics.
Source: `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-assignment-proposal-gate.md`
```
