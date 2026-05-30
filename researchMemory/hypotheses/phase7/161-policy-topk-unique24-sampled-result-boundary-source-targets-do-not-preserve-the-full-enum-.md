# Policy-topk+unique24 sampled result-boundary source targets do not preserve the full-enum boundary source signal.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-sampled-result-boundary-source-gate.md

Summary:

- Added candidate-scored result-boundary target training, where each prompt scores the current policy top-8 results plus unique sampled candidates for `24/39` total result classes. In the 200-step upstream-open source gate, true-candidate coverage rose from `0.6025` to `0.9600`, but learned-best/source calculator accuracy reached only `0.3425` in the training curve, snapshot calculator accuracy `0.3675`, and final eval `141/400 = 0.3525`. This is materially below the matched full-enum hard-best result-boundary comparators (`0.5450`/`0.5475` in the soft-target gate, `0.4625`/`0.4225` in the regret-set gate).

Questions:

- What did we learn about Policy-topk+unique24 sampled result-boundary source targets do not preserve the full-enum boundary source signal?
- Has Policy-topk+unique24 sampled result-boundary source targets do not preserve the full-enum boundary source signal been tested?
- Should we repeat Policy-topk+unique24 sampled result-boundary source targets do not preserve the full-enum boundary source signal?
- What is the status of Policy-topk+unique24 sampled result-boundary source targets do not preserve the full-enum boundary source signal?
- Why did Policy-topk+unique24 sampled result-boundary source targets do not preserve the full-enum boundary source signal fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-sampled-result-boundary-source-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not ladder `result-boundary-target-sample-count`, top-k count, or unique-sampling variants around this same policy-topk sampled target as novelty. The failure is not mainly candidate coverage; the sparse/candidate hard-best target gives a weaker source signal.

Next Allowed:

- Result-boundary source work needs active proposal/training co-design, a stronger online/state-calibrated proposal, or a different target construction; otherwise move to another less-prescriptive credit-assignment family.

Full Text:

```text
MIXED-NEGATIVE: Policy-topk+unique24 sampled result-boundary source targets do not preserve the full-enum boundary source signal.
Conclusion: Added candidate-scored result-boundary target training, where each prompt scores the current policy top-8 results plus unique sampled candidates for `24/39` total result classes. In the 200-step upstream-open source gate, true-candidate coverage rose from `0.6025` to `0.9600`, but learned-best/source calculator accuracy reached only `0.3425` in the training curve, snapshot calculator accuracy `0.3675`, and final eval `141/400 = 0.3525`. This is materially below the matched full-enum hard-best result-boundary comparators (`0.5450`/`0.5475` in the soft-target gate, `0.4625`/`0.4225` in the regret-set gate).
Do not repeat: Do not ladder `result-boundary-target-sample-count`, top-k count, or unique-sampling variants around this same policy-topk sampled target as novelty. The failure is not mainly candidate coverage; the sparse/candidate hard-best target gives a weaker source signal.
Next allowed test: Result-boundary source work needs active proposal/training co-design, a stronger online/state-calibrated proposal, or a different target construction; otherwise move to another less-prescriptive credit-assignment family.
Source: `aiAgentWorkHistory/phase7/2026-05-30-sampled-result-boundary-source-gate.md`
```
