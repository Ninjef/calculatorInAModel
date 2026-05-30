# Random-prompt learned-proposal pretraining does not cleanly rescue streaming local targets.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-pretrained-learned-proposal-gate.md

Summary:

- Added optional `_wN` proposal pretraining for learned local-target branches, using random prompt/result forced-loss observations before model training. In a 200-step streaming batch-16 screen, raw `u32` reached `0.0700` exact calc / `0.0703` sampled normal, the online learned branch reached `0.0925` / `0.0938`, `_w20` reached `0.0975` / `0.0625`, and `_w50` reached `0.0950` / `0.0547`. In the 800-step streaming stress, raw `u32` reached `0.2350` exact calc / `0.2734` sampled normal, while `learned_policy_reweighted_t1_u4_p28_h32_e1_w20` reached `0.2625` exact calc but only `0.1797` sampled normal. The pretraining can slightly raise policy accuracy, but it did not produce a clean functional streaming lift.

Questions:

- What did we learn about Random-prompt learned-proposal pretraining does not cleanly rescue streaming local targets?
- Has Random-prompt learned-proposal pretraining does not cleanly rescue streaming local targets been tested?
- Should we repeat Random-prompt learned-proposal pretraining does not cleanly rescue streaming local targets?
- What is the status of Random-prompt learned-proposal pretraining does not cleanly rescue streaming local targets?
- Why did Random-prompt learned-proposal pretraining does not cleanly rescue streaming local targets fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-pretrained-learned-proposal-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not keep tuning `_w20/_w50` warmup counts, pretrain batch size, or the same polynomial-feature MLP as novelty.

Next Allowed:

- If continuing learned proposals, change the generalization mechanism itself, such as heldout-range validation, a proposal state tied to evolving model features, uncertainty-aware candidate sets, or a different target construction. Otherwise pivot away from learned-proposal warmups.

Full Text:

```text
MIXED-NEGATIVE: Random-prompt learned-proposal pretraining does not cleanly rescue streaming local targets.
Conclusion: Added optional `_wN` proposal pretraining for learned local-target branches, using random prompt/result forced-loss observations before model training. In a 200-step streaming batch-16 screen, raw `u32` reached `0.0700` exact calc / `0.0703` sampled normal, the online learned branch reached `0.0925` / `0.0938`, `_w20` reached `0.0975` / `0.0625`, and `_w50` reached `0.0950` / `0.0547`. In the 800-step streaming stress, raw `u32` reached `0.2350` exact calc / `0.2734` sampled normal, while `learned_policy_reweighted_t1_u4_p28_h32_e1_w20` reached `0.2625` exact calc but only `0.1797` sampled normal. The pretraining can slightly raise policy accuracy, but it did not produce a clean functional streaming lift.
Do not repeat: Do not keep tuning `_w20/_w50` warmup counts, pretrain batch size, or the same polynomial-feature MLP as novelty.
Next allowed test: If continuing learned proposals, change the generalization mechanism itself, such as heldout-range validation, a proposal state tied to evolving model features, uncertainty-aware candidate sets, or a different target construction. Otherwise pivot away from learned-proposal warmups.
Source: `aiAgentWorkHistory/phase7/2026-05-29-pretrained-learned-proposal-gate.md`
```
