# A simple online learned loss proposal improves fixed-grid sparse local targets but does not solve streaming scalability.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-learned-proposal-local-target-gate.md

Summary:

- Added `learned_policy_reweighted_t<T>_u<U>_p<P>_h<H>_e<E>`, which trains a small parametric forced-loss predictor on observed candidate scores and proposes low predicted-loss result classes alongside uniform exploration. In the 200-step full-grid gate at the same 32 forced scores per step, raw `u32` reached `0.3350` exact calc / `0.3438` sampled normal, while `learned_policy_reweighted_t1_u4_p28_h32_e1` reached `0.5850` / `0.5703`, with proposal true-candidate coverage `1.0000`, target argmax `0.9175`, injection-zero `0.0234`, and forced-random `0.0156`; other learned 32-score branches reached `0.4850-0.5050` calc. But streaming minibatches removed the lift: at batch `16`, 200 steps gave exact `0.1100`, raw `u32` `0.0700`, and learned `0.0925`; at 800 steps, raw `u32` and learned tied at `0.2350` exact calc, with sampled normal `0.2734` vs `0.2656`.

Questions:

- What did we learn about A simple online learned loss proposal improves fixed-grid sparse local targets but does not solve streaming scalability?
- Has A simple online learned loss proposal improves fixed-grid sparse local targets but does not solve streaming scalability been tested?
- Should we repeat A simple online learned loss proposal improves fixed-grid sparse local targets but does not solve streaming scalability?
- What is the status of A simple online learned loss proposal improves fixed-grid sparse local targets but does not solve streaming scalability?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-learned-proposal-local-target-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same polynomial-feature online MLP proposal branches (`u4_p28_h32_e1`, `u8_p24_h32_e1`, `u16_p16_h32_e1`, `u8_p24_h64_e3`) on the fixed grid as novelty, and do not claim fixed-grid proposal coverage is scalability evidence without streaming/generalization lift.

Next Allowed:

- Learned proposal work needs an explicit streaming/generalization mechanism or validation objective, such as proposal training across heldout prompt ranges, replayed/off-policy proposal data that is not prompt-keyed, or a target construction that uses proposal uncertainty. Otherwise pivot local-target work away from proposal knobs.

Full Text:

```text
PARTIAL: A simple online learned loss proposal improves fixed-grid sparse local targets but does not solve streaming scalability.
Conclusion: Added `learned_policy_reweighted_t<T>_u<U>_p<P>_h<H>_e<E>`, which trains a small parametric forced-loss predictor on observed candidate scores and proposes low predicted-loss result classes alongside uniform exploration. In the 200-step full-grid gate at the same 32 forced scores per step, raw `u32` reached `0.3350` exact calc / `0.3438` sampled normal, while `learned_policy_reweighted_t1_u4_p28_h32_e1` reached `0.5850` / `0.5703`, with proposal true-candidate coverage `1.0000`, target argmax `0.9175`, injection-zero `0.0234`, and forced-random `0.0156`; other learned 32-score branches reached `0.4850-0.5050` calc. But streaming minibatches removed the lift: at batch `16`, 200 steps gave exact `0.1100`, raw `u32` `0.0700`, and learned `0.0925`; at 800 steps, raw `u32` and learned tied at `0.2350` exact calc, with sampled normal `0.2734` vs `0.2656`.
Do not repeat: Do not rerun the same polynomial-feature online MLP proposal branches (`u4_p28_h32_e1`, `u8_p24_h32_e1`, `u16_p16_h32_e1`, `u8_p24_h64_e3`) on the fixed grid as novelty, and do not claim fixed-grid proposal coverage is scalability evidence without streaming/generalization lift.
Next allowed test: Learned proposal work needs an explicit streaming/generalization mechanism or validation objective, such as proposal training across heldout prompt ranges, replayed/off-policy proposal data that is not prompt-keyed, or a target construction that uses proposal uncertainty. Otherwise pivot local-target work away from proposal knobs.
Source: `aiAgentWorkHistory/phase7/2026-05-29-learned-proposal-local-target-gate.md`
```
