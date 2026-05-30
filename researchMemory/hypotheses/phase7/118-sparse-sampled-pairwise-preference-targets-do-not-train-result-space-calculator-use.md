# Sparse sampled pairwise-preference targets do not train result-space calculator use.

Status: DISPROVEN.

Source: aiAgentWorkHistory/phase7/2026-05-30-sampled-pairwise-preference-target-gate.md

Summary:

- Added `sampled_pairwise_preference_uN[_gG]` to the Phase 7 local-target Stage
  1 runner. The branch samples result candidates, scores forced answer losses,
  and trains policy logits to prefer lower-loss sampled candidates over
  higher-loss sampled candidates.
- In a 200-step fixed-grid gate, pairwise `u8` and `u16` failed completely:
  final exact-grid calc `0.0050` and sampled normal `0.0078`.
- Pairwise `u32` had high true-candidate coverage (`0.8450`) but still reached
  only `0.0425` exact-grid calc and `0.0234` sampled normal.
- The same-budget sparse policy-reweighted comparator `sampled_policy_reweighted_t1_k0_u32`
  reproduced the known positive baseline at `0.3350` exact-grid calc and
  `0.3438` sampled normal.
- Simple sparse pairwise preference is therefore not a useful replacement for
  full-enum or policy-reweighted target construction.

Questions this memory answers:

- Did sampled pairwise-preference targets train calculator use?
- Is pairwise preference better than sparse policy-reweighted targets at the
  same candidate budget?
- Should future agents sweep pairwise candidate counts or loss-gap thresholds?
- What branch syntax was added for sampled pairwise preference?

Do not repeat:

- Do not rerun `sampled_pairwise_preference_u8/u16/u32` on the same 200-step
  fixed-grid gate, and do not run simple candidate-count or `_gG` loss-gap
  sweeps as novelty.

Next allowed test:

- Pairwise-style work needs a materially different mechanism: policy-aware
  weighting, uncertainty-aware active sampling, accumulated preferences, or a
  different target construction. Otherwise return to broader target
  construction or source-geometry questions.

Ledger entry:

DISPROVEN: Sparse sampled pairwise-preference targets do not train result-space calculator use. Conclusion: Added `sampled_pairwise_preference_uN[_gG]`, which scores sparse forced-result candidates and trains the policy to rank lower answer-loss candidates above higher-loss candidates. In the 200-step fixed-grid Stage 1 gate, pairwise `u8` and `u16` stayed at `0.0050` final exact-grid calc / `0.0078` sampled normal, and pairwise `u32` reached only `0.0425` calc / `0.0234` normal despite `0.8450` true-candidate coverage. The same-budget `sampled_policy_reweighted_t1_k0_u32` comparator reached `0.3350` calc / `0.3438` normal. Simple sparse pairwise preference is therefore not a useful target construction here.
Do not repeat: Do not rerun `sampled_pairwise_preference_u8/u16/u32` on the same 200-step fixed-grid gate, and do not run simple candidate-count or `_gG` loss-gap sweeps as novelty.
Next allowed test: Pairwise-style work needs a materially different mechanism: policy-aware weighting, uncertainty-aware active sampling, accumulated preferences, or a different target construction. Otherwise return to broader target construction or source-geometry questions.
Source: `aiAgentWorkHistory/phase7/2026-05-30-sampled-pairwise-preference-target-gate.md`
