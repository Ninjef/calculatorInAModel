# Sparse sampled pairwise-preference targets do not train result-space calculator use.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-sampled-pairwise-preference-target-gate.md

Summary:

- Added `sampled_pairwise_preference_uN[_gG]`, which scores sparse forced-result candidates and trains the policy to rank lower answer-loss candidates above higher-loss candidates. In the 200-step fixed-grid Stage 1 gate, pairwise `u8` and `u16` stayed at `0.0050` final exact-grid calc / `0.0078` sampled normal, and pairwise `u32` reached only `0.0425` calc / `0.0234` normal despite `0.8450` true-candidate coverage. The same-budget `sampled_policy_reweighted_t1_k0_u32` comparator reached `0.3350` calc / `0.3438` normal. Simple sparse pairwise preference is therefore not a useful target construction here.

Questions:

- What did we learn about Sparse sampled pairwise-preference targets do not train result-space calculator use?
- Has Sparse sampled pairwise-preference targets do not train result-space calculator use been tested?
- Should we repeat Sparse sampled pairwise-preference targets do not train result-space calculator use?
- What is the status of Sparse sampled pairwise-preference targets do not train result-space calculator use?
- Why did Sparse sampled pairwise-preference targets do not train result-space calculator use fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-sampled-pairwise-preference-target-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun `sampled_pairwise_preference_u8/u16/u32` on the same 200-step fixed-grid gate, and do not run simple candidate-count or `_gG` loss-gap sweeps as novelty.

Next Allowed:

- Pairwise-style work needs a materially different mechanism: policy-aware weighting, uncertainty-aware active sampling, accumulated preferences, or a different target construction. Otherwise return to broader target construction or source-geometry questions.

Full Text:

```text
DISPROVEN: Sparse sampled pairwise-preference targets do not train result-space calculator use.
Conclusion: Added `sampled_pairwise_preference_uN[_gG]`, which scores sparse forced-result candidates and trains the policy to rank lower answer-loss candidates above higher-loss candidates. In the 200-step fixed-grid Stage 1 gate, pairwise `u8` and `u16` stayed at `0.0050` final exact-grid calc / `0.0078` sampled normal, and pairwise `u32` reached only `0.0425` calc / `0.0234` normal despite `0.8450` true-candidate coverage. The same-budget `sampled_policy_reweighted_t1_k0_u32` comparator reached `0.3350` calc / `0.3438` normal. Simple sparse pairwise preference is therefore not a useful target construction here.
Do not repeat: Do not rerun `sampled_pairwise_preference_u8/u16/u32` on the same 200-step fixed-grid gate, and do not run simple candidate-count or `_gG` loss-gap sweeps as novelty.
Next allowed test: Pairwise-style work needs a materially different mechanism: policy-aware weighting, uncertainty-aware active sampling, accumulated preferences, or a different target construction. Otherwise return to broader target construction or source-geometry questions.
Source: `aiAgentWorkHistory/phase7/2026-05-30-sampled-pairwise-preference-target-gate.md`
```
