# 2026-06-02 Training Method Count Status

## Why This Review

The user asked whether the work has been trying genuinely different
from-scratch calculator-training algorithms or getting bogged down in local
tweaks. This updates the May 30/31 method-family reviews after the amortized
prior replay branch.

## Count

Strict count: about fourteen actual calculator-policy training method families.
This counts changes to the learning signal, target source, generalization
mechanism, or staged training structure. It does not count seed/range repeats,
threshold ladders, hidden-size changes, checkpoint selectors, handoff probes,
or routed/shared-output architecture validation as separate algorithms.

Training-method families tried:

1. Plain answer-loss/no-handoff discovery.
2. Vanilla result-space policy gradient and exact expected answer-loss gradients.
3. Decoder calibration as the main fix.
4. Direct feedback/output-projection boundary feedback.
5. Learned shadow-gradient feedback.
6. Hard improvement assignment/full-grid local targets.
7. Approximate or sampled local targets and proposal variants.
8. Policy-topk plus unique sampled assignment.
9. Staged bottleneck-to-additive transfer with freezing/anchoring/protection.
10. Source objectives for additive handoff geometry: forced-true, forced-margin,
    and recovery schedules.
11. Answer-derived result-boundary/zero-improvement targets.
12. Additive semantic distillation as a readout-geometry auxiliary.
13. Sparse online hard result-boundary memory.
14. Integrated amortized numeric-prior replay for fresh-prompt target
    generalization.

## Current Status

The strongest current branch is family 14 combined with sparse online hard
memory, additive semantic distillation, routed shared-output hooks, and
post-memory-fill prior refresh. It trains calculator-result policies from
scratch in bottleneck source runs, transfers to frozen-policy additive
non-bottleneck handoff, and now clears op29 heldout prompts. A quality-gated
`2000`-update cap has fresh-seed trusted handoff replication, though source
heldout dropped from `0.9611` to `0.9111`. It is still not the final
scalable/non-prescriptive solution because it uses answer-derived sparse
forced-result scoring and a costly prior-fit stabilizer.

The latest error-stratified coreset replay run is not a fifteenth algorithm.
It is a cost-reduction variant inside family 14. It preserved op29 source and
handoff, but worsened prior-update cost, so future work should change the
refresh/replay structure rather than tune the same error-focused coreset.
Likewise, the fresh-seed capped-prior handoff replication is robustness
evidence, not a new algorithm.

## Steering

- Treat future work as a new algorithm only if it changes how calculator
  targets are created, generalized, or credited.
- Do not count more op19/op29 source/handoff replications, hidden-size bumps,
  stop-threshold ladders, or coreset batch-size sweeps as method progress.
- The next high-leverage question is whether family 14 can be made cheaper and
  less prescriptive, or whether a new credit-assignment signal can bypass
  answer-derived candidate scoring entirely.
- After the fresh-seed cap replication, prioritize explicit many-calculator
  cost accounting or removal of candidate scoring over another cap/seed ladder.
