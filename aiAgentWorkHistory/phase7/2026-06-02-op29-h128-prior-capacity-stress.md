# 2026-06-02 - Op29 H128 Prior Capacity Stress

## Question

Can increasing the numeric amortized prior from h64 to h128 fix the op29
constant-batch eval-only target-stratified source miss without increasing the
number of examples per prior fit update?

This is a mechanism check on prior capacity, not another fit-batch or
validation ladder.

## Pre-Run Check

Read `CLAUDE.md`, `RESEARCH_STATE.md`, and `HYPOTHESIS_LEDGER.md`. Searched
memory for op29 h128 numeric prior / target-stratified / eval-only source
capacity and found no prior online h128 source run. The closest memory was the
h64 op29 mixed-negative plus a post-hoc h128 full-memory diagnostic.

## Source Run

Run directory:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_op29_prior_h128_fit160_targetstrat_val20_evalonly_stopval90pat100_src5000/2026-06-02_113036_130205_model-c-op0-29-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-c46e34c022/model-c-2digit-seed9
```

Key change from the h64 op29 stress:

```text
--result-boundary-target-amortized-prior-hidden-size 128
```

Everything else stayed matched to the h64 op29 stress: op29, four routed
shared-output hooks, prompt-keyed online hard memory, target-stratified
fit batch `160`, eval-only validation, every-2 prior fitting, and patience-100
validation stopping.

## Results

- Overall exact/calc: `879/900 = 0.9767`.
- Train exact/calc: `719/720 = 0.9986`.
- Heldout exact/calc: `155/180 = 0.8611`.
- Online prior train/heldout accuracy: `0.8097` / `0.7111`.
- Prior updates: `2501`; validation stop did not fire.
- Forced-result evals: `294,912`.
- Prompt memory entries: `720/720`.
- Heldout controls: injection-zero `0.0278`, forced-zero `0.0000`,
  forced-random `0.0111`.

No trusted handoff was run because heldout missed the source gate.

## Post-Hoc Prior Diagnostic

Ran:

```text
python3 scripts/diagnose_amortized_prior_from_trace.py <run_dir> --operand-vocab-size 30 --result-vocab-size 59 --feature-mode numeric --hidden-size 128 --steps 2500 --seed 17
```

Output:

```text
posthoc_full_memory_numeric_h128_steps2500_prior_diag.json
```

Metrics:

- Train targets matching true sums: `0.9986`.
- Post-hoc h128 train accuracy: `0.9944`.
- Post-hoc h128 heldout accuracy: `0.9278`.
- Post-hoc h128 memory-fit accuracy: `0.9958`.

## Interpretation

Capacity helps offline, but h128 does not fix the online constant-batch source
gate. The source improves over h64 overall (`0.9767` vs `0.9622`) and train is
nearly solved, but heldout remains below gate (`0.8611`). The online prior is
worse than the source split metrics imply, while a full-memory h128 diagnostic
on the same trace recovers `0.9278` heldout.

The durable lesson is that op29 needs changed prior fit dynamics, not a hidden
size bump under the same constant target-stratified fit batch.

## Anti-Rerun Guidance

Do not run more op29 constant-batch hidden-size bumps or validation threshold
ladders as novelty. Do not run trusted handoff from this heldout-missed source.

Next allowed tests should change the online fit dynamic directly: post-memory
full refresh, staged full-fit then cheaper replay, or coverage-aware /
proportional fitting with explicit cost accounting.
