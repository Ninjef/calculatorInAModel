# 2026-05-31 - Random Half-Memory Prior Fit Gate

## Question

Can random half-memory prior fit batches reduce per-update prior cost while
preserving the sustained-convergence heldout source gate?

## Setup

This tested the coreset/minibatch lever after the sustained train-memory
convergence stop gate established a safe full-memory benchmark:

- every-2 full-memory fit with patience `100`;
- source overall `0.9950`;
- heldout exact/calc `0.9125`;
- trusted handoff `0.9925`;
- prior updates `1889`.

The new run kept the every-2/patience-100 recipe but used a random half-memory
prior fit batch:

```text
--result-boundary-target-amortized-prior-fit-batch-size 160
--result-boundary-target-amortized-prior-fit-every 2
--result-boundary-target-amortized-prior-stop-train-accuracy 1.0
--result-boundary-target-amortized-prior-stop-patience 100
```

## Source Run

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fit160_every2_stop1pat100_src5000/2026-05-31_140446_144494_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-61979115ea/model-c-2digit-seed9
```

Results:

- Overall exact/calc `387/400 = 0.9675`.
- Train exact/calc `319/320 = 0.996875`.
- Heldout exact/calc `65/80 = 0.8125`.
- Heldout controls: injection-zero `0.0500`, forced-zero `0.0000`,
  forced-random `0.0125`.
- Prior updates `2501`; convergence stop never activated.
- Prior train/heldout accuracy `0.909375` / `0.7750`.
- Forced-result evals stayed `86,016`.

Training curve checkpoints:

- Step 1000: prior train `0.5125`, heldout replay pseudo `0.28125`.
- Step 2000: prior train `0.75625`, heldout replay pseudo `0.46875`.
- Step 3000: prior train `0.8375`, heldout replay pseudo `0.6875`.
- Step 4000: prior train `0.834375`, heldout replay pseudo `0.765625`.
- Step 5000: prior train `0.909375`, heldout replay pseudo `0.765625`.

## Interpretation

Random half-memory prior fitting improves over the old batch64 underfit but
still fails the heldout source gate. The prior underfits both train memory and
heldout prompts, so the issue is not just stopping too early.

No trusted handoff was run because the source missed the heldout gate.

## Next

Do not run random prior fit-batch-size ladders as novelty. A useful next
coreset/reservoir experiment must change the sampling mechanism, for example
structured operand coverage, balanced sum/result coverage, or a validation-aware
stopping/fitting signal.
