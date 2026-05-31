# 2026-05-31 Integrated Amortized-Prior Source Gate

## Question

Can numeric amortized-prior replay repair the prompt-keyed heldout failure
during source training itself, rather than as a post-hoc result-head replay?

## Prior Evidence

The prompt-keyed streaming heldout run solved seen prompts but failed absent
prompts: train exact/calc `0.996875`, heldout exact/calc `0.0875`.

Post-hoc numeric-prior replay showed the target-prior idea was viable after the
fact: train stayed `0.990625` and heldout rose to `0.9125`. But that was not
end-to-end source training.

## Integrated Minibatch-Prior Gate

First run:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_replay_src5000/2026-05-31_083401_114954_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-c03208a82b/model-c-2digit-seed9
```

Settings:

- op19, four `left_operand_mod` routed hooks, shared output projection.
- Streaming train split: `320` train prompts, `80` heldout prompts.
- Sparse zero-improvement online hard memory on train prompts only.
- Numeric amortized-prior replay on heldout prompts.
- Train replay weight `1`.
- Prior fit batch inherited the model replay batch size: `64`.

Results:

- Overall exact/calc: `376/400 = 0.9400`.
- Train exact/calc: `317/320 = 0.990625`.
- Heldout exact/calc: `57/80 = 0.7125`.
- Heldout controls: injection-zero `0.0500`, forced-zero `0.0000`,
  forced-random `0.0125`.
- Online prior accuracy: train `0.8531`, heldout `0.7000`.

Follow-up diagnostic:

```text
runs/amortized_prior_trace_diagnostics/integrated_src5000_numeric_prior_fit.json
```

Offline full-batch numeric prior fit from the final train trace reached train
`0.990625` and heldout `0.9125` against true sums, while fitting train memory
at `1.0000`. This showed the integrated run was limited by online prior fitting,
not by target quality.

## Full-Memory-Prior Gate

Implementation change:

- Added `--result-boundary-target-amortized-prior-fit-batch-size`.
- Default `-1` preserves the old behavior: use the model replay batch size for
  prior fitting.
- Setting `0` trains the prior on all current memory entries while keeping model
  replay at `64`.

Run:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fitfull_src5000/2026-05-31_122456_419991_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-f0e1656264/model-c-2digit-seed9
```

Results:

- Overall exact/calc: `398/400 = 0.9950`.
- Train exact/calc: `320/320 = 1.0000`.
- Heldout exact/calc: `73/80 = 0.9125`.
- Heldout controls: injection-zero `0.0500`, forced-zero `0.0000`,
  forced-random `0.0125`.
- Prompt memory entries: `320/320`.
- Forced-result evals: `86,016`, slightly below the heldout-failed prompt-memory
  source's `87,552`.
- Online prior accuracy: train `1.0000`, heldout `0.9250`.
- All train routed hooks reached calculator-result accuracy `1.0000`; heldout
  routed hook accuracies were `0.8696`, `0.9130`, `0.9524`, and `0.9231`.

## Trusted Additive Handoff

Run:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fitfull_handoff600/2026-05-31_124025_113211_model-c-op0-19-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed9
```

Settings:

- Loaded the full-memory-prior source final checkpoint with compatible loading.
- Switched to non-bottleneck mode: `calculator_bottleneck_mode=none`.
- Used `calculator_estimator=ste`.
- Froze the calculator policy.
- Trained downstream/readout for `600` steps.

Results:

- Final eval: `400/400 = 1.0000`.
- Final 128-sample controls: injection-zero `0.0234`, forced-zero `0.0078`,
  forced-random `0.0156`.
- Diagnostic calculator-result accuracy: `0.984375`.
- Routed diagnostic normal accuracy was `1.0000` for every hook; routed
  calculator-result accuracy was `0.9574`, `1.0000`, `1.0000`, `1.0000`.

## Interpretation

This is the first integrated source-training positive for the deterministic
heldout-prompt gate in this branch. The model is not merely replaying prompt
keys: heldout prompts were absent from source minibatches and prompt memory, yet
the numeric prior supplied targets that trained the calculator result policy.

The trusted frozen-policy additive handoff also passes, so the learned
calculator policy can function in the non-bottleneck setting.

This is still not the final scalable solution. Full-memory prior fitting is a
costly stabilizer, and the method still depends on sparse forced-result scoring
for train prompts before memory fills. The next question is whether the
full-memory prior-fit effect can be approximated cheaply enough for many
calculators and larger models.
