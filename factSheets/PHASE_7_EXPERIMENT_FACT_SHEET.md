# Phase 7 Experiment Fact Sheet

## Direction

Phase 7 tests natural `0..19` result-level calculator use.

The phase starts from the Phase 6 closure:

```text
Deterministic hard-forward / soft-backward Concrete can discover and retain an
identifiable `sum_left_operand` hard calculator protocol, but natural sum-only
addition failed because the answer identifies a result group rather than a
unique operand pair.
```

Phase 7 should therefore prioritize structured joint-pair or result-space
interfaces that match the result-level information available in natural answer
loss.

## Retention Guardrail

Target-off retention is not new by itself. Earlier phases already established
the general pattern:

- Phase 4: direct operand supervision can teach an identifiable
  `sum_left_operand` calculator-query protocol, and answer loss retains it
  after `aux_operand_loss_weight=0.0` across seeds.
- Phase 5: upstream-open answer-only continuations can preserve or complete
  partially taught identifiable protocols, while no-handoff answer-only
  discovery still fails.
- Phase 6: answer-derived identifiable bridges, including deterministic
  Concrete, can retain after local/relaxed objectives are exactly off.

For Phase 7, the only reason to run target-off retention is to test a new
natural result-level interface or a real stability question. Do not spend new
tasks simply re-proving that retention-after-teaching is possible. If a Stage 1
teaching signal is already known to work, the next task should either diagnose
retention fragility or change the learning signal/action parameterization.

## Current State After Exact-Grid Seed Replication

As of `2026-05-13`, the exact-grid upstream-open boundary-target branch has a
split replication result:

```text
exact_grid_seed_replication_negative
```

Run root:

```text
runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_seed_replication
```

No code changes were required for this task.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
91 passed
```

Preflight reused the existing seed-2 Stage 0 full-grid parity artifact:

```text
runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_retention/stage0_full_grid_parity_gate/stage0_full_grid_parity_summary.json
```

The artifact still reports `400` grid examples, `0` duplicate ordered pairs,
hard-best result parity with true sum `1.0`, tie-aware true-result best
fraction `1.0`, and semantic decoder delta `0.0`.

Note on seed naming: the CLI runs used `--seed 4` and `--seed 5`, matching the
task. `scripts/overfit_one_batch.py` stores `seed=args.seed + num_digits`, so
the output directories are `model-c-2digit-seed6` and `model-c-2digit-seed7`.

Stage 1 exact-grid teaching replicated strongly:

| CLI seed | Selected checkpoint | Hard result acc | Full-enum learned-result best fraction | Mean learned-result minus best gap | Canonical normal / calc result | Injection-zero | Forced-random | Oracle-at-eval |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4` | `stage1_seed_4/.../model-c-2digit-seed6/checkpoint_snapshots/step_00700_weights.pt` | `1.0000` | `1.0000` | `0.0000` | `1.0000 / 1.0000` | `0.0550` | `0.0225` | `1.0000` |
| `5` | `stage1_seed_5/.../model-c-2digit-seed7/checkpoint_snapshots/step_00750_weights.pt` | `0.9975` | `0.9975` | `0.0059` | `0.9975 / 0.9975` | `0.0550` | `0.0225` | `1.0000` |

Stage 1 parameter movement:

| CLI seed | Group | L2 delta | Max abs | Changed tensors |
| --- | --- | ---: | ---: | ---: |
| `4` | semantic decoder | `0.0` | `0.0` | `0/5` |
| `4` | `calculator_hook.result_proj` | `93.1714` | `6.0694` | `2/2` |
| `4` | upstream encoder | `4.8055` | `0.1961` | `14/29` |
| `4` | other interface groups | `0.0` | `0.0` | `0/0` |
| `5` | semantic decoder | `0.0` | `0.0` | `0/5` |
| `5` | `calculator_hook.result_proj` | `90.6737` | `5.4942` | `2/2` |
| `5` | upstream encoder | `5.1313` | `0.2050` | `14/29` |
| `5` | other interface groups | `0.0` | `0.0` | `0/0` |

Stage 2 target-off continuation retained above the final `0.70` floor for both
seeds, but failed the stricter exact-grid best-post-start `90%` retention
criterion for both seeds:

| CLI seed | Stage 1 selected hard acc | Best post-start checkpoint | Best post-start exact-grid hard acc | Exact-grid retention ratio | Final exact-grid hard acc | Final full-enum learned-result best fraction | Final mean learned-result minus best gap | Strict Stage 2 gate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `4` | `1.0000` | `stage2_seed_4/.../model-c-2digit-seed6/checkpoint_snapshots/step_00100_weights.pt` | `0.8700` | `0.8700` | `0.8350` | `0.8350` | `0.8142` | fail |
| `5` | `0.9975` | `stage2_seed_5/.../model-c-2digit-seed7/checkpoint_snapshots/step_00150_weights.pt` | `0.8800` | `0.8822` | `0.7925` | `0.7925` | `0.9376` | fail |

Sampled canonical diagnostics remained consistent with causal calculator-result
dependence. For the selected Stage 2 best/final checkpoints, injection-zero was
`0.0550`, forced-random was `0.0225`, and oracle-at-eval was `1.0000`.

Stage 2 parameter movement:

| CLI seed | Span | Group | L2 delta | Max abs | Changed tensors |
| --- | --- | --- | ---: | ---: | ---: |
| `4` | start -> best step `100` | semantic decoder | `0.0` | `0.0` | `0/5` |
| `4` | start -> best step `100` | `calculator_hook.result_proj` | `1.3996` | `0.2410` | `2/2` |
| `4` | start -> best step `100` | upstream encoder | `0.0725` | `0.0081` | `14/29` |
| `4` | start -> final step `400` | semantic decoder | `0.0` | `0.0` | `0/5` |
| `4` | start -> final step `400` | `calculator_hook.result_proj` | `2.1418` | `0.2933` | `2/2` |
| `4` | start -> final step `400` | upstream encoder | `0.2140` | `0.0278` | `14/29` |
| `5` | start -> best step `150` | semantic decoder | `0.0` | `0.0` | `0/5` |
| `5` | start -> best step `150` | `calculator_hook.result_proj` | `1.7317` | `0.1574` | `2/2` |
| `5` | start -> best step `150` | upstream encoder | `0.1051` | `0.0117` | `14/29` |
| `5` | start -> final step `400` | semantic decoder | `0.0` | `0.0` | `0/5` |
| `5` | start -> final step `400` | `calculator_hook.result_proj` | `2.0321` | `0.3115` | `2/2` |
| `5` | start -> final step `400` | upstream encoder | `0.2185` | `0.0264` | `14/29` |

Interpretation:

- Exact-grid result-boundary teaching itself is robust across the tested seeds:
  both CLI seeds `4` and `5` learned hard result requests near `1.0`.
- Target-off continuation is materially retained in the weaker sense that both
  finals stayed above `0.70`, with semantic decoder movement exactly `0.0`.
- The strict replication gate does not pass because neither seed retained at
  least `90%` of its selected Stage 1 exact-grid hard result accuracy at the
  best post-start checkpoint.

Recommendation:

Do not proceed directly to canonical-query/protocol stabilization as if the
seed-2 retained positive had fully replicated. The next task should analyze
this seed fragility and compare against multi-sample result-space policy
gradient with per-prompt or leave-one-out baselines, rather than rerunning
oracle/readout checks or frozen-head boundary-target variants.

## Previous Selected Next Task

As of `2026-05-14`, the selected next task is:

```text
aiAgentProjectTasks/2026-05-14-phase-7-eighth-task-Multi-sample-result-space-policy-gradient-gate.md
```

Rationale:

- The useful Phase 7 signal is no longer whether answer-derived
  boundary-target teaching can fit the exact grid. It can.
- The unresolved blocker is whether a learning signal closer to true
  non-differentiable tool use can discover and retain the natural result
  request robustly.
- The boundary-target branch should now be treated as a supervised
  ceiling/control for estimator-family comparisons.
- Multi-sample result-space policy gradient is the most direct next big bet:
  it keeps the natural result-level action space, uses exact-grid coverage,
  and replaces scalar single-sample REINFORCE with per-prompt or
  leave-one-out baselines.

Guardrail:

This is not a repeat of Phase 1 single-sample REINFORCE. Phase 1 already tried
single-sample independent-operand REINFORCE with a moving scalar baseline and
got a negative. The next task must use result-space actions, multi-sample
per-prompt advantages, and a gradient-agreement diagnostic against the known
boundary-target ceiling before spending long-run training budget.

## 2026-05-14 Multi-Sample Result-Space Policy-Gradient Gate

Task:

```text
aiAgentProjectTasks/2026-05-14-phase-7-eighth-task-Multi-sample-result-space-policy-gradient-gate.md
```

Run root:

```text
runs/2026-05-14_phase7_multisample_result_space_policy_gradient_gate
```

Code changes:

- `calculator_action_head=result_space` now supports
  `calculator_estimator=reinforce`.
- Result-space REINFORCE samples from `calculator_hook.result_proj`, maps the
  sampled result to the deterministic canonical calculator pair, and records
  `result_logp`; `sampled_logp` is now the sampled result log-probability for
  result-space policies.
- `scripts/overfit_one_batch.py` now supports
  `reinforce_baseline_mode={global_ema,per_prompt_mean,leave_one_out}`,
  `reinforce_num_samples_per_prompt`, and a diagnostic-only
  PG-vs-boundary gradient gate.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
92 passed
```

Stage 0 exact-grid gradient gate used natural `0..19`, exhaustive `400`
prompts, `calculator_action_head=result_space`,
`calculator_estimator=reinforce`, frozen semantic decoder, no oracle operands,
no boundary-target update, and `K=16` samples per prompt with the
leave-one-out baseline.

Artifact:

```text
runs/2026-05-14_phase7_multisample_result_space_policy_gradient_gate/stage0_gradient_gate/2026-05-14_075911_897578_model-c-op0-19-fullgrid-reinforce-result_space-K16-leave_one_out-inlr0.01-uplr0.0003-graddiag-answer_decoder-adec-product/model-c-2digit-seed4/reinforce_gradient_diagnostic_summary.json
```

Stage 0 results:

| Metric | Value |
| --- | ---: |
| answer loss | `7.9035` |
| PG objective / policy loss | `0.001336 / 0.001336` |
| advantage mean / std | `0.000000006 / 3.4309` |
| result entropy | `3.6636` |
| sampled result accuracy | `0.0278` |
| PG result-proj grad L2 | `0.1551` |
| PG upstream grad L2 | `0.0574` |
| PG semantic decoder grad L2 | `0.0` |
| boundary result-proj grad L2 | `0.0897` |
| boundary upstream grad L2 | `0.0332` |
| boundary semantic decoder grad L2 | `0.0` |
| PG-vs-boundary result-proj cosine | `-0.0945` |
| PG-vs-boundary upstream cosine | `-0.1108` |
| global EMA advantage std | `3.7048` |
| per-prompt mean advantage std | `3.1542` |
| leave-one-out advantage std | `3.4382` |

Interpretation:

- The policy-gradient plumbing is live: result-proj and upstream PG gradients
  are nonzero, and semantic decoder gradient remains exactly `0.0`.
- Per-prompt and leave-one-out baselines reduce advantage standard deviation
  versus the legacy global EMA baseline on this fixed-grid diagnostic.
- The decisive gate fails because the result-proj PG gradient is negatively
  aligned with the known answer-derived boundary-target ceiling
  (`cosine=-0.0945`).

Decision:

```text
multisample_result_space_policy_gradient_stage0_alignment_negative
```

Stage 1 and Stage 2 were intentionally not run. Per the task gate, vanilla
multi-sample result-space policy gradient should not receive long-run training
budget while its fixed-grid estimate is anti-aligned with the boundary-target
ceiling. Next work should move to actor-critic/NVIL-style learned baselines
only if the gradient alignment can be improved, or to surrogate/shadow
calculator gradients, synthetic gradients/direct feedback alignment, or a
stricter decoder-phase bottleneck.

## Selected Task After Policy-Gradient Gate

As of `2026-05-14`, before the exact result-marginal gate above, the selected
next task was:

```text
aiAgentProjectTasks/2026-05-14-phase-7-ninth-task-Exact-result-marginal-answer-loss-gradient-gate.md
```

Rationale:

- The multi-sample result-space PG negative left an important ambiguity: the
  sampled estimator may be too noisy, or the exact expected answer-loss
  objective over result actions may itself be misaligned.
- Before moving to actor-critic/NVIL/RELAX, compute the exact result-marginal
  answer-loss gradient over the small `0..38` result action space and compare
  it on the exact `20 x 20` grid against both the sampled PG gradient and the
  boundary-target ceiling.
- If the exact result-marginal gradient aligns while sampled PG does not, the
  immediate blocker is estimator variance/control variates; exact enumeration
  can be used as the fastest controlled training signal while the result space
  remains small.
- If the exact result-marginal gradient is also negative or near-zero, stop
  expected-cost/score-function work and pivot to a different mechanism such as
  surrogate/shadow calculator gradients, synthetic gradients/direct feedback
  alignment, or a stricter decoder-phase bottleneck.

Guardrail:

This is not a repeat of the Phase 6 independent-head expected answer-loss
negative. That branch enumerated `20 x 20` operand pairs and collapsed to wrong
hard actions. The new task must use `calculator_action_head=result_space`,
exact enumeration over only `39` result classes, and a Stage 0 gradient
alignment gate before any long training run.

## 2026-05-14 Exact Result-Marginal Answer-Loss Gradient Gate

Task:

```text
aiAgentProjectTasks/2026-05-14-phase-7-ninth-task-Exact-result-marginal-answer-loss-gradient-gate.md
```

Run root:

```text
runs/2026-05-14_phase7_exact_result_marginal_answer_loss_gradient_gate
```

Code changes:

- `calculator_action_head=result_space` now supports
  `calculator_estimator=full_enum_expected_answer_loss`.
- The expected answer-loss objective now has a result-space branch that
  enumerates the `0..38` forced result classes, computes detached answer-NLL
  costs, and minimizes the model result policy's exact expected cost.
- Added `--expected-answer-loss-gradient-diagnostic-only`, which reports exact
  result-marginal, sampled result-space PG, and boundary-target gradients on
  the same fixed batch.
- Added tests covering result-space expected-loss metrics and
  result-projection gradients.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
94 passed
```

Primary raw-cost Stage 0 artifact:

```text
runs/2026-05-14_phase7_exact_result_marginal_answer_loss_gradient_gate/stage0_gradient_gate/2026-05-14_093048_129116_model-c-op0-19-fullgrid-full_enum_expected_answer_loss-result_space-inlr0.01-uplr0.0003-expanspolt1-expanschunk64-expansgraddiag-answer_decoder-adec-product/model-c-2digit-seed4/expected_answer_loss_gradient_diagnostic_summary.json
```

Raw-cost gate table:

| Gate | Value | Pass? |
| --- | ---: | --- |
| exact-grid prompts | `400` | yes |
| exact result-proj grad L2 > 0 | `0.1465` | yes |
| exact upstream grad L2 > 0 | `0.0549` | yes |
| exact semantic decoder grad L2 == 0.0 | `0.0` | yes |
| exact-vs-boundary result-proj cosine > 0.0 | `-0.0978` | no |
| exact-vs-boundary upstream cosine > 0.0 | `-0.1231` | no |
| sampled PG-vs-exact result-proj cosine | `0.9577` | diagnostic |
| sampled PG-vs-exact upstream cosine | `0.9736` | diagnostic |

Raw-cost diagnostic details:

| Metric | Value |
| --- | ---: |
| exact expected answer-loss objective | `7.8521` |
| raw expected NLL | `7.8521` |
| best/true result NLL | `0.0004 / 0.0004` |
| learned result NLL | `8.5497` |
| expected-minus-best gap | `7.8517` |
| best/true result probability under policy | `0.02565 / 0.02565` |
| hard learned result accuracy | `0.0225` |
| boundary true-result probability | `0.8003` |
| sampled PG-vs-boundary result/upstream cosine | `-0.0945 / -0.1108` |

Z-score cost normalization was checked because it is an allowed detached
normalization:

```text
runs/2026-05-14_phase7_exact_result_marginal_answer_loss_gradient_gate/stage0_gradient_gate_zscore/2026-05-14_093119_897295_model-c-op0-19-fullgrid-full_enum_expected_answer_loss-result_space-inlr0.01-uplr0.0003-expanspolt1-expanschunk64-zscore-expansgraddiag-answer_decoder-adec-product/model-c-2digit-seed4/expected_answer_loss_gradient_diagnostic_summary.json
```

It did not clear the strict upstream-open gate. Result-proj exact-vs-boundary
cosine became weakly positive (`0.0764`), but upstream cosine remained
non-positive (`-0.0007`). Stage 1 exact-marginal training was therefore
skipped.

Decision:

```text
result_space_expected_answer_loss_alignment_negative
```

Interpretation:

- The previous sampled PG negative was not primarily a finite-sample
  variance/control-variate artifact. With raw answer NLL costs, sampled PG was
  strongly aligned with the exact result-marginal gradient, and both were
  anti-aligned with the supervised boundary-target ceiling.
- Detached z-score normalization weakens the negative at the result head but
  still does not produce a positive upstream-open alignment gate.
- Do not spend long-run budget on raw exact expected-cost training, vanilla
  result-space PG, or learned-baseline variants that only estimate the same
  raw expected-cost gradient. The next best branch is a qualitatively different
  learning signal: surrogate/shadow-calculator gradients, synthetic
  gradients/direct feedback alignment, stricter decoder-phase bottlenecks, or
  another estimator that first passes the same three-way gradient gate.

## Selected Task After Exact Result-Marginal Gate

As of `2026-05-14`, this task has been run.

```text
aiAgentProjectTasks/2026-05-14-phase-7-tenth-task-Gradient-friendly-result-decoder-alignment-gate.md
```

Rationale:

- The exact result-marginal gate showed that the current frozen product
  decoder is not merely noisy for upstream result-policy learning; its raw
  answer-loss expected-cost gradient is locally anti-aligned with the known
  good boundary-target direction.
- Actor-critic, learned baselines, or RELAX/NVIL-style control variates are
  therefore not the fastest next move if they preserve the same expected
  gradient.
- The next most direct architectural question is whether the downstream
  decoder can be made gradient-friendly, not just forced-result accurate.
- This keeps the same exact-grid discipline: any decoder candidate must first
  pass exact result-marginal vs boundary-target gradient alignment before long
  model-side training.
- If a result-calibrated decoder passes, Stage 1 can test exact
  result-marginal answer-loss discovery with true-result labels,
  boundary-target CE/KL, oracle operands, and semantic decoder movement all
  off.
- If it fails, Phase 7 should move away from ordinary answer-loss geometry and
  toward explicitly biased backward channels such as synthetic gradients,
  direct feedback alignment, or learned shadow-gradient modules.

## 2026-05-14 Gradient-Friendly Result Decoder Alignment Gate

Task:

```text
aiAgentProjectTasks/2026-05-14-phase-7-tenth-task-Gradient-friendly-result-decoder-alignment-gate.md
```

Run root:

```text
runs/2026-05-14_phase7_gradient_friendly_result_decoder_gate
```

Code changes:

- Added `scripts/run_phase7_gradient_friendly_result_decoder_gate.py`.
- The runner trains two narrow result-calibrated decoder candidates while
  updating only semantic decoder tensors:
  `answer_offset_emb`, `answer_decoder`, and
  `calculator_hook.output_proj`.
- It then loads each candidate with
  `semantic_decoder_checkpoint_load_scope=semantic_decoder_only`, freezes the
  semantic decoder, and reuses the exact result-marginal / sampled PG /
  boundary-target gradient diagnostic on the exhaustive `20 x 20` grid.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py scripts/run_phase7_gradient_friendly_result_decoder_gate.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
94 passed
```

Stage 0 artifact:

```text
runs/2026-05-14_phase7_gradient_friendly_result_decoder_gate/2026-05-14_113835_814589/stage0_gradient_friendly_decoder_gate_summary.json
```

Stage 0 decoder-geometry table:

| Decoder | Forced/oracle exact | Hard-best=true | Tie-aware true-best | Raw expected NLL | Best/true NLL | Learned NLL | Exact result/upstream grad L2 | Semantic grad L2 | Exact-vs-boundary result/upstream cosine | PG-vs-exact result/upstream cosine | PG-vs-boundary result/upstream cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `1.000 / 1.000` | `1.000` | `1.000` | `7.8521` | `0.0004 / 0.0004` | `8.5497` | `0.1465 / 0.0549` | `0.0` | `-0.0978 / -0.1231` | `0.9577 / 0.9736` | `-0.0945 / -0.1108` |
| soft calibration | `1.000 / 1.000` | `1.000` | `1.000` | `7.8441` | `0.0004 / 0.0004` | `8.5417` | `0.1465 / 0.0547` | `0.0` | `-0.0911 / -0.1175` | `0.9579 / 0.9737` | `-0.0876 / -0.1044` |
| contrastive margin | `1.000 / 1.000` | `1.000` | `1.000` | `14.4892` | `0.0000 / 0.0000` | `15.7934` | `0.2562 / 0.0824` | `0.0` | `0.1204 / 0.0484` | `0.9560 / 0.9640` | `0.0949 / 0.0410` |

The contrastive-margin decoder passed the formal Stage 0 sign gate:

```text
gradient_friendly_decoder_alignment_pass
```

Selected decoder:

```text
runs/2026-05-14_phase7_gradient_friendly_result_decoder_gate/2026-05-14_113835_814589/contrastive_margin_best_weights.pt
```

Stage 1 exact result-marginal training was therefore run with the aligned
decoder frozen and all true-result, boundary-target, oracle-action, aux, and
semantic-decoder movement signals off.

Stage 1 run:

```text
runs/2026-05-14_phase7_gradient_friendly_result_decoder_gate/stage1_exact_marginal_discovery/2026-05-14_113930_411831_model-c-op0-19-fullgrid-full_enum_expected_answer_loss-result_space-inlr0.01-uplr0.0003-expanspolt1-expanschunk64-answer_decoder-adec-product/model-c-2digit-seed4
```

Stage 1 result:

| Checkpoint | Normal / calc-result acc | Injection-zero | Forced-random | Oracle | Learned-best | Entropy | Learned-best NLL gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best sampled diagnostic, step `275` | `0.105 / 0.105` | `0.0325` | `0.0225` | `1.000` | not logged in snapshot | not logged in snapshot | not logged in snapshot |
| best training-curve learned-result, step `300` | sampled normal `0.068` | sampled zero `0.033` | not logged | `1.000` | `0.0750` | `0.0367` | `8.2436` |
| final, step `800` | `0.090 / 0.090` | `0.0375` | `0.0275` | `1.000` | `0.0750` | `0.00003` | `8.2003` |

Final `metrics.json` exact-match was `0.085`, and final loss was `8.2003`.
The learned result policy collapsed to a few wrong classes rather than
discovering the true-sum request.

Decision:

```text
gradient_friendly_decoder_stage0_pass_stage1_exact_marginal_discovery_negative
```

Interpretation:

- The contrastive decoder shows that downstream decoder geometry can flip the
  local exact expected-cost gradient positive against the boundary ceiling.
- That local sign improvement was too weak to rescue discovery: exact
  expected-cost training still collapsed under the strict no-teacher,
  semantic-frozen downstream test.
- Do not treat ordinary expected-cost or score-function training as recovered
  by decoder calibration alone. The next Phase 7 branch should use an
  explicitly biased backward channel, such as synthetic gradients/direct
  feedback alignment or a learned shadow-gradient module, and keep the same
  exact-grid boundary-ceiling gate.

## Current State After Full-Grid Boundary Retention Gate

As of `2026-05-13`, exact full-grid upstream-open boundary-target training has
produced the first Phase 7 natural `0..19` retained positive:

```text
full_grid_upstream_open_result_boundary_retained_positive
```

Run root:

```text
runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_retention
```

Code changes:

- Added `--exhaustive-grid-batch` to `scripts/overfit_one_batch.py`.
- Added `make_exhaustive_range_batch(...)`, which builds every ordered
  `(a, b) in 0..operand_max x 0..operand_max` exactly once using the same
  tokenization, padding, and target masking as `make_range_batch`.
- Recorded `exhaustive_grid_batch` and `exhaustive_grid_size` in `config.json`
  and `metrics.json`.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
91 passed
```

Stage 0 full-grid parity gate passed:

| Metric | Value |
| --- | ---: |
| grid examples / duplicate pairs | `400 / 0` |
| hard-best result equals true sum | `1.0000` |
| tie-aware true-result best fraction | `1.0000` |
| soft target true-result probability | `0.99989` |
| target entropy / effective result count | `0.00105 / 1.00105` |
| initial hard learned result accuracy | `0.0225` |
| result-proj gradient L2 | `0.08966` |
| upstream gradient L2 | `0.03320` |
| semantic decoder gradient/delta L2 | `0.0 / 0.0` |

Stage 1 exact-grid upstream-open teaching:

```text
runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_retention/stage1_primary_full_grid/2026-05-13_153947_011891_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

- `exhaustive_grid_batch=true`, `exhaustive_grid_size=400`
- upstream open, semantic decoder frozen
- `answer_loss_weight=0.0`
- `result_boundary_target_loss_weight=1.0`
- `calculator_result_head_hidden_size=0`
- `input_proj_lr=0.01`, `upstream_lr=0.0003`
- `steps=800`, dense snapshots/checkpoints every `25`

Selected checkpoint:

```text
checkpoint_snapshots/step_00800_weights.pt
```

Stage 1 results:

| Metric | Value |
| --- | ---: |
| hard learned calculator-result accuracy | `0.9675` |
| full-enum learned-result best fraction | `0.9675` |
| mean learned-result minus best-result gap | `0.1108` |
| canonical normal exact / calculator-result accuracy | `0.9600 / 0.9600` |
| injection-zero exact | `0.0550` |
| forced-random exact | `0.0225` |
| oracle-at-eval exact | `1.0000` |
| final eval exact | `0.9530` |

Stage 1 parameter movement from step `0` to step `800`:

| Group | L2 delta | Max abs | Changed tensors |
| --- | ---: | ---: | ---: |
| semantic decoder | `0.0` | `0.0` | `0/5` |
| `calculator_hook.result_proj` | `81.5030` | `4.3182` | `2/2` |
| upstream encoder | `4.6336` | `0.1954` | `14/29` |
| other interface groups | `0.0` | `0.0` | `0/0` |

Stage 2 target-off retention:

```text
runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_retention/stage2_target_off_full_grid/2026-05-13_154541_041524_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

- initialized from Stage 1 step `800`
- `answer_loss_weight=1.0`
- `result_boundary_target_loss_weight=0.0`
- aux/adaptive/expected/relaxed-entropy/anchor objectives all `0.0`
- upstream open, semantic decoder frozen
- `exhaustive_grid_batch=true`, `steps=400`

Stage 2 results:

| Metric | Value |
| --- | ---: |
| best post-start hard result accuracy | `0.8800` at step `375` |
| best post-start full-enum learned-result best fraction | `0.8800` |
| retention vs Stage 1 selected hard accuracy | `0.9096` |
| final hard result accuracy | `0.8325` |
| final full-enum learned-result best fraction | `0.8325` |
| final canonical normal exact / calculator-result accuracy | `0.8275 / 0.8275` |
| final injection-zero exact | `0.0550` |
| final forced-random exact | `0.0225` |
| final oracle-at-eval exact | `1.0000` |

Stage 2 movement from Stage 1 selected checkpoint to final step `400`:

| Group | L2 delta | Max abs | Changed tensors |
| --- | ---: | ---: | ---: |
| semantic decoder | `0.0` | `0.0` | `0/5` |
| `calculator_hook.result_proj` | `2.4398` | `0.2809` | `2/2` |
| upstream encoder | `0.2393` | `0.0372` | `14/29` |
| other interface groups | `0.0` | `0.0` | `0/0` |

Interpretation:

- Exact ordered-grid coverage was enough to stabilize the upstream-open
  answer-derived result-boundary branch.
- The hard model-side result request survived target-off continuation: final
  hard/full-enum result accuracy stayed above `0.70`, and the best post-start
  target-off checkpoint retained more than `90%` of the selected Stage 1 hard
  result accuracy.
- Semantic decoder movement remained exactly `0.0`.
- Oracle-at-eval and low forced/injection controls are regression checks only;
  the substantive result is retained learned hard calculator-result behavior.

Next recommendation:

Replicate this exact-grid retained positive across additional seeds before
claiming a robust Phase 7 result. If replication holds, proceed to canonical
query/protocol stabilization; if it does not, compare exact-grid retention
against multi-sample result-space policy-gradient methods.

## Prior State After Result Feature Gate

As of `2026-05-13`, strict natural result-level action parameterizations tried
in Phase 7 remain below the pass gate:

- `joint_pair_stage1_negative`: hard learned calculator-result accuracy peaked
  at `0.11`; soft true-result probability stayed near broad initial mass.
- `result_space_stage1_negative`: even a direct `0..38` result request head
  peaked at only `0.0925` hard learned calculator-result accuracy, while soft
  true-result probability moved only `0.02564 -> 0.02920`.
- `result_boundary_target_stage1_negative`: a direct answer-derived result
  boundary target was sharp and valid, but frozen linear `result_proj` teaching
  peaked at only `0.1150`.
- `minimal_upstream_open_boundary_target_partial`: allowing upstream movement
  improved hard result accuracy to `0.5975`, with semantic decoder movement
  exactly `0.0`, but it still failed the `0.70` Stage 1 pass gate.

This means pair underidentification was real but not sufficient to explain the
natural-addition failure. The frozen product decoder/readout and full-enum
result landscape remain healthy. Frozen features contain nonlinear all-grid
result information, but the current production training paths are still not
reliably converting the result-level target into a retained model-side
calculator request.

Current recommendation:

```text
Do not run retention or seed replication from the current partial checkpoint.
Run one exact full-grid upstream-open boundary-target stabilization gate before
moving to multi-sample policy gradient. The prior `batch_size=400` runs use
random resampling rather than a guaranteed full `20 x 20` ordered grid at every
step, so this is the cleanest near-term test of whether the current partial
rescue is limited by stochastic coverage/stability or by the learning signal
itself.
```

Selected next task:

```text
aiAgentProjectTasks/2026-05-13-phase-7-sixth-task-Full-grid-upstream-open-result-boundary-retention-gate.md
```

If exact-grid upstream-open boundary teaching and the single allowed MLP rescue
both fail the `0.70` Stage 1 gate, Phase 7 should stop iterating on
boundary-target capacity/schedule variants and pivot to multi-sample
result-space policy gradient with per-prompt or leave-one-out baselines.

## 2026-05-13 Result Feature Separability And Upstream-Open Boundary Gate

Task:

```text
aiAgentProjectTasks/2026-05-13-phase-7-fifth-task-Frozen-feature-result-separability-and-minimal-upstream-open-boundary-gate.md
```

Run root:

```text
runs/2026-05-13_phase7_result_feature_separability_and_upstream_open
```

Code changes:

- Added `scripts/run_phase7_result_feature_separability.py`.
- Added `calculator_result_head_hidden_size`; `0` preserves the linear
  `calculator_hook.result_proj`, while positive values use a one-hidden-layer
  result-space MLP.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_phase7_result_feature_separability.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
88 passed
```

### Frozen Feature Probe

Artifacts:

```text
runs/2026-05-13_phase7_result_feature_separability_and_upstream_open/result_feature_separability_summary.json
runs/2026-05-13_phase7_result_feature_separability_and_upstream_open/result_feature_probe_all400.csv
runs/2026-05-13_phase7_result_feature_separability_and_upstream_open/result_feature_probe_5fold.csv
```

Setup:

- Strict natural `0..19`, `result_space`, `operand_spans`, span width `2`.
- Phase 6 product decoder checkpoint loaded with
  `semantic_decoder_checkpoint_load_scope=semantic_decoder_only`.
- Targets constructed by forced-result answer NLL enumeration, true sum used
  only after target construction.

Probe results:

| Metric | Value |
| --- | ---: |
| answer-derived target parity with true sum | `1.0000` |
| exact `result_proj` input width | `64` |
| linear all-400 accuracy | `0.9217` |
| linear 5-fold mean / min accuracy | `0.1358` / `0.0375` |
| MLP-64 all-400 / 5-fold mean accuracy | `1.0000` / `0.1400` |
| MLP-128 all-400 / 5-fold mean accuracy | `1.0000` / `0.1458` |
| operand-A span linear accuracy | `1.0000` |
| operand-B span linear accuracy | `1.0000` |

Interpretation:

- The exact frozen operand-span feature is not linearly sufficient by the task
  threshold (`0.9217 < 0.98`).
- A shallow MLP can memorize the finite all-400 natural grid exactly, so the
  frozen representation contains useful nonlinear information for the
  answer-derived result target.
- Held-out fold accuracy is low for both linear and MLP probes, so this is
  finite-grid separability rather than evidence of smooth extrapolating result
  structure.

### Conditional MLP Result Head

Run:

```text
runs/2026-05-13_phase7_result_feature_separability_and_upstream_open/stage1_mlp64_boundary_target/2026-05-13_091415_689135_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rhead64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

- `calculator_result_head_hidden_size=64`
- semantic decoder frozen
- upstream frozen
- `answer_loss_weight=0.0`
- `result_boundary_target_loss_weight=1.0`
- `result_boundary_target_mode=hard_best_result`
- `input_proj_lr=0.01`
- `steps=600`

Result:

| Metric | Value |
| --- | ---: |
| best hard learned calculator-result accuracy | `0.2950` at step `600` |
| best learned-result best fraction | `0.2950` |
| mean learned-result minus best-result gap at best | `3.9422` |
| final eval exact | `0.2425` |

Decision: failed the `0.70` Stage 1 gate, so target-off retention was not run.

### Minimal Upstream-Open Boundary Target

Run:

```text
runs/2026-05-13_phase7_result_feature_separability_and_upstream_open/stage1_upstream_open_boundary_target/2026-05-13_093849_217301_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

- linear result head, `calculator_result_head_hidden_size=0`
- semantic decoder frozen
- upstream open
- `answer_loss_weight=0.0`
- `result_boundary_target_loss_weight=1.0`
- `result_boundary_target_mode=hard_best_result`
- `input_proj_lr=0.01`, `upstream_lr=0.0003`
- `steps=600`

Best checkpoint:

```text
checkpoint_snapshots/step_00575_weights.pt
```

Stage 1 metrics:

| Metric | Value |
| --- | ---: |
| best hard learned calculator-result accuracy | `0.5975` at step `575` |
| best learned-result best fraction | `0.5975` |
| mean learned-result minus best-result gap at best | `2.0629` |
| final hard learned calculator-result accuracy | `0.4275` |
| final eval exact | `0.4625` |

Selected checkpoint diagnostics:

| Diagnostic | Value |
| --- | ---: |
| canonical normal exact | `0.5625` |
| canonical calculator-result accuracy | `0.5625` |
| canonical result-equivalent pair accuracy | `0.5625` |
| canonical pair exact | `0.0350` |
| injection-zero exact | `0.0550` |
| forced-random exact | `0.0225` |
| oracle-at-eval exact | `1.0000` |
| full-enum learned-result best fraction | `0.5900` |
| full-enum learned result matches true sum | `0.5900` |
| mean learned-result minus best-result gap | `2.0806` |
| true result best fraction | `1.0000` |
| tie-aware true best fraction | `1.0000` |
| soft target true result-group probability | `0.99994` |

Parameter movement from step `0` to step `575`:

| Group | L2 delta | Max abs | Changed tensors |
| --- | ---: | ---: | ---: |
| semantic decoder | `0.0` | `0.0` | `0/3` |
| `calculator_hook.result_proj` | `42.2322` | `3.9242` | `2/2` |
| upstream encoder | `3.3516` | `0.1469` | `14/29` |

Interpretation:

- Label: `minimal_upstream_open_boundary_target_partial`.
- Frozen feature probing showed nonlinear all-grid separability, but the
  production MLP head did not teach a usable hard result request under the
  planned Stage 1 budget.
- Allowing upstream movement gave a substantial rescue relative to frozen
  linear and frozen MLP branches, rising to `0.5975` hard result accuracy, but
  it still failed the `0.70` pass gate and drifted down by final.
- Semantic decoder movement remained exactly `0.0`; both result head and
  upstream moved measurably.
- Target-off retention was not run because Stage 1 did not pass.

Recommendation:

Do not run retention or seed replication from this checkpoint. The next task
should either improve the upstream-open boundary target stability/capacity with
a clearly different mechanism, or move to another signal family such as
multi-sample policy gradient with per-prompt baselines, surrogate gradients, or
direct feedback alignment.

## Starting Guardrail

Oracle/readout success is a wiring gate only. Phase 7 progress must be judged
by learned calculator-result behavior under the hard calculator path:

- learned calculator-result accuracy;
- result-equivalent pair accuracy;
- private all-pair result accuracy;
- full-enum learned-result best fraction and learned-result gaps;
- injection-zero and forced-random controls;
- semantic decoder movement exactly `0.0`;
- auxiliary/direct operand supervision exactly `0.0`;
- all discovery-specific objective weights exactly `0.0` for retention claims.

Exact true operand-pair recovery is diagnostic only in natural sum-only
addition, because many valid calculator calls share the same correct result.

The natural `0..19` product decoder/readout is no longer an open question. It
has repeatedly passed oracle/readout and full-enum result-landscape checks.
Future Phase 7 work should not present decoder usability, oracle-at-eval
success, forced-true result success, or injection wiring as new knowledge. Run
those checks only after code or checkpoint changes that could break the path,
and label them as regression checks only.

## First Recommended Track

Start with the Phase 7 overarching plan:

```text
aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md
```

The first task should gate or implement a natural joint-pair result-group
deterministic Concrete bridge before attempting larger operand ranges.

## 2026-05-12 Joint-Pair Result-Group Bridge Implementation Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-12-phase-7-first-task-Natural-joint-pair-result-group-bridge-gate.md
```

Claim tested:

```text
The natural sum-only interface can use a joint 20 x 20 pair policy whose hard
forward calculator call is trained by soft backward mass grouped by calculator
result, without true operand labels or oracle operands.
```

Code changes:

- Enabled `calculator_estimator=gumbel_concrete_interface` with
  `calculator_action_head=joint_pair`.
- Enabled `calculator_read_position=operand_spans` for `joint_pair` and sized
  `pair_proj` as `2 * calculator_read_span_width * n_embd -> V^2`.
- Added joint-pair hard-forward / soft-backward result-group relaxation:
  `p_result[s] = sum_{a+b=s} p_pair[a,b]`, with hard argmax pair used for the
  forward calculator result.
- Extended relaxed calculator metrics for joint-pair policies with pair/result
  entropy, effective result count, hard pair exact, and hard calculator-result
  accuracy.
- Preserved existing independent-head relaxed behavior and existing joint
  full-enum behavior.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests -q
```

Results:

```text
tests/test_model.py: 74 passed
tests/: 83 passed
```

Phase 7 CLI smoke run:

```text
runs/2026-05-12_phase7_joint_pair_result_group_bridge_gate/stage0_cli_smoke/2026-05-12_184116_723657_model-c-op0-19-gumbel_concrete_interface-joint_pair-inlr0.03-uplr0.0003-rtemp2-rfinal2-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

- `digits=2`, `operand_max=19`, `calculator_operand_vocab_size=20`
- `answer_format=sum`, `calculator_output_format=sum`
- `calculator_bottleneck_mode=answer_decoder`
- `answer_decoder_interaction=product`
- `semantic_decoder_checkpoint_load_scope=semantic_decoder_only`
- `freeze_semantic_decoder=true`
- `freeze_upstream_encoder=true`
- `oracle_train=false`, `oracle_warmup_steps=0`
- `answer_loss_weight=1.0`
- aux/adaptive/local/expected/relaxed-entropy/anchor weights all `0.0`
- trainable parameters: `calculator_hook.pair_proj` only (`26,000`)

One-step CLI smoke summary:

| Step | Answer loss | Hard pair exact | Hard result accuracy | Pair entropy | Effective pairs | Result entropy | Effective results |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `8.3140` | `0.0000` | `0.0078` | `5.9915` | `399.9993` | `3.4932` | `32.8908` |
| `1` | `7.2661` | `0.0000` | `0.0234` | `5.9912` | `399.8765` | `3.4913` | `32.8277` |

Gradient gate on a fixed 128-sample natural batch:

| Metric | Value |
| --- | ---: |
| answer loss | `7.6510` |
| pair-proj gradient L2 | `0.04198` |
| pair-proj one-step delta L2 | `4.8305` |
| input-proj gradient L2 | `0.0` |
| semantic decoder gradient L2 | `0.0` |
| upstream gradient L2 | `0.0` |
| semantic output-proj delta L2 | `0.0` |
| `pair_proj.weight` shape | `[400, 64]` |
| initial hard result accuracy | `0.0234` |

Interpretation:

- The implementation gate passed: answer loss sends nonzero gradient through
  result-group mass into `calculator_hook.pair_proj`, while semantic decoder
  and upstream parameters remain frozen.
- This is not a learned-interface success claim yet. The smoke run is only a
  one-step wiring/gradient gate; hard learned result accuracy remains near
  chance as expected from strict initialization.

Recommendation:

Proceed to the Phase 7 Stage 1 natural decoder/full-enum landscape regression
gate, then run the seed-2 strict joint-pair bridge only if the product decoder
and result landscape gates still pass.

## 2026-05-12 Joint-Pair Stage 1 Result Discovery

Task:

```text
aiAgentProjectTasks/2026-05-12-phase-7-second-task-Natural-joint-pair-stage1-result-discovery-and-retention-gate.md
```

Claim tested:

```text
Can answer loss train a natural 20 x 20 joint-pair calculator-query policy to
produce hard calculator calls with correct results, without true operand
labels, oracle operands, hard-best CE, expected-loss enumeration, or semantic
decoder movement?
```

Code changes:

- Added relaxed joint-pair result metrics to `scripts/overfit_one_batch.py`:
  `relaxed_calculator_true_result_probability`,
  `relaxed_calculator_argmax_result_accuracy`, and
  `relaxed_calculator_top3_result_accuracy`.
- Mirrored the same soft result metrics for independent relaxed policies so
  training curves keep a consistent schema.
- Added focused coverage in `tests/test_model.py` for soft-result versus
  hard-result metric reporting.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
75 passed
```

Stage 0 gates:

| Gate | Result |
| --- | ---: |
| oracle/readout exact | `1.000` |
| oracle/readout injection-zero | `0.055` |
| oracle/readout forced-random | `0.0225` |
| full-enum best result group true sum | `1.000` |
| soft target true result group probability | `0.99994` |
| soft target true pair probability | `0.09749` |

Stage 1 run:

```text
runs/2026-05-12_phase7_joint_pair_stage1_result_discovery/stage1_seed2_primary/2026-05-12_192703_156649_model-c-op0-19-gumbel_concrete_interface-joint_pair-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

- `digits=2`, `operand_max=19`, `calculator_operand_vocab_size=20`
- `answer_format=sum`, `calculator_output_format=sum`
- `calculator_action_head=joint_pair`
- `calculator_estimator=gumbel_concrete_interface`
- `calculator_read_position=operand_spans`, span width `2`
- `answer_decoder_interaction=product`
- `semantic_decoder_checkpoint_load_scope=semantic_decoder_only`
- `freeze_semantic_decoder=true`, `freeze_upstream_encoder=true`
- answer loss `1.0`
- aux/adaptive/local/expected/relaxed-entropy/anchor weights all `0.0`
- trainable parameters: `calculator_hook.pair_proj` only (`26,000`)

Training curve summary:

| Step | Hard result acc | Soft true-result prob | Soft argmax result acc | Top-3 result acc | Result entropy | Pair entropy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.0250` | `0.03364` | `0.0650` | `0.1675` | `3.4932` | `5.9915` |
| `150` | `0.0900` | `0.03447` | `0.0475` | `0.1625` | `3.4879` | `5.9906` |
| `300` | `0.0300` | `0.03456` | `0.0600` | `0.1675` | `3.4829` | `5.9887` |
| `450` | `0.1100` | `0.03383` | `0.0475` | `0.1475` | `3.4702` | `5.9821` |
| `600` | `0.0525` | `0.03643` | `0.0325` | `0.1475` | `3.4213` | `5.9542` |

Selected checkpoint:

```text
runs/2026-05-12_phase7_joint_pair_stage1_result_discovery/stage1_seed2_primary/2026-05-12_192703_156649_model-c-op0-19-gumbel_concrete_interface-joint_pair-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00450_weights.pt
```

Selection reason:

- Best relaxed hard learned calculator-result accuracy in the Stage 1 curve:
  `0.1100`.
- This is below the near-pass threshold and near the old natural
  independent-head negative range.

Selected checkpoint diagnostics:

| Diagnostic | Value |
| --- | ---: |
| canonical normal exact | `0.1275` |
| canonical calculator result accuracy | `0.1275` |
| canonical result-equivalent pair accuracy | `0.1275` |
| canonical pair exact | `0.0125` |
| injection-zero exact | `0.055` |
| forced-random exact | `0.0225` |
| oracle-at-eval exact | `1.000` |
| full-enum learned-result best fraction | `0.1125` |
| full-enum learned result matches true sum | `0.1125` |
| mean learned-result minus best-result gap | `5.5218` |
| full-enum best result group true sum | `1.000` |
| full-enum true result group probability | `0.99994` |
| full-enum true pair probability | `0.09749` |

Parameter movement from step `0` to selected step `450`:

| Group | L2 delta | Max abs | Changed tensors |
| --- | ---: | ---: | ---: |
| `calculator_hook.pair_proj` | `38.0941` | `2.2904` | `2/2` |
| semantic decoder | `0.0` | `0.0` | `0/5` |
| upstream encoder | `0.0` | `0.0` | `0/29` |

Interpretation:

- Label: `joint_pair_stage1_negative`.
- The product decoder and result landscape are healthy, but strict seed-2
  joint-pair result-group bridge training did not discover a useful hard
  natural calculator-result protocol.
- The new metrics distinguish this from a soft-positive/hard-handoff failure:
  soft true-result probability stayed near the broad initial result mass
  (`~0.034` to `0.036`), while hard result accuracy peaked at only `0.11`.
- Retention, replication seeds `4`/`5`, upstream-open training, and
  `operand_max=99` were not run because Stage 1 did not pass or near-pass.

Recommendation:

Move next to Track B result-space interface or Track C canonical symmetry
breaker. Do not run Stage 2 retention or seed replication from this checkpoint.

## 2026-05-13 Result-Space Interface Diagnostic

Task:

```text
aiAgentProjectTasks/2026-05-13-phase-7-third-task-Natural-result-space-interface-diagnostic.md
```

Claim tested:

```text
Can natural answer loss train a frozen-upstream model-side `0..38`
calculator-result request when the action space exactly matches the result
class identified by the answer target?
```

Code changes:

- Added `calculator_action_head=result_space`.
- Added `calculator_hook.result_proj`, mapping paired operand-span read
  representations to `calculator_result_vocab_size` logits.
- Hard forward picks `result_pred=argmax(result_logits)` and maps it to a
  deterministic valid canonical query:
  `a=min(result, operand_max)`, `b=result-a`.
- Deterministic Concrete backward uses a soft result distribution directly over
  `0..38`, with hard-forward / soft-backward calculator-output signal through
  the frozen semantic decoder.
- Added trace fields for result confidence and result entropy.
- Extended relaxed metrics and full-enum diagnostics for result-space heads.
- Added focused tests for canonical mapping coverage, result-proj gradients,
  frozen semantic/upstream gradients, relaxed metrics, and CLI/model validation.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/diagnose_private_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
79 passed
```

Stage 1 run:

```text
runs/2026-05-13_phase7_result_space_interface_diagnostic/stage1_seed2_primary/2026-05-12_203621_038904_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

- `digits=2`, `operand_max=19`, `calculator_operand_vocab_size=20`,
  `calculator_result_vocab_size=39`
- `answer_format=sum`, `calculator_output_format=sum`
- `calculator_action_head=result_space`
- `calculator_estimator=gumbel_concrete_interface`
- `calculator_read_position=operand_spans`, span width `2`
- `answer_decoder_interaction=product`
- `semantic_decoder_checkpoint_load_scope=semantic_decoder_only`
- `freeze_semantic_decoder=true`, `freeze_upstream_encoder=true`
- answer loss `1.0`
- aux/adaptive/local/expected/relaxed-entropy/anchor weights all `0.0`
- trainable parameters: `calculator_hook.result_proj` only (`2,535`)

Training curve summary:

| Step | Hard result acc | Soft true-result prob | Soft argmax result acc | Top-3 result acc | Result entropy | Effective results |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.0075` | `0.02564` | `0.0075` | `0.0850` | `3.6636` | `38.9999` |
| `150` | `0.0675` | `0.02613` | `0.0675` | `0.1650` | `3.6627` | `38.9645` |
| `300` | `0.0325` | `0.02641` | `0.0325` | `0.1575` | `3.6602` | `38.8684` |
| `450` | `0.0425` | `0.02733` | `0.0425` | `0.1625` | `3.6523` | `38.5648` |
| `600` | `0.0925` | `0.02920` | `0.0925` | `0.1750` | `3.6163` | `37.2100` |

Selected checkpoint:

```text
runs/2026-05-13_phase7_result_space_interface_diagnostic/stage1_seed2_primary/2026-05-12_203621_038904_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00600_weights.pt
```

Selection reason:

- Best hard learned calculator-result accuracy in the Stage 1 curve:
  `0.0925`.
- This is below the near-pass threshold and near the strict joint-pair
  negative range.

Selected checkpoint diagnostics:

| Diagnostic | Value |
| --- | ---: |
| canonical normal exact | `0.0975` |
| canonical calculator result accuracy | `0.0975` |
| canonical result-equivalent pair accuracy | `0.0975` |
| canonical pair exact | `0.0100` |
| injection-zero exact | `0.0550` |
| forced-random exact | `0.0225` |
| oracle-at-eval exact | `1.0000` |
| mean result confidence | `0.03018` |
| mean result entropy | `3.6502` |
| full-enum learned-result best fraction | `0.0850` |
| full-enum learned result matches true sum | `0.0850` |
| mean learned-result minus best-result gap | `4.7702` |
| best result group matches true sum | `1.0000` |
| mean soft target true result group probability | `0.99994` |
| mean soft target true pair probability | `0.09749` |

Parameter movement from step `0` to selected step `600`:

| Group | L2 delta | Max abs | Changed tensors |
| --- | ---: | ---: | ---: |
| `calculator_hook.result_proj` | `18.0823` | `2.8812` | `2/2` |
| semantic decoder | `0.0` | `0.0` | `0/3` |
| upstream encoder | `0.0` | `0.0` | `0/29` |

Interpretation:

- Label: `result_space_stage1_negative`.
- The natural product decoder/readout path remains healthy as a wiring
  regression (`oracle-at-eval=1.0`), and the full-enum landscape remains
  result-sharp.
- Strict frozen-upstream result-space training did not learn a useful hard
  calculator-result request. Soft true-result probability rose only from
  `0.02564` to `0.02920`, while effective results remained broad at `37.21`.
- This is not a soft-positive/hard-handoff case and not a retention candidate.

Recommendation:

Do not run Stage 2 retention, seed replication, Track C canonical-query
symmetry breaking, or operand-range scaling from this checkpoint. The next work
should move to qualitatively different learning signals: policy-gradient /
REINFORCE-style calculator actions, target propagation or local boundary
targets, differentiable surrogate gradients, synthetic-gradient/direct-feedback
methods, or explicit curriculum handoffs with teacher removal.

## 2026-05-13 Result-Space Boundary-Target Learning Signal

Task:

```text
aiAgentProjectTasks/2026-05-13-phase-7-fourth-task-Natural-result-space-boundary-target-learning-signal.md
```

Claim tested:

```text
Can an answer-derived boundary target over calculator result classes teach a
natural 0..19 model-side result request, with true sums used only for
diagnostics and parity checks?
```

Code changes:

- Added explicit result-boundary target training flags to
  `scripts/overfit_one_batch.py`:
  `--result-boundary-target-loss-weight`,
  `--result-boundary-target-mode`,
  `--result-boundary-target-temperature`,
  `--result-boundary-target-min-probability-floor`, and
  `--result-boundary-target-chunk-size`.
- Added forced-result-class scoring over result classes `0..38`, using the
  frozen product answer decoder to compute answer NLL for each candidate.
- Added hard-best result CE and soft-result CE/KL targets on
  `calculator_hook.result_proj`; target construction does not use true operands
  or true sums.
- Logged result-boundary target settings and metrics separately from prior
  operand-pair local-target metrics.
- Added tests for lowest-NLL target selection, result-proj gradient flow,
  frozen semantic/upstream gradients, parity with direct true-sum CE only after
  target construction, and CLI validation.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
83 passed
```

Stage 0 boundary-target parity gate from the Phase 6 product checkpoint:

| Metric | Value |
| --- | ---: |
| hard-best result equals true sum | `1.0000` |
| tie-aware true-result best fraction | `1.0000` |
| soft target true-result probability | `0.99989` |
| target entropy | `0.00106` |
| effective result count | `1.0011` |
| initial hard learned result accuracy | `0.0250` |
| result-proj gradient L2 | `0.10210` |
| semantic decoder gradient/delta L2 | `0.0 / 0.0` |
| upstream gradient/delta L2 | `0.0 / 0.0` |
| trainable group | `calculator_hook.result_proj` only |

Stage 1 primary run:

```text
runs/2026-05-13_phase7_result_space_boundary_target_signal/stage1_seed2_hard_best/2026-05-13_072413_688763_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

- `answer_loss_weight=0.0`
- `result_boundary_target_loss_weight=1.0`
- `result_boundary_target_mode=hard_best_result`
- `result_boundary_target_temperature=0.25`
- `input_proj_lr=0.03`
- `steps=300`
- frozen semantic decoder and frozen upstream encoder
- trainable parameters: `calculator_hook.result_proj` only (`2,535`)

Primary curve summary:

| Step | Boundary loss | Learned result acc | Learned-best fraction | Learned-minus-best gap | Result entropy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `3.6638` | `0.0075` | `0.0075` | `7.5011` | `3.6637` |
| `75` | `3.2669` | `0.0600` | `0.0600` | `7.0230` | `3.4690` |
| `150` | `3.1388` | `0.0925` | `0.0925` | `6.7583` | `3.4150` |
| `175` | `3.1153` | `0.1150` | `0.1150` | `6.6698` | `3.3820` |
| `300` | `2.9622` | `0.0700` | `0.0700` | `6.7682` | `3.3110` |

Because the primary run did not reach `0.70`, the single allowed optimization
rescue was run with `input_proj_lr=0.01` and `steps=600`:

```text
runs/2026-05-13_phase7_result_space_boundary_target_signal/stage1_seed2_hard_best_lr001_rescue/2026-05-13_072601_947478_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2
```

Rescue result:

| Metric | Value |
| --- | ---: |
| best hard learned calculator-result accuracy | `0.0900` at step `250` |
| final hard learned calculator-result accuracy | `0.0750` |
| final eval exact | `0.0650` |

Selected Stage 1 checkpoint:

```text
runs/2026-05-13_phase7_result_space_boundary_target_signal/stage1_seed2_hard_best/2026-05-13_072413_688763_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00175_weights.pt
```

Selection reason: primary step `175` had the best hard learned
calculator-result accuracy across primary and rescue runs (`0.1150`).

Selected checkpoint diagnostics:

| Diagnostic | Value |
| --- | ---: |
| canonical normal exact | `0.0850` |
| canonical calculator result accuracy | `0.0850` |
| canonical result-equivalent pair accuracy | `0.0850` |
| canonical pair exact | `0.0125` |
| canonical injection-zero exact | `0.0550` |
| canonical forced-random exact | `0.0225` |
| oracle-at-eval exact | `1.0000` |
| mean result confidence | `0.06118` |
| mean result entropy | `3.3898` |
| full-enum learned-result best fraction | `0.0850` |
| full-enum learned result matches true sum | `0.0850` |
| mean learned-result minus best-result gap | `6.8508` |
| best result group matches true sum | `1.0000` |
| mean soft target true result-group probability | `0.99995` |
| mean effective result count | `1.0009` |

Parameter movement from Stage 1 step `0` to selected step `175`:

| Group | L2 delta | Max abs | Changed tensors |
| --- | ---: | ---: | ---: |
| `calculator_hook.result_proj` | `113.5894` | `5.1529` | `2/2` |
| semantic decoder | `0.0` | `0.0` | `0/3` |
| upstream encoder | `0.0` | `0.0` | `0/29` |
| other interface groups | `0.0` | `0.0` | `0/2` |

Interpretation:

- Label: `result_boundary_target_stage1_negative`.
- The result-boundary target itself is valid and sharp: Stage 0 hard-best and
  tie-aware true-result gates were both `1.0`, and selected-checkpoint
  full-enum diagnostics still show the true result group as best with
  probability `0.99995`.
- Despite that, the frozen operand-span features plus `result_proj` did not
  learn a useful hard result request. The best hard result accuracy was only
  `0.1150`, and the single allowed LR rescue reached only `0.0900`.
- Stage 2 target-off retention was skipped because Stage 1 did not pass or
  near-pass.

Recommendation:

Do not replicate this branch or run Stage 2 from these checkpoints. The next
task should pivot to a different signal family or capacity/feature diagnosis:
multi-sample policy gradient with per-prompt baselines, surrogate gradients,
direct feedback alignment, or a direct separability test of whether frozen
operand-span representations can linearly predict the answer-derived result
target.

## 2026-05-13 Next Selected Task: Result Feature Separability And Minimal Upstream-Open Gate

Task document:

```text
aiAgentProjectTasks/2026-05-13-phase-7-fifth-task-Frozen-feature-result-separability-and-minimal-upstream-open-boundary-gate.md
```

Decision:

```text
Before moving to policy gradient or surrogate-gradient families, directly test
whether the exact frozen operand-span features consumed by result_proj can
linearly or shallowly recover the answer-derived result target.
```

Rationale:

- The Phase 7 boundary-target objective already provided a sharp supervised
  result target without using true sums for target construction.
- That objective still failed with only the frozen linear `result_proj`
  trainable, peaking at `0.1150` hard learned calculator-result accuracy.
- This makes frozen-feature availability and head capacity the most urgent
  ambiguity. If the exact `result_proj` input is not separable, more frozen
  deterministic Concrete or frozen boundary-target schedules are low value.
- A controlled probe is cheap and decisive: it separates linear-head failure,
  shallow-capacity failure, and representation failure.

Task structure:

1. Add a result separability diagnostic over the exhaustive natural `0..19`
   grid, using answer-derived best-result targets and true sums only for
   post-hoc parity.
2. Train controlled linear and one-hidden-layer probes on the exact paired
   operand-span feature consumed by `calculator_hook.result_proj`.
3. If a linear probe passes, debug the mismatch with the in-model boundary
   target rather than changing estimator families.
4. If only a shallow probe passes, test the smallest MLP result head under the
   same boundary-target objective.
5. If frozen probes fail, run the minimal upstream-open boundary-target branch
   with semantic decoder frozen, then attempt target-off retention only if
   Stage 1 reaches the result-level gate.

Interpretation guardrail:

This task is not a learned calculator-use claim by itself. The probe is a
diagnostic gate. A project-level positive still requires a hard learned
calculator-result protocol under the real calculator path and, for retention
claims, all result-boundary/local/auxiliary/expected/anchor objectives exactly
`0.0`.

## 2026-05-28 Boundary-Feedback Result-Space Gradient Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-eleventh-task-Boundary-feedback-result-space-gradient-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_boundary_feedback_gradient_gate
```

Code changes:

- Added `calculator_estimator=direct_feedback_alignment` for result-space
  calculator actions.
- Added a boundary-feedback objective that computes answer-loss gradients at
  the calculator injection under the frozen answer decoder, maps them to
  result-logit feedback, and applies the detached feedback as a surrogate
  result-space gradient.
- Added `--boundary-feedback-weight`, `--boundary-feedback-mode`,
  `--boundary-feedback-seed`, and
  `--boundary-feedback-gradient-diagnostic-only`.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
96 passed
```

Stage 0 output-projection feedback passed the formal local alignment gate:

| Metric | Value |
| --- | ---: |
| feedback result-proj grad L2 | `0.01867` |
| feedback upstream grad L2 | `0.00544` |
| feedback semantic decoder grad L2 | `0.0` |
| result-proj cosine vs boundary | `0.27723` |
| upstream cosine vs boundary | `0.43823` |

Decision:

```text
boundary_feedback_output_projection_stage0_alignment_pass
```

Stage 1 output-projection feedback discovery failed:

| Metric | Value |
| --- | ---: |
| best snapshot normal exact / calc-result accuracy | `0.155` at step `800` |
| final exact match | `0.160` |
| final learned calc-result accuracy in training curve | `0.150` |
| final injection-zero exact match | `0.0625` |
| final forced-random exact match | `0.0156` |
| final oracle-at-eval exact match | `1.0` |

Decision:

```text
boundary_feedback_stage0_output_projection_alignment_pass_stage1_discovery_negative
```

Stage 0 fixed-random direct feedback with seed `0` failed the result-head
alignment gate:

| Metric | Value |
| --- | ---: |
| feedback result-proj grad L2 | `0.00378` |
| feedback upstream grad L2 | `0.00117` |
| feedback semantic decoder grad L2 | `0.0` |
| result-proj cosine vs boundary | `-0.00363` |
| upstream cosine vs boundary | `0.45997` |

Decision:

```text
fixed_random_direct_feedback_stage0_result_head_alignment_negative
```

Interpretation:

- A biased boundary-feedback channel can be locally aligned when using the
  frozen calculator output projection as the feedback matrix.
- That local alignment is still insufficient for natural result request
  discovery; the Stage 1 run stayed far below the `0.70` discovery floor.
- A single fixed-random DFA matrix did not pass Stage 0 at the result head.
- Next work should use a learned shadow-gradient/synthetic-gradient module or
  stronger feedback objective, with the same exact-grid Stage 0 gate and an
  early Stage 1 lift check before long-run budget.

## 2026-05-28 Linear Shadow-Feedback Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-twelfth-task-Linear-shadow-feedback-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_shadow_feedback_gradient_gate
```

Code changes:

- Split shadow feedback into fit/apply functions.
- Added `--shadow-feedback-weight`, `--shadow-feedback-ridge`, and
  `--shadow-feedback-gradient-diagnostic-only`.
- Stage 1 shadow-feedback training fits the linear map once before training,
  saves it, and does not recompute boundary targets in the training loop.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
97 passed
```

Stage 0 linear shadow feedback passed the local model-update alignment gate:

| Metric | Value |
| --- | ---: |
| shadow result-proj grad L2 | `0.08958` |
| shadow upstream grad L2 | `0.03303` |
| shadow semantic decoder grad L2 | `0.0` |
| result-proj cosine vs boundary | `0.99834` |
| upstream cosine vs boundary | `0.98543` |
| linear feedback fit cosine | `0.46028` |

Decision:

```text
linear_shadow_feedback_stage0_alignment_pass
```

The 200-step frozen-map Stage 1 early-lift smoke failed:

| Metric | Value |
| --- | ---: |
| best snapshot normal exact / calc-result accuracy | `0.070` at step `75` |
| final exact match | `0.040` |
| final learned calc-result accuracy in training curve | `0.045` |
| best injection-zero exact match | `0.065` |
| oracle-at-eval exact match | `1.0` |

Decision:

```text
linear_shadow_feedback_stage0_alignment_pass_stage1_early_lift_negative
```

Interpretation:

- A fit-once linear shadow map can produce nearly boundary-aligned gradients at
  initialization, but this is not enough for discovery under a frozen map.
- The early Stage 1 smoke performed worse than the previous output-projection
  boundary-feedback baseline (`0.040` final exact vs `0.160`).
- Do not continue this exact branch to 800 steps. Next work should use a
  heldout-validated or online-trained shadow module, with early Stage 1 lift
  required before long-run budget.

## 2026-05-28 Heldout Linear Shadow-Feedback Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-thirteenth-task-Heldout-linear-shadow-feedback-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_shadow_feedback_heldout_gate
```

Code changes:

- Added `--shadow-feedback-heldout-fraction`.
- The shadow-feedback diagnostic can now fit on one deterministic split and
  report train/heldout gradient agreement separately.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
97 passed
```

Stage 0 heldout diagnostic:

| Metric | Value |
| --- | ---: |
| fit batch / heldout batch | `320 / 80` |
| fit linear feedback cosine | `0.46857` |
| train result-proj cosine vs boundary | `0.99813` |
| heldout result-proj cosine vs boundary | `0.26221` |
| train upstream cosine vs boundary | `0.98449` |
| heldout upstream cosine vs boundary | `0.51012` |
| train-heldout result-proj cosine gap | `0.73591` |
| train-heldout upstream cosine gap | `0.47437` |
| heldout result/upstream relative norm | `1.1164 / 1.0291` |
| heldout semantic decoder grad L2 | `0.0` |

Decision:

```text
heldout_linear_shadow_feedback_stage0_generalization_negative
```

Interpretation:

- The prior same-batch linear shadow Stage 0 pass was over-optimistic.
- The heldout result-proj cosine is below the proposed online-shadow go
  threshold, and the train-heldout gap is too large.
- Do not run more fit-once linear shadow training variants from this setup.
- Next work should add an online MLP shadow-feedback module with result-policy
  state, heldout warmup validation, and only then a 200-step early-lift smoke.

## 2026-05-28 Online MLP Shadow-Feedback Warmup Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-fourteenth-task-Online-MLP-shadow-feedback-warmup-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_gate
```

Code changes:

- Added `--shadow-feedback-mode {fit_once_linear,online_mlp}`.
- Added online MLP shadow-feedback warmup flags:
  `--shadow-feedback-hidden-size`, `--shadow-feedback-online-lr`,
  `--shadow-feedback-warmup-steps`, and
  `--shadow-feedback-updates-per-step`.
- The online MLP diagnostic trains only the shadow module while the main model
  is frozen, uses per-example-scaled answer injection gradients plus current
  result logits as inputs, and evaluates induced model-gradient agreement on a
  deterministic heldout split.
- `online_mlp` is diagnostic-only for now; `--shadow-feedback-weight > 0` is
  rejected in that mode.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
98 passed
```

Stage 0B heldout diagnostic, hidden size `64`:

| Metric | Value |
| --- | ---: |
| fit batch / heldout batch | `320 / 80` |
| final fit MSE / prediction cosine | `0.01747 / 0.55395` |
| train result-proj cosine vs boundary | `0.98498` |
| heldout result-proj cosine vs boundary | `0.71673` |
| train upstream cosine vs boundary | `0.98030` |
| heldout upstream cosine vs boundary | `0.76010` |
| train-heldout result-proj cosine gap | `0.26825` |
| train-heldout upstream cosine gap | `0.22020` |
| heldout result/upstream relative norm | `1.2678 / 1.2188` |
| heldout semantic decoder grad L2 | `0.0` |

Stage 0B anti-overfit variant, hidden size `16`:

| Metric | Value |
| --- | ---: |
| final fit MSE / prediction cosine | `0.02372 / 0.24175` |
| train result-proj cosine vs boundary | `0.61852` |
| heldout result-proj cosine vs boundary | `0.62555` |
| train upstream cosine vs boundary | `0.50238` |
| heldout upstream cosine vs boundary | `0.66675` |
| train-heldout result-proj cosine gap | `-0.00703` |
| train-heldout upstream cosine gap | `-0.16437` |
| heldout result/upstream relative norm | `1.5462 / 1.1396` |

Decision:

```text
online_mlp_shadow_feedback_stage0b_partial_alignment_no_clean_gate
```

Interpretation:

- The online MLP shadow module is a materially better direction than fit-once
  linear shadow feedback: hidden size `64` crossed the headline heldout
  cosine thresholds (`result >= 0.70`, `upstream >= 0.60`).
- It did not clear the full go gate because the train-heldout gaps remained
  above the planned `0.15` limit.
- Reducing capacity to hidden size `16` reduced overfit, but the heldout
  result-proj cosine fell below the go threshold.
- No Stage 1 run was launched from these warmups. Next work should improve
  shadow generalization, not rerun this exact MLP shape.
