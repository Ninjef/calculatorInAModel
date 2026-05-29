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

## 2026-05-28 Online MLP Shadow-Feedback Validation Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-fifteenth-task-Online-MLP-shadow-feedback-validation-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_validation_gate
```

Code changes:

- Added `--shadow-feedback-validation-fraction` and
  `--shadow-feedback-validation-every`.
- Online MLP warmup can now split the non-test pool into fit and validation
  subsets, select the best shadow checkpoint by validation
  `min(result_cosine, upstream_cosine)`, restore that shadow state, and then
  report the final gate on the untouched heldout test split.
- The diagnostic records final, selected train, validation, and heldout-test
  gradient-agreement metrics plus validation history and selection metadata.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
98 passed
```

Stage 0B validation-selected diagnostic, hidden size `64`:

| Metric | Value |
| --- | ---: |
| fit / validation / heldout-test batch | `280 / 40 / 80` |
| selected step / update | `60 / 60` |
| selected validation score | `0.47207` |
| selected train result/upstream cosine | `0.96499 / 0.96796` |
| selected validation result/upstream cosine | `0.48413 / 0.47207` |
| selected heldout-test result/upstream cosine | `0.64485 / 0.72658` |
| selected train-test result/upstream gap | `0.32013 / 0.24138` |
| validation-test result/upstream gap | `-0.16073 / -0.25451` |
| selected heldout-test result/upstream relative norm | `1.3604 / 1.2857` |
| final unselected heldout-test result/upstream cosine | `0.69549 / 0.76165` |

Decision:

```text
online_mlp_shadow_feedback_validation_selection_negative
```

Interpretation:

- Validation checkpoint selection did not rescue the simple online MLP shadow
  module.
- The selected checkpoint missed the result-proj heldout-test threshold and
  still had too-large train-test gaps.
- The unselected final checkpoint was close to the result threshold but still
  below `0.70` and had large gaps.
- No Stage 1 run was launched. Next work should change the target/state or add
  stronger regularization, not rely on validation selection alone.

## 2026-05-28 Online MLP Shadow-Feedback Target-Normalization Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-sixteenth-task-Online-MLP-shadow-feedback-target-normalization-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_target_norm_gate
```

Code changes:

- Added `--shadow-feedback-target-normalization`.
- Added `fit_zscore_per_result`, which fits per-result target mean/std on the
  fit split only, trains the online MLP on normalized shadow targets, and
  unnormalizes predictions before inducing model gradients.
- The diagnostic records normalization epsilon, target mean/scale summaries,
  clamped scale count, normalized fit metrics, and raw model-gradient metrics.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
99 passed
```

Shared target-normalization stats:

| Metric | Value |
| --- | ---: |
| target mean L2 | `0.09148` |
| target scale min / median / max | `0.0000617 / 0.15612 / 0.21795` |
| target scale mean | `0.14742` |
| clamped scale count | `0` |

Stage 0B validation-selected diagnostics:

| Hidden | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| ---: | ---: | ---: | ---: | --- |
| `64` | `0.71287 / 0.77383` | `0.22750 / 0.18105` | `1.1929 / 1.1339` | gap fail |
| `32` | `0.71194 / 0.75472` | `0.19129 / 0.18739` | `1.2276 / 1.2244` | gap fail |
| `16` | `0.72595 / 0.75493` | `0.17235 / 0.14584` | `1.4146 / 1.1848` | near miss, result-gap fail |
| `8` | `0.55824 / 0.58359` | `0.19213 / 0.15603` | `1.5886 / 1.3430` | cosine fail |

Decision:

```text
online_mlp_shadow_feedback_target_normalization_partial_no_go
```

Interpretation:

- Target normalization materially improved heldout-test gradient agreement
  compared with validation selection alone.
- The best capacity point (`h16`) cleared the heldout cosine thresholds and
  the upstream gap threshold, but still missed the result gap threshold
  (`0.17235 > 0.15`).
- No Stage 1 run was launched. Next work should change shadow input/state or
  objective more substantially, not rerun this sweep.

## 2026-05-28 Online MLP Shadow-Feedback Policy-State Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-seventeenth-task-Online-MLP-shadow-feedback-policy-state-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_policy_state_gate
```

Code changes:

- Added `--shadow-feedback-feature-mode`.
- Added `injection_grad_policy_state`, which appends result probabilities,
  result log-probabilities, and result entropy to the existing
  answer-gradient plus result-logit shadow input.
- The diagnostic records feature dimension and per-feature-block norms.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
99 passed
```

Stage 0B validation-selected diagnostics with target normalization:

| Hidden | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| ---: | ---: | ---: | ---: | --- |
| `16` | `0.68622 / 0.73909` | `0.23942 / 0.20316` | `1.1162 / 0.8952` | result cosine and gap fail |
| `32` | `0.70372 / 0.76105` | `0.28529 / 0.21305` | `1.3519 / 1.2201` | gap fail |

Feature observations:

- `injection_grad_policy_state` feature dimension was `134` for the natural
  `0..19` result-space model.
- Log-probability block norms dominated the raw feature vector
  (`382.84` fit L2 vs `69.50` input-gradient L2), which likely makes raw
  appended policy features a poor unnormalized state.

Decision:

```text
online_mlp_shadow_feedback_policy_state_raw_features_negative
```

Interpretation:

- Appending raw policy-state features did not improve the target-normalized
  online MLP gate.
- No Stage 1 run was launched.
- Next work should focus on feature scaling/standardization, regularization,
  a different synthetic-gradient loss, or a more stable target construction.

## 2026-05-28 Online MLP Shadow-Feedback Feature-Standardization Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-eighteenth-task-Online-MLP-shadow-feedback-feature-standardization-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_feature_norm_gate
```

Code changes:

- Added `--shadow-feedback-feature-normalization`.
- Added `fit_zscore_per_feature`, which fits shadow input feature mean/std on
  the fit split only and applies that transform before the online MLP.
- Model-gradient diagnostics still compare denormalized predictions against
  the raw boundary-target gradient ceiling.
- Added tests for fit-only feature-normalizer statistics and CLI defaults.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
100 passed
```

Stage 0B validation-selected diagnostics with target normalization and feature
standardization:

| Feature mode | Hidden | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `injection_grad_policy_state` | `16` | `0.59421 / 0.39967` | `0.28987 / 0.46921` | `1.7669 / 1.1802` | cosine and gap fail |
| `injection_grad_policy_state` | `32` | `0.43401 / 0.40230` | `0.53978 / 0.57451` | `1.8823 / 1.6198` | cosine and gap fail |
| `injection_grad_logits` | `16` | `0.64364 / 0.47630` | `0.07113 / 0.18319` | `1.6959 / 1.3566` | cosine fail |
| `injection_grad_logits` | `32` | `0.66913 / 0.70278` | `0.28301 / 0.26578` | `1.7016 / 1.4895` | result cosine and gap fail |

Feature-normalization observations:

- Policy-state feature scales were extremely uneven even before normalization:
  fit scale min/median/mean/max was
  `0.00000182 / 0.002204 / 0.1171 / 1.5177`.
- Simple logits feature scales were less extreme but still small in the median:
  `0.001190 / 0.002598 / 0.2836 / 1.5177`.
- Z-scoring did not improve heldout model-gradient agreement; it generally
  increased relative norms and widened overfit gaps.

Decision:

```text
online_mlp_shadow_feedback_feature_standardization_negative
```

Interpretation:

- Plain fit-split feature z-scoring is not enough to rescue the online MLP
  shadow-feedback gate.
- No Stage 1 run was launched.
- Next work should change objective, regularization, or target construction
  rather than rerunning feature scaling alone.

## 2026-05-28 Online MLP Shadow-Feedback Directional-Loss Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-nineteenth-task-Online-MLP-shadow-feedback-directional-loss-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_directional_loss_gate
```

Code changes:

- Added `--shadow-feedback-loss-mode`.
- Added `cosine`, which optimizes normalized-target direction.
- Added `mse_plus_cosine`, which combines componentwise MSE with the
  directional objective.
- Preserved `mse` as the default loss mode.
- Added tests for directional loss behavior and CLI defaults.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
101 passed
```

Stage 0B validation-selected diagnostics with target normalization,
`injection_grad_logits` features, and no feature normalization:

| Loss mode | Hidden | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `cosine` | `8` | `0.59905 / 0.58589` | `0.17300 / 0.15366` | `1.5006 / 1.2359` | cosine fail |
| `cosine` | `16` | `0.76465 / 0.80065` | `0.19855 / 0.15470` | `1.1896 / 1.0965` | gap fail |
| `cosine` | `32` | `0.79370 / 0.82697` | `0.20239 / 0.15449` | `1.0114 / 1.0177` | gap fail |
| `mse_plus_cosine` | `8` | `0.58189 / 0.58367` | `0.17915 / 0.15090` | `1.5508 / 1.2926` | cosine fail |
| `mse_plus_cosine` | `16` | `0.77848 / 0.81119` | `0.20450 / 0.16076` | `1.1373 / 1.0900` | gap fail |
| `mse_plus_cosine` | `32` | `0.78528 / 0.81739` | `0.20909 / 0.16364` | `1.0599 / 1.0650` | gap fail |

Decision:

```text
online_mlp_shadow_feedback_directional_loss_partial_no_go
```

Interpretation:

- Directional losses materially improved heldout model-gradient direction.
- `cosine` h32 produced the cleanest relative norms (`1.0114 / 1.0177`) and
  the best heldout cosines, but still overfit the fit split too much.
- No Stage 1 run was launched because the result train-heldout gap remained
  above `0.15`.
- Next work should combine the directional signal with explicit norm/gap
  regularization or a more stable target construction.

## 2026-05-28 Online MLP Shadow-Feedback Gap-Penalized Selection Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-twentieth-task-Online-MLP-shadow-feedback-gap-penalized-selection-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_gap_penalized_selection_gate
```

Code changes:

- Added `--shadow-feedback-selection-score-mode`.
- Added `gap_penalized_min_cosine`, which subtracts a train-validation
  cosine-gap penalty from the validation min-cosine score during checkpoint
  selection.
- Added `--shadow-feedback-selection-gap-penalty`.
- Validation history now records train-validation result/upstream cosine gaps.
- The heldout test split remains untouched for final gate reporting.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
101 passed
```

Stage 0B diagnostics with target normalization, `injection_grad_logits`
features, no feature normalization, and directional losses:

| Loss mode | Hidden | Gap penalty | Step | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cosine` | `16` | `1.0` | `90` | `0.76465 / 0.80065` | `0.19855 / 0.15470` | `1.1896 / 1.0965` | gap fail |
| `cosine` | `16` | `3.0` | `80` | `0.74504 / 0.78002` | `0.18433 / 0.14696` | `1.2853 / 1.1359` | result-gap fail |
| `cosine` | `16` | `4.0` | `70` | `0.71652 / 0.74394` | `0.16727 / 0.13527` | `1.4029 / 1.1966` | result-gap fail |
| `cosine` | `16` | `5.0` | `60` | `0.68723 / 0.69794` | `0.15107 / 0.12203` | `1.5371 / 1.2740` | cosine fail |
| `cosine` | `32` | `1.0` | `90` | `0.79370 / 0.82697` | `0.20239 / 0.15449` | `1.0114 / 1.0177` | gap fail |
| `mse_plus_cosine` | `16` | `1.0` | `90` | `0.76966 / 0.80241` | `0.19924 / 0.15997` | `1.1803 / 1.0974` | gap fail |
| `mse_plus_cosine` | `32` | `1.0` | `90` | `0.78528 / 0.81739` | `0.20909 / 0.16364` | `1.0599 / 1.0650` | gap fail |

Decision:

```text
online_mlp_shadow_feedback_gap_penalized_selection_tradeoff_no_go
```

Interpretation:

- Gap-penalized selection exposes a smooth tradeoff between heldout cosine and
  train-heldout gap.
- Checkpoint selection alone does not cross both gates simultaneously.
- No Stage 1 run was launched.
- Next work should use training-time regularization, a more stable target
  construction, or a different learned-gradient state.

## 2026-05-28 Online MLP Shadow-Feedback Dropout Regularization Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-twenty-first-task-Online-MLP-shadow-feedback-dropout-regularization-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_dropout_gate
```

Code changes:

- Added `--shadow-feedback-dropout`.
- Added `--shadow-feedback-weight-decay`.
- The online shadow MLP can now apply dropout after the hidden activation.
- The online shadow optimizer now uses explicit `AdamW` weight decay from the
  diagnostic config.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
101 passed
```

Stage 0B diagnostics with target normalization, `injection_grad_logits`
features, no feature normalization, `cosine` loss, and ordinary min-cosine
validation selection:

| Hidden | Dropout | Weight decay | Step | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `16` | `0.1` | `0.01` | `100` | `0.77227 / 0.80618` | `0.20170 / 0.15624` | `1.1453 / 1.0777` | gap fail |
| `16` | `0.2` | `0.01` | `100` | `0.76423 / 0.79834` | `0.19773 / 0.15298` | `1.1706 / 1.0851` | gap fail |
| `32` | `0.1` | `0.01` | `100` | `0.79199 / 0.82479` | `0.20391 / 0.15643` | `1.0226 / 1.0241` | gap fail |
| `32` | `0.2` | `0.01` | `100` | `0.79098 / 0.81871` | `0.20355 / 0.16115` | `1.0373 / 1.0340` | gap fail |

Decision:

```text
online_mlp_shadow_feedback_dropout_regularization_no_go
```

Interpretation:

- Dropout preserved the useful directional-loss heldout cosines.
- It did not reduce result train-heldout gap below the `0.15` gate.
- No Stage 1 run was launched.
- Next work should change target construction or learned-gradient state, or
  add explicit training-time gap/norm penalties instead of ordinary dropout.

## 2026-05-28 Online MLP Shadow-Feedback Target Transform Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-twenty-second-task-Online-MLP-shadow-feedback-target-transform-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_target_transform_gate
```

Code changes:

- Added `--shadow-feedback-target-transform`.
- Added `unit_norm_per_example`, which normalizes each shadow target row before
  fit-split target normalization.
- Recorded target-transform metrics in the online MLP diagnostic summaries.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
102 passed
```

Stage 0B diagnostics with target normalization, `unit_norm_per_example` target
transform, `injection_grad_logits` features, no feature normalization, and
ordinary min-cosine validation selection:

| Loss mode | Hidden | Step | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cosine` | `16` | `90` | `0.76497 / 0.80097` | `0.19828 / 0.15457` | `1.2043 / 1.1108` | gap fail |
| `cosine` | `32` | `90` | `0.79363 / 0.82695` | `0.20246 / 0.15451` | `1.0244 / 1.0309` | gap fail |
| `mse_plus_cosine` | `16` | `100` | `0.77871 / 0.81139` | `0.20433 / 0.16068` | `1.1510 / 1.1032` | gap fail |
| `mse_plus_cosine` | `32` | `90` | `0.78550 / 0.81756` | `0.20889 / 0.16350` | `1.0733 / 1.0784` | gap fail |

Decision:

```text
online_mlp_shadow_feedback_target_unit_norm_no_go
```

Interpretation:

- Row-wise target normalization did not change the core overfit mode.
- Heldout cosines stayed useful, but result gaps remained near `0.20`.
- No Stage 1 run was launched.
- Next work should use more structural target stabilization, a different
  learned-gradient state, or explicit train-time gap/norm penalties.

## 2026-05-28 Online MLP Shadow-Feedback Result-Prototype Target Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-twenty-third-task-Online-MLP-shadow-feedback-target-prototype-gate.md
```

Run roots:

```text
runs/2026-05-28_phase7_online_shadow_feedback_target_prototype_gate
runs/2026-05-28_phase7_online_shadow_feedback_target_prototype_gap_selection_gate
```

Code changes:

- Added `fit_result_prototype` to `--shadow-feedback-target-transform`.
- Added a boundary-target helper that returns the boundary-best result class.
- Added fit-split target prototypes keyed by boundary-best result class.
- Heldout diagnostics still compare induced model gradients against the
  original boundary-target gradients.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
103 passed
```

Stage 0B diagnostics with target normalization, `fit_result_prototype` target
transform, `injection_grad_logits` features, no feature normalization, and
ordinary min-cosine validation selection:

| Loss mode | Hidden | Step | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cosine` | `16` | `90` | `0.77715 / 0.80938` | `0.18232 / 0.14664` | `1.1946 / 1.0812` | result-gap fail |
| `cosine` | `32` | `80` | `0.80402 / 0.82434` | `0.19088 / 0.15571` | `1.0286 / 1.0208` | gap fail |
| `mse_plus_cosine` | `16` | `100` | `0.78792 / 0.81720` | `0.19386 / 0.15637` | `1.1355 / 1.0696` | gap fail |
| `mse_plus_cosine` | `32` | `90` | `0.79237 / 0.82132` | `0.20187 / 0.15964` | `1.0651 / 1.0565` | gap fail |

Narrow gap-selection follow-up:

| Hidden | Loss mode | Gap penalty | Step | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `16` | `cosine` | `3.0` | `80` | `0.75397 / 0.78550` | `0.17047 / 0.14094` | `1.2922 / 1.1240` | result-gap fail |
| `16` | `cosine` | `4.0` | `80` | `0.75397 / 0.78550` | `0.17047 / 0.14094` | `1.2922 / 1.1240` | result-gap fail |
| `16` | `cosine` | `5.0` | `80` | `0.75397 / 0.78550` | `0.17047 / 0.14094` | `1.2922 / 1.1240` | result-gap fail |
| `32` | `cosine` | `4.0` | `80` | `0.80402 / 0.82434` | `0.19088 / 0.15571` | `1.0286 / 1.0208` | gap fail |

Decision:

```text
online_mlp_shadow_feedback_target_prototype_partial_no_go
```

Interpretation:

- Result-prototype averaging is the first target-stabilization branch to push
  heldout result cosine above `0.80`.
- It still leaves result train-heldout gaps above the `0.15` gate.
- Gap-penalized selection improves h16 but still misses with result gap
  `0.1705`.
- No Stage 1 run was launched.
- Next work should change learned-gradient state or use explicit train-time
  gap/norm penalties rather than more prototype/selection variants.

## 2026-05-28 Online MLP Shadow-Feedback Result-Input State Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-twenty-fourth-task-Online-MLP-shadow-feedback-result-input-state-gate.md
```

Run roots:

```text
runs/2026-05-28_phase7_online_shadow_feedback_result_input_state_gate
runs/2026-05-28_phase7_online_shadow_feedback_result_input_gap_selection_gate
```

Code changes:

- Added `calculator_read_result_logits_and_input`.
- Added `injection_grad_logits_result_input` to
  `--shadow-feedback-feature-mode`.
- The new state appends the calculator result-projection input vector to the
  answer-gradient plus result-logit shadow features.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
103 passed
```

Stage 0B diagnostics with target normalization, `injection_grad_logits_result_input`
features, no feature normalization, and ordinary min-cosine validation
selection:

| Loss mode | Hidden | Step | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cosine` | `16` | `100` | `0.76756 / 0.83718` | `0.19578 / 0.12689` | `1.1918 / 1.2309` | result-gap fail |
| `cosine` | `32` | `100` | `0.78949 / 0.82941` | `0.20794 / 0.15326` | `1.0190 / 1.0438` | gap fail |
| `mse_plus_cosine` | `16` | `100` | `0.74513 / 0.82490` | `0.19641 / 0.12421` | `1.2097 / 1.2548` | result-gap fail |
| `mse_plus_cosine` | `32` | `100` | `0.77821 / 0.82574` | `0.21862 / 0.15835` | `1.0516 / 1.0729` | gap fail |

Narrow h16/`cosine` gap-selection follow-up:

| Gap penalty | Step | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| `3.0` | `100` | `0.76756 / 0.83718` | `0.19578 / 0.12689` | `1.1918 / 1.2309` | result-gap fail |
| `4.0` | `100` | `0.76756 / 0.83718` | `0.19578 / 0.12689` | `1.1918 / 1.2309` | result-gap fail |
| `5.0` | `100` | `0.76756 / 0.83718` | `0.19578 / 0.12689` | `1.1918 / 1.2309` | result-gap fail |

Decision:

```text
online_mlp_shadow_feedback_result_input_state_negative
```

Interpretation:

- Appending result-projection input improves upstream heldout alignment.
- It does not improve result-head generalization; result gaps remain near
  `0.20`.
- Gap-penalized selection did not move the selected checkpoint.
- No Stage 1 run was launched.
- Next work should use explicit train-time gap/norm penalties or a genuinely
  different learned-gradient state/target.

## 2026-05-28 Online MLP Shadow-Feedback Validation-Loss Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-twenty-fifth-task-Online-MLP-shadow-feedback-validation-loss-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_validation_loss_gate
```

Code changes:

- Added `--shadow-feedback-validation-loss-weight`.
- The online MLP shadow warmup can now add a validation-split prediction loss
  into each fit update.
- The heldout split remains untouched for final Stage 0B evaluation.
- Diagnostic summaries now record the validation-loss weight, final total
  objective, and final validation regularization objective.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
103 passed
```

Stage 0B diagnostics with target normalization, `injection_grad_logits`
features, `cosine` loss, validation fraction `0.1`, and validation-loss
regularization:

| Hidden | Validation-loss weight | Step | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `16` | `0.5` | `70` | `0.74678 / 0.76350` | `0.17935 / 0.13529` | `1.2700 / 1.1495` | result-gap fail |
| `16` | `1.0` | `60` | `0.72741 / 0.73810` | `0.15953 / 0.11499` | `1.3346 / 1.2494` | tradeoff fail |
| `32` | `0.5` | `100` | `0.79530 / 0.82329` | `0.19874 / 0.15688` | `0.9644 / 0.9768` | gap fail |
| `32` | `1.0` | `100` | `0.79154 / 0.81952` | `0.19886 / 0.15922` | `0.9613 / 0.9919` | gap fail |

Decision:

```text
online_mlp_shadow_feedback_validation_loss_regularization_no_go
```

Interpretation:

- Ordinary train-time validation prediction loss does not close the persistent
  result-head generalization gap.
- h32 preserves the useful heldout direction signal, but result gaps stay near
  `0.199`.
- h16/weight `1.0` reduces gaps somewhat, but loses heldout/norm quality.
- No Stage 1 run was launched.
- Next work should use a direct split-gradient gap/norm objective,
  Jacobian-conditioned state, or a richer target construction.

## 2026-05-28 Online MLP Shadow-Feedback Validation-Gradient Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-twenty-sixth-task-Online-MLP-shadow-feedback-validation-gradient-gate.md
```

Run roots:

```text
runs/2026-05-28_phase7_online_shadow_feedback_validation_gradient_gate
runs/2026-05-28_phase7_online_shadow_feedback_validation_gradient_gate/stage1_online_shadow_feedback_early_lift
runs/2026-05-28_phase7_online_shadow_feedback_validation_gradient_gate/stage1_online_shadow_feedback_weight_sweep
```

Code changes:

- Added `--shadow-feedback-validation-gradient-loss-weight`.
- Added `--shadow-feedback-validation-gradient-norm-weight`.
- Added differentiable validation model-gradient regularization for the online
  MLP shadow warmup.
- Split feature extraction from target construction so fixed online shadow
  feedback can train without recomputing boundary targets inside the training
  loop.
- Enabled `--shadow-feedback-mode online_mlp --shadow-feedback-weight > 0` by
  fitting the module once before Stage 1, saving
  `online_shadow_feedback_module.pt`, and applying it as fixed feedback.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
103 passed
```

Stage 0B diagnostics with target normalization, `injection_grad_logits`
features, `cosine` prediction loss, validation-gradient weight `0.5`, and
ordinary min-cosine validation selection:

| Hidden | Norm weight | Step | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `16` | `0.0` | `100` | `0.81317 / 0.81645` | `0.07823 / 0.04347` | `1.5099 / 1.3677` | norm concern |
| `16` | `0.1` | `100` | `0.80523 / 0.81281` | `0.09737 / 0.04914` | `1.3299 / 1.2108` | pass but high norm |
| `32` | `0.0` | `100` | `0.80489 / 0.80824` | `0.12059 / 0.13201` | `1.1647 / 1.1000` | pass |
| `32` | `0.1` | `100` | `0.80683 / 0.80826` | `0.12274 / 0.13430` | `1.1276 / 1.0736` | pass |

Stage 1 fixed-module early-lift smoke from the h32/norm `0.1` calibration:

| Shadow weight | Final exact match | Best snapshot exact | Main failure |
| ---: | ---: | ---: | --- |
| `1.0` | `0.075` | `0.0525` | feedback norm blow-up |
| `0.01` | `0.005` | `0.0400` | feedback norm blow-up |
| `0.001` | `0.035` | `0.0550` | feedback norm blow-up |

Decision:

```text
online_mlp_shadow_feedback_validation_gradient_stage0b_pass_stage1_fixed_module_negative
```

Interpretation:

- Direct validation model-gradient regularization solves the immediate Stage
  0B heldout gap problem for the online shadow module.
- A fixed calibrated module is not stable enough for Stage 1; model movement
  pushes the shadow features out of distribution and the feedback norm
  explodes.
- Next work should keep the direct gradient objective but make Stage 1
  feedback on-policy or trust-region constrained.

## 2026-05-28 Online MLP Shadow-Feedback Apply-Norm Clamp Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-twenty-seventh-task-Online-MLP-shadow-feedback-apply-norm-clamp-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_apply_norm_clamp_gate
```

Code changes:

- Added `--shadow-feedback-apply-max-norm`.
- Fixed online MLP shadow Stage 1 apply can now scale predicted feedback down
  to a maximum L2 norm.
- Training metrics now report applied feedback norm, unclamped feedback norm,
  and the apply scale.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
103 passed
```

Stage 1 fixed h32 validation-gradient module with `shadow_feedback_weight=1.0`:

| Apply max norm | Final exact match | Best snapshot exact | Final applied norm | Final unclamped norm | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| `3.5` | `0.075` | `0.0525` | `3.5` | `77802.8` | no lift |
| `10` | `0.075` | `0.0525` | `10.0` | `79123.9` | no lift |

Decision:

```text
online_mlp_shadow_feedback_apply_norm_clamp_stage1_negative
```

Interpretation:

- Simple output-vector L2 clamping prevents the obvious feedback norm blow-up.
- It does not fix stale fixed-module direction; Stage 1 remains below the
  `0.16` output-projection boundary-feedback baseline.
- Next work should use on-policy refresh or a trust-region gate that checks
  refreshed gradient agreement under model movement.

## 2026-05-28 Online MLP Shadow-Feedback On-Policy Refresh Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-twenty-eighth-task-Online-MLP-shadow-feedback-on-policy-refresh-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_on_policy_refresh_gate
```

Code changes:

- Added `--shadow-feedback-refresh-every`.
- Online MLP shadow Stage 1 can now periodically refit the shadow module
  against the current model.
- Refresh summaries are saved in
  `online_shadow_feedback_refresh_history.json`.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
103 passed
```

Stage 1 h32 validation-gradient module, `shadow_feedback_weight=1.0`, refresh
every `50`, no apply clamp:

| Refresh step | Heldout result/upstream cosine | Train-heldout gap |
| ---: | ---: | ---: |
| `0` | `0.8068 / 0.8083` | `0.1227 / 0.1343` |
| `50` | `0.9820 / 1.0000` | `0.0034 / 0.0000` |
| `100` | `0.9971 / 0.9999` | `0.0029 / 0.0001` |
| `150` | `0.9978 / 0.9991` | `0.0013 / 0.0008` |
| `200` | `0.9716 / 0.9997` | `0.0017 / 0.0001` |

Stage 1 result:

| Metric | Value |
| --- | ---: |
| final exact match | `0.025` |
| best snapshot exact match | `0.0475` |
| final shadow feedback norm | `27.627` |

Decision:

```text
online_mlp_shadow_feedback_on_policy_refresh_alignment_pass_stage1_negative
```

Interpretation:

- On-policy refresh solves stale gradient agreement.
- It does not solve the training dynamics; the model still collapses to a
  single learned result and stays below the `0.16` baseline.
- Next work should add step-level trust regions, entropy/diversity
  stabilization, or a richer target/state.

## 2026-05-28 Result-Policy Soft Diversity Stabilization Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-twenty-ninth-task-Result-policy-soft-diversity-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_shadow_refresh_result_policy_diversity_gate
```

Code changes:

- Added `--result-policy-entropy-weight`.
- Added `--result-policy-batch-diversity-weight`.
- Added `--result-policy-stabilization-temperature`.
- Added `--result-policy-stabilization-decay-steps`.
- Training curves now record soft entropy/effective results, batch-marginal
  effective results, hard-marginal effective results, and argmax/top-3 result
  accuracy.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
103 passed
```

Stage 1 refreshed h32 validation-gradient online shadow module:

| Entropy | Diversity | Apply clamp | Final exact | Best snapshot | Final hard effective results | Final soft marginal effective results |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.01` | `1.0` | none | `0.015` | `0.0475` | `1.00` | `1.00` |
| `0.01` | `1.0` | `10` | `0.005` | `0.0400` | `1.00` | `1.00` |
| `0.0` | `100.0` | `10` | `0.070` | `0.0800` | `9.14` | `35.11` |

Decision:

```text
result_policy_soft_diversity_stabilization_stage1_negative
```

Interpretation:

- Low soft diversity did not prevent hard single-result collapse.
- High soft diversity plus feedback clamp mechanically broadened result usage.
- Broader usage still did not align prompts to useful calculator results, and
  stayed below the `0.16` output-projection feedback baseline.
- Next work should use a hard/assignment-style usage constraint, a step-level
  trust region, Jacobian-conditioned state, or a richer target.

## 2026-05-28 Optimizer Step Trust Region Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-thirtieth-task-Optimizer-step-trust-region-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_shadow_refresh_optimizer_trust_region_gate
```

Code changes:

- Added `--optimizer-step-max-delta-norm`.
- When enabled, the training loop snapshots trainable parameters before
  `optimizer.step()`, computes the realized update L2 norm, and rescales the
  update back to the requested radius.
- Training curves record `optimizer_step_delta_l2`,
  `optimizer_step_unclamped_delta_l2`, `optimizer_step_trust_scale`, and
  `optimizer_step_max_delta_norm`.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
103 passed
```

Stage 1 refreshed h32 validation-gradient online shadow module,
`shadow_feedback_weight=1.0`, feedback clamp `10`:

| Max delta | Final exact | Best snapshot | Min/median/last trust scale | Final learned calc | Final shadow norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.05` | `0.075` | `0.060` | `0.251 / 0.300 / 0.259` | `0.0475` | `5.70` |
| `0.10` | `0.040` | `0.045` | `0.550 / 0.624 / 0.594` | `0.0325` | `10.00` |

Decision:

```text
optimizer_step_trust_region_stage1_negative
```

Interpretation:

- The trust region mechanically bounded actual AdamW movement.
- It stabilized shadow-feedback norms and maintained strong refresh
  agreement.
- Bounding parameter movement alone still did not lift above the `0.16`
  boundary-feedback baseline.
- Next work should use a trust region that validates per-step improvement, a
  hard/assignment-style usage constraint, Jacobian-conditioned state, or
  richer targets.

## 2026-05-28 Answer-Loss Step Acceptance Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-thirty-first-task-Answer-loss-step-acceptance-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_shadow_refresh_answer_loss_acceptance_gate
```

Code changes:

- Added `--optimizer-step-acceptance-mode`.
- Added `--optimizer-step-acceptance-tolerance`.
- `answer_loss_decrease` snapshots trainable parameters before the optimizer
  step and restores them if hard-path answer loss worsens beyond tolerance.
- Training curves now record acceptance before/after answer loss, delta,
  accepted flag, cumulative attempts, cumulative accepted count, and
  acceptance rate.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
103 passed
```

Stage 1 refreshed h32 validation-gradient online shadow module,
`shadow_feedback_weight=1.0`, feedback clamp `10`:

| Tolerance | Accepted steps | Final exact | Best snapshot | Final learned calc | Final shadow norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.0` | `6/200` (`3%`) | `0.050` | `0.070` | `0.0475` | `3.12` |
| `0.1` | `6/200` (`3%`) | `0.050` | `0.070` | `0.0450` | `3.12` |

Decision:

```text
answer_loss_step_acceptance_stage1_negative
```

Interpretation:

- Most refreshed-shadow proposed steps worsen real hard-path answer loss.
- Reverting those steps stabilizes the run but does not create calculator
  discovery.
- Next work needs to repair or construct useful directions, not merely reject
  bad ones; plausible branches are hard/assignment-style usage constraints,
  Jacobian-conditioned state, or richer targets.

## 2026-05-28 Answer-Loss Line Search Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-thirty-second-task-Answer-loss-line-search-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_shadow_refresh_answer_loss_line_search_gate
```

Code changes:

- Added `--optimizer-step-line-search-scales`.
- Added `answer_loss_line_search` as an optimizer step acceptance mode.
- The line-search mode snapshots the proposed optimizer update, evaluates
  configured scales of that fixed update under hard-path answer loss, and
  applies the best improving scale.
- Training curves now record the configured scales and selected scale.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
git diff --check
```

Result:

```text
103 passed
```

Stage 1 refreshed h32 validation-gradient online shadow module,
`shadow_feedback_weight=1.0`, feedback clamp `10`:

| Scales | Accepted steps | Final exact | Best snapshot | Final learned calc | Final shadow norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1,0.5,0.25,0.1,0` | `5/200` (`2.5%`) | `0.060` | `0.0925` | `0.0650` | `3.28` |

Decision:

```text
answer_loss_line_search_step_repair_stage1_negative
```

Interpretation:

- Hard answer-loss line search is a mild improvement over plain accept/reject,
  but still far below the `0.16` boundary-feedback baseline.
- Only `5/200` proposed refreshed-shadow steps had a useful positive scale.
- Step-size repair is not enough; the next branch should construct better
  directions, use hard/assignment-style usage constraints,
  Jacobian-conditioned state, or richer targets.

## 2026-05-28 Output-Jacobian Shadow Feature Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-thirty-third-task-Output-jacobian-shadow-feature-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_output_jacobian_shadow_feature_gate
```

Code changes:

- Added `injection_grad_logits_output_jacobian` as an online shadow feedback
  feature mode.
- The feature appends `J_output^T answer_grad` scores from the calculator
  output projection, giving one local sensitivity score per result class.
- Added a unit test that verifies the appended feature slice equals the
  output-projection transpose applied to the answer-loss injection gradient.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
104 passed
```

Stage 0B validation-gradient diagnostics:

| Hidden | Feature norm | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm |
| ---: | --- | ---: | ---: | ---: |
| `16` | none | `0.6703 / 0.7245` | `0.1013 / 0.0938` | `1.8176 / 1.7362` |
| `32` | none | `0.7957 / 0.8237` | `0.0994 / 0.1079` | `1.2553 / 1.1170` |
| `32` | `fit_zscore_per_feature` | `0.9073 / 0.9011` | `0.0639 / 0.0736` | `1.3044 / 1.2598` |

Stage 1 refreshed h32 feature-normalized smoke:

| Refresh | Clamp | Final exact | Best snapshot | Final learned calc | Final shadow norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `50` | `10` | `0.055` | `0.065` | `0.0475` | `10.00` |

Decision:

```text
output_jacobian_shadow_feature_stage0b_pass_stage1_negative
```

Interpretation:

- The output-Jacobian feature is a real Stage 0B improvement once feature
  z-scored.
- The refreshed Stage 1 failure persists despite current-model refresh
  cosines near `0.999`.
- A state-only Jacobian feature is therefore not enough; next work should use
  hard assignment-style usage constraints, richer targets, or a learned update
  path that changes the proposed direction rather than only describing state.

## 2026-05-28 Hard Improvement Assignment Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-thirty-fourth-task-Hard-improvement-assignment-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_hard_improvement_assignment_gate
```

Code changes:

- Added `--result-policy-improvement-assignment-weight`.
- Added `--result-policy-improvement-assignment-min-improvement`.
- Added `--result-policy-improvement-assignment-quota-multiplier`.
- Added a hard assignment target that chooses answer-loss-improving result
  classes under a per-result quota.
- Allowed result-policy stabilization weights to serve as a
  `direct_feedback_alignment` training objective without boundary/shadow
  feedback.
- Added unit coverage for quota-respecting assignment construction and CLI
  parsing.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
105 passed
```

Stage 1 200-step exact-grid smokes:

| Setup | Assignment weight | Final exact | Best snapshot | Assigned fraction | Target accuracy | Hard effective results |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| refreshed h32 shadow, clamp `10` | `1` | `0.0475` | `0.0650` | `0.9250` | `0.8189` | `1.00` |
| refreshed h32 shadow, clamp `10` | `10` | `0.1700` | `0.2425` | `0.7925` | `0.9117` | `14.12` |
| no shadow feedback | `10` | `0.4000` | `0.3500` | `0.6175` | `0.9474` | `18.85` |

Decision:

```text
hard_improvement_assignment_stage1_lift_partial
```

Interpretation:

- Hard assignment pressure is mechanically active and can lift Stage 1 above
  the `0.16` baseline.
- The no-shadow ablation being stronger means the lift comes primarily from
  the assignment target, not from refreshed shadow feedback.
- This is not yet scalable final success: the target scores forced result
  classes during training and still needs target-off retention, seed
  replication, longer convergence, and lower-cost approximations.

## 2026-05-28 Hard Improvement Assignment Retention Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-thirty-fifth-task-Hard-improvement-assignment-retention-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_hard_improvement_assignment_retention_gate
```

Question:

Can natural answer loss retain a result interface taught by hard improvement
assignment after the assignment weight decays to zero?

Configuration:

- no shadow feedback;
- assignment weight `10`;
- assignment min improvement `0`;
- assignment quota multiplier `1`;
- assignment decay steps `200`;
- answer loss weight `1`;
- 400 total steps.

Result:

| Step | Assignment weight | Snapshot exact | Result-policy accuracy | Hard effective results |
| ---: | ---: | ---: | ---: | ---: |
| `100` | `5.0` | `0.2700` | `0.2650` | `18.30` |
| `175` | `1.25` | `0.3700` | n/a | n/a |
| `200` | `0.0` | `0.3475` | `0.3575` | `18.54` |
| `250` | `0.0` | `0.1050` | `0.0975` | `8.78` |
| `400` | `0.0` | `0.1050` | `0.0975` | `8.73` |

Final eval exact:

```text
0.1075
```

Decision:

```text
hard_improvement_assignment_decay_retention_negative
```

Interpretation:

- Assignment pressure still teaches the interface during the first 200 steps.
- Plain answer loss does not retain that interface after the assignment target
  decays away.
- Do not treat hard improvement assignment as solved; next work should test
  longer always-on convergence, seed replication, a stronger handoff bridge,
  and lower-cost assignment approximations.

## 2026-05-28 Hard Improvement Assignment Convergence Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-thirty-sixth-task-Hard-improvement-assignment-convergence-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_hard_improvement_assignment_convergence_gate
```

Question:

If the hard improvement-assignment objective stays on, does the natural
result-space calculator interface keep improving, and does that improvement
replicate across seeds?

Configuration:

- no shadow feedback;
- assignment weight `10`;
- assignment min improvement `0`;
- assignment quota multiplier `1`;
- answer loss weight `0` for the 1600-step seed sweep;
- frozen product semantic decoder;
- exact-grid natural `0..19` batch;
- 800 or 1600 total steps.

Result:

| CLI seed | Steps | Final eval exact | Best snapshot | Last snapshot | Last result-policy acc | Last injection-zero | Last oracle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2` | `800` | `0.860` | `0.8525` at `800` | `0.8525` | `0.8350` | `0.0375` | `1.0000` |
| `2` | `1600` | `0.915` | `0.9475` at `1300` | `0.9150` | `0.9025` | `0.0375` | `1.0000` |
| `4` | `1600` | `0.860` | `0.8700` at `1600` | `0.8700` | `0.8450` | `0.0575` | `1.0000` |
| `5` | `1600` | `0.820` | `0.9200` at `1500` | `0.8325` | `0.8250` | `0.0575` | `1.0000` |

Answer-loss ablation:

The 800-step seed-2 runs with `answer_loss_weight=0` and `1` were numerically
identical on the reported curve and both ended at `0.860` final eval exact.
In this setup, the natural answer-loss gradient is not visibly changing the
discrete result policy while assignment remains on.

Decision:

```text
hard_improvement_assignment_convergence_seed_replication_mixed_partial
```

Interpretation:

- Always-on hard improvement assignment can train a learned natural
  result-space calculator interface from scratch across multiple seeds.
- The path is not stable enough to call solved: seed `5` peaked high and then
  drifted down, and the earlier target-off decay run collapsed after the
  assignment objective reached zero.
- This is not yet scalable or non-prescriptive. The assignment target scores
  forced result classes during training, so next work should focus on cheaper
  assignment approximations, better handoff/retention, stability selection,
  and the non-bottleneck setting.

## 2026-05-28 Non-Bottleneck Hard Assignment Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-thirty-seventh-task-Non-bottleneck-hard-assignment-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_non_bottleneck_hard_assignment_gate
```

Code change:

- Allowed `calculator_action_head=result_space` with `calculator_estimator=ste`
  so additive non-bottleneck result-space runs can use answer loss and
  result-policy stabilization without the strict answer decoder.

Question:

Does the hard improvement-assignment signal that works in the answer-decoder
bottleneck transfer to an additive non-bottleneck model, where a normal neuron
path can also solve the task?

Configuration:

- additive path: `calculator_bottleneck_mode=none`;
- `calculator_estimator=ste`;
- `calculator_action_head=result_space`;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- CLI seed `2`, 800 steps.

Result:

| Setup | Final eval exact | Best normal snapshot | Best/last injection-zero | Last learned calc | Best result-policy acc | Last assignment target acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| answer loss only | `0.615` | `0.9725` at `600` | `0.560 / 0.3575` | `0.0250` | n/a | n/a |
| answer loss + assignment `10` | `0.700` | `0.8200` at `650` | `0.740 / 0.6500` | `0.0325` | `0.0575` | `0.0033` |

Decision:

```text
non_bottleneck_hard_assignment_transfer_negative
```

Interpretation:

- The additive model can improve answer accuracy through the normal path, as
  shown by high injection-zero accuracy and near-chance calculator-result
  accuracy.
- Hard assignment does not rescue calculator use in this setting. The forced
  result-class landscape becomes a bad teacher when the neuron path can
  bypass the calculator: assignment target accuracy fell to `0.0033` by step
  `800`.
- Do not treat the bottleneck assignment result as a non-bottleneck result.
  Future non-bottleneck work needs explicit causal calculator-use pressure,
  staged bottleneck-to-additive handoff, or a target construction that remains
  valid under bypass.

## 2026-05-28 Non-Bottleneck Causal Gap Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-thirty-eighth-task-Non-bottleneck-causal-gap-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_non_bottleneck_causal_gap_gate
```

Code change:

- Added `--calculator-causal-gap-weight`.
- Added `--calculator-causal-gap-margin`.
- The training curve now logs `calculator_causal_gap`,
  `calculator_causal_gap_objective`, `calculator_causal_gap_zero_loss`, and
  `calculator_causal_gap_normal_loss`.

Question:

Can a cheap, non-prescriptive zero-injection causal-use hinge rescue additive
non-bottleneck hard assignment?

Configuration:

- additive path: `calculator_bottleneck_mode=none`;
- `calculator_estimator=ste`;
- `calculator_action_head=result_space`;
- answer loss weight `1`;
- assignment weight `10`;
- causal-gap margin `0.5`;
- causal-gap weights `10` and `50`;
- exact-grid natural `0..19`;
- CLI seed `2`, 800 steps.

Result:

| Setup | Final eval exact | Best normal snapshot | Last zero-injection | Last learned calc | Best result-policy acc | Last causal gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| assignment `10`, no gap | `0.700` | `0.8200` at `650` | `0.6500` | `0.0325` | `0.0575` | n/a |
| assignment `10`, gap weight `10` | `0.560` | `0.5750` at `750` | `0.3375` | `0.0000` | `0.0300` | `1.2717` |
| assignment `10`, gap weight `50` | `0.4225` | `0.4800` at `750` | `0.2750` | `0.0425` | `0.0450` | `0.8372` |

Decision:

```text
non_bottleneck_causal_gap_pressure_negative
```

Interpretation:

- The hinge can mechanically create a causal gap by making zero-injection
  behavior worse.
- It does not by itself teach correct calculator requests: learned
  calculator-result accuracy and result-policy accuracy remain near chance.
- Non-bottleneck progress needs a causal target tied to correct result-level
  utility, or a staged bottleneck-to-additive handoff, not merely pressure for
  the calculator path to matter.

## 2026-05-28 Bottleneck-to-Additive Transfer Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-thirty-ninth-task-Bottleneck-to-additive-transfer-gate.md
```

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_gate
```

Code changes:

- Added `--semantic-decoder-checkpoint-load-scope compatible_model`, which
  loads only tensors whose names exist in the target model and whose shapes
  match. This allows a bottleneck checkpoint to initialize an additive model
  while skipping incompatible modules such as `answer_decoder`.
- Added `--freeze-calculator-policy`, which freezes embeddings, pre-hook
  transformer blocks, and the calculator action head while leaving the
  calculator output projection, post-hook block, final norm, and answer head
  trainable.

Question:

Can a natural result policy trained in the answer-decoder bottleneck be handed
to an additive non-bottleneck model, and can the downstream additive path learn
to depend on the calculator instead of the bypass?

Source checkpoint:

```text
runs/2026-05-28_phase7_hard_improvement_assignment_convergence_gate/answer0_w10_steps1600/2026-05-28_164332_598334_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed4/final_weights.pt
```

Configuration:

- additive path: `calculator_bottleneck_mode=none`;
- `calculator_estimator=ste`;
- `calculator_action_head=result_space`;
- compatible load from the bottleneck final checkpoint above;
- answer loss weight `1`;
- no assignment target;
- exact-grid natural `0..19`;
- CLI seed `2`, 800 steps.

Result:

| Setup | Final eval exact | Best normal snapshot | Last injection-zero | Last forced-random | Last oracle | Last learned calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| compatible load, no freeze | `0.7825` | `0.8075` at `800` | `0.7675` | `0.7375` | `0.7825` | `0.0250` |
| compatible load, freeze policy | `0.9400` | `0.9475` at `800` | `0.0175` | `0.0500` | `0.9600` | `0.9200` |

Policy-retention trace:

| Setup | Step `0` learned calc | Step `50` learned calc | Step `800` learned calc |
| --- | ---: | ---: | ---: |
| compatible load, no freeze | `0.9125` | `0.0300` | `0.0250` |
| compatible load, freeze policy | `0.9125` | `0.8925` | `0.9200` |

Decision:

```text
bottleneck_to_additive_freeze_policy_handoff_partial_positive
```

Interpretation:

- The unfrozen run proves that compatible loading works but answer-only
  additive training immediately destroys the transferred calculator policy.
- Freezing the policy creates strong non-bottleneck calculator dependence:
  normal accuracy is high, injection-zero and forced-random are near chance,
  oracle is high, and learned result accuracy remains high.
- This is not yet the final goal because the policy is inherited from a
  bottleneck-trained, forced-assignment phase and then frozen. Next work should
  replicate across seeds/checkpoints, test staged unfreezing, and find a
  scalable or less prescriptive way to acquire and preserve the policy.

## 2026-05-28 Bottleneck-to-Additive Transfer Replication

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-fortieth-task-Bottleneck-to-additive-transfer-replication.md
```

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_replication
```

Question:

Does the frozen-policy bottleneck-to-additive handoff replicate across additive
seeds and source bottleneck checkpoints?

Configuration:

- additive path: `calculator_bottleneck_mode=none`;
- `calculator_estimator=ste`;
- `calculator_action_head=result_space`;
- compatible load from bottleneck hard-assignment checkpoints;
- `--freeze-calculator-policy`;
- answer loss weight `1`;
- no assignment target;
- exact-grid natural `0..19`;
- 800 steps.

Result:

| Cell | Final eval | Best normal | Last injection-zero | Last forced-random | Last oracle | Step 0 learned calc | Last learned calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src2_add2` | `0.9400` | `0.9475` at `800` | `0.0175` | `0.0500` | `0.9600` | `0.9125` | `0.9200` |
| `src2_add4` | `0.9525` | `0.9325` at `750` | `0.0200` | `0.0425` | `0.9600` | `0.9000` | `0.9150` |
| `src4_add2` | `0.3025` | `0.3150` at `800` | `0.0000` | `0.0375` | `0.3125` | `0.8650` | `0.8725` |
| `src4_add4` | `0.3375` | `0.3200` at `750` | `0.0000` | `0.0375` | `0.3075` | `0.8325` | `0.8575` |
| `src5_add5` | `0.5550` | `0.5725` at `800` | `0.0125` | `0.0425` | `0.5600` | `0.8475` | `0.8000` |

Decision:

```text
bottleneck_to_additive_freeze_policy_source_quality_mixed
```

Interpretation:

- The strong source checkpoint from the seed-2 1600-step bottleneck run
  replicated across additive seeds. This strengthens the claim that frozen
  handoff can create real non-bottleneck calculator dependence.
- Weaker source checkpoints preserved action accuracy after freezing but did
  not produce high answer accuracy by step `800`. For `src4`, oracle stayed
  near normal around `0.31`, suggesting the downstream/readout side did not
  acquire a broadly useful result representation even when the frozen action
  policy often selected the correct result.
- Frozen handoff is therefore source-quality sensitive. Next work should test
  source checkpoint selection/quality metrics, stronger downstream readout
  adaptation, and controlled unfreezing. Do not repeat these exact matrix cells
  as novelty.

## 2026-05-28 Bottleneck-to-Additive Downstream Adaptation Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-forty-first-task-Bottleneck-to-additive-downstream-adaptation.md
```

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation
```

Question:

Do weak-source frozen handoffs fail because the source representation is
unusable, or because downstream additive adaptation needs more optimization
time?

Configuration:

- Continued from the additive final weights of weak-source frozen handoffs.
- Used `--semantic-decoder-checkpoint-load-scope full_model` to resume the
  additive model.
- Kept `--freeze-calculator-policy` on.
- answer loss weight `1`;
- exact-grid natural `0..19`;
- 800 additional steps.

Result:

| Run | Final eval before | Final eval after | Best normal after | Last injection-zero | Last forced-random | Last oracle | Last learned calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` continue | `0.3025` | `0.6050` | `0.5725` at `550` | `0.0025` | `0.0625` | `0.5725` | `0.8725` |
| `src5_add5` continue | `0.5550` | `0.8175` | `0.8150` at `800` | `0.0000` | `0.0425` | `0.8075` | `0.8000` |

Decision:

```text
bottleneck_to_additive_longer_downstream_adaptation_partial
```

Interpretation:

- Weak-source handoffs are not hard failures. Longer downstream adaptation can
  substantially improve answer accuracy while preserving causal calculator
  dependence.
- The source-quality issue is still real. After the same total 1600-step
  additive adaptation budget, the weak sources remained below the strong
  source handoffs (`~0.95` final eval).
- Next work should test source checkpoint selection, better downstream
  adaptation objectives, or controlled unfreezing rather than merely repeating
  one more longer frozen continuation.

## 2026-05-28 Bottleneck-to-Additive Low-LR Unfreeze Probe

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-forty-second-task-Bottleneck-to-additive-low-lr-unfreeze.md
```

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_unfreeze_probe
```

Question:

After downstream adaptation has learned to use a frozen calculator path, can a
low-LR full-policy unfreeze improve or preserve the handoff?

Configuration:

- Continued from the adapted weak-source checkpoints.
- Loaded with `--semantic-decoder-checkpoint-load-scope full_model`.
- Removed `--freeze-calculator-policy`.
- global LR `3e-4`;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- 400 steps.

Result:

| Run | Final eval before | Final eval after | Best normal after | Last injection-zero | Last forced-random | Last oracle | Learned calc before | Learned calc after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` unfreeze | `0.6050` | `0.5200` | `0.6500` at `0` | `0.0225` | `0.1200` | `0.7100` | `0.8725` | `0.3000` |
| `src5_add5` unfreeze | `0.8175` | `0.8100` | `0.8325` at `0` | `0.0225` | `0.1125` | `0.7900` | `0.8000` | `0.2525` |

Decision:

```text
bottleneck_to_additive_low_lr_unfreeze_policy_collapse_negative
```

Interpretation:

- Low-LR full unfreeze does not preserve the learned calculator policy.
- Normal accuracy can partly survive through the already-trained downstream
  path, but learned calculator-result accuracy collapses and forced-random
  accuracy rises.
- Future unfreezing work needs selective parameter movement, explicit policy
  retention, or a gate that monitors calculator-result accuracy during
  unfreeze.

## 2026-05-28 Bottleneck-to-Additive Policy-Anchor Unfreeze Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-forty-third-task-Bottleneck-to-additive-policy-anchor-unfreeze.md
```

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_unfreeze
```

Code change:

- Added `--result-policy-anchor-weight`.
- Added `--result-policy-anchor-decay-steps`.
- Added `--result-policy-anchor-temperature`.
- Added `--result-policy-anchor-mode {kl,mse}`.
- The anchor snapshots the initial fixed-grid result-space policy and applies
  a KL or logit-MSE drift penalty during training. It currently requires
  `--exhaustive-grid-batch` so anchor rows stay fixed.

Question:

Can explicit result-policy anchoring prevent the policy collapse seen under
plain low-LR full unfreezing while still allowing downstream/non-bottleneck
adaptation?

Configuration:

- Continued from the adapted weak-source checkpoints.
- Loaded with `--semantic-decoder-checkpoint-load-scope full_model`.
- Removed `--freeze-calculator-policy`.
- global LR `3e-4`;
- `--result-policy-anchor-weight 10`;
- `--result-policy-anchor-mode kl`;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- 400 steps.

Result:

| Run | Frozen adapted final | Plain unfreeze final | Anchored final | Anchored best normal | Last injection-zero | Last forced-random | Last oracle | Last learned calc | Last anchor agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` anchor | `0.6050` | `0.5200` | `0.7475` | `0.7375` at `150` | `0.0100` | `0.0950` | `0.7875` | `0.8075` | `0.9800` |
| `src5_add5` anchor | `0.8175` | `0.8100` | `0.9525` | `0.9650` at `400` | `0.0000` | `0.0450` | `0.9375` | `0.7950` | `0.9850` |

Decision:

```text
bottleneck_to_additive_policy_anchor_unfreeze_partial
```

Interpretation:

- The result-policy anchor prevented the learned calculator-result collapse
  that occurred under plain low-LR full unfreeze.
- Anchored unfreeze improved both weak-source adapted handoffs, and `src5_add5`
  reached the strong-source accuracy band while preserving calculator
  dependence.
- This is still not the final project goal because it uses a staged and
  anchored policy. Next work should test an anchor off-ramp/decay, selective
  unfreezing, or less prescriptive source-policy acquisition.

## 2026-05-28 Bottleneck-to-Additive Anchor Decay Off-Ramp Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-forty-fourth-task-Bottleneck-to-additive-anchor-decay-offramp.md
```

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_decay_unfreeze
```

Question:

Can the constant-anchor controlled-unfreeze result become self-sustaining if
the KL anchor is removed halfway through training?

Configuration:

- Continued from the adapted weak-source checkpoints.
- Loaded with `--semantic-decoder-checkpoint-load-scope full_model`.
- Removed `--freeze-calculator-policy`.
- global LR `3e-4`;
- `--result-policy-anchor-weight 10`;
- `--result-policy-anchor-decay-steps 200`;
- `--result-policy-anchor-mode kl`;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- 400 steps.

Result:

| Run | Frozen adapted final | Constant-anchor final | Decay final | Decay best normal | Step-200 calc | Final calc | Final anchor agree | Final injection-zero | Final forced-random | Final oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` decay200 | `0.6050` | `0.7475` | `0.5925` | `0.7250` at `250` | `0.8300` | `0.5950` | `0.6200` | `0.0325` | `0.0975` | `0.7400` |
| `src5_add5` decay200 | `0.8175` | `0.9525` | `0.6750` | `0.9575` at `200` | `0.8225` | `0.3850` | `0.4300` | `0.0375` | `0.0800` | `0.7925` |

Decision:

```text
bottleneck_to_additive_anchor_decay_offramp_negative
```

Interpretation:

- The policies were still useful when the anchor reached zero at step `200`.
- During the following no-anchor tail, both policies drifted substantially;
  the final calculator-result accuracy was much worse than constant anchoring.
- Constant anchoring is therefore doing real policy-retention work. A fast
  linear off-ramp is not enough to make the adapted non-bottleneck calculator
  policy self-sustaining.
- Next work should test slower/floored/gated anchors, selective unfreezing, or
  less prescriptive source-policy acquisition rather than repeating this decay.

## 2026-05-28 Bottleneck-to-Additive Reduced Anchor Strength Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-forty-fifth-task-Bottleneck-to-additive-reduced-anchor-strength.md
```

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_strength_unfreeze
```

Question:

Does non-bottleneck full-policy unfreeze require a large constant anchor, or
can a weaker policy-retention term preserve calculator use?

Configuration:

- Continued from the adapted weak-source checkpoints.
- Loaded with `--semantic-decoder-checkpoint-load-scope full_model`.
- Removed `--freeze-calculator-policy`.
- global LR `3e-4`;
- `--result-policy-anchor-mode kl`;
- anchor weights `1.0` and `0.1`;
- no anchor decay;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- 400 steps.

Result:

| Run | Frozen adapted final | Anchor-10 final | Reduced-anchor final | Best normal | Final injection-zero | Final forced-random | Final oracle | Final calc | Final anchor agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` anchor `1.0` | `0.6050` | `0.7475` | `0.7775` | `0.7550` at `400` | `0.0225` | `0.0950` | `0.8025` | `0.8050` | `0.9625` |
| `src5_add5` anchor `1.0` | `0.8175` | `0.9525` | `0.9925` | `0.9825` at `400` | `0.0075` | `0.0475` | `0.9350` | `0.7925` | `0.9625` |
| `src4_add2` anchor `0.1` | `0.6050` | `0.7475` | `0.8325` | `0.8275` at `400` | `0.0250` | `0.1100` | `0.8650` | `0.8075` | `0.9225` |
| `src5_add5` anchor `0.1` | `0.8175` | `0.9525` | `0.9750` | `0.9700` at `400` | `0.0000` | `0.0525` | `0.9150` | `0.7725` | `0.9075` |

Decision:

```text
bottleneck_to_additive_reduced_anchor_strength_partial
```

Interpretation:

- Constant anchors `1.0` and `0.1` preserved useful calculator-result
  accuracy, unlike no-anchor unfreeze and unlike the fast shutoff tail.
- Lower anchor strength did not merely preserve policy; it improved final eval
  over the original anchor-10 cells in this two-cell gate.
- Injection-zero stayed near chance, so the answers remained calculator
  dependent.
- This improves the scalability story for retention regularization, but it is
  still staged and actively anchored. It is not an anchor-free or from-scratch
  non-bottleneck solution.

## 2026-05-28 Bottleneck-to-Additive Anchor Threshold Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-forty-sixth-task-Bottleneck-to-additive-anchor-threshold.md
```

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_threshold_unfreeze
```

Question:

Is constant KL anchor `0.01` enough to preserve the transferred
non-bottleneck calculator policy, or is it below the useful retention
threshold?

Configuration:

- Continued from the adapted weak-source checkpoints.
- Loaded with `--semantic-decoder-checkpoint-load-scope full_model`.
- Removed `--freeze-calculator-policy`.
- global LR `3e-4`;
- `--result-policy-anchor-mode kl`;
- anchor weight `0.01`;
- no anchor decay;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- 400 steps.

Result:

| Run | Frozen adapted final | Anchor-0.1 final | Anchor-0.01 final | Best normal | Final injection-zero | Final forced-random | Final oracle | Final calc | Final anchor agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` anchor `0.01` | `0.6050` | `0.8325` | `0.7850` | `0.7850` at `400` | `0.0050` | `0.1250` | `0.8200` | `0.7625` | `0.8825` |
| `src5_add5` anchor `0.01` | `0.8175` | `0.9750` | `0.9375` | `0.9250` at `400` | `0.0000` | `0.0950` | `0.9525` | `0.6425` | `0.7050` |

Decision:

```text
bottleneck_to_additive_anchor_0p01_threshold_mixed
```

Interpretation:

- Anchor `0.01` avoided the full no-anchor collapse and kept injection-zero
  near chance, so answers were still calculator dependent.
- It did not cleanly preserve the transferred policy. Final calculator-result
  accuracy and anchor agreement were materially worse than with anchor `0.1`,
  especially for `src5_add5`.
- Future schedules should treat `0.1` as the first plausible lightweight floor
  in this setup, not assume anchor weights can be reduced to near zero.

## 2026-05-28 Bottleneck-to-Additive Anchor Floor Schedule Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-forty-seventh-task-Bottleneck-to-additive-anchor-floor-schedule.md
```

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_floor_unfreeze
```

Code change:

- Added `--result-policy-anchor-floor`.
- The result-policy anchor schedule can now decay to a nonzero floor instead
  of only to zero.
- Added a focused schedule unit test.

Question:

Can a floored schedule keep the useful lightweight retention of anchor `0.1`
while allowing early stronger protection and avoiding the failed zero-off-ramp?

Configuration:

- Continued from the adapted weak-source checkpoints.
- Loaded with `--semantic-decoder-checkpoint-load-scope full_model`.
- Removed `--freeze-calculator-policy`.
- global LR `3e-4`;
- `--result-policy-anchor-mode kl`;
- anchor weight `1.0`;
- `--result-policy-anchor-decay-steps 200`;
- `--result-policy-anchor-floor 0.1`;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- 400 steps.

Result:

| Run | Frozen adapted final | Anchor-0.1 final | Floor-schedule final | Best normal | Final injection-zero | Final forced-random | Final oracle | Final calc | Final anchor weight | Final anchor agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` floor `0.1` | `0.6050` | `0.8325` | `0.7925` | `0.7725` at `200` | `0.0250` | `0.1000` | `0.8000` | `0.8175` | `0.1000` | `0.9225` |
| `src5_add5` floor `0.1` | `0.8175` | `0.9750` | `0.9775` | `0.9650` at `350` | `0.0075` | `0.0550` | `0.9300` | `0.7800` | `0.1000` | `0.8975` |

Decision:

```text
bottleneck_to_additive_anchor_floor_schedule_partial
```

Interpretation:

- The nonzero floor avoided the policy-collapse pattern seen when the anchor
  decayed to zero.
- It preserved calculator dependence and useful calculator-result accuracy in
  both cells.
- It did not outperform constant anchor `0.1` in this two-cell gate, so the
  floor mechanism is useful scheduling infrastructure rather than a new
  performance breakthrough.
- Next work should prefer calculator-accuracy-gated retention, adaptive floors,
  selective unfreezing, or less prescriptive source-policy acquisition.

## 2026-05-28 Bottleneck-to-Additive Freeze Action Head Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-forty-eighth-task-Bottleneck-to-additive-freeze-action-head.md
```

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_selective_unfreeze
```

Code change:

- Added `--freeze-calculator-action-head`.
- For `calculator_action_head=result_space`, this freezes only
  `calculator_hook.result_proj`.
- Added a focused unit test verifying the action head is frozen while
  surrounding model parameters remain trainable.

Question:

Is the result-projection head itself the fragile part of policy collapse, or
can upstream representation drift destroy the transferred policy even if the
head is frozen?

Configuration:

- Continued from the adapted weak-source checkpoints.
- Loaded with `--semantic-decoder-checkpoint-load-scope full_model`.
- Removed `--freeze-calculator-policy`.
- Added `--freeze-calculator-action-head`.
- No result-policy anchor.
- global LR `3e-4`;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- 400 steps.

Result:

| Run | Frozen adapted final | Plain unfreeze final | Freeze-action-head final | Best normal | Final injection-zero | Final forced-random | Final oracle | Final calc | Trainable groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `src4_add2` freeze action head | `0.6050` | `0.5200` | `0.5200` | `0.6500` at `0` | `0.0225` | `0.1200` | `0.7100` | `0.3000` | `upstream` |
| `src5_add5` freeze action head | `0.8175` | `0.8100` | `0.8100` | `0.8325` at `0` | `0.0225` | `0.1125` | `0.7900` | `0.2525` | `upstream` |

Decision:

```text
bottleneck_to_additive_freeze_action_head_unfreeze_negative
```

Interpretation:

- Freezing only `result_proj` did not preserve the learned calculator-result
  policy.
- The final metrics matched the earlier low-LR no-anchor unfreeze collapse,
  while metrics reported only `upstream` as trainable.
- Upstream representation drift alone is sufficient to break the transferred
  policy. Behavior-level retention, full policy-path freezing, or gated
  anchoring remains necessary.
