# Phase 7 Direction Memory

Status: active synthesis
Last updated: 2026-05-31

This file consolidates Phase 7 lessons by research direction. It is meant to
be easier to retrieve than the chronological fact sheet.

## Central Lesson

Natural `0..19` result-level calculator use is not blocked by architecture or
downstream decoding. It is blocked by scalable credit assignment into the
calculator-query policy.

The strongest positives use scaffolding, assignment, or staged transfer. The
missing result is non-prescriptive from-scratch discovery.

## Direction: Oracle And Wiring Checks

Status: paused

Memory:

- Correct calculator outputs or oracle queries let downstream answer layers
  solve the task.
- This is a wiring/control result only.
- Do not present oracle success, oracle-at-eval recovery, injection-zero, or
  forced-random checks as strategic progress except when validating new wiring.

Representative evidence:

- `CLAUDE.md` guardrails
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`

## Direction: Generic Retention After Teaching

Status: paused

Memory:

- Earlier phases established that identifiable/scaffolded protocols can often
  be retained after direct supervision or local targets are removed.
- Phase 7 target-off work is only strategic when it tests a new interface or a
  real stability question.

Representative evidence:

- `RESEARCH_STATE.md`
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`

## Direction: Plain Answer-Loss Discovery

Status: paused without a new mechanism

Memory:

- Vanilla result-space REINFORCE failed because its gradient aligned with the
  raw expected-cost gradient, and that gradient anti-aligned with the boundary
  ceiling.
- Exact result-marginal answer loss did not fix the direction.
- Rank-normalizing forced-result costs only weakly flipped the result-head
  cosine and left upstream cosine essentially zero, so it did not justify
  Stage 1.
- Decoder calibration improved local signs but still collapsed in Stage 1.
- Conclusion: the failure is not mainly finite-sample variance or a slightly
  weak decoder.

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-14-exact-result-marginal-answer-loss-gradient-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-rank-normalized-expected-loss-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-14-gradient-friendly-result-decoder-alignment-gate.md`
- `HYPOTHESIS_LEDGER.md`

## Direction: Direct Feedback And Shadow Gradients

Status: paused without a new dynamics mechanism

Memory:

- Output-projection direct feedback passed a local Stage 0 alignment gate but
  failed Stage 1 discovery.
- Fixed fit-once linear shadow feedback produced excellent same-batch alignment
  but failed early lift and failed heldout generalization.
- Many simple online MLP shadow variants improved heldout alignment metrics but
  did not create useful training dynamics.
- Stage 0 gradient alignment alone is not a sufficient go signal.

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-boundary-feedback-gradient-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-28-linear-shadow-feedback-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-gradient-gate.md`
- `HYPOTHESIS_LEDGER.md`

## Direction: Hard Improvement Assignment

Status: active but constrained

Memory:

- Hard answer-loss improvement assignment is the strongest bottleneck source
  training signal so far.
- Always-on assignment can train natural result-level calculator policies.
- Plain target-off decay failed, and the method is expensive/prescriptive
  because it scores candidate calculator results.
- A direct uniform-sampling cost reduction is negative: on the op19 `rhead64`
  200-step source gate, exact assignment scored `39/39` results and reached
  best snapshot `0.8625`, while sample16 reached only `0.3650` and sample32
  only `0.4050`. Step-200 true coverage and target accuracy were too low
  (`0.6125`/`0.4581` for sample16, `0.7400`/`0.6773` for sample32), and wall
  time savings were modest.
- Fixed-cadence exact assignment refresh is also mixed-negative. Refresh2 and
  refresh5 reduce full assignment refreshes but reach only `0.5875` and
  `0.4950` best snapshot normal/calc on the same op19 `rhead64` gate, versus
  `0.8625` exact, with little full-run wall-clock gain.
- Unique sampled assignment is mixed-positive but below ceiling. Removing
  duplicate candidates lifts sample32 from `0.4050` to `0.6250` best snapshot
  and step-200 true coverage from `0.7400` to `0.9275`, but unique32 still
  scores most of the vocabulary and misses exact (`0.8625` best snapshot).
- Policy-aware top-k proposals are the first lower-cost positive. Topk8 plus
  unique random candidates reached true coverage `1.0000` at `16/39`, `24/39`,
  and `32/39` scored classes; topk8+unique24 reached final `0.7500` and
  topk8+unique32 final `0.8600`, versus exact final `0.7350`.
- The first staged validation is positive: topk8+unique24 source630 reached
  `1.0000` final/source calc while scoring `24/39` result classes, and the
  trusted frozen-policy additive handoff reached `1.0000` final / step-600
  normal with low zero/random controls.
- Fresh-seed staged validation replicated: CLI seed `45` / effective seed `47`
  also reached `1.0000` source final and `1.0000` trusted handoff final, with
  step-600 handoff learned calc `1.0000`, injection-zero `0.0475`, and
  forced-random `0.0250`.
- Op29 range validation is positive on the exact-ceiling seed: topk8+unique24
  scored `24/59` result classes yet reached `1.0000` source final and
  `1.0000` trusted handoff final, with step-600 handoff learned calc `1.0000`,
  injection-zero `0.0356`, and forced-random `0.0189`.
- Op29 range validation replicated on effective seed `33`: sparse source final
  was `899/900 = 0.9989`, trusted handoff final was `900/900 = 1.0000`, and
  step-600 handoff controls stayed causal (`0.0333` injection-zero, `0.0111`
  forced-random, `0.9989` learned calc).
- Many-calculator accounting clarifies what the policy-aware proposal does and
  does not solve. Under the current single-hook design, independent calculators
  would multiply scorer cost by active calculator count: for op29 over 630
  assignment steps, exact scoring is `33,453,000` forced evaluations per
  calculator and topk8+unique24 is `13,608,000`; at 16 calculators that becomes
  `535,248,000` versus `217,728,000`. Result-head parameters also scale
  linearly if each calculator has an independent `rhead64` head (`12,091` each
  at op29). This supports topk as a scorer-cost baseline, not as a complete
  many-calculator architecture.
- A same-layer multi-hook forward path now exists: `GPTConfig.calculator_hook_count`
  instantiates independent calculator hooks, diagnostics report
  `calculator_active_hook_count` and per-hook injections, and the training CLI
  accepts `--calculator-hook-count`. A zero-step smoke with `hook_count=3`
  wrote matching config/metrics. This is prerequisite tooling, not evidence
  that routed/differentiated many-calculator policies can train.
- A first routed variant also exists: `left_operand_mod` activates one hook per
  example by final left-operand digit modulo hook count, reports route IDs and
  counts, and records routing in training config/metrics. It enables a
  task-partitioned diagnostic but still has no per-hook training result.
- Routed diagnostic snapshots now report active-route distribution and per-hook
  quality fields (`hook_{i}_route_count`, `hook_{i}_calculator_result_accuracy`,
  etc.). This makes the next routed training run measurable but is not itself
  evidence of specialization.
- Routed source training exposed a semantic-interface prerequisite. Uncloned
  extra hooks kept random frozen output projections and collapsed hook 1 despite
  balanced routes. Cloning the primary output projection into extra hooks made
  exact route targets accurate and let topk8+unique24 train both hooks on op19:
  step-200 normal `0.9250`, hook calc `0.9315/0.9171`, scored `24/39` results.
  This is source-only and still needs handoff/fresh-seed validation.
- The next routed gate exposed leakage. Strict handoff from the 200-step routed
  source had high normal but high injection-zero (`0.4925`). A fair `embd32`
  routed source630 trained both hooks (`1.0000/0.9944`) but still had high
  injection-zero (`0.4600`), while frozen-upstream source200 reduced leakage
  (`0.1875`) only while undertrained (`0.4150` normal). Extending frozen
  upstream to source630 recovered learning (`0.9750` normal, `0.9955/0.9494`
  hook calc) but injection-zero returned (`0.4400` snapshot, `0.5000` final
  counterfactual). This leakage conclusion was superseded by a control bug:
  the temporary injection-scale helper zeroed only the primary hook. After
  scaling every hook, source200 rerun had `0.9225` normal / `0.0200` zero,
  source630 reload had `0.9950` normal / `0.0250` zero, and strict handoff600
  reload had `0.9250` normal / `0.0000` zero. Routed source/handoff are causal
  under corrected controls. The stronger routed `embd32` source630 then cleared
  a trusted 600-step additive handoff: final/step-600 normal `1.0000`, step-600
  injection-zero `0.0550`, forced-random `0.0300`, and hook calc
  `1.0000/0.9955`. A four-hook routed stress also cleared source and handoff:
  source630 `0.9950` final / `0.0275` zero, handoff600 `1.0000` final /
  `0.0400` zero, all four hooks perfect on the handoff snapshot.
- Active-only routed execution is now implemented for both model forward hooks
  and source-training result-logit reads. With `left_operand_mod` routing, the
  model invokes only hooks that have examples in the batch, scatters their
  traces/injections back into full-batch diagnostics, and records both
  configured (`calculator_active_hook_count`) and invoked
  (`calculator_invoked_hook_count`) hook counts. The training helper applies
  each hook's `result_proj` only to routed examples instead of stacking all
  hooks over the full batch. Regression tests verify a 4-hook batch routed only
  to hooks `0` and `2` calls/projects only those hooks. This removes the
  all-hooks-forward waste from routed batches, but cloned/independent output
  projections still leave parameter scaling unresolved.
- Shared routed output projection support is now implemented. The
  `calculator_share_output_proj` config / `--share-calculator-output-proj` CLI
  flag ties every extra hook's result-to-residual `output_proj` module to the
  primary hook, so many routed calculators can share one semantic output
  interface instead of cloning one matrix per hook. Tests verify object
  identity, parameter-count reduction, older untied-checkpoint compatibility,
  and config/metrics recording in a zero-step routed CLI smoke. This resolves
  the known cloned-output parameter-slope issue, but tied-output source/handoff
  training has not yet been validated.
- The tied-output training gate is mixed. Replacing cloned output projections
  with `--share-calculator-output-proj` in the known four-hook op19 `embd32`
  topk8+unique24 recipe still trained the source perfectly: final eval
  `1.0000`, step-630 normal/calc `1.0000`, injection-zero `0.0275`, and all
  hooks at calculator-result accuracy `1.0000`. But the trusted 600-step
  frozen-policy additive handoff reached only `0.7625` final / `0.7800`
  step-600 normal, with calculator-result accuracy `0.9950`; a continuation
  to another 600 steps reached only `0.7925` final / `0.8050` snapshot normal.
  A later audit found the first A/B had one config mismatch
  (`additive_forced_margin_start_step=50` for cloned, default `0` for shared),
  but a matched delayed-margin shared-output rerun still missed: source final
  `0.9975`, diagnostic calculator-result accuracy `0.9922`, trusted handoff
  final `0.7475`, step-600 normal `0.7225`, injection-zero `0.0875`, and
  learned calculator-result accuracy `0.9900`.
  Shared output projections therefore remove parameter growth and preserve
  source trainability, but they are not a drop-in replacement for cloned
  output projections in the current non-bottleneck handoff geometry.
- Post-shared-output steering review: this scaling branch has reached a clear
  boundary. More same-recipe shared-output seeds, continuations, or
  delayed-margin reruns are low value. Mainline compute should pivot to
  less-prescriptive answer-derived credit assignment unless shared-output work
  introduces a real new transfer-geometry mechanism validated by trusted
  handoff.
- A less-prescriptive transfer-geometry mechanism now exists for shared output:
  four-hook `left_operand_mod` routed online hard memory plus additive semantic
  distillation, with `--share-calculator-output-proj`, reached `1.0000` source
  final/calc, trained all four hooks to calculator-result accuracy `1.0000`,
  and cleared trusted 600-step handoff at `1.0000` final / step-600 normal
  with low controls. This is the first shared-output routed handoff pass, but
  it uses the handoff-friendly seed lineage and still relies on fixed-grid
  per-prompt memory before fill.
- Fresh routed/shared replication cleared the same gate on CLI seed `7` /
  effective seed `9`, the seed lineage where single-hook semantic-distilled
  source had missed trusted handoff. The four-hook shared-output source
  reached final/calc `1.0000`, memory froze after `86,400` forced evals, all
  four hooks reached calc `1.0000`, and the trusted handoff reached `1.0000`
  final / step-600 normal with low controls. This means the routed/shared
  positive is not just the original handoff-friendly seed.
- The routed/shared semantic-distilled method also clears op29 range stress
  with `operand_spans` readout and shallow result heads. At `operand_max=29`
  over a `900`-prompt grid, the source reached final/calc `1.0000`, memory
  froze by step `50` with `367,200` cumulative forced-result evals, all four
  routed hooks reached calc `1.0000`, and the trusted frozen-policy additive
  handoff reached `900/900 = 1.0000` final / step-600 normal. Step-600 controls stayed causal:
  injection-zero `0.0133`, forced-zero `0.0022`, forced-random `0.0156`.
  Fixed-grid routed/shared op19 and op29 are now proven for this mechanism;
  the next bottleneck is streaming/fresh-prompt memory, not another
  fixed-grid range or seed repeat.
- Prompt-keyed streaming minibatch memory is viable if the update budget is
  exposure-matched. The first batch64 source for 800 steps filled/froze all
  `400` prompt entries with true targets but undertrained the policy
  (`0.6325` final, diagnostic calc `0.5781`). The matched-exposure batch64
  source for 5000 steps used the same mechanism, filled/froze all entries after
  `173,568` forced evals, and reached source final/calc `1.0000`. Its trusted
  frozen-policy additive handoff also reached `1.0000` final / step-600 normal,
  with low controls (`0.0781` final injection-zero, `0.0078` forced-zero,
  `0.0156` forced-random) and all four hooks at calculator-result accuracy
  `1.0000`. This proves stochastic minibatch training can learn the routed
  shared-output calculator, but the cost shifts to more optimizer updates; the
  next high-leverage gate is fresh/heldout prompts or cheaper streaming uptake,
  not another fixed-grid or same-exposure repeat.
- Prompt-keyed memory does not generalize to heldout prompts. A deterministic
  80/20 op19 split trained only on `320` prompts for 5000 batch64 steps, filled
  and froze exactly those `320` entries after `87,552` forced evals, and reached
  train exact/calc `0.9969`. The `80` heldout prompts, absent from both
  training minibatches and prompt memory, reached only `0.0875` exact/calc
  with low forced controls. This is a clear transductive-memory boundary: the
  next mechanism must amortize target discovery or otherwise supply
  fresh-prompt credit, not merely store prompt-keyed hard targets.
- First amortized-prior diagnostic: an operand-embedding prior trained from the
  `320` discovered train-memory targets fit train memory but got `0.0000`
  heldout target accuracy, while the same prior with normalized numeric operand
  features reached `0.9125` heldout target accuracy on the `80` unscored
  prompts. This is not a source-policy success yet; tiny integrated smoke runs
  only verify the replay path executes. The next gate is whether numeric-prior
  pseudo-targets raise heldout calculator-result accuracy in the model.
- The open question is scalability: can this be approximated or replaced
  without losing the source-policy result? Uniform random result sampling is
  ruled out as the simple answer, and fixed stale exact targets are not enough;
  next work should reduce prescriptiveness, pursue non-enumerative credit
  assignment, or change shared-output transfer geometry with a predeclared
  mechanism.

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-hard-improvement-assignment-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-28-hard-improvement-assignment-convergence-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-routed-multi-hook-snapshot-metrics.md`
- `aiAgentWorkHistory/phase7/2026-05-30-routed-cloned-output-source-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-routed-source-leakage-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-sampled-hard-assignment-cost-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-exact-assignment-refresh-cadence-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-unique-sampled-assignment-coverage-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-assignment-proposal-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-source-handoff-validation.md`
- `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-fresh-seed-validation.md`
- `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-op29-range-validation.md`
- `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-op29-fresh-seed-validation.md`
- `aiAgentWorkHistory/phase7/2026-05-30-many-calculator-assignment-scaling-accounting.md`
- `aiAgentWorkHistory/phase7/2026-05-30-same-layer-multi-hook-forward-support.md`
- `aiAgentWorkHistory/phase7/2026-05-30-left-operand-routed-multi-hook-support.md`
- `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-routed-shared-output.md`
- `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-routed-shared-output-fresh-seed.md`
- `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-routed-shared-output-op29.md`
- `aiAgentWorkHistory/phase7/2026-05-31-prompt-keyed-online-hard-memory-streaming.md`
- `aiAgentWorkHistory/phase7/2026-05-31-prompt-keyed-online-hard-memory-heldout.md`
- `aiAgentWorkHistory/phase7/2026-05-31-amortized-prior-heldout-diagnostic.md`
- `researchReviews/2026-05-30-assignment-cost-reduction-review.md`
- `researchReviews/2026-05-30-many-calculator-scaling-accounting.md`
- `researchReviews/2026-05-31-prompt-keyed-streaming-memory-review.md`
- `researchReviews/2026-05-31-prompt-keyed-heldout-memory-review.md`
- `HYPOTHESIS_LEDGER.md`

## Direction: Non-Bottleneck Direct Training

Status: paused unless adding a new causal-use mechanism

Memory:

- Direct additive non-bottleneck training with answer loss plus hard assignment
  learned mostly through the neuron path; calculator-result accuracy stayed
  near chance.
- Simple zero-injection causal-gap pressure made bypass worse but did not make
  the calculator query correct.
- Non-bottleneck success currently comes from staged transfer, not direct
  discovery.

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-non-bottleneck-hard-assignment-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-28-non-bottleneck-causal-gap-gate.md`

## Direction: Bottleneck-To-Additive Handoff

Status: active but constrained

Memory:

- A bottleneck-trained calculator policy can be loaded into an additive
  non-bottleneck model and used causally if the policy is frozen/protected.
- Unprotected compatible transfer destroys the policy quickly.
- Source quality and representation geometry matter; source accuracy alone is
  not enough.
- This proves non-bottleneck viability but is not yet scalable or
  non-prescriptive.

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-transfer-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-transfer-replication.md`
- `aiAgentWorkHistory/phase7/2026-05-29-src6-selected-continuation-readout.md`

## Direction: Source Checkpoint Selection

Status: paused for cheap proxies; actual handoff gates remain trusted

Memory:

- Source normal/calculator accuracy is not a reliable selector.
- Actual 600-step frozen-policy additive handoff is the most reliable tested
  source-checkpoint gate for fresh families.
- Several cheaper proxies failed or were mixed: frozen-state readout,
  forced-result geometry, 25/50/100-step loss slope, ridge over early handoff
  traces, 500-step standalone selection on fresh `src6`, and 500-step embedded
  probe normal score on fresh seed `11`.
- Embedded probes are useful logging/triage, but 500-step normal alone is not
  validated as a selector.
- Selector-cost reduction is no longer the default frontier. New selector work
  must either beat the 600-step gate on fresh families or change source
  training pressure directly.

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-handoff-probe-selector-validation.md`
- `aiAgentWorkHistory/phase7/2026-05-29-new-source-500-selector-validation.md`
- `aiAgentWorkHistory/phase7/2026-05-29-additive-geometry-selector-validation.md`
- `aiAgentWorkHistory/phase7/2026-05-29-handoff-trace-learned-selector-audit.md`
- `aiAgentWorkHistory/phase7/2026-05-29-intraining-probe-source-selection-validation.md`

## Direction: Source Acquisition For Transfer Geometry

Status: active

Memory:

- Stabilized source acquisition can produce transferable policies in some
  seeds, but high source performance does not guarantee transfer.
- Seed-9-style no-decay stabilization plus continuation/readout can clear the
  non-bottleneck gate; seed-10-style runs show hostile geometry can persist.
- The source objective can keep improving learned calculator accuracy while
  worsening additive handoff geometry; seed 10 is the clearest warning.
- A first direct geometry objective is mixed-positive: a forced-true additive
  readout auxiliary during bottleneck source acquisition made the true result
  the best forced additive result on `59%` of a small `0..9` grid versus `0%`
  baseline, but weakened source policy accuracy at the same budget.
- Scheduling fixed the first tradeoff in the same small gate: delaying the
  forced-true additive auxiliary to step `50` improved source calc/final eval
  over baseline (`0.3900`/`0.4000` vs `0.3500`/`0.3800`) while retaining
  strong additive geometry (`forced_best_true=0.5100` vs baseline `0.0000`).
- The first full-grid scheduled gate is positive: at matched 200-step
  `operand_max=19` source checkpoints, scheduled aux nearly tied source
  accuracy but improved forced-result geometry (`forced_best_true=0.2125` vs
  `0.0000`) and standalone 600-step handoff (`0.4150` final eval vs `0.2525`).
- Longer scheduled source training compounded through source step `600`: the
  step-600 checkpoint reached `forced_best_true=0.9800` and standalone
  600-step handoff final eval `0.7725`. Step `800` had perfect forced-result
  geometry but worse handoff (`0.6750`), so final source checkpoint is not
  automatically best.
- Continuation/readout from the scheduled step-600 handoff lineage was
  mixed-positive but below gate: continuation reached `0.7775`, 600-step
  readout reached `0.8175`, and extended readout reached `0.8475`, with low
  controls but learned calc stuck around `0.5391`.
- The source-policy diagnosis was confirmed: a gentle low-LR recovery phase
  from the scheduled step-600 checkpoint (`lr=3e-4`, forced-true weight `0.1`,
  30 steps) raised source calc from `0.5800` to `0.7950` while preserving
  low forced-true loss, improved 600-step handoff to `0.8425`, and cleared
  continuation/readout at `0.9320` with low zero/random controls.
- The recovery effect replicated on fresh seed `14`: scheduled step-600 source
  eval `0.6675` recovered to `0.8850`, and the recovered checkpoint reached
  `0.9600` final eval under the trusted 600-step frozen-policy handoff, with
  learned calc `0.8700`, injection-zero `0.0850`, and forced-random `0.0875`
  at the final snapshot.
- The late-source recovery phase can be automated in a single source run:
  a fixed step-600 switch to LR multiplier `0.1` and forced-true weight `0.1`
  reached source eval `0.8775` and a trusted 600-step handoff final eval
  `0.9400`, with learned calc `0.8725`, injection-zero `0.0800`, and
  forced-random `0.0775` at the final handoff snapshot.
- A simple adaptive source-accuracy trigger is mixed-positive: after the
  adaptive recovery switch was wired to lower both LR and forced-true weight,
  `result_policy_argmax_result_accuracy >= 0.65` fired at step `528` and the
  trusted 600-step handoff reached `0.9850` final eval, exceeding the fixed
  step-600 switch. The tradeoff is higher controls (`0.1325` zero and
  forced-random) and lower source final eval (`0.8250`), so it needs fresh-seed
  replication or a smoother/conjunctive trigger before treating it as a
  validated selector.
- The fresh-seed replication was negative for raw source-accuracy thresholding:
  on seed `17`, the same `>=0.65` trigger never fired, source final eval was
  `0.6100`, and trusted 600-step handoff reached only `0.6825`. A matched fixed
  step-600 control did better (`0.7450` source, `0.7675` handoff) but still
  missed the high gate. Do not treat raw argmax source accuracy as a validated
  adaptive transition criterion.
- Forced-true loss is a better one-metric trigger on the hard seed 17:
  `additive_forced_true_loss <= 0.05` fired at step `500`, improved source final
  to `0.7225`, and reached `0.7625` trusted 600-step handoff with low controls.
  This mostly matched the fixed step-600 control (`0.7675`) and beat the
  no-trigger source-accuracy branch (`0.6825`), but still missed the high gate.
- Adding EMA/patience to the forced-loss trigger improved timing on the same
  hard seed: beta `0.8`, patience `10`, min step `500` fired at step `509`,
  raised source final to `0.7625`, and improved trusted 600-step handoff to
  `0.8025` with low forced-random (`0.0325`) but still below the high gate.
- A hard conjunctive source-accuracy gate was too conservative on the same
  hard seed: forced-loss readiness was satisfied, but requiring
  `result_policy_argmax_result_accuracy >= 0.70` never fired, source final
  stayed `0.6100`, and handoff returned to `0.6825`.
- A budgeted contrastive source-geometry objective is positive in the matched
  full-grid early handoff gate. The full-grid 4-negative forced-margin branch
  was too costly locally, but one sampled negative per prompt reached `0.3225`
  source calc / `0.3600` final eval at step `200`. Geometry was mixed
  (`forced_best_true=0.6725`, 50-step slope final loss `1.4660`), but the
  trusted 600-step frozen-policy handoff reached `0.6600` final eval /
  `0.7050` step-600 normal with low controls, above matched scheduled
  forced-true (`0.4150`) and baseline (`0.2525`).
- Longer one-negative forced-margin training is mixed-positive but not a clean
  replacement for scheduled forced-true. A 600-step source run reached
  `0.5225` source calc and near-perfect forced geometry, but trusted handoff
  was `0.7330` final / `0.7500` step-600 normal. Continuing the exact 200-step
  positive checkpoint found a better intermediate handoff (`0.7400` final /
  `0.7850` step-600 normal) but degraded with another 200 source steps.
  Longer margin therefore improves over the 200-step margin handoff (`0.6600`)
  but does not clearly beat scheduled forced-true step-600 final (`0.7725`).
- Forced-margin branch review: treat one-negative margin as a useful
  constrained auxiliary, not a standalone mainline. Do not continue with
  negative-count tweaks, same-seed longer ladders, start-step tweaks, slope
  proxy selection, or geometry-only checkpoint fishing as novelty. Continue
  only for predeclared source recovery/retention, fresh-seed stability, or as
  evidence feeding less prescriptive/scalable credit-assignment work.
- The predeclared source recovery test was positive: continuing the longer
  one-negative forced-margin step-600 source checkpoint for `30` low-LR steps
  (`lr=0.0003`, margin weight `0.1`) raised source calc from `0.5225` to
  `0.7725` and final source eval to `0.7825`. The trusted frozen-policy
  600-step handoff from recovered step `30` reached `0.8700` final /
  `0.9050` step-600 normal with injection-zero `0.0000`, forced-random
  `0.0313`, and learned calc `0.8594`. This means forced-margin was partly
  source-policy-maturity limited, but it remains prescriptive and below
  automated scheduled-source recovery (`0.9400` final).
- Automated forced-margin recovery now replicates strongly on a fresh seed.
  Adding a late forced-margin weight override let a single 630-step source run
  switch at step `600` (`lr` multiplier `0.1`, margin weight `0.1`), raising
  source calc `0.5825 -> 0.8825`; the trusted 600-step frozen-policy handoff
  reached `0.9875` final / `0.9800` step-600 normal with injection-zero
  `0.0156-0.0250`, forced-random `0.0938`, and learned calc `0.8906`. This is
  strong staged-transfer evidence, but still prescriptive.
- Follow-up review decision: automated one-negative forced-margin recovery is
  now a staged-transfer benchmark, not a local knob branch. Do not tune
  start-step, margin, negative count, or recovery length on the same setup as
  novelty. Future forced-margin compute should stress stability/scale or
  remove prescriptiveness by replacing hard assignment or true-result forcing.
- A second fresh-seed automated forced-margin stability check is mixed-positive:
  CLI seed `19` / effective seed `21` again improved sharply during late
  recovery (`0.5625 -> 0.8325` source calc from step `600` to `630`) and
  cleared trusted handoff (`0.8975` final / `0.9050` step-600 normal,
  zero-injection `0.0000`, forced-random `0.0350`). This confirms the benchmark
  is real but seed-variable, below the prior `0.9875` handoff.
- A wider-model scale stress is positive with a caveat: using an existing
  `n_embd=32`, `n_head=2` non-product semantic decoder, the automated
  forced-margin source reached `0.9125` final eval and the trusted 600-step
  handoff reached `1.0000` final / `1.0000` step-600 normal with low controls
  (`0.0625` zero-injection, `0.0325` forced-random). This supports staged
  scale/stability but does not remove prescriptiveness or prove product-decoder
  parity.
- Product-decoder parity for the wider scale stress is positive. A matching
  `n_embd=32`, `n_head=2`, `answer_decoder_interaction=product` oracle decoder
  reached `1.0000`; the automated forced-margin source improved during late
  recovery from `0.6375` to `0.9475`, and the trusted 600-step frozen-policy
  additive handoff reached `1.0000` final / step-600 normal with `0.0000`
  injection-zero and `0.0225` forced-random at step `600`. This removes the
  non-product decoder caveat but remains prescriptive staged transfer.
- The first larger-range stress is mixed-negative. At `operand_max=29`, the
  wider product oracle decoder reached full-grid `1.0000`, but the same
  automated forced-margin source recovered only from `0.3533` to `0.6889`
  source calc and the trusted handoff reached `0.8533` final / `0.8278`
  step-600 normal with low controls. Range scaling is therefore an unresolved
  source-acquisition/assignment-cost problem; do not jump to op49 with the same
  full-grid recipe as novelty.
- A low-LR op29 source-recovery diagnostic is mixed-positive: continuing the
  op29 step-630 source for `90` steps at `lr=0.0003` raised source calc to
  `0.8211`, and trusted handoff improved to `0.9067` final / `0.8978`
  step-600 normal with low controls. This means the op29 miss was partly
  source-maturity limited, but the rescue adds prescriptive full-grid source
  compute and is not a scalable fix.
- A hidden result-head op29 capacity diagnostic is positive: adding
  `--calculator-result-head-hidden-size 64` raised source final eval to
  `0.9978` and the trusted handoff to `1.0000` final / step-600 normal with
  low controls. A fresh-seed repeat also cleared: source step `630` reached
  `0.9967` and the trusted handoff reached `1.0000` final / step-600 normal
  with low controls. The op29 range miss was strongly source-capacity
  sensitive, but the fix adds per-calculator result-head parameters and keeps
  full-grid hard assignment plus true-result forcing.
- The first op39 `rhead64` stress is mixed-positive and costly. The op39
  product oracle decoder reached `1.0000`, but the exact full-grid source run
  was interrupted after about `33` local CPU minutes with checkpoints through
  step `540`; step `540` eval was only `0.543`. A 90-step continuation rescued
  source eval to `0.940`, and trusted handoff reached `0.9475` final /
  `0.9419` step-600 normal with low controls. This is causal larger-range
  transfer, but not op29-style perfect scaling and not a scalable recipe.
- Forced-margin benchmark review: automated recovery is now the staged-transfer
  benchmark to beat, not the next knob branch. Future forced-margin compute
  must stress a new thesis-relevant axis such as product-decoder parity, larger
  operand range, larger architecture, or many-calculator cost, or remove hard
  assignment / true-result forcing. Do not run more local knob sweeps, same-axis
  seed-only reruns, or cheap selector/proxy work as novelty.
- A less-prescriptive answer-derived bridge is positive but constrained: the
  older full-grid `result_boundary_target` source checkpoint transfers into
  the trusted frozen-policy additive gate at `0.8825` final / `0.8425`
  step-600 normal, with injection-zero `0.0000`, forced-random `0.0391`, and
  learned calc `0.9922`. This shows true-result forced-margin pressure is not
  strictly required for causal staged transfer, but it remains full-enumeration
  candidate scoring plus frozen-policy transfer and is weaker than automated
  forced-margin recovery.
- Hidden-output amortized critics are not the current scalable result-boundary
  bridge: pointwise recovery reached only `0.08-0.26`; pairwise ranking helped
  the trained checkpoint to `0.40` heldout argmin recovery at `k=24`, but that
  already scores most of the 39-class result vocabulary. Do not continue
  pointwise/pairwise/hybrid critic-loss variants as novelty.
- A follow-up uncertainty/proposal diagnostic changed the question from direct
  argmin prediction to "propose a subset, then score it." This is
  mixed-promising but not a solved bridge: at step `800`, a single pairwise
  critic trained from `8` forced scores per train prompt recovered the
  full-enum best on `0.79` of heldout prompts with top-8 proposal rescoring and
  `0.96` with top-16; a four-member ensemble reached `0.84` top-8 and `1.00`
  top-16. But top-16 already scores `16/39` heldout candidates, the ensemble
  uses `32` train scores per prompt, and LCB uncertainty did not beat mean
  proposals. Continue only with adaptive compute/soft targets/streaming
  validation, not beta/count/ensemble tweaks.
- Adaptive compute adds only modest leverage in the same static gate. Expanding
  the most cutoff-margin-uncertain prompts from top-8 to top-16 beats random at
  matched average cost: single critic `0.85` vs `0.82` at mean `10/39` and
  `0.92` vs `0.88` at mean `12/39`; ensemble `0.91` vs `0.88` at mean `10/39`
  and `0.97` vs `0.91` at mean `12/39`. But fixed top-16 remains stronger
  (`0.96-1.00`), ensemble training cost is high, and std/LCB uncertainty are
  weaker than margin. Do not run threshold/beta/fraction sweeps as novelty.
- Static soft result-boundary targets are also negative. In the matched
  200-step full-grid upstream-open source gate, hard-best reached `0.5450`
  learned calc / `0.5475` final eval; soft `t=1` reached only `0.2900` /
  `0.2775`, and broad soft `t=4` reached `0.1350` / `0.1275`. Temperature
  softening diluted rather than improved the answer-derived teaching signal.
- Static full-enum regret-set targets are negative too. Fixed margins up to
  `1.0` collapsed to hard-best, margin `2.0` was nearly hard (`1.06`
  effective results), and the first meaningful set target, margin `4.0`
  (`5.6975` effective results, true result always in set), trained far worse
  than matched hard-best at step `200`: `0.0900` learned calc / `0.0900`
  final eval versus `0.4625` / `0.4225`.
- Steering review: static result-boundary approximation and static set targets
  are paused. The branch has now tested direct critics, proposal rescoring,
  adaptive expansion, soft targets, and fixed-margin regret sets. Continue only
  with evolving-checkpoint validation, calibrated proposal learning, adaptive
  uncertainty/regret selection, or a genuinely different less-prescriptive
  credit-assignment mechanism.
- Cross-checkpoint validation shows frozen sparse proposal critics are
  state-local. Same-state top-8 proposal recovery improved with source
  maturity (`0.48` step100, `0.74` step400, `0.79` step800), but forward
  transfer collapsed: train step100 to eval step400/800 gave only
  `0.11`/`0.12`, and train step400 to eval step800 gave `0.23`. Do not wire a
  frozen critic into source training; result-boundary proposals need online
  refresh, state calibration, or a different mechanism.
- Simple warm-start online calibration is partial-negative. Retargeting the
  critic normalization at the eval checkpoint and fine-tuning on fresh sparse
  scores repaired some forward transfer (`step400 -> step800` from `0.23` to
  `0.59` with `2` fresh scores/prompt), but even `8` fresh scores/prompt only
  reached `0.62`, below same-state step800 `0.79`. Do not tune small
  adapt-lr/epoch/count variants as novelty.
- Direct sampled result-boundary source training with policy-topk candidates is
  also insufficient. Scoring top-8 policy results plus unique candidates for
  `24/39` classes raised true-candidate coverage to `0.9600` by step `200`,
  but learned calc/final eval only reached `0.3425` training-curve learned-best,
  `0.3675` snapshot calc, and `0.3525` final eval, below matched full-enum
  hard-best source comparators. Do not respond with sample-count/top-k ladders;
  change proposal/training co-design or target construction.
- Zero-injection improvement is the active result-boundary lead. It builds a
  target from answer-loss improvement over no calculator injection, not from
  the true sum label. Full enumeration reached step-200 snapshot calc `0.5700`,
  learned-best `0.5475`, and final eval `0.5425`, matching nearby hard-best
  boundary comparators while keeping effective target size `1.2692`. The
  topk8+unique24 sparse version improved over sampled hard-best (`0.4300`
  final vs `0.3525`) but still trailed full enumeration despite `0.9725`
  true-candidate coverage, so sparse scaling remains open.
- Longer zero-improvement source training confirms bottleneck viability but
  exposes a handoff gap. The 1600-equivalent source reached final `0.9850` and
  source calc `0.9725`; its trusted frozen-policy additive handoff was causal
  but sub-gate (`0.6775` final, `0.7150` step-600 normal, `0.0100`
  injection-zero, `0.0525` forced-random). Source maturity helps versus the
  800-step source handoff (`0.3650` final), but this still trails the old
  hard-best boundary handoff (`0.8825` final). Continue only with a new
  handoff-aware geometry mechanism or scalable proposal.
- Naive additive-path zero-improvement is not that mechanism. Scoring
  zero-improvement through the non-bottleneck additive path before the additive
  readout is trained made the source learn a non-arithmetic target: step-200
  learned-best reached `0.6025`, but target best=true was only `0.0325`,
  true-result target probability `0.0225`, and final/snapshot calc stayed
  `0.0200`. Additive-path targets need readout preconditioning/co-training,
  not a longer run from an untrained additive loss table.
- Semantic readout distillation is a partial repair but not enough. A 300-step
  arbitrary-result distill preconditioner raised additive/semantic token
  agreement to `0.7694` and repaired additive target quality (`best=true`
  `0.5225` at source step 0, `0.8200` by step 200 with ongoing distill), but
  source uptake stayed weak (`learned_best=0.1400`, calc `0.0675`). Without
  ongoing distill the table drifted non-arithmetic again (`best=true=0.1575`)
  while learned-best rose to `0.6950`. Next variants must address policy
  uptake or target drift, not just readout semantics.
- Frozen-teacher additive target anchoring solves the drift diagnostic but not
  the uptake problem. A separate teacher checkpoint keeps the additive target
  table at `best=true=0.5225` while the live policy trains; 800 steps improved
  learned-best only to `0.4125`, with source calc/final eval `0.1700`/`0.1750`.
  Head-only anchoring stalled lower (`learned_best=0.188`), and freezing only
  the post-calculator decoder still let pre-hook residual changes drift the
  target (`best=true=0.1575`). Do not run same-checkpoint anchor sweeps as
  novelty; policy uptake needs a different mechanism.
- Cached teacher target tables separate uptake from target quality. Cached
  soft weights reproduce weak uptake even with the higher-quality teacher
  (`0.393` learned-best / `0.2725` final), while cached hard-best from that
  teacher lifts source training to `0.765` learned-best / `0.5825` final at
  1600. Teacher quality matters, but this remains cached/full-enum and below
  the teacher's `0.8200` best-true ceiling. Use cache as a diagnostic for
  better target construction, not as the recipe.
- Online hard result-boundary memory is the sparse answer-derived lead.
  Topk8+unique24 zero-improvement scoring fills a hard per-prompt memory, and
  the freeze-full branch stops rescoring after `86,400` forced evals while
  reaching `0.9675` calc / `0.9725` final at 800. The first trusted additive
  handoff missed (`0.465` final / `0.485` normal) despite preserved frozen calc
  (`0.9575`), so source discovery is strong but handoff/readout geometry is not.
- Adding arbitrary-result additive semantic distillation during that source
  training repairs handoff geometry on one op19 fixed-grid seed. The combined
  source reached `1.0000` final/calc and semantic token agreement `0.7459`;
  the trusted frozen-policy additive handoff reached `1.0000` final /
  step-600 normal with low controls. Fresh-seed source acquisition replicated
  (`1.0000` final/calc, memory frozen after `76,800` forced evals), but its
  trusted handoff was source/geometry sensitive: the fresh source got `0.6475`
  final / `0.6625` step-600 normal, and an alternate downstream seed also
  missed (`0.6325`), while the original good source passed with that failed
  handoff seed (`1.0000`). Continuation from the fresh-source miss improved to
  `0.823` final / `0.850` normal. The auxiliary teaches arbitrary result
  semantics without specifying prompt results, but robust handoff is not
  solved. A many-calculator/shared-output stress on the handoff-friendly seed
  is positive and replicates on the handoff-sensitive fresh seed: four routed
  hooks with shared output reached source/handoff `1.0000` and low controls on
  both seeds. Next work must test streaming/fresh prompts or larger range; do
  not tune the same op19 weight/sample/length or repeat routed op19 seeds as
  novelty.
- Uniform sampled hard assignment does not fix the exact full-grid cost:
  sample16 and sample32 destroyed the op19 `rhead64` source signal relative to
  exact assignment while saving only modest wall time. Future cost reduction
  must improve candidate coverage/target quality or change the credit signal.
- Fixed-cadence exact target refresh also weakens source acquisition; refresh2
  is better than sampled candidates but still far below exact. Do not treat
  stale exact targets as the cheap assignment answer without adaptive freshness.
- Unique sampled assignment improves coverage and source learning, but still
  misses exact despite scoring `32/39` classes. Candidate proposals need to be
  smarter than duplicate-free uniform coverage.
- Policy-topk plus unique random candidates is now the proposal to validate:
  it preserves true-result coverage and source learning much better than
  uniform unique sampling at matched scored count. It now has op19 handoff
  validation, but still needs fresh-seed, range, or many-calculator validation.
- Forced-result geometry alone remains a triage signal; actual handoff/readout
  gates remain decisive.

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-source-acquisition-stabilization-floor.md`
- `aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-continuation-readout.md`
- `aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-seed10-replication.md`
- `aiAgentWorkHistory/phase7/2026-05-29-seed10-source-checkpoint-geometry-sweep.md`
- `aiAgentWorkHistory/phase7/2026-05-29-source-assignment-weight5-transfer-probe.md`
- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-source-aux-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-schedule-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-op19-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-long-source-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-scheduled-source-continuation-readout.md`
- `aiAgentWorkHistory/phase7/2026-05-29-scheduled-source-low-lr-recovery.md`
- `aiAgentWorkHistory/phase7/2026-05-29-fresh-scheduled-source-recovery-replication.md`
- `aiAgentWorkHistory/phase7/2026-05-29-automated-scheduled-source-recovery.md`
- `aiAgentWorkHistory/phase7/2026-05-29-adaptive-source-recovery-trigger.md`
- `aiAgentWorkHistory/phase7/2026-05-29-fresh-adaptive-recovery-trigger-replication.md`
- `aiAgentWorkHistory/phase7/2026-05-29-forced-loss-adaptive-recovery-trigger.md`
- `aiAgentWorkHistory/phase7/2026-05-29-smoothed-forced-loss-recovery-trigger.md`
- `aiAgentWorkHistory/phase7/2026-05-29-conjunctive-recovery-trigger.md`
- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-margin-source-aux-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-margin-op19-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-margin-long-source-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-forced-margin-low-lr-source-recovery.md`
- `aiAgentWorkHistory/phase7/2026-05-30-automated-forced-margin-source-recovery.md`
- `aiAgentWorkHistory/phase7/2026-05-30-forced-margin-second-fresh-seed-stability.md`
- `aiAgentWorkHistory/phase7/2026-05-30-forced-margin-wider-model-scale-stress.md`
- `aiAgentWorkHistory/phase7/2026-05-30-forced-margin-product-decoder-parity.md`
- `aiAgentWorkHistory/phase7/2026-05-30-forced-margin-op29-range-stress.md`
- `aiAgentWorkHistory/phase7/2026-05-30-op29-low-lr-source-recovery-diagnostic.md`
- `aiAgentWorkHistory/phase7/2026-05-30-op29-hidden-result-head-capacity-diagnostic.md`
- `aiAgentWorkHistory/phase7/2026-05-30-op29-rhead64-fresh-seed-replication.md`
- `aiAgentWorkHistory/phase7/2026-05-30-op39-rhead64-range-stress.md`
- `researchReviews/2026-05-30-forced-margin-range-stress-review.md`
- `researchReviews/2026-05-30-forced-margin-benchmark-direction-review.md`
- `aiAgentWorkHistory/phase7/2026-05-30-answer-derived-boundary-handoff.md`
- `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-amortized-critic-diagnostic.md`
- `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-uncertainty-proposal-diagnostic.md`
- `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-adaptive-proposal-diagnostic.md`
- `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-soft-target-training-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-regret-set-training-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-cross-checkpoint-critic-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-online-calibrated-critic-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-sampled-result-boundary-source-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-zero-improvement-boundary-source-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-zero-improvement-boundary-handoff.md`
- `aiAgentWorkHistory/phase7/2026-05-30-additive-zero-improvement-source-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-semantic-distilled-additive-zero-improvement.md`
- `aiAgentWorkHistory/phase7/2026-05-30-frozen-teacher-additive-target-anchor.md`
- `aiAgentWorkHistory/phase7/2026-05-30-cached-teacher-target-table.md`
- `aiAgentWorkHistory/phase7/2026-05-30-high-quality-cached-teacher-table.md`
- `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-result-boundary.md`
- `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-handoff.md`
- `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-fresh-seed.md`
- `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-routed-shared-output.md`
- `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-routed-shared-output-fresh-seed.md`
- `researchReviews/2026-05-30-sampled-result-boundary-steering-review.md`
- `researchReviews/2026-05-30-result-boundary-approximation-review.md`
- `researchReviews/2026-05-30-result-boundary-static-approximation-steering-review.md`
- `researchReviews/2026-05-30-result-boundary-set-target-steering-review.md`
- `aiAgentWorkHistory/phase7/2026-05-30-sampled-hard-assignment-cost-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-exact-assignment-refresh-cadence-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-unique-sampled-assignment-coverage-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-assignment-proposal-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-source-handoff-validation.md`
- `researchReviews/2026-05-30-assignment-cost-reduction-review.md`
- `researchReviews/2026-05-29-scheduled-source-geometry-review.md`
- `researchReviews/2026-05-29-forced-margin-branch-review.md`
- `researchReviews/2026-05-30-forced-margin-recovery-review.md`

## Direction: Target Propagation / Local Targets

Status: paused as a mainline; active only with estimator, target-construction, or validated generalization changes

Memory:

- This direction has not yet received the same empirical treatment as the
  answer-loss and shadow-gradient families.
- It is strategically interesting because it changes the credit-assignment
  family rather than tuning another proxy.
- The first exact-grid Stage 0 gate is partially positive: sharp
  current-policy-reweighted targets and weakly proximal logit-descent targets
  produce gradients aligned with the hard boundary ceiling, while expected
  answer loss remains anti-aligned.
- A 200-step Stage 1 lift gate showed `policy_reweighted_t1` reaches `0.5600`
  exact-grid calculator-result accuracy and `0.5391` sampled normal accuracy,
  beating the failed expected-loss baseline and slightly beating the
  same-budget hard-boundary run.
- An 800-step target-training plus 200-step answer-only retention gate showed
  `policy_reweighted_t1` is nonmonotonic during target training but can finish
  retention strongest: `0.8925` exact-grid calculator-result accuracy and
  `0.8750` sampled normal, with injection-zero and forced-random controls low.
- A first sparse approximation gate was mixed-negative: naive top-k/uniform
  sampled candidate sets underperformed badly unless they scored nearly the
  full result vocabulary (`u32` only `0.3350` exact-grid calc at 200 steps,
  `u36` `0.4100`, while full-vocabulary `u39` reached `0.6250`).
- A simple adaptive loss-neighborhood proposal was negative: at similar scoring
  budgets it underperformed raw uniform `u32` because expansion clustered into
  fewer unique candidates and lower true-result coverage.
- A replay-memory proposal was the first positive sparse approximation: scoring
  `8` fresh uniform results per step and reusing cached low-loss candidates
  beat raw uniform `u32` at 200 steps (`0.5900` vs `0.3350` exact-grid calc)
  and reached `0.8600` calc / `0.8750` sampled normal after an 800+200
  answer-only retention gate.
- Lowering the fresh scoring budget improved the short gate through `u2_m30`:
  at 200 steps `u2_m30` reached `0.6025` exact-grid calc and `0.6016` sampled
  normal, while `u1_m31` weakened to `0.4075`/`0.4219`. The long `u2_m30`
  retention gate was positive but weaker than `u8_m24`: `0.7850` calc /
  `0.7656` normal after answer-only retention.
- Simple cached-candidate rescoring did not improve that retention weakness:
  `u2_m30_r2` tied no-rescore in both 200-step and 800+200 gates, while
  heavier `r4/r8` rescoring hurt short-gate learning.
- Finite reset windows exposed a transductive dependency: at 199 steps,
  no-reset `u2_m30` reached `0.5925` exact calc / `0.5938` sampled normal,
  while `reset100` reached only `0.4575` / `0.4453` and `reset50` only
  `0.2575` / `0.2812`, even though target true-candidate coverage mostly
  recovered between resets.
- Streaming minibatches with prompt-keyed caches removed the strong replay
  lift. At 800 steps with batch `16`, exact `policy_reweighted_t1` and raw
  uniform `u32` both reached `0.2450` exact calc, `u8_m24` was only comparable
  at `0.2650`, and `u2_m30` lagged at `0.1850`.
- A simple estimator/target correction also failed: preserving current policy
  mass on unscored result classes with imputed mean/current/max losses reached
  at best `0.2500` exact calc / `0.2500` sampled normal at `u16`, below raw
  uniform `u32` (`0.3350`/`0.3438`) and far below exact policy-reweighted
  (`0.5600`/`0.5391`).
- A simple online learned loss proposal is partial: with the same 32 forced
  scores per step, `learned_policy_reweighted_t1_u4_p28_h32_e1` beat raw
  `u32` on the fixed-grid 200-step gate (`0.5850` exact calc / `0.5703`
  sampled normal vs `0.3350`/`0.3438`) and achieved full true-candidate
  proposal coverage, but it did not beat raw `u32` under streaming minibatches
  (`0.2350` calc for both at 800 steps, sampled normal `0.2656` learned vs
  `0.2734` raw).
- Random-prompt proposal pretraining is mixed-negative: `_w20` warmup gave a
  small 800-step streaming exact-calc nudge over raw `u32` (`0.2625` vs
  `0.2350`) but hurt sampled normal badly (`0.1797` vs `0.2734`), and `_w20`
  / `_w50` did not lift the 200-step streaming screen.
- A sparse pairwise-preference target is negative as a different target
  construction: `sampled_pairwise_preference_u8/u16` stayed at `0.0050` exact
  calc, and `u32` reached only `0.0425` calc / `0.0234` sampled normal despite
  true-candidate coverage `0.8450`, while same-budget policy-reweighted `u32`
  reached `0.3350` / `0.3438`.
- Local-target proposal branch review: pause simple proposal approximation as
  a mainline branch. Exact `policy_reweighted_t1` remains a useful ceiling, but
  raw/adaptive candidates, fixed replay, imputed sparse targets, online learned
  proposals, random-prompt proposal pretraining, and simple pairwise
  preferences have all failed scalability or Stage 1 stress. Continue only
  with a different estimator, materially different target construction, or
  learned proposal validation explicitly tied to
  streaming/full-grid generalization.
- Follow-up direction review after the pairwise negative: local-target
  approximation is now a ceiling/diagnostic branch, not the current scalable
  path. Do not run more sparse count ladders, replay-cache tuning, imputation
  variants, learned-proposal hyperparameter sweeps, or pairwise count/gap
  sweeps as novelty. Future local-target compute must predeclare the new
  estimator/target mechanism and its streaming or heldout-generalization gate.

Representative evidence:

- `SOLUTION_IDEAS.md`
- `RESEARCH_STATE.md`
- `researchReviews/2026-05-29-phase7-local-target-approximation-review.md`
- `researchReviews/2026-05-29-replay-memory-branch-review.md`
- `aiAgentWorkHistory/phase7/2026-05-29-local-target-propagation-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-local-target-stage1-lift-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-local-target-convergence-retention-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-sampled-local-target-approximation-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-adaptive-local-target-proposal-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-local-target-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-lower-budget-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-rescore-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-reset-stress-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-streaming-prompt-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-corrected-sparse-local-target-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-learned-proposal-local-target-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-pretrained-learned-proposal-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-30-sampled-pairwise-preference-target-gate.md`
- `researchReviews/2026-05-29-local-target-proposal-branch-review.md`
- `researchReviews/2026-05-30-local-target-approximation-direction-review.md`
