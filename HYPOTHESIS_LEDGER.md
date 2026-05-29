# Hypothesis Ledger

Tiny claims and outcomes to prevent retesting settled branches.

Use this file tactically. The strategic synthesis lives in
`RESEARCH_STATE.md`; this ledger prevents local reruns and records which
families are paused. Direction-level synthesis lives in `researchMemory/`.

Maintenance rule: individual entries may accumulate here, but when several
entries point to the same lesson, consolidate that lesson into
`researchMemory/` and update the family status index rather than relying on
future agents to infer it from chronology.

## Family Status Index

| Family | Status | Strategic implication |
| --- | --- | --- |
| Oracle calculator wiring | Paused | Use only as a wiring/control check; not progress toward the thesis. |
| Generic retention-after-teaching | Paused | Already proven in identifiable/scaffolded settings; only run for a new interface or stability question. |
| Vanilla score-function / expected answer-loss discovery | Paused | Sampling variance and decoder calibration were not the main blockers. |
| Simple decoder calibration | Paused | Local gradient sign improvements did not produce Stage 1 discovery. |
| Simple direct feedback / fixed shadow gradients | Paused | Stage 0 alignment alone was not enough; do not repeat without a new dynamics mechanism. |
| Simple online shadow-gradient variants | Paused | Normalization, validation selection, dropout, and directional losses did not clear the useful-training gate. |
| Hard improvement assignment | Active but constrained | Strong bottleneck source ceiling, but scalability and prescriptiveness remain unresolved. |
| Bottleneck-to-additive staged handoff | Active but constrained | Proves non-bottleneck viability, but source quality and policy protection remain bottlenecks. |
| Cheap source-checkpoint selectors | Paused | Frozen-state, geometry, short-slope, ridge, and embedded 500-step probes are not reliable replacements for actual handoff gates. |
| Source acquisition for transfer geometry | Active | Current best strategic direction if it directly targets handoff/readout behavior. |
| Target propagation / local targets | Active candidate, constrained | Exact/full-enum local-target gates are positive, but raw sampled/top-k and simple adaptive-neighborhood approximations are paused; continue only with learned proposals, estimator correction, or a different target construction. |

Rule: if a proposed experiment belongs to a paused family, it needs a new
mechanism and should be reconciled with `RESEARCH_STATE.md` before running.

## Phase 7

DISPROVEN: Vanilla result-space policy gradient is mainly blocked by finite-sample variance.
Conclusion: Exact result-marginal gradients align with sampled PG but both anti-align with the boundary ceiling.
Do not repeat: Longer vanilla PG or learned-baseline runs that estimate the same raw expected-cost gradient.
Next allowed test: A qualitatively different backward channel with a fixed-grid alignment gate.
Source: `aiAgentWorkHistory/phase7/2026-05-14-exact-result-marginal-answer-loss-gradient-gate.md`

DISPROVEN: Decoder calibration alone rescues ordinary expected-cost discovery.
Conclusion: Contrastive-margin decoder passed local sign alignment, then Stage 1 collapsed to wrong low-entropy results.
Do not repeat: Decoder-only sharpening/calibration without a stronger backward channel.
Next allowed test: Synthetic gradients, direct feedback alignment, or learned shadow-gradient modules.
Source: `aiAgentWorkHistory/phase7/2026-05-14-gradient-friendly-result-decoder-alignment-gate.md`

DISPROVEN: Output-projection boundary feedback is sufficient for natural result-space discovery.
Conclusion: Stage 0 aligned with the boundary ceiling, but Stage 1 reached only `0.155` best snapshot calculator-result accuracy and `0.160` final exact match.
Do not repeat: Plain output-projection feedback with the same weight/schedule as a mainline long run.
Next allowed test: A learned shadow-gradient module or stronger feedback training objective that must pass Stage 0 and show early Stage 1 lift.
Source: `aiAgentWorkHistory/phase7/2026-05-28-boundary-feedback-gradient-gate.md`

DISPROVEN: One fixed-random direct-feedback matrix is enough to clear the Stage 0 result-head gate.
Conclusion: Seed `0` fixed-random feedback had result-head cosine `-0.0036` against the boundary ceiling despite positive upstream cosine.
Do not repeat: Single-seed fixed-random DFA long training without a positive result-head Stage 0 gate.
Next allowed test: Multi-seed random-feedback screening or learned feedback, but only with Stage 0 gating.
Source: `aiAgentWorkHistory/phase7/2026-05-28-boundary-feedback-gradient-gate.md`

DISPROVEN: A fit-once linear shadow map from injection gradients to boundary result-logit gradients is enough for early natural result discovery.
Conclusion: Stage 0 model-update alignment was very high (`0.9983` result-proj, `0.9854` upstream), but 200-step Stage 1 reached only `0.070` best snapshot accuracy and `0.040` final exact match.
Do not repeat: Frozen fit-once linear shadow feedback with the same exact-grid calibration and weight/schedule.
Next allowed test: Heldout-validated or online-trained shadow modules with an early-lift gate, not a fixed linear map fit once at initialization.
Source: `aiAgentWorkHistory/phase7/2026-05-28-linear-shadow-feedback-gate.md`

DISPROVEN: Same-batch fit-once linear shadow alignment is a reliable Stage 0 gate.
Conclusion: With a deterministic `320/80` split, train result-proj cosine was `0.9981` but heldout result-proj cosine fell to `0.2622`, with a `0.7359` train-heldout gap.
Do not repeat: Treating same-batch linear shadow alignment as sufficient for training budget.
Next allowed test: Online MLP shadow feedback that includes result-policy state and must pass heldout warmup before Stage 1.
Source: `aiAgentWorkHistory/phase7/2026-05-28-heldout-linear-shadow-feedback-gate.md`

DISPROVEN: A simple online MLP shadow module with injection-gradient plus result-logit state cleanly passes the heldout warmup gate.
Conclusion: Hidden size `64` reached heldout result/upstream cosines `0.7167/0.7601`, but train-heldout gaps were `0.2683/0.2202`; hidden size `16` reduced the gap but heldout result cosine fell to `0.6255`.
Do not repeat: Launching Stage 1 from these simple online-MLP warmups, or rerunning the same `h64`/`h16`, `lr=1e-3`, `100`-step gate as novelty.
Next allowed test: Add a genuinely stronger shadow-generalization mechanism, such as validation early stopping, regularization, target normalization, richer policy state, or a different synthetic-gradient objective, and gate it heldout before Stage 1.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-warmup-gate.md`

DISPROVEN: Validation-selected early stopping is enough to rescue the simple online MLP shadow module.
Conclusion: With `h64`, `lr=1e-3`, `100` steps, `0.1` validation and `0.2` heldout test, the selected step `60` reached test result/upstream cosines `0.6449/0.7266` with train-test gaps `0.3201/0.2414`.
Do not repeat: Treating validation-best checkpoints from this same simple MLP as a Stage 1 go signal.
Next allowed test: Change the learned-gradient target or state itself, such as target normalization, regularization, richer policy features, or a different synthetic-gradient objective.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-gate.md`

DISPROVEN: Fit-split per-result z-scored shadow targets are sufficient to clear the online MLP heldout warmup gate.
Conclusion: Target normalization improved heldout cosines, but `h64/h32/h16` still missed the train-heldout gap gate; best near miss was `h16` with heldout `0.7259/0.7549` and gaps `0.1723/0.1458`.
Do not repeat: The same per-result z-score target-normalized `h64/h32/h16/h8`, `lr=1e-3`, `100`-step validation-selected Stage 0B sweep as novelty.
Next allowed test: Change the shadow input/state or objective more substantially, e.g. richer policy features, explicit regularization, a different loss, or a more stable target construction.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-target-normalization-gate.md`

DISPROVEN: Appending raw result policy-state features rescues target-normalized online MLP shadow feedback.
Conclusion: Adding result probabilities, log-probabilities, and entropy to the shadow input did not clear the heldout gap gate; `h32` reached heldout `0.7037/0.7611` but gaps were `0.2853/0.2131`, and `h16` missed the result threshold.
Do not repeat: Raw `injection_grad_policy_state` features with per-result target z-score, `h16/h32`, `lr=1e-3`, `100` steps as novelty.
Next allowed test: Feature scaling/standardization, explicit regularization, a different synthetic-gradient loss, or a more stable target construction.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-policy-state-gate.md`

DISPROVEN: Fit-split per-feature z-score standardization rescues target-normalized online MLP shadow feedback.
Conclusion: Feature z-scoring hurt the raw policy-state branch (`h16` heldout `0.5942/0.3997`, `h32` `0.4340/0.4023`) and did not rescue the simpler logits branch (`h32` heldout `0.6691/0.7028`, gaps `0.2830/0.2658`).
Do not repeat: Plain `fit_zscore_per_feature` with `injection_grad_logits` or `injection_grad_policy_state`, per-result target z-score, `h16/h32`, `lr=1e-3`, `100` steps as novelty.
Next allowed test: Change objective/regularization or target construction, not just raw feature scale.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-feature-standardization-gate.md`

DISPROVEN: Directional cosine shadow losses alone cleanly clear the online MLP heldout warmup gate.
Conclusion: `cosine` and `mse_plus_cosine` improved heldout cosines for the simple logits state (`h16/h32` around `0.76-0.79` result, `0.80-0.83` upstream), but result train-heldout gaps stayed around `0.20`; h8 missed heldout cosine.
Do not repeat: Plain `cosine` or `mse_plus_cosine` with `injection_grad_logits`, per-result target z-score, `h8/h16/h32`, `lr=1e-3`, `100` steps as novelty.
Next allowed test: Add explicit norm/gap regularization, a more stable target construction, or a qualitatively different learned-gradient state.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-directional-loss-gate.md`

DISPROVEN: Gap-penalized validation selection alone resolves directional-loss overfit.
Conclusion: Gap penalties moved `cosine` h16 earlier, but penalty `4` still had result gap `0.1673`, while penalty `5` reduced gap to `0.1511/0.1220` and dropped heldout to `0.6872/0.6979`.
Do not repeat: Gap-penalized selection on the same directional-loss `injection_grad_logits`, target-normalized h16/h32 setup with penalties `1/3/4/5`.
Next allowed test: Use training-time regularization, target stabilization, or a different learned-gradient state, not checkpoint selection alone.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-gap-penalized-selection-gate.md`

DISPROVEN: Simple dropout regularization rescues directional-loss online MLP shadow overfit.
Conclusion: Dropout `0.1/0.2` with `weight_decay=0.01` preserved heldout cosines on the target-normalized `cosine` branch, but h16/h32 still had result train-heldout gaps near `0.20`; best h32/dropout `0.1` reached heldout `0.7920/0.8248` with gaps `0.2039/0.1564`.
Do not repeat: Dropout-only h16/h32 sweeps on the same `injection_grad_logits`, target-normalized, `cosine`, `lr=1e-3`, `100`-step setup as novelty.
Next allowed test: Change target construction or learned-gradient state, or add explicit training-time gap/norm penalties rather than ordinary dropout alone.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-dropout-regularization-gate.md`

DISPROVEN: Per-example unit-norm target stabilization rescues directional-loss online MLP shadow overfit.
Conclusion: Unit-normalizing each target row before fit-split z-scoring preserved the same heldout cosines but kept result gaps near `0.20`; best h32/cosine reached heldout `0.7936/0.8270` with gaps `0.2025/0.1545`.
Do not repeat: `unit_norm_per_example` target transform on the same `injection_grad_logits`, target-normalized h16/h32, `cosine`/`mse_plus_cosine`, `lr=1e-3`, `100`-step setup as novelty.
Next allowed test: More substantial target stabilization, a different learned-gradient state, or explicit train-time gap/norm penalties.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-target-transform-gate.md`

DISPROVEN: Fit-split result-prototype target averaging cleanly rescues directional-loss online MLP shadow overfit.
Conclusion: Prototype targets slightly improved the tradeoff but not enough; h32/cosine reached heldout `0.8040/0.8243` with gaps `0.1909/0.1557`, and h16/cosine plus gap selection reached `0.7540/0.7855` with gaps `0.1705/0.1409`.
Do not repeat: `fit_result_prototype` target transform on the same `injection_grad_logits`, target-normalized h16/h32, `cosine`/`mse_plus_cosine`, `lr=1e-3`, `100`-step setup, including gap penalties `3/4/5`, as novelty.
Next allowed test: Different learned-gradient state, explicit train-time gap/norm penalties, or a target construction richer than boundary-best class prototypes.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-target-prototype-gate.md`

DISPROVEN: Appending the raw result-projection input rescues directional-loss online MLP shadow overfit.
Conclusion: The result-input state improved upstream heldout alignment, but result gaps remained high; h16/cosine reached heldout `0.7676/0.8372` with gaps `0.1958/0.1269`, and h32/cosine reached `0.7895/0.8294` with gaps `0.2079/0.1533`.
Do not repeat: `injection_grad_logits_result_input` with target z-score, h16/h32, `cosine`/`mse_plus_cosine`, `lr=1e-3`, `100` steps, including h16 gap penalties `3/4/5`, as novelty.
Next allowed test: Explicit train-time gap/norm penalties, Jacobian-conditioned state, or another genuinely different learned-gradient target/state.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-result-input-state-gate.md`

DISPROVEN: Train-time validation prediction-loss regularization rescues directional-loss online MLP shadow overfit.
Conclusion: h32 with validation-loss weight `0.5/1.0` kept heldout cosines high (`0.7953/0.8233`, `0.7915/0.8195`) but result gaps stayed near `0.199`; h16/weight `1.0` reduced gaps to `0.1595/0.1150` but dropped heldout to `0.7274/0.7381` and inflated norms.
Do not repeat: Validation-loss weights `0.5/1.0` on the same `injection_grad_logits`, target-normalized h16/h32, `cosine`, `lr=1e-3`, `100`-step setup as novelty.
Next allowed test: A direct split-gradient gap/norm objective, Jacobian-conditioned state, or a richer target construction.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-loss-gate.md`

PARTIAL: Direct validation model-gradient regularization can clear Stage 0B, but a fixed calibrated module does not produce Stage 1 lift.
Conclusion: h32/validation-gradient `0.5`/norm `0.1` reached heldout `0.8068/0.8083` with gaps `0.1227/0.1343` and norms `1.1276/1.0736`; fixed-module Stage 1 weights `1.0/0.01/0.001` ended at `0.075/0.005/0.035` exact match.
Do not repeat: The same h16/h32 validation-gradient `0.5`, norm `0/0.1` Stage 0B grid or fixed-module Stage 1 weights `1.0/0.01/0.001` as novelty.
Next allowed test: Keep the direct gradient objective, but refresh the shadow module on-policy, add trust-region/norm clamps, or condition on state that remains valid after model movement.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-gradient-gate.md`

DISPROVEN: Simple fixed-module feedback L2 clamping rescues validation-gradient online shadow Stage 1.
Conclusion: Apply clamps `3.5` and `10` kept feedback norm bounded, but both runs ended at `0.075` final exact match with best snapshot `0.0525`, unchanged from unclamped weight `1.0`.
Do not repeat: The same fixed h32 validation-gradient module with simple apply max-norm clamps `3.5` or `10` as novelty.
Next allowed test: On-policy shadow refresh or a trust region that refreshes gradient agreement, not only output-vector norm.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-apply-norm-clamp-gate.md`

DISPROVEN: Periodic on-policy shadow refresh alone rescues validation-gradient online shadow Stage 1.
Conclusion: Refresh every `50` steps restored excellent current-model heldout gradient agreement (`0.982-0.998` result cosine, ~`1.0` upstream), but Stage 1 ended at `0.025` final exact match with best snapshot `0.0475`.
Do not repeat: Same h32 validation-gradient module with refresh every `50`, `shadow_feedback_weight=1.0`, no apply clamp, and 200-step budget as novelty.
Next allowed test: Add training-dynamics constraints such as step-level trust region, entropy/diversity stabilization, or a target/state that avoids single-result collapse.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-on-policy-refresh-gate.md`

DISPROVEN: Soft result-policy entropy/batch-diversity stabilization rescues refreshed online-shadow Stage 1.
Conclusion: Low diversity weight `1.0` still collapsed to one hard result with final exact `0.015` unbounded and `0.005` clamped; high diversity weight `100` plus clamp `10` kept hard usage broader (`9.14` effective hard results) but reached only `0.070` final and `0.080` best snapshot.
Do not repeat: Same refreshed h32 validation-gradient module with soft result-policy diversity weights `1` or `100`, optional tiny entropy, 200-step budget, and clamp `0/10` as novelty.
Next allowed test: A hard/assignment-style usage constraint, step-level trust region, Jacobian-conditioned state, or richer target that links diverse requests to per-example improvement.
Source: `aiAgentWorkHistory/phase7/2026-05-28-result-policy-soft-diversity-gate.md`

DISPROVEN: Bounding actual optimizer parameter-step L2 norm rescues refreshed online-shadow Stage 1.
Conclusion: Trust caps `0.05` and `0.10` scaled proposed AdamW deltas from about `0.17-0.20`, stabilized shadow norms and refresh agreement, but ended at only `0.075`/`0.040` final exact with best snapshots `0.060`/`0.045`.
Do not repeat: Same refreshed h32 validation-gradient module with feedback clamp `10`, optimizer step max deltas `0.05` or `0.10`, and 200-step budget as novelty.
Next allowed test: Trust region that validates per-step improvement, hard assignment-style usage constraints, Jacobian-conditioned state, or richer targets.
Source: `aiAgentWorkHistory/phase7/2026-05-28-optimizer-step-trust-region-gate.md`

DISPROVEN: Hard-path answer-loss step acceptance rescues refreshed online-shadow Stage 1.
Conclusion: Accept/reject gating with tolerances `0.0` and `0.1` accepted only `6/200` proposed steps (`3%`) and ended at `0.050` final exact with best snapshot `0.070`.
Do not repeat: Same refreshed h32 validation-gradient module with feedback clamp `10`, answer-loss acceptance tolerance `0.0` or `0.1`, and 200-step budget as novelty.
Next allowed test: A mechanism that repairs/constructs useful directions rather than simply rejecting most shadow steps, hard assignment-style usage constraints, Jacobian-conditioned state, or richer targets.
Source: `aiAgentWorkHistory/phase7/2026-05-28-answer-loss-step-acceptance-gate.md`

DISPROVEN: Hard-path answer-loss line search over proposed shadow step scales rescues refreshed online-shadow Stage 1.
Conclusion: Scales `1,0.5,0.25,0.1,0` accepted only `5/200` steps (`2.5%`); best snapshot improved to `0.0925`, but final exact was only `0.060`.
Do not repeat: Same refreshed h32 validation-gradient module with feedback clamp `10`, answer-loss line-search scales `1,0.5,0.25,0.1,0`, and 200-step budget as novelty.
Next allowed test: Construct better directions, hard assignment-style usage constraints, Jacobian-conditioned state, or richer targets rather than selecting among mostly harmful proposed shadow steps.
Source: `aiAgentWorkHistory/phase7/2026-05-28-answer-loss-line-search-gate.md`

PARTIAL: Output-Jacobian-conditioned shadow features improve Stage 0B but do not rescue refreshed online-shadow Stage 1.
Conclusion: `injection_grad_logits_output_jacobian` with feature z-scoring cleared Stage 0B at `0.9073/0.9011` heldout result/upstream cosines, but refreshed clamp-`10` Stage 1 ended at `0.055` final exact with best snapshot `0.065`.
Do not repeat: h16/h32 raw output-Jacobian features or h32 fit-split feature z-scoring with validation-gradient `0.5`, norm `0.1`, refresh every `50`, clamp `10`, and 200-step budget as novelty.
Next allowed test: Hard assignment-style usage constraints, richer targets, or a more substantial learned-gradient update path; do not treat this state-only Jacobian feature as enough.
Source: `aiAgentWorkHistory/phase7/2026-05-28-output-jacobian-shadow-feature-gate.md`

PARTIAL: Hard answer-loss improvement assignments can produce Stage 1 lift.
Conclusion: Assignment weight `10` reached `0.170` final exact with refreshed shadow and `0.400` final exact without shadow at 200 steps; weight `1` collapsed at `0.0475`.
Do not repeat: Same seed-2/seed-4 exact-grid 200-step assignment weights `1` or `10` as novelty.
Next allowed test: Longer convergence, target-off retention, seed replication, or cheaper/scalable assignment construction; do not claim final success because this still scores forced result classes during training.
Source: `aiAgentWorkHistory/phase7/2026-05-28-hard-improvement-assignment-gate.md`

DISPROVEN: Plain linear target-off decay lets answer loss retain the hard improvement-assignment interface.
Conclusion: Assignment weight `10 -> 0` over 200 steps with `answer_loss_weight=1` peaked at `0.370` before shutoff, was `0.3475` at step `200`, collapsed to `0.105` by step `250`, and ended at `0.1075`.
Do not repeat: Same seed-2/seed-4 exact-grid no-shadow assignment decay over `200` steps with 400-step budget as novelty.
Next allowed test: Longer always-on convergence, seed replication, a stronger handoff bridge, or lower-cost assignment approximation.
Source: `aiAgentWorkHistory/phase7/2026-05-28-hard-improvement-assignment-retention-gate.md`

PARTIAL: Always-on hard improvement assignment can train a natural result-space calculator interface across seeds.
Conclusion: With assignment weight `10` kept on, three 1600-step exact-grid seeds ended at `0.915`, `0.860`, and `0.820` final exact; best snapshots reached `0.9475`, `0.870`, and `0.920`.
Do not repeat: The same no-shadow 800/1600-step always-on assignment runs on CLI seeds `2/4/5` as novelty.
Next allowed test: Cheaper/scalable assignment construction, stronger target-off handoff, stability/selection to avoid late drift, or a non-bottleneck version of the gate.
Source: `aiAgentWorkHistory/phase7/2026-05-28-hard-improvement-assignment-convergence-gate.md`

DISPROVEN: The bottleneck hard-assignment signal transfers directly to an additive non-bottleneck model.
Conclusion: Non-bottleneck answer+assignment reached `0.700` final exact, but calculator-result accuracy stayed near chance (`0.0275` final) and assignment target accuracy fell to `0.0033`; the answer-only baseline also solved substantially with high injection-zero.
Do not repeat: Same additive result-space `ste`, answer-loss `1`, assignment weight `10`, 800-step seed-2 exact-grid gate as novelty.
Next allowed test: Add causal calculator-use pressure, staged bottleneck-to-additive handoff, or a target that remains tied to true calculator utility when the neuron path can bypass.
Source: `aiAgentWorkHistory/phase7/2026-05-28-non-bottleneck-hard-assignment-gate.md`

DISPROVEN: A zero-injection causal-gap hinge is enough to make non-bottleneck hard assignment learn calculator use.
Conclusion: Gap weights `10/50` with margin `0.5` produced final causal gaps `1.27/0.84`, but final calculator-result accuracy stayed `0.000/0.0425` and final exact fell to `0.560/0.4225`.
Do not repeat: Same additive assignment-weight `10`, causal-gap margin `0.5`, weights `10/50`, 800-step seed-2 exact-grid gate as novelty.
Next allowed test: A staged bottleneck-to-additive handoff or causal target that rewards correct result-level utility, not merely making zero-injection worse.
Source: `aiAgentWorkHistory/phase7/2026-05-28-non-bottleneck-causal-gap-gate.md`

PARTIAL: A frozen bottleneck-trained result policy can be handed to an additive non-bottleneck model.
Conclusion: Compatible checkpoint loading preserved a `0.9125` bottleneck result policy; without freezing it collapsed to `0.0300` by step `50`, but freezing embeddings/pre-hook block/result head kept final calculator-result accuracy `0.9200` and produced `0.9475` normal versus `0.0175` injection-zero.
Do not repeat: Same seed/checkpoint compatible transfer without freezing, or same frozen-policy 800-step handoff as novelty.
Next allowed test: Seed/checkpoint replication, staged unfreezing, or a scalable/non-prescriptive way to acquire and preserve the policy.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-transfer-gate.md`

PARTIAL: Frozen handoff replicates for a strong source checkpoint but is sensitive to source checkpoint quality.
Conclusion: Strong source `src2` transferred to additive seeds `2/4` with final eval `0.9400/0.9525` and learned calc `0.9200/0.9150`; weaker sources `src4/src5` preserved learned calc around `0.80-0.87` but reached only `0.3025-0.5550` final eval by 800 steps.
Do not repeat: The same frozen 800-step matrix cells `src2_add2`, `src2_add4`, `src4_add2`, `src4_add4`, or `src5_add5` as novelty.
Next allowed test: Source checkpoint selection/quality metrics, longer or stronger downstream readout adaptation, staged unfreezing, or a less prescriptive source-policy training method.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-transfer-replication.md`

PARTIAL: Longer downstream adaptation helps weak frozen handoffs but does not erase source sensitivity.
Conclusion: Continuing weak cells for another 800 steps improved `src5_add5` final eval `0.5550 -> 0.8175` and `src4_add2` `0.3025 -> 0.6050`, while injection-zero stayed near chance and learned calc stayed `0.8000/0.8725`.
Do not repeat: The same `src4_add2` or `src5_add5` one-extra-800-step continuation as novelty.
Next allowed test: Better source checkpoint selection, stronger readout adaptation, controlled unfreezing, or source-policy training that produces more handoff-friendly representations.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-downstream-adaptation.md`

DISPROVEN: Low-LR full-policy unfreezing preserves adapted non-bottleneck calculator use.
Conclusion: From adapted weak-source checkpoints, unfreezing all policy parameters at LR `3e-4` for 400 steps collapsed learned calc from `0.8725 -> 0.3000` and `0.8000 -> 0.2525`; answer accuracy did not improve.
Do not repeat: The same `src4_add2` or `src5_add5` adapted-checkpoint low-LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Selective unfreezing, explicit policy-retention regularization, or unfreeze schedules gated by calculator-result accuracy.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-low-lr-unfreeze.md`

PARTIAL: Result-policy anchoring can make full-policy unfreeze useful after staged handoff.
Conclusion: KL anchor weight `10` at LR `3e-4` preserved learned calc (`0.8075/0.7950`) and improved final eval over frozen adapted baselines (`src4_add2 0.6050 -> 0.7475`, `src5_add5 0.8175 -> 0.9525`).
Do not repeat: Same adapted `src4_add2/src5_add5`, anchor weight `10`, LR `3e-4`, 400-step KL-anchor full unfreeze as novelty.
Next allowed test: Anchor decay/off-ramp, selective unfreeze, source checkpoint selection, or less prescriptive source-policy acquisition.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-policy-anchor-unfreeze.md`

DISPROVEN: A fast KL-anchor off-ramp makes the adapted non-bottleneck policy self-sustaining.
Conclusion: Decaying anchor weight `10 -> 0` over the first `200/400` unfreeze steps preserved calc accuracy at shutoff (`0.8300/0.8225`) but final calc fell to `0.5950/0.3850`, with final eval `0.5925/0.6750`.
Do not repeat: Same adapted `src4_add2/src5_add5`, anchor weight `10`, decay `200`, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Slower or floored anchor schedules, calculator-accuracy-gated unfreezing, selective unfreeze, or a source policy that is robust without anchoring.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-anchor-decay-offramp.md`

PARTIAL: Reduced constant KL anchors can preserve non-bottleneck calculator use.
Conclusion: Anchor weights `1.0` and `0.1` at LR `3e-4` kept final calc near `0.77-0.81`, final eval `0.7775/0.9925` for weight `1` and `0.8325/0.9750` for weight `0.1`, with injection-zero near chance.
Do not repeat: Same adapted `src4_add2/src5_add5`, anchor weights `1.0` or `0.1`, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Even weaker/floored/gated anchors, selective unfreeze, or source-policy training that reduces the need for an anchor.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-reduced-anchor-strength.md`

MIXED: Constant KL anchor `0.01` is below the clean policy-retention region.
Conclusion: Anchor `0.01` kept injection-zero near chance and final eval `0.7850/0.9375`, but final calc fell to `0.7625/0.6425` and anchor agreement to `0.8825/0.7050`.
Do not repeat: Same adapted `src4_add2/src5_add5`, anchor weight `0.01`, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Floored or gated schedules around the `0.1` region, selective unfreezing, or policy acquisition that reduces active anchoring needs.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-anchor-threshold.md`

PARTIAL: A nonzero anchor floor rescues the failed zero-off-ramp pattern.
Conclusion: Anchor `1.0 -> 0.1` over 200 steps kept final calc `0.8175/0.7800`, final eval `0.7925/0.9775`, and injection-zero `0.0250/0.0075`, but did not beat constant anchor `0.1`.
Do not repeat: Same adapted `src4_add2/src5_add5`, anchor `1.0`, decay `200`, floor `0.1`, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Calculator-accuracy-gated retention, adaptive floors, selective unfreeze, or source-policy acquisition that reduces active anchoring needs.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-anchor-floor-schedule.md`

DISPROVEN: Freezing only the calculator action head preserves transferred policy during unfreeze.
Conclusion: With `result_proj` frozen and only upstream trainable, adapted `src4_add2/src5_add5` still collapsed to final calc `0.3000/0.2525` and final eval `0.5200/0.8100`, matching the earlier plain unfreeze failure.
Do not repeat: Same adapted `src4_add2/src5_add5`, `--freeze-calculator-action-head`, no anchor, LR `3e-4`, 400-step unfreeze as novelty.
Next allowed test: Behavior-level anchoring/gating, freezing the upstream policy path, or a more targeted selective parameter set that prevents upstream representation drift.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-freeze-action-head.md`

PARTIAL: Behavior-gated anchoring can reduce average anchor weight but does not beat fixed `0.1`.
Conclusion: Base anchor `0.01` with agreement gate `<0.9 -> 0.1` ended with final calc `0.7700/0.7700` and final eval `0.8050/0.9675`; it improved over constant `0.01` but was roughly comparable to, not better than, constant `0.1`.
Do not repeat: Same adapted `src4_add2/src5_add5`, base anchor `0.01`, gate threshold `0.9`, gate weight `0.1`, argmax-agreement gate, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Better gate metric/threshold, adaptive continuous weights, calculator-accuracy-gated retention, or source-policy acquisition that reduces active anchoring needs.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-behavior-gated-anchor.md`

MIXED: Simple calculator-accuracy-gated anchoring is threshold-sensitive and does not beat fixed `0.1`.
Conclusion: Base anchor `0.01` with `current_argmax_accuracy` gates `<0.80` or `<0.82 -> 0.1` reached `src5` final eval `0.9825` at both thresholds, but `src4` reached only `0.7725/0.7900`, below fixed anchor `0.1` (`0.8325`).
Do not repeat: Same adapted `src4_add2/src5_add5`, base anchor `0.01`, gate thresholds `0.80` or `0.82`, gate weight `0.1`, calculator-accuracy gate, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Continuous/adaptive anchor control, selective policy-path unfreezing, stronger source acquisition, or a retention signal that combines calculator accuracy with answer utility.
Source: `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-accuracy-gated-anchor.md`

PARTIAL: Continuous calculator-accuracy anchor control can lower average retention weight but is not a clean fixed-`0.1` replacement.
Conclusion: Linear gate with base `0.01`, threshold `0.85`, band `0.10`, and max `0.1` reached `src4` final eval `0.8375` with mean weight `0.0385`, but `src5` ended `0.9725`, slightly below fixed `0.1`/discrete accuracy gates.
Do not repeat: Same adapted `src4_add2/src5_add5`, base `0.01`, linear `current_argmax_accuracy` gate threshold `0.85`, band `0.10`, gate weight `0.1`, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Source-policy acquisition, selective policy-path unfreezing, or a retention controller that combines calculator accuracy with answer utility instead of metric-only anchor scaling.
Source: `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-continuous-anchor-gate.md`

PARTIAL: Freezing the policy backbone prevents no-anchor policy collapse, but action-head/readout adaptation alone is weaker than anchored unfreezing.
Conclusion: `--freeze-calculator-policy-backbone` with no anchor preserved final learned calc `0.8200/0.8025` and improved adapted weak handoffs to final eval `0.7250/0.8700`, but stayed below lightweight anchor results.
Do not repeat: Same adapted `src4_add2/src5_add5`, no anchor, `--freeze-calculator-policy-backbone`, LR `3e-4`, 400-step unfreeze as novelty.
Next allowed test: Combine policy-backbone freezing with lightweight/utility-aware retention, improve source-policy acquisition, or test whether a different movable parameter set improves readout without policy drift.
Source: `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-policy-backbone-freeze.md`

NO-GAIN: Tiny anchoring is redundant when the policy backbone is frozen.
Conclusion: `--freeze-calculator-policy-backbone` plus KL anchor `0.01` kept anchor agreement `1.0000/0.9975` and learned calc `0.8200/0.8000`, but final eval `0.7125/0.8600` was slightly below no-anchor backbone freeze.
Do not repeat: Same adapted `src4_add2/src5_add5`, `--freeze-calculator-policy-backbone`, result-policy anchor `0.01`, LR `3e-4`, 400-step unfreeze as novelty.
Next allowed test: Improve downstream/readout adaptation under stable policy, use answer-utility-aware retention, or improve source-policy acquisition; tiny action-policy anchoring is not the missing ingredient here.
Source: `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-policy-backbone-tiny-anchor.md`

MIXED: Longer stable-policy readout adaptation helps good sources but does not erase weak-source sensitivity.
Conclusion: `--freeze-calculator-policy-backbone`, no anchor, 1600 steps lifted `src5_add5` to final eval `0.9500` with learned calc `0.8325`, but `src4_add2` reached only `0.7550` despite learned calc `0.8550`.
Do not repeat: Same adapted `src4_add2/src5_add5`, no anchor, `--freeze-calculator-policy-backbone`, LR `3e-4`, 1600-step adaptation as novelty.
Next allowed test: Source-policy acquisition/selection, stronger downstream adaptation targeted at weak sources, or a utility-aware readout objective under stable policy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-policy-backbone-long-adaptation.md`

PARTIAL: Source checkpoint selection improves weak-source handoff but source action accuracy is not sufficient.
Conclusion: Reproducing `src5` with checkpoint snapshots and transferring the source step-1500 checkpoint (`0.9200` source normal/calc) improved immediate frozen-policy additive handoff from the old final-checkpoint baseline `0.5550` to `0.6975`.
Do not repeat: Same `src5` step-1500 selected-source checkpoint into additive seed `5`, frozen-policy, 800-step handoff as novelty.
Next allowed test: Source-selection metrics beyond normal/calc accuracy, source acquisition for handoff-friendly geometry, stronger selected-source replication, or utility-aware stable-policy readout adaptation.
Source: `aiAgentWorkHistory/phase7/2026-05-29-source-checkpoint-selection-gate.md`

DISPROVEN: Highest source normal/calculator accuracy is a reliable source checkpoint selector.
Conclusion: Reproduced `src2` with checkpoints; source step `1300` had higher source normal/calc (`0.9475`) than final (`0.9150`) but transferred worse into additive seed `4` (`0.8675` vs final-control `0.9525`).
Do not repeat: Same `src2` step-1300 versus final additive seed-4 frozen-policy 800-step transfer as novelty.
Next allowed test: Source-quality probes for handoff geometry, source acquisition optimized for transfer/readout learnability, or selected-source replication with a selector beyond source accuracy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-source-selection-metric-replication.md`

PARTIAL: Short additive handoff progress predicts final handoff better than source accuracy.
Conclusion: Across six non-continued frozen-policy transfer cells, normal accuracy at step `400` correlated with final eval at `0.9374`, and step `600` at `0.9935`; step `200` was noisy (`-0.0959`).
Do not repeat: Same trace audit over the current frozen-policy transfer cells as novelty.
Next allowed test: Use a 400/600-step handoff probe for checkpoint selection, build a cheaper readout/linear proxy for that probe, or optimize source acquisition for early additive handoff slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-short-handoff-probe-audit.md`

DISPROVEN: The initial frozen-state readout probe was a reliable handoff proxy.
Conclusion: Reusable script validation exposed that the scratch probe used the wrong `EQ_ID`/leaky position. Correct safe features over five checkpoints had weak correlation with final handoff (`read_pair 0.2118`, `layer2_pair 0.2865`).
Do not repeat: Same five-checkpoint safe frozen-state readout probe as novelty, or the leaky/wrong-position answer-token probe.
Next allowed test: Build a better non-leaky geometry proxy, validate source selectors on unseen checkpoints, or use 400/600-step handoff probes until a cheaper proxy is proven.
Source: `aiAgentWorkHistory/phase7/2026-05-29-frozen-state-readout-probe.md`

POSITIVE: A 600-step handoff probe can select a better source checkpoint than source accuracy.
Conclusion: On `src5`, the 600-step probe selected step `1100` (source normal `0.8400`) over the source-accuracy-selected step `1500` (`0.9200`); full additive handoff improved from `0.6975` to `0.7950`.
Do not repeat: Same `src5` step `1100/1400/1500/final`, additive seed `5`, frozen-policy handoff-probe comparison as novelty.
Next allowed test: Use the 600-step handoff probe on newly acquired source checkpoints, reduce/approximate its cost, or optimize source acquisition for probe score.
Source: `aiAgentWorkHistory/phase7/2026-05-29-handoff-probe-selector-validation.md`

POSITIVE: The 600-step handoff probe rescues weak `src4` source selection.
Conclusion: Reproduced `src4` with snapshots; 600-step probe selected step `1200` (source normal `0.7550`) over final (`0.8700`), and full frozen handoff improved from old final-source `0.3025` to `0.7800`.
Do not repeat: Same `src4` step `1000/1200/final`, additive seed `2`, frozen-policy handoff-probe comparison as novelty.
Next allowed test: Use probe score during source acquisition, reduce probe cost, or test whether probe-selected sources reduce later anchor/long-adaptation needs.
Source: `aiAgentWorkHistory/phase7/2026-05-29-handoff-probe-selector-src4.md`

MIXED-POSITIVE: Probe-selected sources help later stable-policy adaptation for weak handoffs, but are not a universal long-adaptation selector.
Conclusion: 1600-step no-anchor `--freeze-calculator-policy-backbone` adaptation from probe-selected checkpoints lifted `src4` from old final-source long adaptation `0.7550` to `0.8900`, but `src5` reached `0.9250`, below the old final-source long adaptation `0.9500`.
Do not repeat: Same probe-selected `src4` step-1200/add2 or `src5` step-1100/add5 frozen handoff checkpoint into no-anchor policy-backbone-frozen 1600-step adaptation as novelty.
Next allowed test: Add a second-stage long-adaptation/readout-compatibility selector, optimize source acquisition for both 600-step handoff slope and later readout adaptability, or reduce the handoff-probe cost.
Source: `aiAgentWorkHistory/phase7/2026-05-29-probe-selected-policy-backbone-adaptation.md`

DISPROVEN: The `src5` source-accuracy-selected step-1500 checkpoint is the better long-adaptation candidate.
Conclusion: Starting from the existing step-1500 800-step frozen handoff and adapting for 1600 steps with policy backbone frozen reached final eval `0.9100`, below the handoff-probe-selected step-1100 result `0.9250`, despite higher final calc accuracy (`0.9325` vs `0.8275`).
Do not repeat: Same `src5` step-1500 800-step frozen handoff into no-anchor policy-backbone-frozen 1600-step adaptation as novelty.
Next allowed test: Compare against the exact old final-source checkpoint lineage, inspect reproduced-versus-old final-source differences, or optimize source acquisition for downstream readout compatibility.
Source: `aiAgentWorkHistory/phase7/2026-05-29-long-adaptation-selector-probe.md`

POSITIVE: The old `src5` final-source long-adaptation advantage was mostly a continuation-depth fairness issue.
Conclusion: Giving the handoff-probe-selected `src5` step-1100 lineage the same extra 800-step frozen-policy continuation lifted it from `0.7950` to `0.8800`, then 1600-step stable-policy adaptation reached `0.9425`, nearly matching the old final-source `0.9500`.
Do not repeat: Same `src5` step-1100 selected handoff plus extra 800-step frozen-policy continuation plus no-anchor policy-backbone-frozen 1600-step adaptation as novelty.
Next allowed test: Apply the fair continuation recipe to `src4` step-1200, or optimize source acquisition for early handoff and continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-selected-source-continuation-fairness.md`

POSITIVE: Fair continuation also improves the weak `src4` selected-source lineage.
Conclusion: `src4` step-1200 selected handoff improved from `0.7800` to `0.8150` with another 800 frozen-policy steps, then reached `0.9125` after policy-backbone-frozen long adaptation, beating direct selected long (`0.8900`) and old final-source long (`0.7550`).
Do not repeat: Same `src4` step-1200 selected handoff plus extra 800-step frozen-policy continuation plus no-anchor policy-backbone-frozen 1600-step adaptation as novelty.
Next allowed test: Reduce handoff-probe/continuation cost, or optimize source acquisition for early handoff and continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-src4-selected-source-continuation-fairness.md`

MIXED-POSITIVE: Stable readout adaptation can be cut to 600 steps, but 200 steps is source-sensitive.
Conclusion: From continued selected checkpoints, 200-step readout worked for `src5` (`0.9275`) but not `src4` (`0.8775`); 600-step readout passed both (`src4 0.9025`, `src5 0.9325`) with injection-zero near zero and forced-random near chance.
Do not repeat: Same 200/600-step no-anchor policy-backbone-frozen readout adaptation from continued selected `src4` step-1200/add2 and `src5` step-1100/add5 checkpoints as novelty.
Next allowed test: Reduce the 600-step handoff probe or 800-step frozen-policy continuation cost, or optimize source acquisition for early handoff/continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-reduced-readout-budget-validation.md`

MIXED-POSITIVE: The source-selection handoff probe can be cut to 500 steps on current candidates, but not to 400.
Conclusion: Existing traces show 400-step probe would pick `src5` step-1500, but 500-step probe picks the same checkpoints as 600 for both audited families (`src5` step-1100, `src4` step-1200).
Do not repeat: Same trace audit over existing `src5` 1100/1400/1500/final and `src4` 1000/1200/final handoff probes as novelty.
Next allowed test: Validate 500-step selection on new source checkpoints, reduce the 800-step continuation cost, or optimize source acquisition for early handoff/continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-shorter-handoff-probe-trace-audit.md`

MIXED-NEGATIVE: Reducing the frozen-policy continuation from 800 to 600 steps is source-sensitive.
Conclusion: With 600-step readout after reduced continuation, `src5` still passed (`0.9275` vs `0.9325` reference), but weak `src4` fell below gate (`0.8750` vs `0.9025` reference) despite retained calculator dependence.
Do not repeat: Same 600-step continuation plus 600-step policy-backbone-frozen readout from selected `src4` step-1200/add2 and `src5` step-1100/add5 checkpoints as novelty.
Next allowed test: Keep 800 continuation for weak sources, test 700-step continuation only for fine-grained tuning, or optimize source acquisition for continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-reduced-continuation-budget-validation.md`

POSITIVE: The 500-step handoff selector validates on the `src2` source-accuracy counterexample.
Conclusion: For `src2` additive seed `4`, 500-step handoff progress picks final/source step-1600 (`0.6900`) over source-accuracy-favored step-1300 (`0.5875`), matching final handoff (`0.9525` vs `0.8675`).
Do not repeat: Same `src2` step-1300 versus final additive seed-4 400/500/600-step trace audit as novelty.
Next allowed test: Validate 500-step selection on new source checkpoints, or optimize source acquisition for early handoff/continuation slope directly.
Source: `aiAgentWorkHistory/phase7/2026-05-29-src2-500-step-selector-validation.md`

SYNTHESIS: The selected-source non-bottleneck recipe is progressing, not looping.
Conclusion: Recent work moved from disproving source-accuracy selection to validating a 500-step handoff selector on `src2/src4/src5`, keeping 800 continuation for weak sources, and reducing stable readout to 600 steps (`src4 0.9025`, `src5 0.9325`).
Do not repeat: Existing source-accuracy selector tests, existing `src4/src5` 200/600 readout cuts, existing 600-step continuation cut, or existing `src2` 500-step selector trace as novelty.
Next allowed test: Validate the 500-step selector on newly acquired source checkpoints, or train source acquisition directly for early handoff and continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-periodic-review-selected-source-recipe.md`

MIXED-NEGATIVE: The 500-step selector does not generalize cleanly to fresh `src6`.
Conclusion: Fresh `src6` 500-step handoff scores pick step `1500` (`0.7200`) over final (`0.6850`), but full 800-step handoff is better from final (`0.8975` vs `0.8875`); 600-step scores would pick final (`0.8050` vs `0.7800`).
Do not repeat: Same `src6` step-1200/step-1500/final additive seed-6 frozen-policy 800-step comparison as novelty.
Next allowed test: Use 600-step selection for fresh sources, run continuation/readout from fresh `src6` final, or optimize source acquisition for 600-step handoff/continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-new-source-500-selector-validation.md`

POSITIVE: The selected-source continuation/readout recipe clears the gate on fresh `src6`.
Conclusion: Fresh `src6` final-source handoff was near-gate (`0.8975`); 800 frozen-policy continuation reached `0.9625`, and 600 policy-backbone-frozen readout reached `0.9850` with controls far below normal.
Do not repeat: Same `src6` final-source continuation plus 600-step readout as novelty.
Next allowed test: Replicate on another fresh source with 600-step selection, or optimize source acquisition for 600-step handoff/continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-src6-selected-continuation-readout.md`

MIXED: The 600-step selector replicated on fresh `src7`, but the reduced recipe did not clear the gate.
Conclusion: On weak fresh `src7`, 600-step handoff selection picked step `1400` (`0.5025`) over step `1000` (`0.4850`) and final (`0.4150`), matching the full-handoff winner; continuation/readout improved `0.7325 -> 0.8125 -> 0.8825` but missed the `0.90` gate.
Do not repeat: Same `src7` step-1000/step-1400/final handoff comparison or step-1400 continuation/readout chain as novelty.
Next allowed test: Optimize source acquisition for 600-step handoff/continuation slope, or use fresh-source replications only as planned acquisition gates.
Source: `aiAgentWorkHistory/phase7/2026-05-29-src7-600-selector-replication.md`

DISPROVEN: Decaying small entropy/batch-diversity source stabilization to zero will rescue weak fresh-source acquisition.
Conclusion: Fresh seed-9 source acquisition with entropy `0.05`, batch diversity `0.1`, improvement assignment `10`, and decay-to-zero over 1200 steps peaked at `0.7050` around steps `700/900`, then collapsed to final `0.1825`.
Do not repeat: Same decay-to-zero source-only recipe with answer loss off as novelty.
Next allowed test: Keep a nonzero source-objective floor, add policy anchoring, or optimize source acquisition for 600-step handoff/continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-source-acquisition-stabilization-variant.md`

MIXED-POSITIVE: Keeping entropy/diversity source stabilization active fixes collapse; initial handoff is weak, but continuation/readout can unlock the source.
Conclusion: No-decay entropy `0.05` + batch diversity `0.1` + improvement assignment `10` reached source step `1400` normal `0.9100` and final eval `0.8575`; final-source handoff started weak (`0.6500`) but 800-step continuation reached `0.9050` and 600-step readout reached `0.9575`.
Do not repeat: Same no-decay source recipe plus step `1400`/final additive seed-9 handoff, direct readout, continuation, and readout chain as novelty.
Next allowed test: Replicate no-decay stabilized continuation/readout on another seed, reduce continuation cost, or build a cheaper proxy for continuation/readout slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-source-acquisition-stabilization-floor.md`

POSITIVE: The no-decay stabilized final-source lineage clears the non-bottleneck gate after continuation/readout.
Conclusion: Starting from final-source handoff `0.6500`, direct 600-step readout reached `0.8000`, 800-step frozen-policy continuation reached `0.9050`, and 600-step post-continuation readout reached `0.9575` with controls far below normal.
Do not repeat: Same no-decay final-source handoff into direct readout, continuation, and post-continuation readout as novelty.
Next allowed test: Replicate on another fresh stabilized source, reduce continuation cost, or identify a cheap continuation/readout-slope proxy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-continuation-readout.md`

POSITIVE: The no-decay stabilized source can use 600-step continuation and still clear the non-bottleneck gate.
Conclusion: Reading out from the step-600 continuation checkpoint reached final eval `0.9425` with injection-zero `0.0078`, forced-random `0.0781`, and learned calc `0.8750`; this is only `0.0150` below the 800-continuation readout.
Do not repeat: Same no-decay stabilized step-600 continuation checkpoint into 600-step readout as novelty.
Next allowed test: Replicate the 600-continuation recipe on another stabilized source, test 500 continuation, or build a proxy for deciding continuation budget.
Source: `aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-reduced-continuation.md`

MIXED-POSITIVE: The no-decay stabilized source continuation budget can be cut to 400 steps, but not 300 by final eval.
Conclusion: 600-step readout from continuation checkpoints reached final eval `0.9400` at step `500`, `0.9175` at step `400`, and `0.8850` at step `300`; controls stayed far below normal.
Do not repeat: Same no-decay stabilized `600/500/400/300` continuation-checkpoint readout ladder as novelty.
Next allowed test: Replicate the 400/500 boundary on another no-decay stabilized source, validate a readout-snapshot selector, or build a cheap continuation-budget proxy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-continuation-boundary.md`

MIXED-NEGATIVE: No-decay stabilized source acquisition reliably implies transferable non-bottleneck geometry.
Conclusion: Fresh CLI seed `10` reached source final `0.9000` and learned calc `0.8984`, but final-source handoff reached only `0.3275`, direct readout `0.4275`, and 800-step continuation `0.4350`.
Do not repeat: Same seed-10 no-decay source, final-source handoff, direct readout, or 800-step continuation as novelty.
Next allowed test: Compare seed-9 positive vs seed-10 negative geometry, build a transfer/readout proxy, or optimize source acquisition for continuation/readout geometry.
Source: `aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-seed10-replication.md`

MIXED-NEGATIVE: The seed-10 transfer failure is only a bad final-checkpoint selection artifact.
Conclusion: Earlier seed-10 checkpoints improved 600-step handoff over final (`0.4475/0.4325/0.4225` vs `0.3375`), but all stayed below seed-9 final reference (`0.5250` at 600 and `0.6500` final eval); frozen-state linear probing was not a valid selector because it ranked seed-10 final highest (`0.4500`) despite worst handoff.
Do not repeat: Same seed-10 step `1000`/`1300`/`1400` 600-step handoff sweep or frozen-state linear probe over these checkpoints as novelty.
Next allowed test: Build an additive learning-slope or injection-to-answer geometry proxy, or optimize source acquisition for early handoff slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-seed10-source-checkpoint-geometry-sweep.md`

MIXED-NEGATIVE: A direct additive handoff geometry probe can replace the 400/600-step handoff selector.
Conclusion: Forced-result geometry flags seed-10 as hostile (`true_best=0.0`, true top-3 `0.03-0.045`, true-best gap `0.0058-0.0063`) versus seed-9 positive (`true_best=0.0625`, top-3 `0.2125`, gap `0.0034`), but it does not cleanly separate `src6` positive from `src7` boundary-negative and 100-step loss slope is not a reliable selector.
Do not repeat: Same geometry probe over seed-9 final, seed-10 `1000/1300/1400/final`, `src6` final, or `src7` step `1400` as novelty.
Next allowed test: Add forced-result geometry as a source-training snapshot metric, optimize it during source acquisition, or keep using actual handoff probes as selection gates.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-handoff-geometry-probe.md`

DISPROVEN: Additive forced-result geometry can select source checkpoints inside known source families.
Conclusion: On known `src2/src4/src5` handoff comparisons, geometry only partially identified `src5` step `1100`; it tied or favored non-winners for `src4` and tied `src2` step `1300` versus final, while true-best gap selected wrong checkpoints.
Do not repeat: Same geometry scan over `src2` step `1300`/final, `src4` step `1000/1200`/final, or `src5` step `1100/1400/1500`/final as novelty.
Next allowed test: Use geometry only as logging/warning, optimize source acquisition for actual early handoff slope, or design a stronger one/few-update proxy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-geometry-selector-validation.md`

DISPROVEN: A 25/50/100-step downstream loss-slope proxy can replace the additive handoff selector.
Conclusion: On known `src2/src4/src5` handoff comparisons, 100-step loss/loss-drop selects the wrong checkpoint for `src5` and `src4`, though it selects `src2` final correctly; existing exact-match traces show `src5` still needs about 500 steps to select its known winner.
Do not repeat: Same 0/25/50/100-step loss-slope probe over `src2` step `1300`/final, `src4` step `1000/1200`/final, or `src5` step `1100/1400/1500`/final as novelty.
Next allowed test: Keep 500/600-step handoff gates, optimize source acquisition against actual early handoff exact, or train a learned proxy on accumulated handoff traces.
Source: `aiAgentWorkHistory/phase7/2026-05-29-short-slope-selector-validation.md`

DISPROVEN: Lowering the hard improvement-assignment source weight from `10` to `5` rescues transfer-hostile seed-10 geometry.
Conclusion: The weight-5 seed-10 source weakened to final eval `0.6750`; its best source snapshots were around `0.78`, and 600-step additive handoffs from step `1200`/final reached only `0.3425`/`0.2475` snapshots and `0.3000`/`0.2325` final eval.
Do not repeat: Same seed-10 no-decay entropy `0.05`, diversity `0.1`, improvement weight `5`, 1600-step source run or step-1200/final 600-step frozen handoffs as novelty.
Next allowed test: Optimize source acquisition against actual 500/600-step handoff behavior, add a direct handoff/readout geometry term, or train a learned selector validated against the handoff gate.
Source: `aiAgentWorkHistory/phase7/2026-05-29-source-assignment-weight5-transfer-probe.md`

DISPROVEN: A simple learned ridge selector over early handoff trace features can replace the 500/600-step handoff gate.
Conclusion: Leave-family-out ridge over 21 deduped candidates and 8 source families reached `3/8`, `4/8`, `3/8`, and `5/8` winner accuracy at prediction steps `200/300/400/500`; raw early exact matched or beat it at every step and reached `6/8` at step `500`.
Do not repeat: Same ridge selector over normal/zero/oracle/forced-random/calc early trace features on the current Phase 7 handoff trace dataset as novelty.
Next allowed test: Add logging-only in-training additive handoff probes, collect more labeled families, or test a richer learned selector only if it beats raw early exact under leave-family-out validation.
Source: `aiAgentWorkHistory/phase7/2026-05-29-handoff-trace-learned-selector-audit.md`

TOOLING: Logging-only in-training additive handoff probes are implemented.
Conclusion: `overfit_one_batch.py` can now clone current source state into additive non-bottleneck mode, freeze the calculator policy, train a bounded downstream probe, and log probe rows/metrics without feeding probe gradients back into source training.
Do not repeat: One-step smoke under `runs/2026-05-29_phase7_additive_handoff_probe_logging_smoke` as novelty.
Next allowed test: Run a real source-acquisition lineage with meaningful 500-step probe logging and verify selected checkpoints with the established handoff gate.
Source: `aiAgentWorkHistory/phase7/2026-05-29-in-training-additive-handoff-probe-logging.md`

MIXED-NEGATIVE: A 500-step in-training additive handoff probe is not yet a reliable source-checkpoint selector.
Conclusion: On fresh no-decay source seed `11`, in-training probe normal @500 chose source step `400` (`0.5625`) over step `800` (`0.5525`), but standalone 600-step frozen-policy handoff favored step `800` by a wide margin (`0.6925` snapshot, `0.7075` final eval) over step `400` (`0.5975` snapshot, `0.6050` final eval).
Do not repeat: Same source step `400` vs step `800`, 500-step embedded probe plus standalone 600-step verification as novelty.
Next allowed test: Treat embedded 500-step probes as logging/triage only, verify with standalone 600-step handoffs, or run embedded probes with 600 steps / richer trend metrics before using them for selection.
Source: `aiAgentWorkHistory/phase7/2026-05-29-intraining-probe-source-selection-validation.md`

PARTIAL: Local-target propagation style result-boundary targets can pass Stage 0, but sharp settings mostly recover the full-enum boundary teacher.
Conclusion: A new exact-grid Stage 0 diagnostic found that current-policy-reweighted forced-loss targets align with the hard boundary ceiling when sharp (`t=0.25` result/upstream cosine `~1.0/~1.0`) and remain strongly aligned at `t=1.0` (`0.9355/0.8766`), while ordinary expected answer loss stayed anti-aligned (`-0.1045/-0.0034`). Local logit-descent targets also aligned when weakly proximal (`p=0.01` `~1.0/~1.0`, `p=0.1` `0.9998/0.9997`), but a stronger proximity setting collapsed toward the failed expected-loss direction (`p=1.0` `-0.0895/-0.0028`).
Do not repeat: The same seed-2 exact-grid Stage 0 sweep over policy-reweighted temperatures `0.25/0.5/1/2` and logit-descent proximity `0.01/0.1/1` as novelty.
Next allowed test: If continuing this family, run a short Stage 1 lift gate for the softer aligned settings (`policy_reweighted_t1` or `logit_descent_p0.1`) against the hard-boundary ceiling, then design an approximation that avoids full result-class enumeration before calling it scalable.
Source: `aiAgentWorkHistory/phase7/2026-05-29-local-target-propagation-gate.md`

PARTIAL-POSITIVE: The softer policy-reweighted local target produces Stage 1 lift above the failed expected-loss baseline.
Conclusion: In a 200-step exact-grid Stage 1 gate, `policy_reweighted_t1` reached `0.5600` exact-grid calculator-result accuracy and `0.5391` sampled normal accuracy with controls low (`injection_zero=0.0234`, `forced_random=0.0156`), slightly above the hard-boundary ceiling run at the same budget (`0.5500` calc, `0.4844` normal). Ordinary expected loss collapsed to near chance (`0.0025` calc, `0.0000` normal), while `logit_descent_p0.1` improved but lagged (`0.2950` calc, `0.1953` normal).
Do not repeat: The same seed-2, 200-step Stage 1 comparison of `hard_boundary`, `expected_loss`, `policy_reweighted_t1`, and `logit_descent_p0.1` as novelty.
Next allowed test: Replicate or extend `policy_reweighted_t1` to a longer convergence/retention gate, then design a sampled/top-k/learned approximation that avoids full forced-result enumeration.
Source: `aiAgentWorkHistory/phase7/2026-05-29-local-target-stage1-lift-gate.md`

PARTIAL-POSITIVE: `policy_reweighted_t1` local targets can survive and improve during answer-only retention, but target training is nonmonotonic.
Conclusion: In an 800-step target-training plus 200-step answer-only retention gate, `policy_reweighted_t1` trailed hard-boundary at target step 800 (`0.7050` vs `0.8200` exact-grid calc) after peaking at step 600 (`0.8925`), but finished retention at `0.8925` exact-grid calc and `0.8750` sampled normal versus hard-boundary `0.8050`/`0.8281`. Controls remained causal (`injection_zero=0.0234`, `forced_random=0.0156`, oracle `1.0000`).
Do not repeat: The same seed-2, 800-target-step plus 200-retention-step comparison of `hard_boundary` and `policy_reweighted_t1` as novelty.
Next allowed test: Seed-replicate only if stability is the explicit question; otherwise approximate `policy_reweighted_t1` with sampled/top-k/learned targets that avoid full forced-result enumeration.
Source: `aiAgentWorkHistory/phase7/2026-05-29-local-target-convergence-retention-gate.md`

MIXED-NEGATIVE: Naive sparse sampled `policy_reweighted_t1` targets do not preserve the full-enum learning signal unless candidate coverage is near-full.
Conclusion: In a 200-step no-replacement sparse approximation gate, exact `policy_reweighted_t1` reached `0.5600` exact-grid calc and `0.5391` sampled normal. Sparse uniform branches improved with coverage but lagged badly: `u16` reached `0.1975` calc, `u24` `0.2800`, `u32` `0.3350`, and near-full `u36` `0.4100`; only full-vocabulary `u39` recovered/exceeded the signal at `0.6250`. Top-k plus uniform was worse (`k8_u8` `0.0925` calc).
Do not repeat: The same seed-2 200-step sparse candidate ladder over `k8_u8/k0_u16/k0_u24/k0_u32/k0_u36/k0_u39` as novelty.
Next allowed test: Use a smarter proposal/learned candidate generator or importance-corrected target that improves true-result coverage without near-full forced-result enumeration.
Source: `aiAgentWorkHistory/phase7/2026-05-29-sampled-local-target-approximation-gate.md`

DISPROVEN: Simple loss-ranked neighborhood expansion rescues sparse `policy_reweighted_t1` local targets.
Conclusion: In a 200-step adaptive proposal gate, raw uniform `sampled_policy_reweighted_t1_k0_u32` reached `0.3350` exact-grid calc and `0.3438` sampled normal. Adaptive low-loss-neighborhood branches underperformed at similar raw scoring budgets: `u8_b4_r2` `0.2025` calc, `u8_b4_r3` `0.2600`, and `u12_b4_r2` `0.2700`; the adaptive branches had lower unique coverage (`18.42-22.08` unique results) and lower true-result coverage (`0.6350-0.7700`) than raw `u32` (`32` unique, `0.8450` coverage).
Do not repeat: The same seed-2 200-step adaptive neighborhood gate over `u8_b4_r2/u8_b4_r3/u12_b4_r2` as novelty.
Next allowed test: Use a learned proposal or importance/bias-corrected sampled target; otherwise pivot to source-acquisition-for-handoff geometry instead of more local sampled-candidate variants.
Source: `aiAgentWorkHistory/phase7/2026-05-29-adaptive-local-target-proposal-gate.md`

MIXED-POSITIVE: A forced-true additive readout auxiliary can shape transfer geometry during bottleneck source acquisition.
Conclusion: In a reduced `operand_max=9`, 100-step seed-13 source gate, adding `--additive-forced-true-loss-weight 0.5` made the true result the best forced additive result on `0.5900` of the grid (`top3=0.6900`) versus baseline `0.0000`/`0.0000`, and lowered 50-step additive slope final loss (`0.7367` vs `1.5305`). It also weakened source policy acquisition at the same budget (`0.2800` source calc and `0.2800` final eval vs baseline `0.3500` calc and `0.3800` final eval).
Do not repeat: The same small `operand_max=9`, 100-step seed-13 baseline vs aux-weight `0.5` source/geometry gate as novelty.
Next allowed test: Use a scheduled/gated auxiliary or retention anchor to avoid competing with source policy acquisition, then verify on `operand_max=19` with targeted standalone 600-step additive handoff gates.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-source-aux-gate.md`

POSITIVE: Delaying the forced-true additive auxiliary can preserve source acquisition while shaping additive geometry.
Conclusion: On the same reduced `operand_max=9`, 100-step seed-13 gate, turning on `--additive-forced-true-loss-weight 0.5` only after step `50` beat both baseline and always-on aux on source acquisition (`0.3900` source calc and `0.4000` final eval, vs baseline `0.3500`/`0.3800` and always-on `0.2800`/`0.2800`) while keeping a large additive geometry gain (`forced_best_true=0.5100`, `top3=0.5600`, 50-step slope final loss `0.7979` vs baseline `0.0000`/`0.0000`/`1.5305`).
Do not repeat: The same small `operand_max=9`, seed-13, 100-step, start-step-50 schedule gate as novelty.
Next allowed test: Scale to `operand_max=19` with source-only checkpointing first, then verify promising scheduled-aux checkpoints with targeted standalone 600-step additive handoff.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-schedule-gate.md`

POSITIVE: The scheduled forced-true additive source objective improves full-grid standalone handoff.
Conclusion: On `operand_max=19`, seed-13, 200-step source acquisition, scheduled aux (`weight=0.5`, start step `50`) approximately tied baseline source policy (`0.2800` vs `0.2875` train calc; `0.2750` vs `0.2825` final eval) but strongly improved additive geometry (`forced_best_true=0.2125` vs `0.0000`, 50-step slope loss `1.0360` vs `1.8058`) and the trusted 600-step frozen-policy handoff (`0.4150` final eval / `0.3925` step-600 snapshot vs baseline `0.2525` / `0.2625`).
Do not repeat: The same seed-13, `operand_max=19`, 200-step baseline vs scheduled step-50 source gate plus 600-step handoff as novelty.
Next allowed test: Extend scheduled source acquisition to longer horizons (`400/600/800`) and verify selected checkpoints with standalone 600-step additive handoff; add a policy-retention anchor if source accuracy drifts.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-op19-gate.md`

POSITIVE: Longer scheduled forced-true source acquisition compounds handoff geometry through step 600.
Conclusion: Extending the seed-13 scheduled source to `800` steps improved forced-result geometry from step `200` to `600` (`forced_best_true 0.2125 -> 0.9800`, 50-step slope loss `1.0360 -> 0.4719`) and the step-600 source checkpoint reached `0.7725` final eval under the trusted 600-step frozen-policy additive handoff. Step `800` had perfect forced-result geometry but worse handoff (`0.6750` final), so final source checkpoint is not automatically best.
Do not repeat: The same seed-13 scheduled source `200/400/600/800` geometry ladder or step-600 vs step-800 handoff comparison as novelty.
Next allowed test: Run continuation/readout from the step-600 handoff lineage to test whether the scheduled source can clear the high non-bottleneck gate; replicate on a fresh seed only if stability is the explicit question.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-long-source-gate.md`

MIXED-POSITIVE: The scheduled step-600 source lineage remains calculator-dependent but misses the high continuation/readout gate.
Conclusion: Starting from scheduled step-600 handoff final `0.7725`, 800-step frozen-policy continuation reached only `0.7775`, 600-step readout after continuation reached `0.8175`, and an extra 1000 stable-policy readout steps reached `0.8475`; controls stayed low (`injection_zero <=0.0547`, forced-random <=`0.0391`), but learned calc stayed around `0.5391`.
Do not repeat: The same scheduled step-600 handoff -> 800 continuation -> 600 readout -> extra 1000 readout chain as novelty.
Next allowed test: Improve source policy accuracy while preserving scheduled geometry, or run continuation/readout only after a scheduled source checkpoint shows both strong handoff geometry and materially higher learned calculator accuracy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-scheduled-source-continuation-readout.md`

POSITIVE: Gentle low-LR recovery from a scheduled source checkpoint restores learned calc and clears readout.
Conclusion: Continuing the scheduled step-600 source for 30 CPU steps with LR `3e-4` and lower forced-true weight `0.1` raised source calc from `0.5800` to `0.7950` while keeping forced-true loss low; the resulting frozen-policy 600-step handoff reached `0.8425`, 800-step continuation reached `0.8900` final / `0.9175` best snapshot, and 600-step readout reached `0.9320` final eval with final diagnostic controls normal `0.9225`, injection-zero `0.0300`, forced-random `0.0325`, learned calc `0.7925`.
Do not repeat: The same seed-13 scheduled step-600 -> 30-step low-LR `aux=0.1` recovery -> 600 handoff -> 800 continuation -> 600 readout chain as novelty.
Next allowed test: Replicate on a fresh scheduled source seed or integrate the low-LR/lower-aux recovery as an automatic late-source phase, then verify with the trusted 600-step handoff and continuation/readout gates.
Source: `aiAgentWorkHistory/phase7/2026-05-29-scheduled-source-low-lr-recovery.md`

POSITIVE: The gentle scheduled-source recovery recipe replicates on a fresh seed and clears handoff directly.
Conclusion: Fresh seed-14 scheduled source training reached step-600 source eval `0.6675`; the same 30-step CPU recovery (`lr=3e-4`, forced-true weight `0.1`) raised source eval to `0.8850`, and the trusted 600-step frozen-policy additive handoff reached `0.9600` final eval / `0.9650` step-600 snapshot with learned calc `0.8700`, injection-zero `0.0850`, and forced-random `0.0875`.
Do not repeat: The same seed-14 scheduled source -> 30-step low-LR recovery -> 600-step frozen-policy handoff as novelty.
Next allowed test: Automate the late-source transition or test a third seed only if the explicit question is stability; keep the 600-step handoff/readout gates as arbiter and monitor the somewhat higher seed-14 zero/random controls.
Source: `aiAgentWorkHistory/phase7/2026-05-29-fresh-scheduled-source-recovery-replication.md`

POSITIVE: The late scheduled-source recovery phase can be automated in one source run.
Conclusion: Adding an in-run late-source recovery switch at step `600` (LR multiplier `0.1`, forced-true weight override `0.1`) preserved the seed-14 recovery effect without manual checkpoint relaunch: final source eval reached `0.8775`, and the trusted 600-step frozen-policy handoff reached `0.9400` final eval / `0.9475` step-600 snapshot with learned calc `0.8725`, injection-zero `0.0800`, and forced-random `0.0775`.
Do not repeat: The same seed-14 automated fixed-step-600 recovery plus 600-step handoff as novelty.
Next allowed test: Replace the fixed recovery step with adaptive transition criteria, or move the source branch toward less prescriptive/scalable assignment while preserving the handoff/readout gates.
Source: `aiAgentWorkHistory/phase7/2026-05-29-automated-scheduled-source-recovery.md`

MIXED-POSITIVE: A simple source-accuracy trigger can replace the fixed late-source recovery step on seed 14, but controls worsen.
Conclusion: After wiring adaptive recovery to switch both LR and forced-true weight, `result_policy_argmax_result_accuracy >= 0.65` with min step `500` fired at step `528`; source final eval was only `0.8250`, but the trusted 600-step frozen-policy handoff reached `0.9850` final eval / `0.9775` step-600 snapshot with learned calc `0.8425`, injection-zero `0.1325`, and forced-random `0.1325`. This beats the fixed step-600 handoff final eval (`0.9400`) but has higher controls than the fixed-step run (`0.0800`/`0.0775`).
Do not repeat: The same seed-14 `argmax_result_accuracy >= 0.65`, min-step-500 adaptive recovery plus 600-step handoff as novelty.
Next allowed test: Replicate the adaptive trigger on a fresh scheduled source, or use a smoothed/conjunctive trigger that preserves the handoff lift while reducing zero/random controls.
Source: `aiAgentWorkHistory/phase7/2026-05-29-adaptive-source-recovery-trigger.md`

MIXED-NEGATIVE: The simple source-accuracy adaptive recovery trigger does not replicate on a fresh seed.
Conclusion: On a fresh seed-17 source run, the same `result_policy_argmax_result_accuracy >= 0.65` min-step-500 trigger never fired; source final eval reached only `0.6100`, and trusted 600-step frozen-policy handoff reached `0.6825` final eval / `0.6925` step-600 snapshot with learned calc `0.6075`, injection-zero `0.0400`, and forced-random `0.0500`. A matched fixed step-600 control did better but still missed the high gate: source final `0.7450`, handoff `0.7675` final / `0.7850` snapshot, learned calc `0.7350`, injection-zero `0.0500`, forced-random `0.0375`.
Do not repeat: The same fresh seed-17 `argmax_result_accuracy >= 0.65`, min-step-500 adaptive run or matched fixed step-600 control as novelty.
Next allowed test: Use a smoothed/conjunctive recovery trigger or a different transition metric; do not treat raw source argmax accuracy thresholding as validated.
Source: `aiAgentWorkHistory/phase7/2026-05-29-fresh-adaptive-recovery-trigger-replication.md`
