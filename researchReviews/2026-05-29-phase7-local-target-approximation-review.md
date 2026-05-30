# 2026-05-29 Phase 7 Local-Target Approximation Review

## Why This Review Exists

The local-target branch produced a real positive and then immediately risked
becoming a local variant loop.

Exact `policy_reweighted_t1` targets showed Stage 1 lift and answer-only
retention, but the method still scores broad forced-result candidates. The
last cluster tested whether simple sampled/adaptive candidate sets could make
that signal scalable. They did not.

This review freezes the lesson before future agents keep trying nearby
hand-coded candidate proposals.

## What Changed

The local-target branch is no longer merely a feasibility idea.

- Stage 0 showed current-policy-reweighted targets align with the hard
  boundary ceiling while ordinary expected answer loss anti-aligns.
- A 200-step Stage 1 gate showed `policy_reweighted_t1` reaches `0.5600`
  exact-grid calculator-result accuracy and `0.5391` sampled normal.
- An 800-step target-training plus 200-step answer-only retention gate showed
  the branch can recover and finish at `0.8925` exact-grid calculator-result
  accuracy and `0.8750` sampled normal.

But the scalability story got worse:

- Naive sparse no-replacement sampling did not preserve the learning signal
  until candidate coverage approached full enumeration. `u32` scored about
  `82%` of the result vocabulary and still reached only `0.3350` exact-grid
  calc at 200 steps.
- Full-vocabulary sparse-path `u39` recovered the signal, confirming the
  implementation was sound and the failure was coverage/proposal quality.
- Current-policy top-k plus uniform sampling was worse than uniform-only
  sampling at comparable budget.
- Loss-ranked neighborhood expansion also underperformed raw uniform `u32`
  because it clustered into fewer unique candidates and lower true-result
  coverage.

## Belief Changes

The local-target branch remains strategically interesting because it changes
the credit-assignment family. It is the best recent evidence that an
answer-derived local target can train and retain a natural result-level
calculator policy.

However, simple hand-coded candidate proposal is not the path to scalability.
The target seems to need either high true-result candidate coverage or a
different estimator/target construction that does not collapse when the true
useful result is absent from the candidate set.

The main uncertainty is now sharper:

```text
Can we learn or correct the proposal/estimator enough to avoid near-full
forced-result scoring?
```

If not, the local-target branch remains another prescriptive full-enum
assignment method, not the final scalable answer.

## Branches That Should Stop

Stop running hand-coded sampled-candidate ladders as novelty:

- raw uniform/top-k count ladders;
- top-k plus uniform variants without a new correction mechanism;
- low-loss integer-neighborhood expansion;
- longer versions of the same sparse candidate proposals;
- seed replications of these sparse proposals unless a new estimator or
  proposal mechanism changes the expected failure mode.

The exact `policy_reweighted_t1` branch also should not be rerun merely to
reconfirm retention. It is positive but non-scalable.

## Branches That Deserve Compute

Three local-target follow-ups still deserve compute if they are made explicit:

1. A learned candidate proposal with a predeclared coverage and Stage 1 gate.
   It should beat raw uniform `u32` on true-result coverage and calculator
   learning at similar scoring cost.
2. An importance-corrected or bias-corrected sampled target that changes the
   estimator, not just the candidate count.
3. A different local-target construction that does not require the useful
   result to be present in a small candidate set.

If none of these is concrete, the better mainline path is source acquisition
for additive handoff/readout geometry, where the project already has a staged
non-bottleneck recipe but needs better source objectives.

## Are We Closer To The Goal?

Yes, but mostly by ruling out a tempting scalability shortcut.

The exact local-target branch shows that a non-backpropagated answer-derived
target can train calculator use. The sparse and adaptive negatives show that
making this scalable is not as simple as sampling a few candidate results.

The project is closer because future work can avoid re-entering the
candidate-count loop and focus on either learned/corrected proposals or the
source-acquisition-for-handoff branch.

## What Counts As Success Next

A next local-target result should satisfy all of:

- score materially fewer result classes than full enumeration;
- beat raw uniform `u32` and adaptive-neighborhood baselines at comparable
  scoring cost;
- show Stage 1 calculator-result lift, not only target coverage;
- report true-result candidate coverage, target argmax accuracy, sampled
  normal, injection-zero, forced-random, and oracle controls.

A next source-acquisition result should instead improve fresh-family
standalone 600-step handoff or continuation/readout behavior, not merely source
calculator accuracy.

## Steering Decision

Keep target propagation/local targets active, but narrow the allowed work:

- exact-grid `policy_reweighted_t1` is a positive ceiling, not a scalable
  method;
- hand-coded sparse/adaptive proposals are paused;
- local-target approximation needs a learned proposal, estimator correction, or
  different target construction;
- absent that, pivot mainline compute to source acquisition optimized for
  additive handoff/readout geometry.

## Addendum: Replay-Memory Proposal

A first nontrivial approximation worked after this review.

The new `memory_policy_reweighted_t1_u8_m24` branch scores `8` fresh uniform
result candidates per step, caches observed forced-result losses per prompt,
and builds the target from fresh candidates plus `24` low-loss cached
candidates.

Results:

- At 200 steps, replay memory beat raw uniform `u32` while using one quarter
  the fresh scoring per step: exact-grid calc `0.5900` and sampled normal
  `0.5391` versus `0.3350`/`0.3438`.
- Its final target true-candidate coverage was `1.0000`, target argmax
  accuracy `0.9850`, and controls stayed low (`0.0234` injection-zero,
  `0.0156` forced-random).
- In an 800+200 retention gate, replay memory reached target `0.9600` exact
  calc / `0.9766` sampled normal and retained `0.8600` calc / `0.8750`
  sampled normal under answer-only training.

Updated steering: replay memory deserves follow-up because it changes the
candidate mechanism rather than rerunning a sparse count ladder. The important
caveat is that the current test is transductive: on the fixed exhaustive grid,
the memory eventually observes all `39` result classes. The next tests should
stress lower fresh scoring, stale-loss aging/rescoring, and generalization
beyond a fixed prompt grid before treating this as scalable.

## Addendum: Lower Fresh-Scoring Budget

The first budget stress is positive, with a useful floor:

- At 200 steps, `memory_policy_reweighted_t1_u2_m30` was best despite scoring
  only `2` fresh results per step: exact-grid calc `0.6025`, sampled normal
  `0.6016`, true-candidate coverage `0.9925`, and target argmax `0.9600`.
- The same gate showed `u8_m24` at `0.5900`/`0.5391`, `u4_m28` at
  `0.5100`/`0.4844`, and `u1_m31` at only `0.4075`/`0.4219`; raw uniform
  `u32` remained `0.3350`/`0.3438`.
- An 800+200 `u2_m30` retention gate reached target `0.9000` exact calc /
  `0.8750` sampled normal and retained `0.7850` calc / `0.7656` normal. This
  is weaker than `u8_m24` retention (`0.8600`/`0.8750`) but still far above the
  sparse uniform baselines.

Updated steering: do not run more simple replay-memory budget ladders as
novelty. Treat `u2_m30` as the best current low-fresh-score point and `u1_m31`
as below the useful 200-step budget floor. The next replay-memory work should
attack transduction directly: stale-cache aging/rescoring, memory reset,
streaming/non-exhaustive prompts, or learned/generalized candidate memory.

## Addendum: Cached-Candidate Rescoring

Simple top-cached-candidate rescoring did not fix the low-budget retention
tradeoff:

- Added optional `_rN` syntax to replay-memory branches, e.g.
  `memory_policy_reweighted_t1_u2_m30_r4`.
- At 200 steps, `u2_m30_r2` exactly tied no-rescore `u2_m30` at
  `0.6025` exact calc / `0.6016` sampled normal while doubling forced-score
  cost from `2` to `4` per step.
- Heavier rescoring hurt the short gate: `r4` reached `0.5300` calc /
  `0.5781` normal, and `r8` reached `0.4675` / `0.4609`.
- The 800+200 `u2_m30_r2` retention gate also exactly tied no-rescore:
  target `0.9000` calc / `0.8750` normal and retention `0.7850` calc /
  `0.7656` normal.

Updated steering: stale-cache rescoring by itself is not the missing piece.
Do not spend more turns tuning rescore counts. The next replay-memory work
should target the transductive assumption directly: finite/reset memory,
streaming/non-exhaustive prompts, or learned/generalized candidate memory.

## Addendum: Finite Reset Memory Stress

Resetting the replay-memory cache exposed that the positive depends heavily on
persistent prompt-identity memory:

- Added optional `_resetN` syntax, e.g.
  `memory_policy_reweighted_t1_u2_m30_reset50`, which clears cached losses
  every `N` target-loss calls.
- In the 200-step reset stress, no-reset `u2_m30` reached `0.6025` exact calc /
  `0.6016` sampled normal. `reset50` fell to `0.2500` / `0.2578`, `reset25`
  to `0.1650` / `0.2188`, and `reset10` to `0.0950` / `0.1406`.
- A 199-step boundary check avoided ending exactly on a reset: no-reset reached
  `0.5925` / `0.5938`, `reset100` reached only `0.4575` / `0.4453`, and
  `reset50` only `0.2575` / `0.2812`.
- The boundary check is especially important because `reset100` had nearly
  full final target coverage (`0.9925` true-candidate coverage) yet still
  underperformed, so the damage is not just a final empty-cache snapshot.

Updated steering: do not tune reset intervals as the next local fix. Treat
plain replay memory as a useful but transductive approximation. The next
local-target scalability test should use streaming/non-exhaustive prompts or a
learned/generalized proposal that cannot rely on a durable per-prompt cache.

## Addendum: Streaming Prompt Stress

Streaming minibatches removed the strong fixed-grid replay-memory lift:

- Added `--streaming-train-batch-size`, which trains on freshly sampled
  minibatches while evaluating on the full grid. Replay-memory branches switch
  to prompt-keyed caches in this mode so rows cannot reuse another prompt's
  cache.
- At 200 steps with batch `16`, exact `policy_reweighted_t1` reached only
  `0.1100` exact calc, raw uniform `u32` `0.0700`, `u2_m30` `0.0475`, and
  `u8_m24` `0.0950`.
- At 800 steps with batch `16`, exact and raw `u32` both reached `0.2450`
  exact calc / `0.2578` sampled normal. `u8_m24` was comparable at `0.2650`
  calc / `0.2500` normal, while `u2_m30` lagged at `0.1850` / `0.2031`.
- A batch-64, 200-step check stayed weak: exact `0.1650` calc, raw `u32`
  `0.1475`, `u8_m24` `0.1475`, and `u2_m30` `0.0975`.

Updated steering: prompt-keyed replay memory is not the scalable answer. It is
still a useful clue that cached candidate proposals can matter, but next
local-target work needs a learned/generalized proposal, estimator correction,
or a different target construction. Otherwise, mainline compute should return
to source objectives that improve additive handoff/readout geometry.

## Addendum: Corrected Sparse Target

A simple estimator/target correction did not rescue sparse local targets:

- Added `corrected_policy_reweighted_t<T>_u<U>_b<mean|current|max>`.
- The branch scores uniform candidates but keeps unscored result classes in the
  target with an imputed baseline loss, instead of forcing all target mass onto
  the sampled candidate set.
- In the 200-step gate, exact `policy_reweighted_t1` reached `0.5600` exact
  calc / `0.5391` sampled normal; raw uniform `u32` reached `0.3350` /
  `0.3438`.
- Corrected branches underperformed: `u8_bmean` `0.1150` / `0.0938`,
  `u8_bcurrent` `0.1100` / `0.0938`, `u8_bmax` `0.0675` / `0.0625`,
  `u16_bmean` `0.2100` / `0.2500`, and `u16_bcurrent` `0.2500` / `0.2500`.
- True-candidate coverage remained the binding issue (`0.1850` for `u8`,
  `0.4050` for `u16`), and preserving unscored mass mostly diluted target
  pressure rather than correcting the sparse estimator.

Updated steering: do not tune imputed mean/current/max sparse targets or treat
"preserve unscored mass" as the missing estimator correction. Local-target
approximation still needs a learned/generalized proposal, a correction with a
stronger bias/variance argument, or a target construction that can create
useful pressure without requiring high true-result coverage.

## Addendum: Learned Loss Proposal

A first parametric learned proposal is a real fixed-grid improvement but not a
streaming scalability answer.

- Added `learned_policy_reweighted_t<T>_u<U>_p<P>_h<H>_e<E>`. The branch
  trains a small online MLP to predict forced-result answer loss from observed
  candidate scores, then proposes low predicted-loss result classes plus
  uniform exploration. It uses no true-sum label and no prompt-keyed cache.
- In the fixed-grid 200-step gate at 32 forced scores per step,
  `learned_policy_reweighted_t1_u4_p28_h32_e1` reached `0.5850` exact calc /
  `0.5703` sampled normal, beating raw uniform `u32` (`0.3350` / `0.3438`)
  and matching the exact-grid ceiling neighborhood (`policy_reweighted_t1`
  `0.5600` / `0.5391`).
- The fixed-grid learned branch achieved `1.0000` proposal true-candidate
  coverage, target argmax `0.9175`, and low controls (`0.0234`
  injection-zero, `0.0156` forced-random).
- Other 32-score learned branches were also above raw `u32` but weaker:
  `u8_p24_h32_e1` `0.4925` / `0.4141`, `u16_p16_h32_e1` `0.5050` /
  `0.4141`, and `u8_p24_h64_e3` `0.4850` / `0.4766`.
- Streaming minibatches removed the useful lift. At batch `16`, 200 steps gave
  exact `0.1100`, raw `u32` `0.0700`, and learned `0.0925`; at 800 steps,
  raw `u32` and learned tied at `0.2350` exact calc, with sampled normal
  `0.2734` raw vs `0.2656` learned.

Updated steering: keep the learned-proposal idea alive, but do not tune this
simple online polynomial-feature MLP on the fixed grid. The next learned
proposal must explain how it will generalize under streaming/non-exhaustive
training, or local-target work should pivot to estimator/target construction
rather than more proposal knobs.

## Addendum: Pretrained Learned Proposal

Random-prompt pretraining did not turn the simple learned proposal into a
streaming solution.

- Added optional `_wN` syntax, e.g.
  `learned_policy_reweighted_t1_u4_p28_h32_e1_w20`. Before model training, the
  proposal MLP receives `N` batches of random prompt/result forced-loss
  observations, then continues online as before.
- In a 200-step batch-16 streaming screen, raw `u32` reached `0.0700` exact
  calc / `0.0703` sampled normal. The online learned branch reached `0.0925`
  / `0.0938`; `_w20` reached `0.0975` / `0.0625`; `_w50` reached `0.0950` /
  `0.0547`.
- In the 800-step batch-16 stress, raw `u32` reached `0.2350` exact calc /
  `0.2734` sampled normal. `_w20` reached a slightly higher `0.2625` exact
  calc but much lower `0.1797` sampled normal.

Updated steering: do not keep turning the `_wN` dial. The result suggests
offline proposal data can slightly shape the result policy, but this simple
polynomial-feature predictor does not produce robust functional calculator
use under streaming. A next learned proposal needs a different generalization
story, not more warmup batches.
