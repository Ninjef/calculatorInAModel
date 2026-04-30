# Overview

This is the next task queue under `2026-04-27-1828-Steps-next.md`, updated for the latest finding:

Low true-operand exact match does not necessarily mean the calculator is unused. At least one tiny Model C checkpoint loses substantial answer accuracy when calculator injection is removed or corrupted, and it also gets worse when true oracle operands are forced. That suggests the model learned a useful non-human calculator-result code.

2026-04-30 status update:

- Added per-prompt forced result-class sweeps to `scripts/diagnose_calculator_protocol.py`.
- Ran the 1-digit weird-protocol autopsy on the primary and sanity checkpoints.
- Added a high-priority 2-digit track because the 1-digit regime required models so small that the calculator interface may be artificially cramped.
- First 2-digit result: `2 layers, 1 head, n_embd=16, mlp_expansion=1, hook_after_layer=1` is a useful candidate regime. Oracle Model C reaches `0.988` exact match, while matched Model A/B are `0.711` / `0.787`.
- Learned 2-digit Model C with plain STE currently collapses (`0.023` exact match), with confident but wrong operands. This points toward the learned input protocol/optimization as the next bottleneck, not the calculator output side.
- Added a new follow-up task for a calculator-required bottleneck experiment:
  `aiAgentProjectTasks/2026-04-30-Calculator-required-bottleneck-experiment.md`.

Current completion state:

1. Task 1 is substantially complete.
2. Task 2 is partially complete and has been redirected toward 2-digit and bottleneck experiments.
3. Task 3 is not started.

Recommended next work:

1. Run the calculator-required bottleneck experiment, oracle first.
2. Continue the 2-digit capacity/protocol track, using oracle and probe checks before interpreting learned Model C failures.
3. Only then compare training pressures for human-readable true-operand protocols versus useful private protocols.

# Task 1 - Learned Calculator Codebook Analysis

Status: substantially complete as of 2026-04-30.

Outputs:

```text
runs/2026-04-29_125836_910885_model-c-op0-9-1/model-c-1digit-seed2/forced-result-sweep/
runs/2026-04-29_125836_910885_model-c-op0-9/model-c-1digit-seed1/forced-result-sweep/
aiAgentWorkHistory/2026-04-30-next-calculator-protocol-experiments.md
```

Summary:

- Implemented `--forced-result-sweep`.
- Generated `forced_result_sweep.csv`, `forced_result_summary.json`, and `result_codebook.csv`.
- The strong 1-digit checkpoint uses a coarse private answer code:
  - learned class `4`: sum `5`;
  - learned class `6`: sums `0, 1, 2, 3, 4, 6`;
  - learned class `11`: sums `7..18`.
- Learned class is causally better than true-sum class for the strong checkpoint:
  - learned class best-forced fraction: `0.244`;
  - true-sum class best-forced fraction: `0.049`;
  - mean learned-minus-true-sum target log probability: `+0.655`.
- This is a useful non-human/private calculator-result protocol, not the intended true-operand protocol.

## Research question

What information is carried by the learned calculator result classes, and how does the downstream model use those classes to improve answer accuracy?

Answer this causally, not just correlationally.

## Primary checkpoint

Start with the strongest current weird-protocol checkpoint:

```text
runs/2026-04-29_125836_910885_model-c-op0-9-1/model-c-1digit-seed2/final_weights.pt
```

This checkpoint had:

- normal learned calculator exact match around `0.904`;
- injection-scale-zero exact match around `0.645`;
- oracle true-operand exact match around `0.637`;
- true operand exact match around `0.014`;
- calculator result accuracy around `0.146`.

Also run a smaller sanity check on:

```text
runs/2026-04-29_125836_910885_model-c-op0-9/model-c-1digit-seed1/final_weights.pt
```

## Experiments

### Per-prompt result-class sweep

For each held-out prompt `a+b=`, force every possible calculator result class `0..18` at the `=` position while keeping the rest of the checkpoint fixed.

Record for each forced result class:

- generated answer;
- exact-match correctness;
- probability/logit of the correct next answer token;
- full generated answer probability if straightforward to compute;
- whether the forced class matches the learned class;
- whether the forced class matches the true sum.

Output:

- `forced_result_sweep.csv`;
- summary table of best forced class per prompt;
- aggregate accuracy under each forced class.

### Learned-class codebook table

Using normal learned calculator behavior, group rows by learned calculator result class.

For each result class, report:

- count;
- distribution of true sums;
- distribution of first answer digit;
- distribution of full target answer;
- answer accuracy;
- most common `(true_a, true_b)` pairs;
- mean answer confidence.

Interpretation goal:

- Is the learned code mostly a coarse sum bucket?
- Is it a first-digit or carry code?
- Is it an operand-region code?
- Is it only meaningful in combination with the downstream residual state?

### Causal importance of learned class

For each prompt, compare:

- normal learned class;
- true-sum class;
- best forced class from the sweep;
- random class;
- zero class.

Report:

- how often the learned class is also the best forced class;
- how much log probability drops when replacing learned with true-sum;
- how much log probability drops when replacing learned with random;
- examples where learned class is clearly better than true-sum.

### Optional trajectory analysis

If the existing checkpoints are enough, inspect saved training curves or intermediate runs to see when the code emerges.

If not enough checkpoint history exists, add lightweight periodic diagnostic snapshots in a future training run, but do not turn this task into a full retraining campaign.

## Implementation notes

Prefer adding a focused diagnostic script or extending `scripts/diagnose_calculator_protocol.py` with a clearly named mode such as:

```bash
python3 scripts/diagnose_calculator_protocol.py \
  --checkpoint runs/.../final_weights.pt \
  --digits 1 \
  --operand-max 9 \
  --samples 512 \
  --forced-result-sweep
```

Keep normal diagnostic behavior unchanged.

The current `--calculator-result-override` modes are coarse global counterfactuals. This task needs per-example forced result classes.

## Measurements

Save:

- config;
- diagnostic CSVs;
- summary JSON;
- representative examples;
- short written interpretation in `aiAgentWorkHistory/`.

At minimum include:

- normal exact match;
- injection-zero exact match;
- oracle exact match;
- best-forced-class exact match;
- true-sum-forced exact match;
- learned-class mutual information with true sum, first answer digit, and carry/no-carry.

## Acceptance criteria

The writeup should answer:

- What does the learned calculator result class encode?
- Is the learned class causally better than the true-sum class for this checkpoint?
- Does the downstream model use the calculator result as a coarse code, a prompt-specific key, or something else?
- Is this phenomenon strong enough to deserve training-objective work, or is it an artifact of one tiny checkpoint?

# Task 2 - Clean Capacity Regime For Calculator Use

Status: partially complete as of 2026-04-30.

The original 1-digit capacity strategy is now lower priority because making 1-digit addition difficult required models so small that the calculator interface may be artificially cramped. Initial 2-digit experiments are more informative.

Current best candidate:

```text
digits=2
operand range=0..99
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
steps=1000
```

Completed 2-digit runs:

| Variant | Exact match | Final loss | Run |
| --- | ---: | ---: | --- |
| Model C oracle | `0.988` | `0.0049` | `runs/2026-04-29_220949_913725_model-c-oracle/model-c-2digit-seed2` |
| Model A | `0.711` | `0.1380` | `runs/2026-04-29_221055_780936_model-a/model-a-2digit-seed2` |
| Model B | `0.787` | `0.1632` | `runs/2026-04-29_221055_780881_model-b/model-b-2digit-seed2` |
| Model C learned STE | `0.023` | `1.3661` | `runs/2026-04-29_221445_951260_model-c/model-c-2digit-seed2` |

Interpretation:

- This is cleaner than the 1-digit `n_embd=4` regime because Model A/B struggle naturally while oracle Model C succeeds.
- Learned STE Model C fails confidently under the 100-class operand interface:
  - true operand exact match: `0.000`;
  - calculator result accuracy: about `0.008`;
  - operand confidences: near `1.0`;
  - oracle-at-eval, injection-zero, and forced-zero all stay around `0.010` exact match.
- This points to learned input-protocol/optimization failure, not an output-side calculator injection failure.

Additional 2-digit check:

| Variant | Exact match | Final loss | Run |
| --- | ---: | ---: | --- |
| `32d` Model C oracle | `1.000` | `0.0001` | `runs/2026-04-29_221246_375002_model-c-oracle/model-c-2digit-seed2` |
| `32d` Model A | `0.086` | `0.9114` | `runs/2026-04-29_221246_374987_model-a/model-a-2digit-seed2` |
| `32d` Model B | `0.047` | `0.9213` | `runs/2026-04-29_221246_374983_model-b/model-b-2digit-seed2` |

The `32d` oracle path works, but A/B optimization was worse and hook-position probes did not improve. Bigger is not automatically cleaner.

Remaining work:

- Run more seeds for the `16d` 2-digit regime.
- Try smaller 2-digit curricula such as fixed-width `operand_max=19` or `49`.
- Add or run additional probe positions as diagnostics, while keeping the actual hook read position as the main criterion.
- Run the calculator-required bottleneck experiment in `2026-04-30-Calculator-required-bottleneck-experiment.md`.

## Research question

Can we find a small architecture/training-budget/task split where Model A/B struggle, residual probes show operand information is available, and Model C can use the calculator path?

The previous capacity sweep found two unhelpful extremes:

- `1 layer, 1 head, n_embd=8, mlp_expansion=1` can still solve `0..9` single-digit addition too well by the ordinary transformer path.
- `1 layer, 1 head, n_embd=4, mlp_expansion=1` makes Model A/B struggle, but residual probes are weak, so it may be too narrow to expose operands cleanly at the hook.

## Candidate strategies

Try these in order, stopping once a clean regime is found.

### Keep `n_embd=8`, reduce training budget

The `n_embd=8` model solved `0..9` after `1000` steps. Run Model A/B/C with shorter budgets:

- `100` steps;
- `200` steps;
- `400` steps;
- optionally `700` steps if needed.

Use:

```text
n_layer=1
n_head=1
n_embd=8
mlp_expansion=1
calculator_hook_after_layer=1
operand_max=9
```

Run at least seeds `0, 1, 2` for promising settings.

### Keep train easy, eval harder

Train on a restricted subset and evaluate systematically on all `0..9`.

Possible splits:

- train operands `0..7`, eval `0..9`;
- train sums `0..12`, eval all sums;
- hold out a checkerboard pattern such as `(a+b) % 2 == 1`.

Only add split support if it is cleanly scoped.

Goal:

Make ordinary memorization/interpolation less sufficient while preserving single-digit interface simplicity.

### Try a slightly richer but still tiny 2-layer model

If 1-layer placement is too cramped, try:

```text
n_layer=2
n_head=1
n_embd=8
mlp_expansion=1
calculator_hook_after_layer=1 or 2
```

The hook-after-layer choice matters:

- after layer 1 gives downstream capacity to use the code;
- after layer 2 tests whether a late injection is still usable.

## Required controls

For every promising Model C setting, run matching:

- Model A;
- Model B with calculator hook off;
- Model C with learned calculator;
- Model C with injection scale zero at eval;
- Model C with oracle operands at eval;
- residual probes at the hook read position.

## Metrics

Report:

- answer exact match;
- final loss;
- parameter count;
- true operand exact match;
- calculator result accuracy;
- operand entropy/confidence;
- injection-zero exact match;
- oracle exact match;
- residual probe accuracy for operand A/B;
- representative trace rows.

## Decision rules

A clean capacity regime is promising if:

- Model A and Model B are below roughly `0.85` exact match under the chosen budget/split;
- Model B is close enough to Model A that hook-off plumbing is not suspicious;
- residual probes are meaningfully above chance;
- Model C improves over A/B or shows strong counterfactual calculator dependence.

A regime is not clean if:

- Model A/B are already near perfect;
- residual probes are near chance;
- Model B diverges materially from Model A without an explanation;
- Model C improves only on answer accuracy and calculator counterfactuals show no causal dependence.

## Implementation notes

Keep all defaults compatible with previous runs.

If adding train/eval split support, save the split definition in every config and summary. Avoid hidden dataset assumptions.

Do not commit large checkpoint artifacts unless this repo already tracks them intentionally.

## Acceptance criteria

Write up:

- the best clean regime found;
- why it is cleaner than the 4-dim model;
- whether Model C shows intended operand use, non-human calculator-code use, or no useful calculator dependence;
- which architecture/budget should be the default for the next protocol-learning experiments.

# Task 3 - Training Pressure For Human-Readable Vs Private Calculator Protocols

Status: not started as of 2026-04-30.

Do not start broad auxiliary-loss or oracle-distillation schedules until the 2-digit/bottleneck diagnostics clarify whether the current learned failure is mostly:

- lack of operand accessibility;
- STE saturation/optimization;
- too many operand classes;
- or ordinary bypass/co-adaptation.

## Research question

Which training interventions preserve calculator usefulness while steering the model toward the intended true-operand protocol?

The latest counterfactual diagnostics suggest Model C can learn a useful private calculator protocol. That is interesting, but it raises a new question:

Are we trying to encourage any useful calculator protocol, or specifically the human-readable true-operand protocol?

## Prerequisites

Prefer completing or partially completing Task 1 and Task 2 first.

This task is still useful on the existing 4-dim checkpoint regime, but results will be easier to interpret in a cleaner capacity setting.

## Candidate interventions

### Auxiliary operand supervision schedules

Try gentler schedules than the previous constant `0.1` and too-strong `1.0` attempts:

- constant `0.01`;
- constant `0.03`;
- warmup `0.1` then decay slowly over all training;
- warmup `0.1` for first 20-30% of steps, then floor at `0.01` instead of decaying to zero.

Report both:

- answer accuracy;
- true operand protocol accuracy.

Also run counterfactual evals to ensure any answer gain is calculator-dependent.

### Freeze or bottleneck the downstream bypass

Test whether the private code emerges because the downstream transformer can co-adapt too freely.

Possible diagnostics:

- freeze non-calculator transformer blocks for a short phase after operand-head warmup;
- freeze the calculator output projection after oracle pretraining;
- reduce or regularize downstream residual capacity only if it can be done cleanly.

Keep this narrow. Do not introduce a large training framework.

### Oracle-to-learned distillation

Train with oracle operands for a phase so the output side learns to use true sums. Then switch to learned operands with an auxiliary operand loss.

Compare:

- oracle throughout;
- oracle warmup then learned;
- learned from scratch with equivalent aux schedule.

Key question:

Does pretraining the downstream side on true calculator semantics make the intended protocol easier to maintain?

### Reward/probe private code explicitly as a control

If the codebook analysis finds a stable private code, train a diagnostic classifier from learned result classes or hook logits to answer features.

This is not the main target, but it helps separate:

- private protocol discovery;
- true-operand protocol discovery;
- ordinary transformer bypass.

## Required evals

For each intervention:

- normal exact match;
- injection-zero exact match;
- true-oracle exact match;
- forced-zero/random result exact match;
- true operand exact match;
- calculator result accuracy;
- residual probe accuracy;
- mean operand entropy/confidence;
- representative trace rows.

## Success definitions

Three outcomes are all scientifically useful, but label them clearly.

### Intended protocol success

- high answer exact match;
- high true operand exact match;
- high calculator result accuracy;
- injection-zero hurts;
- oracle operands do not hurt.

### Private protocol success

- high answer exact match;
- low true operand exact match;
- injection-zero or result corruption hurts;
- oracle true operands hurt or do not help.

### Bypass

- high answer exact match;
- low true operand exact match;
- injection-zero does not hurt materially;
- result corruption does not hurt materially.

## Implementation notes

Add only minimal flags needed for the tested schedule. Save schedule parameters in configs and summaries.

Be especially careful with comparisons:

- use matching seeds;
- preserve Model A/B controls;
- verify hook-off and injection-zero behavior;
- avoid interpreting answer accuracy without counterfactuals.

## Acceptance criteria

Write a short work history entry answering:

- Which intervention best encourages true operands?
- Which intervention most strongly encourages private calculator-code use?
- Does oracle warmup make learned true operands maintainable?
- Should the project embrace private protocols as a research target, or keep pushing for human-readable protocols first?

# Overall caution

Do not collapse future evaluation back into true-operand metrics only. The point of the next phase is to distinguish:

- intended true-operand calculator protocols;
- useful non-human/private calculator protocols;
- ordinary transformer bypass.
