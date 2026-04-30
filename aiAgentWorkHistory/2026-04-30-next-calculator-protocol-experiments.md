# 2026-04-30 - Next calculator protocol experiments

Task: Implement the revised plan for `aiAgentProjectTasks/2026-04-29-1320-Next-calculator-protocol-experiments.md`: add a forced result-class diagnostic, run the 1-digit weird-protocol autopsy, and start the 2-digit capacity-regime track.

## Code changes

- Added eval-only forced calculator result classes through `CalculatorHook.forward`, `TinyGPT.forward`, and `scripts/diagnose_calculator_protocol.py`.
- Added `--forced-calculator-result-class` for single-class counterfactual evals.
- Added `--forced-result-sweep`, which saves:
  - `forced_result_sweep.csv`
  - `forced_result_summary.json`
  - `result_codebook.csv`
- The sweep records generated answer, exact match, correct first-token probability, teacher-forced target log probability, learned-class match, and true-sum match for every forced class.
- Added tests for forced-class behavior and the diagnostic CLI sweep.

Verification:

```bash
python3 -m pytest
```

Result: `29 passed`.

## 1-digit codebook autopsy

Primary checkpoint:

```text
runs/2026-04-29_125836_910885_model-c-op0-9-1/model-c-1digit-seed2/final_weights.pt
```

Outputs:

```text
runs/2026-04-29_125836_910885_model-c-op0-9-1/model-c-1digit-seed2/forced-result-sweep/
```

Key results on 512 samples:

- Normal exact match: `0.904`
- True operand exact match: `0.014`
- Calculator result accuracy: `0.146`
- Learned class is best forced class: `0.244`
- True-sum class is best forced class: `0.049`
- Mean learned-minus-true-sum target log probability: `+0.655`
- Mutual information:
  - learned class vs true sum: `1.062` bits
  - learned class vs first answer token: `0.973` bits
  - learned class vs carry: `0.291` bits

The strong checkpoint emits only three result classes:

| Learned class | Count | True sums represented |
| --- | ---: | --- |
| `4` | 32 | exactly `5` |
| `6` | 109 | `0, 1, 2, 3, 4, 6` |
| `11` | 371 | `7..18` |

Interpretation: this is a coarse private answer-code, not the intended true-sum protocol. It carries meaningful answer-region information and is causally better than forcing true sums for this checkpoint, but it is not a perfect discrete code: another forced class is often slightly better per prompt.

Sanity checkpoint:

```text
runs/2026-04-29_125836_910885_model-c-op0-9/model-c-1digit-seed1/final_weights.pt
```

Outputs:

```text
runs/2026-04-29_125836_910885_model-c-op0-9/model-c-1digit-seed1/forced-result-sweep/
```

Key results:

- Normal exact match: `0.418`
- Learned class is best forced class: `0.424`
- True-sum class is best forced class: `0.037`
- Mean learned-minus-true-sum target log probability: `+0.423`
- Mutual information with true sum: `1.718` bits

The weaker checkpoint also uses a non-human code, but with lower answer accuracy and a more fragmented class partition.

## 2-digit track

Motivation: 1-digit addition may force models to be so small that they cannot learn a usable calculator interface. The 2-digit track makes the task harder while allowing less cramped models.

### Candidate A: `2L/1H/16d/mlp1`, hook after layer 1

All runs used fixed-width 2-digit addition, full operand range `0..99`, 1000 steps, eval samples `512`, seed argument `0` (run seed `2`), and hook after layer 1.

| Variant | Exact match | Final loss | Run |
| --- | ---: | ---: | --- |
| Model C oracle | `0.988` | `0.0049` | `runs/2026-04-29_220949_913725_model-c-oracle/model-c-2digit-seed2` |
| Model A | `0.711` | `0.1380` | `runs/2026-04-29_221055_780936_model-a/model-a-2digit-seed2` |
| Model B | `0.787` | `0.1632` | `runs/2026-04-29_221055_780881_model-b/model-b-2digit-seed2` |
| Model C learned STE | `0.023` | `1.3661` | `runs/2026-04-29_221445_951260_model-c/model-c-2digit-seed2` |

Residual probes at layer-1 `=` position, 1024 samples and 400 probe steps:

| Variant | Operand A probe | Operand B probe |
| --- | ---: | ---: |
| Model A | `0.029` | `0.039` |
| Model B | `0.034` | `0.039` |

The regime is promising because oracle works and A/B struggle naturally. The concern is that operand linear accessibility at the hook is still weak.

Learned STE Model C collapsed confidently:

- Diagnostic exact match: `0.008` on 128 trace samples
- Operand exact match: `0.000`
- Calculator result accuracy: `0.008`
- Mean operand confidences: `1.000` / `0.996`
- Injection-zero exact match: `0.010`
- Oracle-at-eval exact match: `0.010`
- Forced-zero result exact match: `0.010`

Interpretation: the learned run did not find a useful private 2-digit protocol. It saturated into confidently wrong operand classes, and neither removing nor fixing the calculator result helped after training. This is an input-protocol/optimization failure, not an output-side impossibility, because oracle training in the same architecture works.

### Candidate B: `2L/1H/32d/mlp1`, hook after layer 1

| Variant | Exact match | Final loss | Run |
| --- | ---: | ---: | --- |
| Model C oracle | `1.000` | `0.0001` | `runs/2026-04-29_221246_375002_model-c-oracle/model-c-2digit-seed2` |
| Model A | `0.086` | `0.9114` | `runs/2026-04-29_221246_374987_model-a/model-a-2digit-seed2` |
| Model B | `0.047` | `0.9213` | `runs/2026-04-29_221246_374983_model-b/model-b-2digit-seed2` |

Residual probes at layer-1 `=` position:

| Variant | Operand A probe | Operand B probe |
| --- | ---: | ---: |
| Model A | `0.034` | `0.034` |
| Model B | `0.024` | `0.029` |

The `32d` oracle path is excellent, but A/B optimization was worse than `16d` in this run and probes did not improve. Bigger is not automatically cleaner here.

## Current conclusion

The project should move toward 2-digit math, but plain STE with a 100-class operand interface is too brittle as currently configured. The best next default is the `2L/1H/16d/mlp1` 2-digit regime because it satisfies the most important controls:

- Model A/B are below `0.85`.
- Model B is in the same broad band as Model A.
- Oracle Model C works very well.
- Learned Model C failure is diagnostic rather than ambiguous output-side failure.

Next recommended work:

- Improve the learned input protocol before broad architecture sweeps:
  - oracle warmup then learned operands;
  - low-weight auxiliary operand schedules;
  - smaller operand vocab curriculum such as 2-digit formatting with `operand_max=19` or `49`;
  - batched forced-result sweep implementation before applying it to 199 result classes.
- Add probe positions for operand-token locations as a diagnostic, but keep the main success criterion at the actual hook read position.
