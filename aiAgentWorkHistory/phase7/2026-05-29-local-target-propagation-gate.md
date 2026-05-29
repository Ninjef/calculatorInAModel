# 2026-05-29 Local-Target Propagation Gate

## Aim

Start the active target-propagation/local-target direction with a small Stage 0
feasibility gate before any long run.

The test asks whether locally constructed result-boundary targets can create a
model update direction aligned with the known hard boundary-target ceiling,
without simply following the failed ordinary expected answer-loss gradient.

## Code

Added:

```text
scripts/run_phase7_local_target_propagation_gate.py
```

The script loads the standard Phase 6 product semantic decoder, freezes the
semantic decoder, builds the natural `0..19` exact grid, scores all forced
result classes, and compares gradient directions for:

- current-policy-reweighted forced-loss targets,
  `softmax(log current_policy - forced_loss / temperature)`;
- local logit-descent targets, where a free per-example result-logit vector is
  optimized against forced-result losses plus a proximity penalty;
- the hard best-result boundary target ceiling;
- the ordinary expected answer-loss baseline.

This is a Stage 0 diagnostic only. It still uses full forced-result scoring,
so it is not yet scalable.

## Runs

Smoke:

```text
runs/2026-05-29_phase7_local_target_propagation_gate/smoke_op3_scaled_descent
```

Full exact-grid gate:

```text
runs/2026-05-29_phase7_local_target_propagation_gate/full_grid_seed2_final
```

Command:

```text
python3 scripts/run_phase7_local_target_propagation_gate.py \
  --output-root runs/2026-05-29_phase7_local_target_propagation_gate/full_grid_seed2_final
```

## Result

The hard boundary target remains valid on this grid:

| Metric | Value |
| --- | ---: |
| hard-best equals true sum | `1.0000` |
| soft ceiling true-result probability | `0.8003` |
| ordinary expected-loss vs boundary result/upstream cosine | `-0.1045 / -0.0034` |

Current-policy-reweighted targets:

| Target | Result cosine | Upstream cosine | True prob | Effective results |
| --- | ---: | ---: | ---: | ---: |
| `t=0.25` | `1.0000` | `1.0000` | `0.9999` | `1.0011` |
| `t=0.5` | `0.9989` | `0.9978` | `0.9873` | `1.0893` |
| `t=1.0` | `0.9355` | `0.8766` | `0.8002` | `2.7229` |
| `t=2.0` | `0.5976` | `0.4902` | `0.3624` | `13.3178` |

Local logit-descent targets:

| Target | Result cosine | Upstream cosine | True prob | Effective results |
| --- | ---: | ---: | ---: | ---: |
| `proximity=0.01` | `1.0000` | `1.0000` | `0.9845` | `1.1439` |
| `proximity=0.1` | `0.9998` | `0.9997` | `0.9140` | `1.8351` |
| `proximity=1.0` | `-0.0895` | `-0.0028` | `0.0330` | `38.8377` |

## Decision

Label:

```text
local_target_propagation_stage0_partial_positive
```

Local-target propagation style targets can pass the Stage 0 gradient-alignment
gate, and the aligned directions are clearly different from ordinary expected
answer loss. The softer `policy_reweighted_t1` target is the most interesting
tested point because it remains strongly aligned while keeping nontrivial
target entropy.

This is not yet a final method. The sharpest targets are almost the known hard
boundary teacher, and every tested target still depends on full forced-result
enumeration.

## Anti-Rerun Note

Do not repeat this same seed-2 exact-grid Stage 0 sweep over policy-reweighted
temperatures `0.25/0.5/1/2` and logit-descent proximity weights
`0.01/0.1/1` as novelty.

Next useful tests:

- run a short Stage 1 lift gate for `policy_reweighted_t1` and/or
  `logit_descent_p0.1`, compared against the hard-boundary ceiling and the
  failed expected-loss baseline;
- if Stage 1 works, design a cheaper approximation that avoids full
  result-class enumeration;
- do not claim scalability from this Stage 0 gate alone.

## Verification

Commands completed:

```text
python3 -m py_compile scripts/run_phase7_local_target_propagation_gate.py
python3 scripts/run_phase7_local_target_propagation_gate.py --operand-max 3 --output-root runs/2026-05-29_phase7_local_target_propagation_gate/smoke_op3_scaled_descent
python3 scripts/run_phase7_local_target_propagation_gate.py --output-root runs/2026-05-29_phase7_local_target_propagation_gate/full_grid_seed2_final
```

The full run wrote:

```text
runs/2026-05-29_phase7_local_target_propagation_gate/full_grid_seed2_final/local_target_propagation_summary.json
runs/2026-05-29_phase7_local_target_propagation_gate/full_grid_seed2_final/local_target_propagation_rows.csv
```
