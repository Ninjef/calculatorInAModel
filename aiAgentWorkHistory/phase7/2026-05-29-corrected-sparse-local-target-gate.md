# 2026-05-29 Corrected Sparse Local-Target Gate

## Question

Can a sparse `policy_reweighted_t1` local target be rescued by correcting the
target construction, rather than changing the candidate proposal?

Prior sampled branches forced all target mass onto the sampled candidate set.
That makes missing the useful result especially damaging. This gate tests a
simple correction: keep unscored result classes in the target with an imputed
baseline loss, and only reweight scored candidates by observed forced-result
answer loss.

## Code

Extended:

```text
scripts/run_phase7_local_target_stage1_lift_gate.py
```

New branch family:

```text
corrected_policy_reweighted_t<T>_u<U>_b<mean|current|max>
```

The branch:

1. Uniformly samples `U` result classes per prompt.
2. Scores those candidates with forced-result answer loss.
3. Fills unscored result classes with a per-row baseline loss:
   - `mean`: mean sampled loss
   - `current`: current-policy-weighted sampled loss
   - `max`: max sampled loss
4. Builds a dense policy-reweighted target over all result classes.

## Commands

Smoke:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --operand-max 3 \
  --steps 2 \
  --eval-every 1 \
  --control-eval-every 2 \
  --eval-samples 32 \
  --branches corrected_policy_reweighted_t1_u4_bmean,corrected_policy_reweighted_t1_u4_bcurrent \
  --output-root runs/2026-05-29_phase7_corrected_local_target_gate/smoke_op3
```

Focused gate:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --steps 200 \
  --eval-every 25 \
  --control-eval-every 100 \
  --eval-samples 128 \
  --branches policy_reweighted_t1,sampled_policy_reweighted_t1_k0_u32,corrected_policy_reweighted_t1_u8_bmean,corrected_policy_reweighted_t1_u8_bcurrent,corrected_policy_reweighted_t1_u8_bmax,corrected_policy_reweighted_t1_u16_bmean,corrected_policy_reweighted_t1_u16_bcurrent \
  --output-root runs/2026-05-29_phase7_corrected_local_target_gate/corrected_sparse_200
```

## Results

Final 200-step metrics:

| Branch | Scored results | True coverage | Target argmax | Target true prob | Exact-grid calc | Sampled normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `policy_reweighted_t1` | full | n/a | `0.9925` | `0.9371` | `0.5600` | `0.5391` |
| `sampled_policy_reweighted_t1_k0_u32` | `32` | `0.8450` | `0.8350` | `0.7827` | `0.3350` | `0.3438` |
| `corrected_policy_reweighted_t1_u8_bmean` | `8` | `0.1850` | `0.1925` | `0.1953` | `0.1150` | `0.0938` |
| `corrected_policy_reweighted_t1_u8_bcurrent` | `8` | `0.1850` | `0.1875` | `0.1799` | `0.1100` | `0.0938` |
| `corrected_policy_reweighted_t1_u8_bmax` | `8` | `0.1850` | `0.1850` | `0.1794` | `0.0675` | `0.0625` |
| `corrected_policy_reweighted_t1_u16_bmean` | `16` | `0.4050` | `0.4050` | `0.3907` | `0.2100` | `0.2500` |
| `corrected_policy_reweighted_t1_u16_bcurrent` | `16` | `0.4050` | `0.4125` | `0.3707` | `0.2500` | `0.2500` |

Additional diagnostics:

| Branch | Scored target mass | Unscored target mass | Target effective results |
| --- | ---: | ---: | ---: |
| `u8_bmean` | `0.7687` | `0.2313` | `4.9629` |
| `u8_bcurrent` | `0.5130` | `0.4870` | `10.5117` |
| `u8_bmax` | `0.9617` | `0.0383` | `2.4750` |
| `u16_bmean` | `0.9375` | `0.0625` | `2.9774` |
| `u16_bcurrent` | `0.7483` | `0.2517` | `5.5869` |

## Interpretation

Negative.

Preserving unscored policy mass did not rescue sparse local targets. At `u16`,
the best corrected branches reached only `0.2500` exact-grid calc / `0.2500`
sampled normal, below raw uniform `u32` (`0.3350` / `0.3438`) and far below
the exact full-vocabulary ceiling (`0.5600` / `0.5391`).

The failure still tracks true-candidate coverage. `u8` saw the true result only
`18.5%` of the time; `u16` only `40.5%`. The correction changed how mass was
distributed over unscored classes, but it did not create useful pressure when
the good result was missing.

## Decision

```text
corrected_sparse_policy_reweighted_target_negative
```

Do not repeat these mean/current/max imputation branches or simple `u8/u16`
tuning as novelty.

Next useful local-target work needs a learned/generalized proposal, a stronger
estimator correction with an explicit bias/variance argument, or a target
construction that does not rely on high true-result coverage.
