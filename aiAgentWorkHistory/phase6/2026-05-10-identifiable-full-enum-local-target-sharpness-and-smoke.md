# Identifiable Full-Enum Local-Target Sharpness and Smoke

## Claim Tested

In the Phase 4/5 identifiable `sum_left_operand` setup, frozen answer-decoder
NLL over the full `20 x 20` action space should identify the true calculator
query sharply enough to serve as a local interface-discovery signal without
direct true-operand supervision.

## Code Changes

- Added estimator alias `identifiable_full_enum_local_target`.
- Added `--action-loss-full-enum-target-mode soft_pair | hard_best_pair`.
- Added hard-best full-enum local target CE for independent operand heads.
- Added target-sharpness metrics: tie-aware true-best, best-left accuracy,
  top-1/top-3/top-5 mass, true-pair rank/probability.
- Added reusable Phase 6 runner:

```text
scripts/run_phase6_identifiable_full_enum_local_target.py
```

Runner subcommands:

```text
summarize-target
run-smoke
diagnostics
summarize
```

## Exact Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/run_full_enum_action_loss_diagnostic.py scripts/run_phase6_identifiable_full_enum_local_target.py
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase6_identifiable_full_enum_local_target.py summarize-target --samples 128 --temperature 0.25
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase6_identifiable_full_enum_local_target.py run-smoke --steps 500 --snapshot-every 50 --input-proj-lr 0.001 --target-mode hard_best_pair
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase6_identifiable_full_enum_local_target.py run-smoke --skip-frozen --include-upstream --target-mode hard_best_pair
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase6_identifiable_full_enum_local_target.py diagnostics
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase6_identifiable_full_enum_local_target.py summarize
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_data.py tests/test_model.py -q
```

The runner also wrote exact subprocess commands to:

```text
runs/2026-05-10_phase6_identifiable_full_enum_local_target/commands.jsonl
```

## Run Paths

Run root:

```text
runs/2026-05-10_phase6_identifiable_full_enum_local_target
```

Stage 0B checkpoint:

```text
/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

Frozen-upstream smoke:

```text
runs/2026-05-10_phase6_identifiable_full_enum_local_target/stage1/frozen_upstream/2026-05-10_165718_640502_model-c-op0-19-identifiable_full_enum_local_target-inlr0.001-uplr0.0003-fullt0.25-fullchunk64-hard_best_pair-answer_decoder-sum_left_operand/model-c-2digit-seed2
```

Upstream-open smoke:

```text
runs/2026-05-10_phase6_identifiable_full_enum_local_target/stage1/upstream_open/2026-05-10_170237_134099_model-c-op0-19-identifiable_full_enum_local_target-inlr0.0003-uplr3e-05-fullt0.25-fullchunk64-hard_best_pair-answer_decoder-sum_left_operand/model-c-2digit-seed2
```

## Target Sharpness

Temperature: `0.25`; samples: `128`; action pairs: `400`.

| Checkpoint | Best=true | Tie-aware true-best | Mean true rank | Effective pairs | True-pair prob | Top-5 mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 0B full-model load | `1.000` | `1.000` | `1.000` | `1.079` | `0.989` | `0.999` |
| Phase 4 retained positive | `1.000` | `1.000` | `1.000` | `1.079` | `0.989` | `0.999` |

Gate decision: pass. The identifiable target is sharp enough to train against.

## Smoke Results

Both branches used direct teacher weights exactly off:

```text
oracle_train=false
oracle_warmup_steps=0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
local_target_loss_weight=1.0
local_target_mode=hard_best_pair
semantic_decoder_checkpoint_load_scope=full_model
freeze_semantic_decoder=true
```

| Branch | Best step | Lightweight normal/operand/pair/calc | Final eval exact | Final aux/local/adaptive/anchor |
| --- | ---: | ---: | ---: | --- |
| frozen upstream | `250` | `0.688 / 0.688 / 0.688 / 0.688` | `0.402` | `0.0 / 1.0 / 1.0 / 0.0` |
| upstream open | `600` | `0.680 / 0.680 / 0.680 / 0.680` | `0.359` | `0.0 / 1.0 / 1.0 / 0.0` |

No Stage 2 retention was run because neither branch reached the required
`>=0.90` fast-gate threshold.

## Selected Checkpoint Diagnostics

| Checkpoint | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-true/best gap | Learned-best | Oracle | Injection-zero | Forced-random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 0B | `0.0156 / 0.0156 / 0.0352` | `0.0125 / 0.0125 / 0.0325` | `9.379 / 9.379` | `0.0078` | `1.000` | `0.000` | `0.0039` |
| frozen best step `250` | `0.566 / 0.566 / 0.566` | `0.5825 / 0.5825 / 0.585` | `1.813 / 1.813` | `0.594` | `1.000` | `0.000` | `0.0234` |
| frozen final | `0.398 / 0.398 / 0.414` | `0.4225 / 0.4225 / 0.4325` | `3.557 / 3.557` | `0.375` | `1.000` | `0.000` | `0.0234` |
| upstream-open best step `600` | `0.734 / 0.734 / 0.734` | `0.6825 / 0.6825 / 0.6825` | `1.041 / 1.041` | `0.633` | `1.000` | `0.000` | `0.0273` |
| upstream-open final | `0.336 / 0.336 / 0.348` | `0.3375 / 0.3375 / 0.345` | `3.193 / 3.193` | `0.336` | `1.000` | `0.000` | `0.0234` |

Full-enum true-best and best-matches-true stayed `1.000` for all selected
diagnostics, confirming that target construction remained sharp even when the
learned interface was partial.

## Parameter Movement

Parameter deltas are versus each branch's step `0` checkpoint.

| Checkpoint | input-proj L2 / max | upstream L2 / max | semantic decoder L2 / max |
| --- | ---: | ---: | ---: |
| frozen best step `250` | `2.3458 / 0.2006` | `0.0 / 0.0` | `0.0 / 0.0` |
| frozen final | `4.4895 / 0.3990` | `0.0 / 0.0` | `0.0 / 0.0` |
| upstream-open best step `600` | `1.6701 / 0.1556` | `0.3410 / 0.0160` | `0.0 / 0.0` |
| upstream-open final | `2.6288 / 0.2561` | `0.6235 / 0.0268` | `0.0 / 0.0` |

Upstream-open changed `14/29` upstream tensors. The semantic decoder stayed
exactly fixed in all selected checkpoints.

## Comparison To Phase 5 No-Handoff Smoke

Phase 5 no-handoff full-model answer-only best partial checkpoints reached:

```text
seed 0: canonical operand/pair/calc 0.4297
seed 3: canonical operand/pair/calc 0.4336
```

This Phase 6 local-target smoke exceeded that:

```text
frozen best selected canonical operand/pair/calc 0.566
upstream-open best selected canonical operand/pair/calc 0.734
```

Because the local target was nonzero, this is not answer-only discovery. It is
evidence that the answer-derived full-enum local target provides a useful
interface training signal.

## Decision And Recommendation

Positive:

- Target sharpness passed decisively.
- Local-target training improved learned protocol metrics without direct
  operand supervision.
- The best upstream-open selected checkpoint materially exceeded the Phase 5
  no-handoff best partial checkpoints.
- Oracle-at-eval stayed `1.0`, injection-zero stayed `0.0`, and semantic
  decoder movement stayed `0.0`.

Negative / not yet solved:

- No snapshot reached the `>=0.90` retention gate.
- Final checkpoints drifted while the local target was still on.
- Full-enum learned-minus-true/best gaps remained positive, and learned-best
  stayed far below `1.0`.

Next step:

Do not move to strict random-upstream discovery yet. Try targeted
optimization/parameterization improvements: stronger hard-best local weight,
input-proj LR variants, a soft or Gumbel/Concrete relaxation, or a joint-pair
head adapted to `operand_spans`.

## Validation

```text
73 passed in 2.49s
```
