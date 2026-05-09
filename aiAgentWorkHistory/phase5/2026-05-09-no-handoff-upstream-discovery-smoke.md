# 2026-05-09 - No-handoff upstream discovery smoke

Task: Phase 5 fourth task, controlled no-handoff upstream discovery smoke.

## Claim

Test whether answer-only training can discover the calculator-query protocol
without any Stage 1 supervised interface handoff, while preserving the strict
Phase 4/5 semantic bottleneck.

## Loader clarity

Updated `scripts/overfit_one_batch.py` with:

```text
--semantic-decoder-checkpoint-load-scope full_model | semantic_decoder_only
```

The default remains `full_model`, preserving old behavior. The new
`semantic_decoder_only` opt-in loads only:

- `answer_offset_emb.*`
- `answer_decoder.*`
- `calculator_hook.output_proj.*`

This task's main smoke used `full_model` explicitly and recorded the current
behavior as `full_model_current_behavior`.

## Runner

Added:

```text
scripts/run_phase5_no_handoff_upstream_discovery_smoke.py
```

Run root:

```text
runs/2026-05-09_phase5_no_handoff_upstream_discovery_smoke
```

The runner:

- records the Stage 0B operand-aware oracle semantic decoder checkpoint;
- records the previous seed `2` step `55` and seed `5` step `25`
  upstream-assisted completion summaries;
- runs the two allowed no-handoff full-model seeds;
- writes `summary.json` and `summary.md`;
- compares final checkpoints against the Stage 0B checkpoint by parameter
  group;
- runs canonical, private-protocol, and full-enum diagnostics on final and
  best dense checkpoints.

## Starting point

- Stage 0B checkpoint:
  `/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt`
- Repo-local Stage 0B checkpoint was absent, so the absolute Phase 4
  fact-sheet path was used.
- Interpretation label: `no_handoff_full_model_init`.
- Semantic checkpoint load scope: `full_model`.

## Shared setup

- `answer_format=sum_left_operand`
- `calculator_output_format=sum_left_operand`
- `calculator_read_position=operand_spans`
- `calculator_read_span_width=2`
- `calculator_bottleneck_mode=answer_decoder`
- `calculator_estimator=adaptive_interface`
- `freeze_semantic_decoder=true`
- `answer_loss_weight=1.0`
- `aux_operand_loss_weight=0.0`
- `adaptive_interface_loss_weight=0.0`
- `input_proj_anchor_weight=0.0`
- `input_proj_lr=0.0003`
- `upstream_lr=0.00003`
- no oracle training and no direct operand labels

## Fast gates

| Condition | Final eval | Best dense step normal/operand/pair/calc | Final normal/operand/pair/calc | Injection-zero | Forced-random | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CLI seed `0` / effective seed `2` | `0.048828` | step `650`: `0.457031` / `0.457031` / `0.457031` / `0.457031` | `0.054688` / `0.054688` / `0.054688` / `0.070312` | `0.0` | `0.003906` | `1.0` |
| CLI seed `3` / effective seed `5` | `0.062500` | step `350`: `0.433594` / `0.433594` / `0.433594` / `0.441406` | `0.050781` / `0.050781` / `0.050781` / `0.082031` | `0.0` | `0.007812` | `1.0` |

## Parameter deltas

Compared with Stage 0B:

| Condition | `calculator_hook.input_proj` L2 / max | upstream L2 / max | semantic decoder L2 |
| --- | ---: | ---: | ---: |
| seed `0` | `0.860573` / `0.390899` | `1.60121` / `0.053195` | `0.0` |
| seed `3` | `0.808462` / `0.322416` | `1.48781` / `0.050160` | `0.0` |

The runs were not no-ops: upstream and input-proj parameters moved measurably
while the semantic decoder stayed frozen.

## Diagnostics

| Selection | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-minus-true/best gaps | Learned-best |
| --- | ---: | ---: | ---: | ---: |
| seed `0` final | `0.0391` / `0.0391` / `0.0625` | `0.0375` / `0.0375` / `0.0625` | `6.1375` / `6.1375` | `0.0391` |
| seed `0` best step `650` | `0.4297` / `0.4297` / `0.4297` | `0.4200` / `0.4200` / `0.4200` | `2.5851` / `2.5851` | `0.4062` |
| seed `3` final | `0.0508` / `0.0508` / `0.0859` | `0.0575` / `0.0575` / `0.0925` | `5.6418` / `5.6418` | `0.0469` |
| seed `3` best step `350` | `0.4336` / `0.4336` / `0.4336` | `0.4500` / `0.4500` / `0.4575` | `2.0637` / `2.0637` | `0.4141` |

## Interpretation

This is a clean no-handoff full-model initialization smoke failure. The fixed
semantic decoder/calculator path remained mechanically viable
(`oracle_at_eval=1.0`), but answer-only training did not discover the true
calculator-query protocol in either allowed seed.

The best checkpoints were partial and still had strongly positive full-enum
learned-minus-true/best gaps. Final checkpoints drifted close to chance learned
actions. Because Stage 1 produced no real no-handoff discovery checkpoint, I
did not run the optional strict random-upstream branch.

## Recommendation

Do not broaden into a seed/LR sweep. Move next to one minimal local-target /
full-enum target-prop style objective or a Gumbel-Softmax estimator, keeping
the same strict identifiable setup and diagnostics.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py scripts/run_phase5_no_handoff_upstream_discovery_smoke.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q -k "semantic_decoder_checkpoint_load_scope or freeze_semantic_decoder_preserves_decoder_but_not_interface"
PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase5_no_handoff_upstream_discovery_smoke.py summarize
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase5_no_handoff_upstream_discovery_smoke.py run --jobs 2
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase5_no_handoff_upstream_discovery_smoke.py diagnostics
PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase5_no_handoff_upstream_discovery_smoke.py summarize
```
