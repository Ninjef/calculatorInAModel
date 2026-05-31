# 2026-05-31 Online Hard Memory Semantic-Distill Handoff

## Question

Can sparse zero-improvement online hard memory be made handoff-compatible by
adding additive semantic distillation during source training?

The prior online-hard-memory source reached high calculator accuracy, but the
trusted frozen-policy additive handoff missed badly. The hypothesis here is
that source accuracy was not enough: the source also needed additive/readout
geometry that makes arbitrary result classes easy for the non-bottleneck path
to decode.

## Mechanism

Combine:

- `--result-boundary-target-mode zero_improvement`
- `--result-boundary-target-sample-count 24`
- `--result-boundary-target-policy-topk-count 8`
- `--result-boundary-target-unique-sampling`
- `--result-boundary-target-online-hard-memory`
- `--result-boundary-target-online-memory-freeze-when-full`
- `--additive-semantic-distill-weight 1`
- `--additive-semantic-distill-sample-count 8`

The online hard memory chooses a hard answer-improving result from sparse
candidate scoring. The semantic-distill auxiliary forces arbitrary result
classes and trains the additive path to match frozen answer-decoder logits. It
does not tell the policy which result to request for a prompt.

## Tooling Fix

The first launches failed before training because the generated directory name
exceeded the filesystem path-component limit. Added deterministic run-suffix
shortening in `scripts/overfit_one_batch.py`, preserving a stable prefix and a
short SHA1 digest. Full settings remain recorded in run metadata.

## Source Run

```text
runs/ohm_semdist_src800/2026-05-30_193507_633991_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rbtonlinehardmem-rbtmem-44b570a4cc/model-c-2digit-seed6
```

Command summary:

- op19 exhaustive `400`-prompt grid.
- Source mode: `calculator_bottleneck_mode=answer_decoder`.
- Policy estimator: `gumbel_concrete_interface`.
- Frozen product semantic decoder loaded with `semantic_decoder_only`.
- `800` source steps, snapshots/checkpoints every `200`.

Results:

| Step | Normal | Zero-inj | Learned calc | Best true | Distill agreement | Forced evals |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0150 | 0.0575 | 0.0225 | 0.9100 | 0.0340 | active |
| 200 | 0.4550 | 0.0425 | 0.4775 | 1.0000 | 0.4700 | 86,400 |
| 400 | 0.9575 | 0.0475 | 0.9525 | 1.0000 | 0.6610 | 86,400 |
| 600 | 1.0000 | 0.0475 | 0.9975 | 1.0000 | 0.7378 | 86,400 |
| 800 | 1.0000 | 0.0450 | 1.0000 | 1.0000 | 0.7459 | 86,400 |

Final source metrics:

- Final eval: `400/400 = 1.0000`.
- Diagnostic calculator-result accuracy: `1.0000`.
- 128-sample counterfactuals: injection-zero `0.0391`, forced-zero `0.0000`,
  forced-random `0.0391`.
- Memory filled/froze by step `50`, so cumulative forced-result evaluations
  stayed capped at `86,400`.

## Trusted Handoff

```text
runs/ohm_semdist_handoff600/2026-05-30_193842_897628_model-c-op0-19-fullgrid-adec-product/model-c-2digit-seed6
```

Command summary:

- Loaded source final checkpoint with `compatible_model`.
- Additive non-bottleneck mode: `calculator_bottleneck_mode=none`.
- `calculator_estimator=ste`.
- `--freeze-semantic-decoder`.
- `--freeze-calculator-policy`.
- `answer_loss_weight=1`.
- `600` downstream/readout steps.

Results:

| Step | Normal | Zero-inj | Forced zero | Forced random | Calc accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1300 | 0.0050 | 0.0000 | 0.0050 | 1.0000 |
| 100 | 0.2475 | 0.0700 | 0.0075 | 0.0525 | 1.0000 |
| 200 | 0.3925 | 0.0825 | 0.0100 | 0.0450 | 1.0000 |
| 300 | 0.4775 | 0.0325 | 0.0000 | 0.0325 | 1.0000 |
| 400 | 0.7700 | 0.0450 | 0.0050 | 0.0175 | 1.0000 |
| 500 | 0.9575 | 0.0325 | 0.0025 | 0.0325 | 1.0000 |
| 600 | 1.0000 | 0.0525 | 0.0050 | 0.0175 | 1.0000 |

Final handoff metrics:

- Final eval: `400/400 = 1.0000`.
- Final 128-sample counterfactuals: injection-zero `0.0547`, forced-zero
  `0.0000`, forced-random `0.0312`.
- Diagnostic calculator-result accuracy: `1.0000`.

## Interpretation

This is a clean positive for the handoff-geometry hypothesis. The previous
online-hard-memory source had high source calculator accuracy but trusted
handoff only `0.4650` final / `0.4850` step-600 normal. Adding additive
semantic distillation made the same sparse online-hard source family transfer
perfectly into the trusted frozen-policy additive handoff.

This does not complete the thesis. The result is still on the fixed op19 grid,
uses per-prompt memory, and scores sparse forced-result candidates before the
memory fills. The next high-leverage tests are fresh-seed replication,
streaming/fresh-prompt memory, larger-range stress, or routed/many-calculator
validation. Do not spend compute on same-seed weight/sample/length tweaks.

## Decision

```text
online_hard_memory_semantic_distill_handoff_positive
```
