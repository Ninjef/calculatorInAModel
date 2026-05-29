# 2026-05-29 Stabilized Source Seed-10 Replication

## Aim

Replicate the no-decay entropy/diversity stabilized source recipe on a fresh
seed, then gate downstream non-bottleneck work on the resulting source and
handoff behavior.

This tests whether the previous seed-9 positive is robust, rather than only a
single favorable source geometry.

## Source Acquisition

Run root:

```text
runs/2026-05-29_phase7_stabilized_source_replication
```

Source cell:

```text
src10_entropy0p05_div0p1_nodecay_steps1600
```

Saved source run:

```text
runs/2026-05-29_phase7_stabilized_source_replication/src10_entropy0p05_div0p1_nodecay_steps1600/2026-05-29_031224_343252_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed12
```

The CLI used `--seed 10`; the script stores `seed=args.seed+num_digits`, so
the saved run directory is `seed12`.

Configuration matched the no-decay stabilized source recipe:

- `result_policy_improvement_assignment_weight=10`
- `result_policy_entropy_weight=0.05`
- `result_policy_batch_diversity_weight=0.1`
- `result_policy_stabilization_decay_steps=0`
- frozen product semantic decoder
- exact-grid natural `0..19`
- 1600 source steps with 100-step snapshots

Source result:

| Step | Source normal | Injection-zero | Oracle | Learned calc |
| --- | ---: | ---: | ---: | ---: |
| `1000` | `0.7500` | `0.0325` | `1.0000` | `0.7500` |
| `1300` | `0.8425` | `0.0375` | `1.0000` | `0.8425` |
| `1400` | `0.8550` | `0.0625` | `1.0000` | `0.8550` |
| `1600` | `0.8850` | `0.0750` | `1.0000` | `0.8850` |
| final eval | `0.9000` | `0.0391` | `1.0000` | `0.8984` |

The no-decay source-acquisition part replicated: this seed did not collapse
and reached a strong bottleneck source.

## Additive Handoff and Diagnostics

All downstream runs started from the final source checkpoint.

| Stage | Final eval | Best snapshot | Injection-zero | Forced-random | Oracle | Calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 800-step final-source handoff | `0.3275` | `0.3375` at handoff step `600` | `0.0313` | `0.0938` | `0.4141` | `0.8984` |
| direct 600-step readout | `0.4275` | `0.4350` at step `500` | n/a | n/a | n/a | n/a |
| 800-step continuation | `0.4350` | `0.4250` at step `500` | `0.0078` | `0.1250` | `0.4922` | `0.8984` |

The direct readout and continuation diagnostics did not justify a
post-continuation readout: even after 800 continuation steps, final eval was
only `0.4350` and oracle-at-eval stayed below `0.50`.

## Decision

Label:

```text
stabilized_source_seed10_source_positive_transfer_negative
```

The no-decay stabilized source-acquisition recipe replicated at the bottleneck
source level, but the non-bottleneck continuation/readout positive did not
replicate on this seed.

## Interpretation

This is a sharp source-geometry boundary. The frozen calculator policy keeps a
good result signal (`~0.90` learned calc), but the additive downstream path
does not learn to use it: handoff, direct readout, and frozen-policy
continuation all remain far below gate.

The current no-decay recipe therefore improves source acquisition reliability
relative to decay-to-zero, but it does not yet reliably produce transferable
non-bottleneck geometry. The next useful work is a selector or proxy that
distinguishes seed-9-like transferable sources from seed-10-like non-readable
sources before spending continuation/readout budget.

## Anti-Rerun Note

Do not repeat this exact CLI seed-10 no-decay source, final-source additive
handoff, direct 600-step readout, or 800-step continuation as novelty.

Next useful tests:

- compare seed-9 positive and seed-10 negative geometry to identify a cheap
  transfer/readout proxy;
- validate another fresh no-decay source only if it is part of a planned
  replication or selector gate;
- optimize the source objective for continuation/readout geometry rather than
  only bottleneck source accuracy.

## Verification

The source run, final-source handoff, direct readout diagnostic, and
frozen-policy continuation diagnostic completed and wrote metrics under the run
root above.
