# Phase 7 Seventy-Ninth Task: Stabilized Source Seed-10 Replication

## Status

Completed 2026-05-29.

## Question

Does the no-decay stabilized source recipe replicate on a fresh seed, including
the downstream non-bottleneck continuation/readout behavior?

## Setup

- Trained a fresh no-decay stabilized bottleneck source with CLI seed `10`.
- Ran an 800-step frozen-policy additive handoff from the final source.
- Ran a direct 600-step policy-backbone-frozen readout diagnostic.
- Ran an 800-step frozen-policy continuation diagnostic.

## Result

| Stage | Final eval | Best snapshot | Injection-zero | Forced-random | Oracle | Calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source final | `0.9000` | `0.8850` | `0.0391` | n/a | `1.0000` | `0.8984` |
| handoff | `0.3275` | `0.3375` | `0.0313` | `0.0938` | `0.4141` | `0.8984` |
| direct readout | `0.4275` | `0.4350` | n/a | n/a | n/a | n/a |
| continuation | `0.4350` | `0.4250` | `0.0078` | `0.1250` | `0.4922` | `0.8984` |

## Decision

```text
stabilized_source_seed10_source_positive_transfer_negative
```

The no-decay source-acquisition recipe replicated at the bottleneck source
level, but the downstream non-bottleneck positive did not replicate.

## Next

Compare seed-9 positive and seed-10 negative geometry to build a cheap
transfer/readout proxy before spending full continuation/readout budget on
future stabilized sources.
