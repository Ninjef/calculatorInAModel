# 2026-05-31 Online Hard Memory Result-Boundary

## Question

The cached-teacher branch showed that hard targets are much easier for the
result policy to imitate than soft zero-improvement weights, but cached teacher
tables are not scalable. Can sparse answer-derived scoring discover hard
targets online, store them, and then train the source policy without full-enum
rescoring?

## Mechanism

Added `--result-boundary-target-online-hard-memory`.

At each step, the mechanism:

1. Scores sparse result-boundary candidates using the existing
   `--result-boundary-target-sample-count`, `--result-boundary-target-unique-sampling`,
   and `--result-boundary-target-policy-topk-count` proposal.
2. For `zero_improvement`/`additive_zero_improvement`, keeps only candidates
   whose forced answer loss improves over injection-zero loss.
3. Stores the best discovered answer-improving result per fixed-grid prompt.
4. Trains result logits with hard CE to the stored result for prompts with a
   discovered target.

Added `--result-boundary-target-online-memory-freeze-when-full` to stop
candidate rescoring once every fixed-grid prompt has a discovered target.

This tests a mechanism-level hypothesis: sparse answer-derived discovery might
be used only to build a high-quality hard target memory, after which ordinary
hard imitation does the policy uptake.

## Runs

Shared setup:

- op19 exhaustive 400-prompt source gate.
- `result_boundary_target_mode=zero_improvement`.
- topk8+unique24 sparse candidate scoring.
- frozen product semantic decoder.
- `answer_loss_weight=0`.
- seed argument `4`, effective run seed `6`.

Run roots:

```text
runs/2026-05-31_phase7_online_hard_memory_result_boundary/online_hard_memory_zero_improvement_topk8_unique24_source200_cpu
runs/2026-05-31_phase7_online_hard_memory_result_boundary/online_hard_memory_zero_improvement_topk8_unique24_source800_cpu
runs/2026-05-31_phase7_online_hard_memory_result_boundary/online_hard_memory_freeze_full_zero_improvement_topk8_unique24_source800_cpu
```

## Results

| Branch | Scoring | Memory/target quality | Calc/final |
| --- | ---: | ---: | ---: |
| old sparse zero-improvement 200 | topk8+unique24 every step | soft true mass `0.9356` | `0.4275` / `0.4300` |
| online hard memory 200 | topk8+unique24 every step | best=true `1.0000` | `0.4550` / `0.4350` |
| online hard memory 800 | topk8+unique24 every step | best=true `1.0000` | `0.9675` / `0.9725` |
| online hard memory freeze-full 800 | stop after full memory | best=true `1.0000` | `0.9675` / `0.9725` |

Freeze-full curve:

| Step | Memory seen | Frozen | Cumulative forced evals | Learned calc |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `0.6650` | `0` | `9,600` | `0.0225` |
| `50` | `1.0000` | `1` | `86,400` | `0.0850` |
| `200` | `1.0000` | `1` | `86,400` | `0.4550` |
| `400` | `1.0000` | `1` | `86,400` | `0.8350` |
| `600` | `1.0000` | `1` | `86,400` | `0.9600` |
| `800` | `1.0000` | `1` | `86,400` | `0.9675` |

Final controls for freeze-full step 800:

- Final eval: `389/400 = 0.9725`.
- Snapshot normal: `0.9650`.
- Injection-zero: `0.0450`.
- Oracle: `1.0000`.
- Forced-random: low by the normal/zero/oracle snapshot pattern; forced-random
  was not separately enabled in this source-only gate.

## Interpretation

This is a substantial positive for less-prescriptive sparse credit assignment.
The old soft sparse zero-improvement target already had high true mass at step
200 but trained only to `0.4300` final. Hard online memory barely helped at
200, but by 800 it reached mature bottleneck source quality.

The freeze-full result is especially important. The memory was full and true
by step 50, so later forced-result scoring was unnecessary for this fixed-grid
gate. Freezing rescoring after memory fill preserved the same `0.9725` final
while capping cumulative forced-result scoring at `86,400`, versus about
`7,689,600` if topk8+unique24 scoring continued through all 801 steps.

This does not yet prove scalability. The memory is keyed by the fixed training
grid, so it may be transductive like earlier replay-memory local targets. It
also has not yet been validated through the trusted additive handoff or across
fresh seeds/many calculators.

## Decision

```text
online_hard_memory_result_boundary_partial_positive
```

Do not tune same-seed op19 length/LR variants. The next valuable tests are:

- fresh-seed replication;
- trusted additive handoff from the source checkpoint;
- streaming/fresh-prompt validation where prompt memory cannot simply memorize
  a closed grid;
- many-calculator cost accounting or routed-hook validation.
