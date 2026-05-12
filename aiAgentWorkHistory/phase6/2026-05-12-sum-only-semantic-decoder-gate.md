# Sum-Only Semantic Decoder Gate

## Task

```text
aiAgentProjectTasks/2026-05-12-phase-6-ninth-task-Sum-only-semantic-decoder-gate-and-natural-bridge-readiness.md
```

## Code Added

- Added `scripts/run_phase6_sum_only_semantic_decoder_gate.py`.
- Added `--exhaustive-grid` support to
  `scripts/run_full_enum_action_loss_diagnostic.py`, so the natural `0..19`
  branch can score all `400` prompts exactly once instead of relying on random
  samples.
- Added a regression test that verifies the exhaustive grid covers every pair
  once and builds the expected fixed-width prompts.

## Validation

```bash
python3 -m py_compile scripts/run_phase6_sum_only_semantic_decoder_gate.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
68 passed
```

## Commands

All commands were run in the sandbox with the usual thread limits:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_phase6_sum_only_semantic_decoder_gate.py stage0-existing
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_phase6_sum_only_semantic_decoder_gate.py stage0-candidates
```

Exact underlying oracle-training, zero-step gate, causal diagnostic, and
full-enum commands are recorded in logs under:

```text
runs/2026-05-12_phase6_sum_only_semantic_decoder_gate
```

## Results

Run root:

```text
runs/2026-05-12_phase6_sum_only_semantic_decoder_gate
```

Primary summaries:

```text
runs/2026-05-12_phase6_sum_only_semantic_decoder_gate/stage0_existing_decoder_diagnosis.json
runs/2026-05-12_phase6_sum_only_semantic_decoder_gate/stage0_existing_decoder_diagnosis.md
runs/2026-05-12_phase6_sum_only_semantic_decoder_gate/stage0_candidate_summary.json
runs/2026-05-12_phase6_sum_only_semantic_decoder_gate/summary.json
runs/2026-05-12_phase6_sum_only_semantic_decoder_gate/summary.md
```

### Stage 0A Existing Decoder Diagnosis

Existing checkpoint:

```text
runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

All-400 gate metrics:

| Metric | Value |
| --- | ---: |
| Normal learned exact | `0.0300` |
| Oracle-at-eval exact | `0.9300` |
| Forced true result exact | `0.9300` |
| Injection-zero exact | `0.0050` |
| Forced-zero exact | `0.0025` |
| Forced-random exact | `0.0325` |
| Full-enum best-result group matches true sum | `0.9075` |
| Full-enum true-result is best result | `0.9075` |
| Mean same-true-sum near-best pair count | `13.35` |
| Mean effective result count | `2.4820` |
| Semantic decoder delta | `0.0` |

Misses were systematic, not random sampling noise:

- Oracle-at-eval missed every prompt with true sum `12`, `31`, and `32`.
- Full-enum best-result matching missed every prompt with true sum `23`, `12`,
  and `31`.
- Forced true result class agreed exactly with oracle operand injection
  (`0.9300`), so the failure is the sum-only semantic decoder/result readout,
  not operand injection shape or learned-interface behavior.
- The April checkpoint metadata matched the current sum-only assumptions:
  `answer_format=sum`, `calculator_output_format=sum`,
  `calculator_read_position=operand_spans`,
  `calculator_read_span_width=2`, and
  `calculator_bottleneck_mode=answer_decoder`.

### Stage 0B Candidate Ladder

No candidate passed the strict gate
`oracle-at-eval >= 0.98`, full-enum best-result group matches true sum
`>= 0.98`, low injection-zero/forced-random controls, and semantic decoder
delta `0.0`.

Best checkpoint per branch:

| Branch | Best checkpoint | Oracle-at-eval | Best result=true | Injection-zero | Forced-random |
| --- | --- | ---: | ---: | ---: | ---: |
| tiny `operand_spans` dense | `step_00500_weights.pt` | `0.9325` | `0.9300` | `0.0050` | `0.0200` |
| tiny `operands` dense | `step_00500_weights.pt` | `0.9150` | `0.9300` | `0.0075` | `0.0175` |
| `n_embd=32`, `n_head=2`, `n_layer=2` | `step_01000_weights.pt` | `0.9275` | `0.9150` | `0.0050` | `0.0200` |
| `n_embd=32`, `n_head=2`, `n_layer=3` | `step_01500_weights.pt` | `0.9350` | `0.9250` | `0.0100` | `0.0125` |

## Decision

Stage 1 natural deterministic relaxed bridge training was not run. The Stage 0
decoder/wiring gate did not pass, and bridge training would be uninterpretable
under the task guardrails.

Interpretation label:

```text
sum_only_decoder_capacity_blocker
```

## Recommendation

Do not proceed to `operand_max=99` and do not treat this as a natural bridge
negative. The blocker is still the strict natural sum-only semantic decoder.
The next useful axis is a decoder/readout redesign for sum-only, not more
answer-only bridge training from a sub-0.98 gate.
