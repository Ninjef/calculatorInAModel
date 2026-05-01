# Phase 2 Experiment Fact Sheet

## Adaptive interface under strict answer-decoder bottleneck

- Date: 2026-05-01.
- Task: `aiAgentProjectTasks/2026-05-01-phase-2-first-task-Adaptive-calculator-interface-bottleneck.md`.
- Work history: `aiAgentWorkHistory/phase2/2026-05-01-adaptive-calculator-interface-bottleneck.md`.
- Starting decoder checkpoint: `runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt`.
- Adaptive run: `runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2`.
- Config: `digits=2`, `operand_max=19`, operand vocab `20`, `2L/1H/16d/mlp1`, hook after layer `1`, `calculator_read_position=operands`, `calculator_bottleneck_mode=answer_decoder`, `calculator_estimator=adaptive_interface`.
- Frozen by default: `calculator_hook.output_proj`, `answer_offset_emb`, and `answer_decoder`.
- Upstream encoder trainable: yes.
- `calculator_hook.input_proj` moved substantially from the oracle checkpoint: weight L2 delta `13.1145`, bias L2 delta `2.6793`.
- Frozen decoder components stayed unchanged: output projection, answer offset embedding, and answer decoder all had L2 delta `0.0000`.

Headline result:

- Eval exact match: `5/512 = 0.0098`.
- Built-in 128-sample diagnostic exact match: `0.0156`.
- Operand exact match: `0.0000`.
- Calculator result accuracy: `0.0156`.
- Adaptive target result accuracy: `0.9375`.
- Learned-target result agreement: `0.0156`.
- Adaptive target operand exact match: `0.1172`.
- Final adaptive interface CE loss: `193.0297`.

Counterfactuals:

- Injection-zero exact: `0.0078`.
- Forced-zero exact: `0.0000`.
- Forced-random exact: `0.0156`.
- Oracle-at-eval exact: `0.9531`.

Track 3:

- Output: `runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2/track3_diagnostics`.
- Classification: `calculator_ignored_or_bypassed`.
- Bottleneck label: `strict_bottleneck_unvalidated` for this checkpoint.
- Learned calculator result collapsed to class `5` for all 64 Track 3 prompts.
- True-sum class was best forced class on `0.9219` of prompts.
- Learned class was best forced class on `0.0156` of prompts.

Track 4:

- Output: `runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2/track4_action_loss`.
- Mean true-action NLL: `0.1181`.
- Mean random-action NLL: `9.9763`.
- Mean shuffled-action NLL: `10.4317`.
- Mean learned-forced NLL: `14.4446`.
- Random minus true gap: `9.8583`.
- Shuffled minus true gap: `10.3136`.
- Learned minus true gap: `14.3265`.
- True best fraction: `0.9844`.
- Learned best fraction: `0.0000`.

Conclusion:

- Useful negative result. The strict oracle decoder and action-loss landscape remain strong: true calculator actions are clearly preferred and oracle-at-eval recovers high answer accuracy.
- The adaptive moving interface did not learn that protocol. It changed substantially but collapsed to an overconfident constant calculator result, with chance answer accuracy and zero operand exact match.
- Recommendation: no-go on this exact adaptive-interface variant. Next variants should stabilize interface optimization before scaling, for example lower LR/loss weight, entropy regularization, target smoothing, upstream-frozen diagnostic training, or a small true-operand aux stabilizer while preserving the strict answer-decoder gates.

## Stabilized adaptive-interface objective ladder

- Date: 2026-05-01.
- Task: `aiAgentProjectTasks/completed/phase2/2026-05-01-phase-2-second-task-Stabilize-adaptive-interface-objective.md`.
- Work history: `aiAgentWorkHistory/phase2/2026-05-01-stabilize-adaptive-interface-objective.md`.
- Starting decoder checkpoint: `runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt`.
- Shared config: `digits=2`, `operand_max=19`, operand vocab `20`, `2L/1H/16d/mlp1`, hook after layer `1`, `calculator_read_position=operands`, `calculator_bottleneck_mode=answer_decoder`, `calculator_estimator=adaptive_interface`.
- Code changes: separate adaptive optimizer LRs (`--input-proj-lr`, `--upstream-lr`), soft result-mass objective (`--adaptive-interface-target-mode soft_result`), and adaptive entropy bonus (`--adaptive-interface-entropy-weight`).
- Frozen semantic components stayed frozen in every run: `calculator_hook.output_proj`, `answer_offset_emb`, and `answer_decoder` all had L2 delta `0.0000`.

Run comparison:

| Variant | Run | Eval exact | Operand exact | Calc result acc | Target acc | Learned-target agreement | Final CE | Track 3 | Track 4 learned-true gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Frozen upstream | `runs/2026-05-01_080805_707417_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2` | `0.1016` | `0.0234` | `0.1016` | `0.9375` | `0.0625` | `2.8654` | `causally_useful_opaque_private_code` | `9.0455` |
| Lower LR | `runs/2026-05-01_080805_707417_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2` | `0.0664` | `0.0156` | `0.0469` | `0.9375` | `0.0703` | `2.2887` | `causally_useful_opaque_private_code` | `10.4115` |
| Soft result | `runs/2026-05-01_080805_707417_model-c-op0-19-adaptive_interface-soft_result-answer_decoder/model-c-2digit-seed2` | `0.0059` | `0.0000` | `0.0156` | `0.9375` | `0.0078` | `226.3104` | `calculator_ignored_or_bypassed` | `8.4456` |
| Lower LR + entropy `0.001` | `runs/2026-05-01_081215_279041_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-ient0.001-answer_decoder/model-c-2digit-seed2` | `0.0410` | `0.0078` | `0.0625` | `0.9375` | `0.0625` | `2.3270` | `causally_useful_opaque_private_code` | `10.2226` |
| Lower LR + entropy `0.003` | `runs/2026-05-01_081215_279020_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-ient0.003-answer_decoder/model-c-2digit-seed2` | `0.0254` | `0.0000` | `0.0469` | `0.9375` | `0.0234` | `2.3133` | `calculator_ignored_or_bypassed` | `9.2909` |
| Lower LR + entropy `0.01` | `runs/2026-05-01_081215_283303_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-ient0.01-answer_decoder/model-c-2digit-seed2` | `0.0391` | `0.0000` | `0.0469` | `0.9375` | `0.0859` | `2.3345` | `calculator_ignored_or_bypassed` | `6.9002` |

Counterfactual and action-loss notes:

- Built-in counterfactuals were identical across the six runs: injection-zero `0.0078`, forced-zero `0.0000`, forced-random `0.0156`, oracle-at-eval `0.9531`.
- Track 3 oracle-at-eval was `0.90625` for every run; true-sum forced result class was best on `0.921875` of prompts for every run.
- Track 4 true-action NLL stayed `0.1181`, random-action NLL `9.9763`, shuffled-action NLL `10.4317`; learned actions were never best (`learned_best_fraction=0.0`) in any run.

Conclusion:

- Useful negative result. Frozen-upstream training improved normal/calculator-result accuracy over the failed baseline, so encoder/interface co-adaptation is likely part of the instability.
- Lower LR reduced `input_proj` movement and held the adaptive CE down, but still learned a constant calculator result under Track 3.
- Soft-result target alone did not remove collapse; it became overconfident and worse than the hard-pair objective.
- Entropy helped keep distributions less sharp but did not produce stable learned calculator actions. High entropy is not success: operand exact remained near zero and Track 4 learned-best stayed `0.0`.
- Recommendation: no-go on adaptive-interface success. Next variants should keep the strict decoder and consider staged/slow interface training, target smoothing or clipping, or a small true-operand auxiliary stabilizer as a diagnostic.
