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
