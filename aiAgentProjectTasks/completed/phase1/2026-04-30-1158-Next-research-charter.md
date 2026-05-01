# Next Research Charter: Probe-Guided Calculator Protocol

## Context

The original vision was to put a real calculator inside a small transformer and see whether the network could learn to feed it useful operands from its own residual stream. The revised vision correctly narrowed the central question: the hard calculator is not the bottleneck; the research object is the internal protocol between neural activations and the discrete tool.

The completed work has made good progress:

- Model A establishes ordinary transformer baselines on synthetic addition.
- Model B verifies that the hook can be wired in without obviously breaking the model.
- Model C proves that calculator injection is technically possible, but learned operands usually fail under downstream answer loss.
- Oracle-operand training works, including in narrow 2-digit regimes, so result injection and downstream use are not fundamentally broken.
- Diagnostics show answer accuracy can hide bypasses or non-human private codes.
- Forced-result sweeps show some 1-digit checkpoints causally use the calculator path, but as a coarse private answer code rather than as true addition.
- Probes now give the most important new signal: operand information is often linearly available at operand-token positions, but weak at the current `=` read position.

The current branch is `experiment/non-bottleneck-protocol-experiments`. There are already local task-doc moves into `aiAgentProjectTasks/completed/`; do not undo those when continuing work.

## External Research Snapshot

The project is adjacent to, but not identical with, modern tool-use LMs. Toolformer showed that LMs can learn API use, including calculator calls, but it did so by creating and filtering textual tool-call traces rather than relying on a hidden continuous-to-discrete protocol to emerge from final answer loss alone:

- https://arxiv.org/abs/2302.04761

ART, PAL, Program-of-Thoughts, and the broader tool-use literature similarly favor explicit program/tool-call surfaces where the model emits inspectable calls and receives results:

- https://arxiv.org/abs/2303.09014
- https://link.springer.com/article/10.1007/s44336-025-00024-x

Recent tool-learning surveys emphasize supervised fine-tuning, RL, execution feedback, step-level rewards, and tool-call validity as central training signals. That matters here because the sandbox keeps showing that final answer loss is too weak and too confounded to make the intended latent calculator language emerge reliably.

Discrete-latent optimization literature also fits the empirical story. Straight-through and Gumbel-style estimators are useful but biased or variance-sensitive, and newer variants mostly try to reduce estimator pathologies rather than eliminate the need for a well-shaped objective:

- https://openreview.net/forum?id=Mk6PZtgAgfq
- https://proceedings.mlr.press/v162/fan22a.html

Arithmetic-specific LM work is also a caution: small transformers can learn arithmetic-like behavior internally, sometimes as encoding-regression-decoding rather than symbolic digit manipulation. That means raw answer accuracy is an especially weak success metric for this project:

- https://www.sciencedirect.com/science/article/pii/S089360802400474X

## Strategy Update

The vision should change slightly, not be abandoned.

Old implicit bet:

> A transformer may discover a human-readable latent calculator protocol through answer loss if the hard tool is present in the forward pass.

Updated bet:

> A transformer may be able to use an internal calculator, but the interface must be made compatible with where the transformer naturally represents operands, and the experiment must measure protocol learning directly rather than infer it from answer accuracy.

This is still a strong and interesting thesis. It is also more precise. If it fails after the next phase, the failure will be much more meaningful.

## Best Next Focus

Implement the narrow read-position interface recommended by the latest work history:

- Preserve current behavior as `calculator_read_position=eq`.
- Add `calculator_read_position=operands`.
- In operand-read mode, read A logits from the final A digit position and B logits from the final B digit position.
- Keep calculator result injection at `=`.
- Keep Model A/B/C/oracle controls unchanged.

This directly tests the strongest current hypothesis: the calculator is being asked to read both operands from a position where the model does not naturally expose them. The probe evidence says the operands are available elsewhere; the next experiment should let the interface meet the representation halfway.

## Evaluation Standard

Do not declare success from answer exact match alone.

For every candidate regime, report at least:

- Model A exact match.
- Model B exact match and whether it remains a benign control.
- Model C exact match.
- Oracle Model C exact match.
- True operand exact match at the calculator read sites.
- Calculator result accuracy.
- Injection-zero, forced-zero, forced-random, and oracle-at-eval counterfactuals.
- Probe accuracy at A token, B token, and `=` positions.

Success means the calculator path helps answer accuracy and the model uses a stable, interpretable operand protocol, or at least a stable protocol whose semantics can be causally decoded.

Failure means more than "Model C did not beat Model A." A meaningful negative result would be:

- oracle works;
- probes show operand information is available at the read sites;
- B remains healthy;
- multiple optimization strategies fail to learn or preserve a protocol;
- counterfactuals show no calculator dependence, or dependence only on uninterpretable/private codes that do not scale.

## Suggested Experiment Ladder

First pass:

- Implement operand-position read mode.
- Re-run `operand_max=19` with `2L/1H/16d/mlp1`.
- Include A/B/C/oracle controls.
- Use the existing snapshot/counterfactual machinery.

If promising:

- Move to `operand_max=49`, then `99`.
- Add 3 seeds only after the first rung shows a credible calculator-use signature.
- Revisit light auxiliary operand supervision as a diagnostic, not as the headline method.

If still failing:

- Try a two-stage protocol curriculum: supervised operand head to convergence, then decay or remove aux while measuring whether the protocol is retained.
- Try a soft/straight-through Gumbel or low-variance ST variant only after the read-position issue is resolved.
- Run the multi-sample action-loss diagnostic proposed after the REINFORCE attempt to measure whether different operand actions actually change downstream loss before training policy-gradient variants further.

## Research Guardrails

- Keep the task synthetic and tiny until protocol learning is demonstrated.
- Avoid GPT-2, HuggingFace abstractions, natural language word problems, or broad tool sets for now.
- Do not optimize for one-off high answer accuracy.
- Treat private codes as interesting evidence, not final success.
- Keep run artifacts local unless explicitly asked to commit them.
- Prefer exact run paths in writeups because this repo already contains many similar exploratory runs.

## North Star

The next phase should answer one question cleanly:

> If the calculator reads from positions where operands are actually represented, can the model learn a reliable internal tool protocol that survives counterfactual testing?

If yes, the project graduates from "can the hook work?" to "what training signal best teaches tool protocols?" If no, the project has a much sharper negative result: even when operand information is available and oracle use works, hidden discrete tool protocols do not emerge reliably from this setup without explicit protocol supervision.
