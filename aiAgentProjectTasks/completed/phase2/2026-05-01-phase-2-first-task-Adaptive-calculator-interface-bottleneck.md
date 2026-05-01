# Adaptive Calculator Interface Under Strict Bottleneck

## Mission

Train the calculator's coded input interface as a moving, learnable adapter under the validated strict `answer_decoder` bottleneck. The calculator operation remains fixed: `(a, b) -> a + b`. The research object is the changing interface from residual activations to interpreted operands:

```text
h -> calculator_hook.input_proj(h) -> operand logits -> hard operands -> calculator
```

This task should test whether counterfactual downstream loss can teach the interface how to reinterpret encoder activations, while preserving the guarantee that the network cannot route around the calculator through answer-token residual context.

## Required Design

- Use `calculator_bottleneck_mode=answer_decoder` for all headline runs.
- Add or use `calculator_estimator=adaptive_interface`.
- Start from an oracle-trained strict-bottleneck Model C checkpoint as the semantic downstream decoder.
- Freeze `calculator_hook.output_proj`, `answer_offset_emb`, and `answer_decoder` by default.
- Keep `calculator_hook.input_proj` trainable; this is the moving coded interface.
- Keep upstream transformer layers trainable by default so the encoder and interface can co-adapt.
- For each batch, compute counterfactual answer loss under every forced calculator result class.
- Select the target result class `r*` that minimizes downstream answer loss.
- Convert `r*` into operand targets `(a*, b*)` satisfying `a* + b* = r*`, tie-broken by the current interface probabilities.
- Train with the adaptive interface CE loss on `(a*, b*)`, allowing gradients to update the interface and flow into upstream layers.

## Starting Regime

Use the strict validated baseline first:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
calculator_read_position=operands
calculator_bottleneck_mode=answer_decoder
```

Do not scale beyond `operand_max=19` until this produces a credible calculator-use signature.

## Validation Gates

- Model B/off fails under the strict bottleneck.
- Oracle Model C succeeds under the strict bottleneck.
- Oracle Model C collapses under injection-zero, forced-zero, and forced-random calculator outputs.
- Adaptive-interface Model C improves substantially over strict learned STE.
- `calculator_hook.input_proj` changes during training.
- Operand exact match and calculator result accuracy improve over strict STE.
- Injection-zero, forced-zero, and forced-random counterfactuals collapse for any claimed positive checkpoint.
- Track 3 classifies the checkpoint as intended true-operand calculator use or a clearly decodable calculator protocol.
- Track 4 shows learned/true operand actions beat random and shuffled actions.

## Required Reporting

Write a work history with:

- exact commands and run paths;
- semantic decoder checkpoint path;
- whether semantic decoder and upstream encoder were frozen;
- answer exact match, operand exact match, and calculator result accuracy;
- adaptive target result accuracy and learned-target agreement;
- injection-zero, forced-zero, forced-random, and oracle-at-eval counterfactuals;
- Track 3 classification and bottleneck label;
- Track 4 action-loss gaps;
- a go/no-go recommendation for further adaptive-interface variants.

## Success Criteria

A positive result is calculator-dependent answer accuracy with improved operand/result protocol metrics relative to strict STE. A useful negative result is also acceptable if the adaptive interface loss decreases and `input_proj` changes, but the checkpoint still fails calculator-dependent answer accuracy. That would show the moving-interface idea was implemented but insufficient in this setup.
