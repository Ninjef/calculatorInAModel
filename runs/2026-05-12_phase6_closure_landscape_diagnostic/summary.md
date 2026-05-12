# Phase 6 Closure Landscape Diagnostic

## Existing Evidence

- Identifiable deterministic Concrete replicated across effective seeds `2`, `4`, and `5`, then retained with all teacher/local/expected/relaxed objectives inactive.
- Natural sum-only product decoder passed the oracle/readout gate, but the learned deterministic Concrete bridge selected only about `0.11` learned-result-best fraction with a learned-result minus best-result gap around `5.57`.

## Paired Full-Enum Landscape

| setting | best pair=true | tie-aware true best | best result=true | true pair rank | effective pairs | effective results | same-true-sum near-best | true pair prob | true result prob | top1/top3/top5 mass | learned result acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identifiable sum_left_operand | 1 | 1 | 1 | 1 | 1.08394 | 1.04621 | 1 | 0.987884 | 0.993087 | 0.987884/0.996166/0.998356 | 0.0325 |
| natural sum-only product | 0.0975 | 1 | 1 | 1 | 13.3573 | 1.00105 | 13.35 | 0.0974936 | 0.99994 | 0.0974936/0.277481/0.437469 | 0.03 |

## One-Step Relaxed Gradient

| setting | true pair prob delta | true result prob delta | best result prob delta | hard pair delta | hard calc/result delta | grad cos vs pair CE | grad cos vs result group | input/upstream/semantic delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identifiable sum_left_operand | 8.54054e-06 | 2.88151e-05 | 2.88151e-05 | 0.00333333 | 0.00333333 | 0.0512509 | 0.0567104 | 1.08929/0/0 |
| natural sum-only product | 1.31081e-05 | 7.72129e-06 | 7.72129e-06 | 0.0075 | 0.0141667 | 0.131723 | 0.21877 | 1.08924/0/0 |

## Closure Decision

Phase 6 should close. The deterministic Concrete positive is real, replicated, and retained in the identifiable setting, but the natural sum-only negative is best explained as an underidentified/diffuse result-action landscape for independent operand heads rather than a broken decoder or broken relaxation implementation.

Recommended Phase 7 first task: test a result-space interface parameterization or structured joint-pair objective for natural `0..19` addition before any `operand_max=99` scaling.
