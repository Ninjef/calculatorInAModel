import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.diagnose_calculator_protocol import (  # noqa: E402
    READ_SITE_INTERVENTIONS,
    diagnostic_rows,
    load_checkpoint,
    pick_device,
    summarize_rows,
    write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Checkpoint-first private-code protocol decoding diagnostics."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--operand-max", type=int, default=19)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def compact_distribution(values: list[Any], *, limit: int = 10) -> str:
    counts = Counter(values)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return json.dumps(dict(ordered[:limit]), sort_keys=True)


def exact_rate(rows: list[dict[str, Any]], predicate) -> float:
    if not rows:
        return 0.0
    return sum(int(predicate(row)) for row in rows) / len(rows)


def write_confusion_matrix(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    true_key: str,
    pred_key: str,
    true_max: int,
    pred_max: int,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true"] + [str(i) for i in range(pred_max + 1)])
        for true_value in range(true_max + 1):
            counts = Counter(
                int(row[pred_key])
                for row in rows
                if int(row[true_key]) == true_value and int(row[pred_key]) >= 0
            )
            writer.writerow(
                [true_value] + [counts.get(pred_value, 0) for pred_value in range(pred_max + 1)]
            )


def majority_mapping(
    rows: list[dict[str, Any]], *, pred_key: str, true_key: str
) -> dict[int, int]:
    grouped: dict[int, Counter[int]] = defaultdict(Counter)
    for row in rows:
        grouped[int(row[pred_key])][int(row[true_key])] += 1
    mapping: dict[int, int] = {}
    for pred_value, counts in grouped.items():
        mapping[pred_value] = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return mapping


def best_affine_mod_mapping(
    rows: list[dict[str, Any]],
    *,
    pred_key: str,
    true_key: str,
    modulus: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for scale in range(modulus):
        if math.gcd(scale, modulus) != 1:
            continue
        for offset in range(modulus):
            correct = sum(
                int(((scale * int(row[pred_key]) + offset) % modulus) == int(row[true_key]))
                for row in rows
            )
            candidates.append(
                {
                    "scale": scale,
                    "offset": offset,
                    "exact": correct / max(len(rows), 1),
                }
            )
    return max(candidates, key=lambda item: item["exact"])


def mapping_rows(mapping: dict[int, int]) -> list[dict[str, int]]:
    return [
        {"learned_class": learned_class, "mapped_true_value": true_value}
        for learned_class, true_value in sorted(mapping.items())
    ]


def add_mapped_fields(rows: list[dict[str, Any]], a_map: dict[int, int], b_map: dict[int, int]) -> None:
    for row in rows:
        mapped_a = a_map.get(int(row["a_pred"]), -1)
        mapped_b = b_map.get(int(row["b_pred"]), -1)
        row["mapped_a"] = mapped_a
        row["mapped_b"] = mapped_b
        row["mapped_result"] = mapped_a + mapped_b if mapped_a >= 0 and mapped_b >= 0 else -1
        row["mapped_operand_exact"] = mapped_a == int(row["true_a"]) and mapped_b == int(row["true_b"])
        row["mapped_result_exact"] = row["mapped_result"] == int(row["true_sum"])


def per_operand_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    output = []
    for value in sorted({int(row[key]) for row in rows}):
        subset = [row for row in rows if int(row[key]) == value]
        output.append(
            {
                key: value,
                "count": len(subset),
                "operand_exact": exact_rate(
                    subset,
                    lambda row: int(row["a_pred"]) == int(row["true_a"])
                    and int(row["b_pred"]) == int(row["true_b"]),
                ),
                "calculator_result_accuracy": exact_rate(
                    subset,
                    lambda row: int(row["calculator_result"]) == int(row["true_sum"]),
                ),
                "answer_exact": exact_rate(subset, lambda row: bool(row["correct"])),
                "mapped_operand_exact": exact_rate(
                    subset, lambda row: bool(row["mapped_operand_exact"])
                ),
                "mapped_result_exact": exact_rate(
                    subset, lambda row: bool(row["mapped_result_exact"])
                ),
            }
        )
    return output


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        "all": rows,
        "carry": [row for row in rows if int(row["true_sum"]) >= 10],
        "no_carry": [row for row in rows if int(row["true_sum"]) < 10],
        "large_operand": [
            row for row in rows if int(row["true_a"]) >= 10 or int(row["true_b"]) >= 10
        ],
        "small_operands": [
            row for row in rows if int(row["true_a"]) < 10 and int(row["true_b"]) < 10
        ],
        "symmetric": [row for row in rows if int(row["true_a"]) == int(row["true_b"])],
    }
    output = []
    for name, subset in groups.items():
        output.append(
            {
                "group": name,
                "count": len(subset),
                "operand_exact": exact_rate(
                    subset,
                    lambda row: int(row["a_pred"]) == int(row["true_a"])
                    and int(row["b_pred"]) == int(row["true_b"]),
                ),
                "calculator_result_accuracy": exact_rate(
                    subset,
                    lambda row: int(row["calculator_result"]) == int(row["true_sum"]),
                ),
                "answer_exact": exact_rate(subset, lambda row: bool(row["correct"])),
                "mapped_operand_exact": exact_rate(
                    subset, lambda row: bool(row["mapped_operand_exact"])
                ),
                "mapped_result_exact": exact_rate(
                    subset, lambda row: bool(row["mapped_result_exact"])
                ),
            }
        )
    return output


def code_stability_rows(rows: list[dict[str, Any]], *, pred_key: str, true_key: str) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row[pred_key])].append(row)
    output = []
    for learned_class, subset in sorted(grouped.items()):
        true_values = [int(row[true_key]) for row in subset]
        mode_value, mode_count = sorted(
            Counter(true_values).items(), key=lambda item: (-item[1], item[0])
        )[0]
        output.append(
            {
                "learned_class": learned_class,
                "count": len(subset),
                "mode_true_value": mode_value,
                "mode_fraction": mode_count / len(subset),
                "true_value_distribution": compact_distribution(true_values, limit=20),
            }
        )
    return output


def intervention_summary(intervention_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    normal_by_sample = {
        int(row["sample"]): row for row in intervention_rows.get("normal", [])
    }
    output = []
    for condition, rows in intervention_rows.items():
        changed = 0
        for row in rows:
            normal = normal_by_sample.get(int(row["sample"]))
            if normal is not None and str(row["prediction"]) != str(normal["prediction"]):
                changed += 1
        output.append(
            {
                "condition": condition,
                "samples": len(rows),
                "answer_exact": exact_rate(rows, lambda row: bool(row["correct"])),
                "prediction_changed_vs_normal": changed / max(len(rows), 1),
                "operand_exact": exact_rate(
                    rows,
                    lambda row: int(row["a_pred"]) == int(row["true_a"])
                    and int(row["b_pred"]) == int(row["true_b"]),
                ),
                "calculator_result_accuracy": exact_rate(
                    rows,
                    lambda row: int(row["calculator_result"]) == int(row["true_sum"]),
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    model, train_config = load_checkpoint(args.checkpoint, device=device, injection_scale=None)
    sample_specs = [
        {"sample": a * (args.operand_max + 1) + b, "true_a": a, "true_b": b}
        for a in range(args.operand_max + 1)
        for b in range(args.operand_max + 1)
    ]
    normal_rows = diagnostic_rows(
        model,
        num_digits=args.digits,
        operand_max=args.operand_max,
        samples=len(sample_specs),
        seed=args.seed,
        device=device,
        oracle=False,
        calculator_result_override="add",
        sample_specs=sample_specs,
    )

    a_map = majority_mapping(normal_rows, pred_key="a_pred", true_key="true_a")
    b_map = majority_mapping(normal_rows, pred_key="b_pred", true_key="true_b")
    result_map = majority_mapping(
        normal_rows, pred_key="calculator_result", true_key="true_sum"
    )
    add_mapped_fields(normal_rows, a_map, b_map)

    write_rows(args.output_dir / "all_pair_rows.csv", normal_rows)
    write_rows(args.output_dir / "a_majority_mapping.csv", mapping_rows(a_map))
    write_rows(args.output_dir / "b_majority_mapping.csv", mapping_rows(b_map))
    write_rows(args.output_dir / "result_majority_mapping.csv", mapping_rows(result_map))
    write_rows(args.output_dir / "per_true_a_summary.csv", per_operand_rows(normal_rows, "true_a"))
    write_rows(args.output_dir / "per_true_b_summary.csv", per_operand_rows(normal_rows, "true_b"))
    write_rows(args.output_dir / "per_true_sum_summary.csv", per_operand_rows(normal_rows, "true_sum"))
    write_rows(args.output_dir / "group_summary.csv", group_summary(normal_rows))
    write_rows(
        args.output_dir / "result_code_stability.csv",
        code_stability_rows(normal_rows, pred_key="calculator_result", true_key="true_sum"),
    )
    write_rows(
        args.output_dir / "a_code_stability.csv",
        code_stability_rows(normal_rows, pred_key="a_pred", true_key="true_a"),
    )
    write_rows(
        args.output_dir / "b_code_stability.csv",
        code_stability_rows(normal_rows, pred_key="b_pred", true_key="true_b"),
    )
    write_confusion_matrix(
        args.output_dir / "confusion_true_a_vs_learned_a.csv",
        normal_rows,
        true_key="true_a",
        pred_key="a_pred",
        true_max=args.operand_max,
        pred_max=args.operand_max,
    )
    write_confusion_matrix(
        args.output_dir / "confusion_true_b_vs_learned_b.csv",
        normal_rows,
        true_key="true_b",
        pred_key="b_pred",
        true_max=args.operand_max,
        pred_max=args.operand_max,
    )
    write_confusion_matrix(
        args.output_dir / "confusion_true_sum_vs_learned_result.csv",
        normal_rows,
        true_key="true_sum",
        pred_key="calculator_result",
        true_max=args.operand_max * 2,
        pred_max=args.operand_max * 2,
    )

    wrong_operands_right_answer = [
        row
        for row in normal_rows
        if bool(row["correct"])
        and not (
            int(row["a_pred"]) == int(row["true_a"])
            and int(row["b_pred"]) == int(row["true_b"])
        )
    ]
    right_operands_wrong_answer = [
        row
        for row in normal_rows
        if not bool(row["correct"])
        and int(row["a_pred"]) == int(row["true_a"])
        and int(row["b_pred"]) == int(row["true_b"])
    ]
    write_rows(
        args.output_dir / "examples_wrong_operands_right_answer.csv",
        wrong_operands_right_answer[:25],
    )
    write_rows(
        args.output_dir / "examples_right_operands_wrong_answer.csv",
        right_operands_wrong_answer[:25],
    )

    intervention_rows = {"normal": normal_rows}
    for intervention in sorted(READ_SITE_INTERVENTIONS):
        intervention_rows[intervention] = diagnostic_rows(
            model,
            num_digits=args.digits,
            operand_max=args.operand_max,
            samples=len(sample_specs),
            seed=args.seed,
            device=device,
            oracle=False,
            calculator_result_override="add",
            calculator_read_intervention=intervention,
            sample_specs=sample_specs,
        )
        write_rows(
            args.output_dir / f"intervention_{intervention}.csv",
            intervention_rows[intervention],
        )
    write_rows(
        args.output_dir / "read_vector_intervention_summary.csv",
        intervention_summary(intervention_rows),
    )

    summary = summarize_rows(normal_rows)
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "device": device,
            "digits": args.digits,
            "operand_max": args.operand_max,
            "pairs": len(normal_rows),
            "train_config_seed": train_config.get("seed"),
            "train_config_run_name": train_config.get("run_name"),
            "mapped_operand_exact_match": exact_rate(
                normal_rows, lambda row: bool(row["mapped_operand_exact"])
            ),
            "mapped_calculator_result_accuracy": exact_rate(
                normal_rows, lambda row: bool(row["mapped_result_exact"])
            ),
            "a_best_affine_mod_mapping": best_affine_mod_mapping(
                normal_rows, pred_key="a_pred", true_key="true_a", modulus=args.operand_max + 1
            ),
            "b_best_affine_mod_mapping": best_affine_mod_mapping(
                normal_rows, pred_key="b_pred", true_key="true_b", modulus=args.operand_max + 1
            ),
            "result_majority_mapped_accuracy": exact_rate(
                normal_rows,
                lambda row: result_map.get(int(row["calculator_result"]), -1)
                == int(row["true_sum"]),
            ),
            "wrong_operands_right_answer_count": len(wrong_operands_right_answer),
            "right_operands_wrong_answer_count": len(right_operands_wrong_answer),
        }
    )
    (args.output_dir / "private_protocol_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
