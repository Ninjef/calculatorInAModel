import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


WINDOW_STEPS = [125, 150, 175, 200, 225]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_csv_by_step(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        return {int(row["step"]): row for row in csv.DictReader(handle)}


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def condition_from_metrics(metrics: dict[str, Any]) -> str:
    if int(metrics.get("adaptive_interface_loss_decay_steps", 0)) > 0:
        return "decayed"
    return "constant"


def checkpoint_for_step(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoint_snapshots" / f"step_{step:05d}_weights.pt"


def trajectory_rows(run_dir: Path) -> list[dict[str, Any]]:
    metrics = load_json(run_dir / "metrics.json")
    config_path = run_dir / "config.json"
    config = load_json(config_path) if config_path.exists() else {}
    stored_seed = metrics.get("seed", config.get("seed"))
    requested_seed = None
    if stored_seed is not None:
        requested_seed = int(stored_seed) - int(metrics.get("num_digits", 2))
    snapshots = load_csv_by_step(run_dir / "diagnostic_snapshots.csv")
    curve = load_csv_by_step(run_dir / "training_curve.csv")
    rows: list[dict[str, Any]] = []
    for step in WINDOW_STEPS:
        if step not in snapshots or step not in curve:
            continue
        snapshot = snapshots[step]
        curve_row = curve[step]
        rows.append(
            {
                "run_dir": str(run_dir),
                "seed": stored_seed,
                "requested_seed": requested_seed,
                "condition": condition_from_metrics(metrics),
                "step": step,
                "checkpoint": str(checkpoint_for_step(run_dir, step)),
                "aux_operand_loss_weight": as_float(
                    curve_row, "aux_operand_loss_weight"
                ),
                "adaptive_interface_loss_weight": as_float(
                    curve_row, "adaptive_interface_loss_weight"
                ),
                "normal_exact_match": as_float(snapshot, "normal_exact_match"),
                "injection_zero_exact_match": as_float(
                    snapshot, "injection_zero_exact_match"
                ),
                "forced_zero_exact_match": as_float(
                    snapshot, "forced_zero_exact_match"
                ),
                "forced_random_exact_match": as_float(
                    snapshot, "forced_random_exact_match"
                ),
                "oracle_exact_match": as_float(snapshot, "oracle_exact_match"),
                "pair_exact_match": as_float(snapshot, "pair_exact_match"),
                "calculator_result_accuracy": as_float(
                    snapshot, "calculator_result_accuracy"
                ),
                "mean_pair_entropy": as_float(snapshot, "mean_pair_entropy"),
                "pair_logit_effective_pairs": as_float(
                    curve_row, "action_loss_full_enum_pair_logit_effective_pairs"
                ),
                "built_in_eval_exact": float(metrics.get("exact_match", 0.0)),
                "final_aux_operand_loss_weight": float(
                    metrics.get("final_aux_operand_loss_weight", 0.0)
                ),
                "final_adaptive_interface_loss_weight": float(
                    metrics.get("final_adaptive_interface_loss_weight", 0.0)
                ),
                "final_input_proj_anchor_weight": float(
                    metrics.get("final_input_proj_anchor_weight", 0.0)
                ),
                "trainable_parameter_groups": [
                    group["name"]
                    for group in metrics.get("trainable_parameter_groups", [])
                ],
            }
        )
    return rows


def selectable(row: dict[str, Any]) -> bool:
    if row["aux_operand_loss_weight"] != 0.0:
        return False
    if row["condition"] == "decayed" and row["adaptive_interface_loss_weight"] != 0.0:
        return False
    return True


def select_checkpoint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if selectable(row)]
    if not candidates:
        raise ValueError("no selectable aux-zero checkpoints")
    return sorted(
        candidates,
        key=lambda row: (
            -row["pair_exact_match"],
            row["injection_zero_exact_match"],
            row["forced_random_exact_match"],
            -row["oracle_exact_match"],
            -row["calculator_result_accuracy"],
            row["step"],
        ),
    )[0]


def find_first_json(base: Path, patterns: list[str]) -> dict[str, Any] | None:
    for pattern in patterns:
        matches = sorted(base.glob(pattern))
        if matches:
            return load_json(matches[0])
    return None


def add_diagnostic_summaries(selected: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(selected["run_dir"])
    step = int(selected["step"])
    enriched = dict(selected)
    canonical = find_first_json(
        run_dir,
        [
            f"step{step}_canonical_causal_diagnostics/diagnostic_summary.json",
            f"selected_step{step}_canonical_causal_diagnostics/diagnostic_summary.json",
        ],
    )
    full_enum = find_first_json(
        run_dir,
        [
            f"step{step}_full_enum_action_loss/**/full_enum_summary.json",
            f"selected_step{step}_full_enum_action_loss/**/full_enum_summary.json",
        ],
    )
    private = find_first_json(
        run_dir,
        [
            f"step{step}_private_protocol_diagnostics/private_protocol_summary.json",
            f"selected_step{step}_private_protocol_diagnostics/private_protocol_summary.json",
        ],
    )
    if canonical is not None:
        counterfactuals = {
            row["condition"]: row["exact_match"]
            for row in canonical.get("counterfactual_exact_match", [])
        }
        enriched["canonical"] = {
            "normal_exact_match": canonical.get("exact_match"),
            "injection_zero_exact_match": counterfactuals.get("injection_zero"),
            "forced_zero_exact_match": counterfactuals.get("forced_zero"),
            "forced_random_exact_match": counterfactuals.get("forced_random"),
            "oracle_exact_match": counterfactuals.get("oracle_at_eval"),
            "pair_exact_match": canonical.get("pair_exact_match"),
            "calculator_result_accuracy": canonical.get(
                "calculator_result_accuracy"
            ),
            "classification": canonical.get("classification"),
            "bottleneck_mode": canonical.get("calculator_bottleneck_mode"),
        }
    if full_enum is not None:
        enriched["full_enum"] = {
            "mean_best_full_enum_nll": full_enum.get("mean_best_full_enum_nll"),
            "mean_learned_nll": full_enum.get("mean_learned_nll"),
            "mean_true_nll": full_enum.get("mean_true_nll"),
            "mean_learned_minus_true_gap": full_enum.get(
                "mean_learned_minus_true_gap"
            ),
            "mean_learned_minus_best_gap": full_enum.get(
                "mean_learned_minus_best_gap"
            ),
            "learned_best_fraction": full_enum.get("learned_best_fraction"),
            "learned_within_1e-3_best_fraction": full_enum.get(
                "learned_within_1e-3_best_fraction"
            ),
            "mean_pair_logit_effective_pair_count": full_enum.get(
                "mean_pair_logit_effective_pair_count"
            ),
        }
    if private is not None:
        enriched["private"] = {
            "all_pair_answer_exact": private.get("exact_match"),
            "operand_exact_match": private.get("operand_exact_match"),
            "pair_exact_match": private.get("pair_exact_match"),
            "calculator_result_accuracy": private.get(
                "calculator_result_accuracy"
            ),
            "mapped_operand_exact_match": private.get(
                "mapped_operand_exact_match"
            ),
            "mapped_calculator_result_accuracy": private.get(
                "mapped_calculator_result_accuracy"
            ),
        }
    return enriched


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif isinstance(value, list):
                values.append(", ".join(str(item) for item in value))
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider] + body)


def aggregate_rows(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for condition in ["constant", "decayed"]:
        rows = [row for row in selected if row["condition"] == condition]
        if not rows:
            continue
        output.append(
            {
                "condition": condition,
                "runs": len(rows),
                "mean_selected_pair_exact": mean(
                    row["pair_exact_match"] for row in rows
                ),
                "mean_selected_calc_result_acc": mean(
                    row["calculator_result_accuracy"] for row in rows
                ),
                "mean_selected_normal_exact": mean(
                    row["normal_exact_match"] for row in rows
                ),
                "mean_selected_injection_zero": mean(
                    row["injection_zero_exact_match"] for row in rows
                ),
                "mean_selected_forced_random": mean(
                    row["forced_random_exact_match"] for row in rows
                ),
                "mean_selected_oracle": mean(row["oracle_exact_match"] for row in rows),
            }
        )
    return output


def summarize_runs(run_dirs: list[Path]) -> dict[str, Any]:
    trajectories = []
    selected = []
    for run_dir in run_dirs:
        rows = trajectory_rows(run_dir)
        trajectories.extend(rows)
        selected.append(add_diagnostic_summaries(select_checkpoint(rows)))
    return {
        "selected": selected,
        "trajectory": trajectories,
        "aggregate": aggregate_rows(selected),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize matched retained checkpoints for the phase-3 ladder."
    )
    parser.add_argument("run_dir", type=Path, nargs="+")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_runs(args.run_dir)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    selected_columns = [
        ("Arg Seed", "requested_seed"),
        ("Seed", "seed"),
        ("Condition", "condition"),
        ("Step", "step"),
        ("Iface", "adaptive_interface_loss_weight"),
        ("Aux", "aux_operand_loss_weight"),
        ("Normal", "normal_exact_match"),
        ("Inj0", "injection_zero_exact_match"),
        ("Rand", "forced_random_exact_match"),
        ("Oracle", "oracle_exact_match"),
        ("Pair", "pair_exact_match"),
        ("Calc", "calculator_result_accuracy"),
    ]
    aggregate_columns = [
        ("Condition", "condition"),
        ("Runs", "runs"),
        ("Mean Pair", "mean_selected_pair_exact"),
        ("Mean Calc", "mean_selected_calc_result_acc"),
        ("Mean Normal", "mean_selected_normal_exact"),
        ("Mean Inj0", "mean_selected_injection_zero"),
        ("Mean Rand", "mean_selected_forced_random"),
        ("Mean Oracle", "mean_selected_oracle"),
    ]
    markdown = "\n\n".join(
        [
            "## Selected Checkpoints",
            markdown_table(summary["selected"], selected_columns),
            "## Aggregate",
            markdown_table(summary["aggregate"], aggregate_columns),
        ]
    )
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
