import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent


FEATURE_COLUMNS = [
    "normal_exact_match",
    "injection_zero_exact_match",
    "oracle_exact_match",
    "forced_random_exact_match",
    "calculator_result_accuracy",
    "normal_minus_zero",
    "oracle_minus_normal",
    "normal_minus_forced_random",
]


@dataclass(frozen=True)
class HandoffRun:
    run_dir: Path
    metrics_path: Path
    snapshots_path: Path
    family: str
    candidate: str
    target_score: float
    final_score: float
    max_snapshot_step: int
    features: dict[str, float]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_snapshot_rows(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = int(row["step"])
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if key in {"step", "learned_result_distribution"}:
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    continue
            rows[step] = parsed
    return rows


def metric_bool(metrics: dict[str, Any], key: str) -> bool:
    return bool(metrics.get(key, False))


def metric_float(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    return float(default if value is None else value)


def is_initial_frozen_handoff(metrics_path: Path, metrics: dict[str, Any]) -> bool:
    text = metrics_path.as_posix()
    blocked_tokens = [
        "continue",
        "continued",
        "continuation",
        "readout",
        "adapted",
        "unfreeze",
        "policy_anchor",
        "policy_backbone",
    ]
    if any(token in text for token in blocked_tokens):
        return False
    if "freeze_policy" not in text and "handoff" not in text:
        return False
    if metrics.get("calculator_bottleneck_mode") != "none":
        return False
    if metrics.get("calculator_injection_mode") != "add":
        return False
    if not metric_bool(metrics, "freeze_calculator_policy"):
        return False
    if metric_bool(metrics, "freeze_calculator_policy_backbone"):
        return False
    if metric_float(metrics, "answer_loss_weight") != 1.0:
        return False
    if metric_float(metrics, "result_policy_anchor_weight") != 0.0:
        return False
    if metric_float(metrics, "final_result_policy_anchor_weight") != 0.0:
        return False
    return True


def infer_family(metrics_path: Path, metrics: dict[str, Any]) -> str:
    candidates = [
        metrics_path.as_posix(),
        str(metrics.get("semantic_decoder_checkpoint", "")),
    ]
    for text in candidates:
        if "src10_entropy0p05_div0p1_improve5" in text:
            return "src10_improve5"
        if "src10_entropy0p05_div0p1_nodecay" in text:
            return "src10_nodecay"
        if "src9_entropy0p05_div0p1_nodecay" in text:
            return "src9_nodecay"
    for text in candidates:
        match = re.search(r"source_seed(\d+)", text)
        if match:
            return f"src{match.group(1)}"
        match = re.search(r"\bsrc(\d+)[_\-/]", text)
        if match:
            return f"src{match.group(1)}"
    semantic = str(metrics.get("semantic_decoder_checkpoint", ""))
    match = re.search(r"src(\d+)_", semantic)
    if match:
        return f"src{match.group(1)}"
    if "seed10" in metrics_path.as_posix() or "src10" in semantic:
        return "src10"
    return "unknown"


def infer_candidate(metrics_path: Path, metrics: dict[str, Any]) -> str:
    text = metrics_path.as_posix()
    for pattern in [r"source_seed\d+_step(\d+)", r"step(\d+)_handoff"]:
        match = re.search(pattern, text)
        if match:
            return f"step{int(match.group(1))}"
    semantic = str(metrics.get("semantic_decoder_checkpoint", ""))
    match = re.search(r"step_(\d+)_weights", semantic)
    if match:
        return f"step{int(match.group(1))}"
    return "final"


def build_features(row: dict[str, float]) -> dict[str, float]:
    normal = row.get("normal_exact_match", 0.0)
    zero = row.get("injection_zero_exact_match", 0.0)
    oracle = row.get("oracle_exact_match", 0.0)
    forced_random = row.get("forced_random_exact_match", 0.0)
    features = {
        "normal_exact_match": normal,
        "injection_zero_exact_match": zero,
        "oracle_exact_match": oracle,
        "forced_random_exact_match": forced_random,
        "calculator_result_accuracy": row.get("calculator_result_accuracy", 0.0),
        "normal_minus_zero": normal - zero,
        "oracle_minus_normal": oracle - normal,
        "normal_minus_forced_random": normal - forced_random,
    }
    return features


def discover_runs(
    *,
    run_glob: str,
    prediction_step: int,
    target_step: int,
) -> list[HandoffRun]:
    runs: list[HandoffRun] = []
    for metrics_path in sorted(REPO_ROOT.glob(run_glob)):
        metrics = read_json(metrics_path)
        if not is_initial_frozen_handoff(metrics_path, metrics):
            continue
        snapshots_path = metrics_path.with_name("diagnostic_snapshots.csv")
        if not snapshots_path.exists():
            continue
        rows = read_snapshot_rows(snapshots_path)
        if prediction_step not in rows or target_step not in rows:
            continue
        family = infer_family(metrics_path, metrics)
        if family == "unknown":
            continue
        runs.append(
            HandoffRun(
                run_dir=metrics_path.parent,
                metrics_path=metrics_path,
                snapshots_path=snapshots_path,
                family=family,
                candidate=infer_candidate(metrics_path, metrics),
                target_score=rows[target_step]["normal_exact_match"],
                final_score=metric_float(metrics, "exact_match"),
                max_snapshot_step=max(rows),
                features=build_features(rows[prediction_step]),
            )
        )
    return runs


def dedupe_runs(runs: list[HandoffRun]) -> list[HandoffRun]:
    best: dict[tuple[str, str], HandoffRun] = {}
    for run in runs:
        key = (run.family, run.candidate)
        current = best.get(key)
        if current is None:
            best[key] = run
            continue
        current_key = (current.max_snapshot_step, current.final_score)
        run_key = (run.max_snapshot_step, run.final_score)
        if run_key > current_key:
            best[key] = run
    return sorted(best.values(), key=lambda run: (run.family, run.candidate))


def eligible_families(runs: list[HandoffRun]) -> list[str]:
    by_family: dict[str, int] = {}
    for run in runs:
        by_family[run.family] = by_family.get(run.family, 0) + 1
    return sorted(family for family, count in by_family.items() if count >= 2)


def standardize_train_test(
    train_x: torch.Tensor, test_x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


def ridge_predict(
    train: list[HandoffRun],
    test: list[HandoffRun],
    *,
    ridge: float,
) -> list[float]:
    train_x = torch.tensor(
        [[run.features[name] for name in FEATURE_COLUMNS] for run in train],
        dtype=torch.float64,
    )
    test_x = torch.tensor(
        [[run.features[name] for name in FEATURE_COLUMNS] for run in test],
        dtype=torch.float64,
    )
    train_y = torch.tensor([run.target_score for run in train], dtype=torch.float64)
    train_x, test_x = standardize_train_test(train_x, test_x)
    train_x = torch.cat(
        [torch.ones(train_x.shape[0], 1, dtype=torch.float64), train_x],
        dim=1,
    )
    test_x = torch.cat(
        [torch.ones(test_x.shape[0], 1, dtype=torch.float64), test_x],
        dim=1,
    )
    penalty = torch.eye(train_x.shape[1], dtype=torch.float64) * ridge
    penalty[0, 0] = 0.0
    weights = torch.linalg.solve(train_x.T @ train_x + penalty, train_x.T @ train_y)
    return (test_x @ weights).tolist()


def winner(runs: list[HandoffRun], scores: list[float]) -> HandoffRun:
    return runs[max(range(len(runs)), key=lambda idx: scores[idx])]


def leave_family_out(
    runs: list[HandoffRun],
    *,
    ridge: float,
) -> list[dict[str, Any]]:
    families = eligible_families(runs)
    results: list[dict[str, Any]] = []
    for family in families:
        test = [run for run in runs if run.family == family]
        train = [run for run in runs if run.family != family and run.family in families]
        if len(train) < len(FEATURE_COLUMNS) + 1:
            continue
        predicted = ridge_predict(train, test, ridge=ridge)
        true_scores = [run.target_score for run in test]
        early_scores = [run.features["normal_exact_match"] for run in test]
        calc_scores = [run.features["calculator_result_accuracy"] for run in test]
        pred_winner = winner(test, predicted)
        early_winner = winner(test, early_scores)
        calc_winner = winner(test, calc_scores)
        true_winner = winner(test, true_scores)
        results.append(
            {
                "family": family,
                "candidates": [
                    {
                        "candidate": run.candidate,
                        "target_score": run.target_score,
                        "final_score": run.final_score,
                        "max_snapshot_step": run.max_snapshot_step,
                        "prediction_step_normal": run.features["normal_exact_match"],
                        "prediction_step_calc": run.features[
                            "calculator_result_accuracy"
                        ],
                        "predicted_target_score": predicted[idx],
                        "run_dir": run.run_dir.relative_to(REPO_ROOT).as_posix(),
                    }
                    for idx, run in enumerate(test)
                ],
                "true_winner": true_winner.candidate,
                "ridge_winner": pred_winner.candidate,
                "early_exact_winner": early_winner.candidate,
                "calc_winner": calc_winner.candidate,
                "ridge_correct": pred_winner == true_winner,
                "early_exact_correct": early_winner == true_winner,
                "calc_correct": calc_winner == true_winner,
            }
        )
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)

    def count(key: str) -> int:
        return sum(1 for row in results if row[key])

    return {
        "families": total,
        "ridge_correct": count("ridge_correct"),
        "early_exact_correct": count("early_exact_correct"),
        "calc_correct": count("calc_correct"),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leave-family-out selector audit over existing additive frozen-policy "
            "handoff traces."
        )
    )
    parser.add_argument(
        "--run-glob",
        default="runs/2026-05-29_phase7_*/**/metrics.json",
        help="Repo-relative glob for metrics.json files.",
    )
    parser.add_argument("--prediction-step", type=int, default=400)
    parser.add_argument("--target-step", type=int, default=600)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    discovered = discover_runs(
        run_glob=args.run_glob,
        prediction_step=args.prediction_step,
        target_step=args.target_step,
    )
    runs = dedupe_runs(discovered)
    families = eligible_families(runs)
    filtered = [run for run in runs if run.family in families]
    results = leave_family_out(filtered, ridge=args.ridge)
    summary = {
        "prediction_step": args.prediction_step,
        "target_step": args.target_step,
        "ridge": args.ridge,
        "discovered_runs": len(discovered),
        "deduped_runs": len(runs),
        "eligible_runs": len(filtered),
        "eligible_families": families,
        **summarize(results),
    }
    print(json.dumps(summary, indent=2))
    for row in results:
        print(
            f"{row['family']}: true={row['true_winner']} "
            f"ridge={row['ridge_winner']} early={row['early_exact_winner']} "
            f"calc={row['calc_winner']}"
    )
    if args.output_root is not None:
        output_root = (
            args.output_root
            if args.output_root.is_absolute()
            else REPO_ROOT / args.output_root
        )
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "selector_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        (output_root / "selector_leave_family_out.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        write_rows(
            output_root / "selector_runs.csv",
            [
                {
                    "family": run.family,
                    "candidate": run.candidate,
                    "target_score": run.target_score,
                    "final_score": run.final_score,
                    "max_snapshot_step": run.max_snapshot_step,
                    **run.features,
                    "run_dir": run.run_dir.relative_to(REPO_ROOT).as_posix(),
                }
                for run in filtered
            ],
        )


if __name__ == "__main__":
    main()
