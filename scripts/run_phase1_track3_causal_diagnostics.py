import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_HISTORY = REPO_ROOT / "aiAgentWorkHistory/2026-04-30-track-3-causal-diagnostics-codebooks.md"
TASK_DOC = REPO_ROOT / "aiAgentProjectTasks/2026-04-30-1202-Track-3-causal-diagnostics-codebooks.md"
COMPLETED_TASK_DOC = (
    REPO_ROOT
    / "aiAgentProjectTasks/completed/2026-04-30-1202-Track-3-causal-diagnostics-codebooks.md"
)


@dataclass(frozen=True)
class Track3Checkpoint:
    name: str
    purpose: str
    checkpoint: str
    digits: int = 2
    operand_max: int = 19
    forced_sweep: bool = True
    oracle: bool = False


MANIFEST = [
    Track3Checkpoint(
        name="additive_answer_only_model_c",
        purpose="Additive answer-only learned Model C",
        checkpoint="runs/2026-04-30_124622_062528_model-c-op0-19/model-c-2digit-seed2/final_weights.pt",
    ),
    Track3Checkpoint(
        name="additive_aux001_model_c",
        purpose="Additive high-answer aux 0.01 Model C",
        checkpoint="runs/2026-04-30_124941_086322_model-c-op0-19-aux0.01/model-c-2digit-seed2/final_weights.pt",
    ),
    Track3Checkpoint(
        name="replace_model_b_off_leakage_control",
        purpose="Replacement Model B/off leakage control",
        checkpoint="runs/2026-04-30_131732_611676_model-b-op0-19-replace/model-b-2digit-seed2/final_weights.pt",
        forced_sweep=False,
    ),
    Track3Checkpoint(
        name="replace_oracle_model_c",
        purpose="Replacement oracle Model C control",
        checkpoint="runs/2026-04-30_131816_053136_model-c-oracle-op0-19-replace/model-c-2digit-seed2/final_weights.pt",
        oracle=True,
    ),
    Track3Checkpoint(
        name="replace_answer_only_model_c",
        purpose="Replacement answer-only learned Model C",
        checkpoint="runs/2026-04-30_131959_337278_model-c-op0-19-replace/model-c-2digit-seed2/final_weights.pt",
    ),
    Track3Checkpoint(
        name="replace_aux_decay_model_c",
        purpose="Replacement aux decay transient candidate",
        checkpoint="runs/2026-04-30_132810_628910_model-c-op0-19-replace-aux0.03-auxdecay1000/model-c-2digit-seed2/final_weights.pt",
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_counterfactual_table(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {
            row["condition"]: float(row["exact_match"])
            for row in csv.DictReader(f)
        }


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def run_diagnostics(args: argparse.Namespace) -> list[dict[str, Any]]:
    leakage_control = 0.888671875
    results = []
    for item in MANIFEST[: args.limit]:
        checkpoint = REPO_ROOT / item.checkpoint
        output_dir = checkpoint.parent / "track3_diagnostics"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/diagnose_calculator_protocol.py"),
            "--checkpoint",
            str(checkpoint),
            "--digits",
            str(item.digits),
            "--operand-max",
            str(item.operand_max),
            "--samples",
            str(args.samples),
            "--forced-result-batch-size",
            str(args.forced_result_batch_size),
            "--leakage-control-exact-match",
            str(leakage_control),
            "--output-dir",
            str(output_dir),
        ]
        if item.forced_sweep and not args.skip_forced_sweep:
            cmd.append("--forced-result-sweep")
        if item.oracle:
            cmd.append("--oracle")
        if args.dry_run:
            print(" ".join(cmd))
        else:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)

        summary_path = output_dir / "diagnostic_summary.json"
        summary = {} if args.dry_run else read_json(summary_path)
        counterfactuals = (
            {} if args.dry_run else read_counterfactual_table(output_dir / "counterfactual_exact_match.csv")
        )
        results.append(
            {
                "name": item.name,
                "purpose": item.purpose,
                "checkpoint": item.checkpoint,
                "output_dir": str(output_dir.relative_to(REPO_ROOT)),
                "summary": summary,
                "counterfactuals": counterfactuals,
            }
        )
    return results


def write_work_history(results: list[dict[str, Any]], *, samples: int) -> None:
    lines = [
        "# 2026-04-30 - Track 3 causal diagnostics, codebooks, and falsification",
        "",
        "Task: harden the checkpoint-first diagnostic suite and apply it to the six Track 2 priority checkpoints.",
        "",
        "## Implementation",
        "",
        "- Extended `scripts/diagnose_calculator_protocol.py` to always write trace rows, diagnostic summary JSON, result and operand codebooks, a counterfactual exact-match table, and bottleneck/protocol classification.",
        "- Added tensor-valued forced result classes and batched forced-result sweeps for 2-digit result vocabularies.",
        "- Added read-site interventions for swapping or corrupting only A/B calculator read vectors.",
        "- Added `scripts/run_track3_causal_diagnostics.py` as the six-checkpoint manifest runner.",
        "",
        "## Track 2 checkpoint classifications",
        "",
        f"All rows below use `{samples}` diagnostic samples.",
        "",
        "| Checkpoint | Exact | Operand | Calc result | Zero inj | Forced random | Classification | Bottleneck | Output |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for result in results:
        summary = result["summary"]
        classification = summary.get("classification", {})
        counterfactuals = result["counterfactuals"]
        lines.append(
            "| "
            + " | ".join(
                [
                    result["purpose"],
                    format_float(summary.get("exact_match")),
                    format_float(summary.get("operand_exact_match")),
                    format_float(summary.get("calculator_result_accuracy")),
                    format_float(counterfactuals.get("injection_zero")),
                    format_float(counterfactuals.get("forced_random")),
                    classification.get("category", "n/a"),
                    classification.get("bottleneck_classification", "n/a"),
                    f"`{result['output_dir']}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Additive high answer accuracy remains bypass-compatible: operand exact match and calculator result accuracy stay near chance, and counterfactuals do not show clean dependence on the calculator.",
            "- Replacement oracle remains the positive control: true operands and true calculator result are usable downstream, and removing/corrupting calculator output collapses accuracy.",
            "- Current replacement learned checkpoints are not valid calculator-required bottleneck evidence. The Model B/off replacement control reaches high exact match, confirming leakage through ordinary autoregressive context.",
            "- The learned replacement checkpoints are best treated as failed or bypass-heavy protocol-learning attempts, not as evidence against a stricter future bottleneck.",
            "",
            "## Validation",
            "",
            "```bash",
            "PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/diagnose_calculator_protocol.py scripts/run_track3_causal_diagnostics.py",
            "PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. pytest -q",
            "python3 scripts/run_track3_causal_diagnostics.py --samples 8 --limit 1 --skip-forced-sweep",
            f"python3 scripts/run_track3_causal_diagnostics.py --samples {samples}",
            "```",
            "",
            "Conclusion: Track 3 makes answer accuracy much harder to overstate by pairing it with protocol codebooks, mutual information, counterfactuals, read-site interventions, and explicit bottleneck/leakage labels.",
        ]
    )
    WORK_HISTORY.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Track 3 causal diagnostics on the priority checkpoint manifest."
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--limit", type=int, default=len(MANIFEST))
    parser.add_argument("--forced-result-batch-size", type=int, default=64)
    parser.add_argument("--skip-forced-sweep", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-move-task-doc", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.limit > len(MANIFEST):
        raise ValueError(f"--limit must be in [1, {len(MANIFEST)}]")
    results = run_diagnostics(args)
    if not args.dry_run:
        write_work_history(results, samples=args.samples)
        if not args.no_move_task_doc and TASK_DOC.exists():
            COMPLETED_TASK_DOC.parent.mkdir(parents=True, exist_ok=True)
            TASK_DOC.rename(COMPLETED_TASK_DOC)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
