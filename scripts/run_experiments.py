from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    name: str
    args: tuple[str, ...]


EXPERIMENTS: dict[str, Experiment] = {
    "poisson": Experiment(
        name="poisson",
        args=("--task", "poisson"),
    ),
    "darcy_iid_mean": Experiment(
        name="darcy_iid_mean",
        args=("--task", "darcy", "--darcy-orientation", "iid", "--darcy-vertex-projection", "mean"),
    ),
    "darcy_iid_none": Experiment(
        name="darcy_iid_none",
        args=("--task", "darcy", "--darcy-orientation", "iid", "--darcy-vertex-projection", "none"),
    ),
    "darcy_blobs_none": Experiment(
        name="darcy_blobs_none",
        args=(
            "--task",
            "darcy",
            "--darcy-orientation",
            "blobs",
            "--darcy-vertex-projection",
            "none",
        ),
    ),
    "darcy_holes_blobs_none": Experiment(
        name="darcy_holes_blobs_none",
        args=(
            "--task",
            "darcy_holes",
            "--darcy-orientation",
            "blobs",
            "--darcy-vertex-projection",
            "none",
        ),
    ),
}


def parse_csv_ints(value: str) -> list[int]:
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def parse_csv_experiments(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(names) - set(EXPERIMENTS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown experiment(s): {', '.join(unknown)}; choices: {', '.join(EXPERIMENTS)}"
        )
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired TNO/vertex experiment sweeps.")
    parser.add_argument(
        "--experiments",
        type=parse_csv_experiments,
        default=["darcy_blobs_none"],
        help=f"Comma-separated experiment names. Choices: {', '.join(EXPERIMENTS)}.",
    )
    parser.add_argument("--seeds", type=parse_csv_ints, default=[7, 8, 9])
    parser.add_argument("--models", choices=["both", "tno", "vertex"], default="both")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report", type=Path, default=Path("docs/experiment_summary.md"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use tiny settings for checking the runner without doing a real sweep.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not refresh docs/experiment_summary.md after the sweep.",
    )
    return parser.parse_args()


def selected_models(choice: str) -> list[str]:
    if choice == "both":
        return ["tno", "vertex"]
    return [choice]


def output_path(outputs_dir: Path, experiment: Experiment, model: str, seed: int, quick: bool) -> Path:
    prefix = "smoke_sweep_" if quick else ""
    return outputs_dir / f"{prefix}{experiment.name}_{model}_seed{seed}.pt"


def base_train_args(args: argparse.Namespace) -> list[str]:
    if args.quick:
        return [
            "--epochs",
            "3",
            "--train-samples",
            "24",
            "--val-samples",
            "8",
            "--batch-size",
            "4",
            "--hidden-dim",
            "16",
            "--layers",
            "2",
            "--nx",
            "8",
            "--ny",
            "8",
        ]
    return [
        "--epochs",
        str(args.epochs),
        "--train-samples",
        str(args.train_samples),
        "--val-samples",
        str(args.val_samples),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--layers",
        str(args.layers),
        "--nx",
        str(args.nx),
        "--ny",
        str(args.ny),
    ]


def train_command(
    args: argparse.Namespace,
    experiment: Experiment,
    model: str,
    seed: int,
    save_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "train.py",
        *experiment.args,
        "--model",
        model,
        "--seed",
        str(seed),
        "--save",
        str(save_path),
        *base_train_args(args),
    ]


def run_command(command: list[str], dry_run: bool) -> None:
    print("$", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def refresh_summary(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "scripts/summarize_results.py",
        "--outputs-dir",
        str(args.outputs_dir),
        "--output",
        str(args.report),
    ]
    if args.quick:
        command.append("--include-smoke")
    run_command(command, args.dry_run)


def main() -> None:
    args = parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    for experiment_name in args.experiments:
        experiment = EXPERIMENTS[experiment_name]
        for seed in args.seeds:
            for model in selected_models(args.models):
                save_path = output_path(args.outputs_dir, experiment, model, seed, args.quick)
                if save_path.exists() and not args.overwrite:
                    print(f"skip existing {save_path}")
                    continue
                command = train_command(args, experiment, model, seed, save_path)
                run_command(command, args.dry_run)
    if not args.no_summary:
        refresh_summary(args)


if __name__ == "__main__":
    main()
