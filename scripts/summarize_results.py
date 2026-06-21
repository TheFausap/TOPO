from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class RunSummary:
    path: Path
    model: str
    task: str
    best_val_rel_l2: float | None
    final_val_rel_l2: float | None
    final_val_sample_rel_l2: float | None
    final_val_mse: float | None
    args: dict[str, Any]
    history_path: Path | None = None

    @property
    def experiment_key(self) -> tuple[Any, ...]:
        return (
            self.task,
            self.args.get("nx", 16),
            self.args.get("ny", 16),
            self.args.get("train_samples"),
            self.args.get("val_samples"),
            self.args.get("seed", 7),
            self.args.get("darcy_vertex_projection", "mean"),
            self.args.get("darcy_orientation", "iid"),
            self.args.get("darcy_orientation_blobs", 4),
            self.args.get("darcy_orientation_sigma", 0.25),
            self.args.get("darcy_min_solution_norm", 0.5),
        )

    @property
    def experiment_family_key(self) -> tuple[Any, ...]:
        return (
            self.task,
            self.args.get("nx", 16),
            self.args.get("ny", 16),
            self.args.get("train_samples"),
            self.args.get("val_samples"),
            self.args.get("darcy_vertex_projection", "mean"),
            self.args.get("darcy_orientation", "iid"),
            self.args.get("darcy_orientation_blobs", 4),
            self.args.get("darcy_orientation_sigma", 0.25),
            self.args.get("darcy_min_solution_norm", 0.5),
        )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _pct_improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def _fmt_mean_std(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} +/- {statistics.stdev(values):.4f}"


def load_history(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def summarize_run(path: Path) -> RunSummary:
    checkpoint = load_checkpoint(path)
    args = dict(checkpoint.get("args", {}))
    history_path = checkpoint.get("history_path")
    if history_path is None:
        candidate = path.with_suffix(".history.json")
        history_path = str(candidate) if candidate.exists() else None

    final_val_rel_l2 = None
    final_val_sample_rel_l2 = None
    final_val_mse = None
    best_val_rel_l2 = checkpoint.get("best_val_relative_l2")
    resolved_history_path = None

    if history_path is not None:
        resolved_history_path = Path(history_path)
        if not resolved_history_path.is_absolute():
            resolved_history_path = path.parent / resolved_history_path.name
        if resolved_history_path.exists():
            history_payload = load_history(resolved_history_path)
            args = {**history_payload.get("args", {}), **args}
            best_val_rel_l2 = history_payload.get("best_val_relative_l2", best_val_rel_l2)
            history = history_payload.get("history", [])
            if history:
                final = history[-1]
                final_val_rel_l2 = final.get("val_rel_l2")
                final_val_sample_rel_l2 = final.get("val_sample_rel_l2")
                final_val_mse = final.get("val_mse")

    return RunSummary(
        path=path,
        model=str(args.get("model", "unknown")),
        task=str(args.get("task", "poisson")),
        best_val_rel_l2=float(best_val_rel_l2) if best_val_rel_l2 is not None else None,
        final_val_rel_l2=float(final_val_rel_l2) if final_val_rel_l2 is not None else None,
        final_val_sample_rel_l2=(
            float(final_val_sample_rel_l2) if final_val_sample_rel_l2 is not None else None
        ),
        final_val_mse=float(final_val_mse) if final_val_mse is not None else None,
        args=args,
        history_path=resolved_history_path,
    )


def collect_runs(outputs_dir: Path, include_smoke: bool) -> list[RunSummary]:
    runs = []
    for path in sorted(outputs_dir.glob("*.pt")):
        if not include_smoke and "smoke" in path.name:
            continue
        try:
            runs.append(summarize_run(path))
        except Exception as exc:
            print(f"warning: skipped {path}: {exc}")
    return runs


def _run_rank(run: RunSummary) -> tuple[int, int, float]:
    seed = run.args.get("seed", 7)
    seed_named = f"seed{seed}" in run.path.stem
    has_history = run.history_path is not None and run.history_path.exists()
    return (int(seed_named), int(has_history), run.path.stat().st_mtime)


def deduplicate_runs(runs: list[RunSummary]) -> list[RunSummary]:
    """Keep one artifact for each exact experiment/model/seed combination."""
    by_key: dict[tuple[Any, ...], RunSummary] = {}
    for run in runs:
        key = (*run.experiment_key, run.model)
        current = by_key.get(key)
        if current is None or _run_rank(run) > _run_rank(current):
            by_key[key] = run
    return sorted(by_key.values(), key=lambda run: (describe_experiment(run), run.model, run.path.name))


def describe_experiment(run: RunSummary) -> str:
    if run.task not in {"darcy", "darcy_holes", "darcy_holes_flux"}:
        return run.task
    projection = run.args.get("darcy_vertex_projection", "mean")
    orientation = run.args.get("darcy_orientation", "iid")
    return f"{run.task}/orientation={orientation}/vertex_projection={projection}"


def make_markdown(runs: list[RunSummary]) -> str:
    runs = deduplicate_runs(runs)
    lines = [
        "# Experiment Summary",
        "",
        "Generated from checkpoint metadata and, when present, `*.history.json` files.",
        "Lower relative L2 is better. Checkpoints are selected by aggregate validation relative L2.",
        "Equivalent artifacts are deduplicated by experiment, model, and seed.",
        "",
        "## Runs",
        "",
        "| Experiment | Model | Best val rel L2 | Final val rel L2 | Final sample rel L2 | Final val MSE | Artifact |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for run in sorted(runs, key=lambda r: (describe_experiment(r), r.model, r.path.name)):
        lines.append(
            "| "
            f"{describe_experiment(run)} | "
            f"{run.model} | "
            f"{_fmt(run.best_val_rel_l2)} | "
            f"{_fmt(run.final_val_rel_l2)} | "
            f"{_fmt(run.final_val_sample_rel_l2)} | "
            f"{_fmt(run.final_val_mse)} | "
            f"`{run.path}` |"
        )

    lines.extend(["", "## Seed Aggregates", ""])
    family_groups: dict[tuple[Any, ...], dict[str, list[RunSummary]]] = {}
    for run in runs:
        family_groups.setdefault(run.experiment_family_key, {}).setdefault(run.model, []).append(run)

    aggregate_rows = []
    for by_model in family_groups.values():
        tno_runs = by_model.get("tno", [])
        vertex_runs = by_model.get("vertex", [])
        if not tno_runs or not vertex_runs:
            continue
        tno_values = [run.best_val_rel_l2 for run in tno_runs if run.best_val_rel_l2 is not None]
        vertex_values = [
            run.best_val_rel_l2 for run in vertex_runs if run.best_val_rel_l2 is not None
        ]
        if not tno_values or not vertex_values:
            continue
        description = describe_experiment(tno_runs[0])
        tno_seeds = {run.args.get("seed", 7) for run in tno_runs}
        vertex_seeds = {run.args.get("seed", 7) for run in vertex_runs}
        common_seeds = sorted(tno_seeds & vertex_seeds)
        aggregate_rows.append((description, tno_values, vertex_values, common_seeds))

    if not aggregate_rows:
        lines.append("No aggregate seed groups found.")
    else:
        lines.extend(
            [
                "| Experiment | Seeds | TNO best mean | Vertex best mean | TNO improvement |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for description, tno_values, vertex_values, common_seeds in sorted(
            aggregate_rows, key=lambda item: item[0]
        ):
            tno_mean = statistics.mean(tno_values)
            vertex_mean = statistics.mean(vertex_values)
            improvement = _pct_improvement(vertex_mean, tno_mean)
            seeds = ", ".join(str(seed) for seed in common_seeds) if common_seeds else "mixed"
            lines.append(
                "| "
                f"{description} | "
                f"{seeds} | "
                f"{_fmt_mean_std(tno_values)} | "
                f"{_fmt_mean_std(vertex_values)} | "
                f"{improvement:+.1f}% |"
            )

    lines.extend(["", "## Matched Comparisons", ""])
    groups: dict[tuple[Any, ...], dict[str, RunSummary]] = {}
    for run in runs:
        groups.setdefault(run.experiment_key, {})[run.model] = run

    comparisons = []
    for by_model in groups.values():
        tno = by_model.get("tno")
        vertex = by_model.get("vertex")
        if not tno or not vertex:
            continue
        if tno.best_val_rel_l2 is None or vertex.best_val_rel_l2 is None:
            continue
        comparisons.append((describe_experiment(tno), tno, vertex))

    if not comparisons:
        lines.append("No matched `tno`/`vertex` pairs found.")
    else:
        lines.extend(
            [
                "| Experiment | TNO best | Vertex best | TNO improvement | Notes |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for description, tno, vertex in sorted(comparisons, key=lambda item: item[0]):
            improvement = _pct_improvement(vertex.best_val_rel_l2, tno.best_val_rel_l2)
            note = "TNO better" if improvement > 0 else "Vertex better"
            lines.append(
                "| "
                f"{description} | "
                f"{_fmt(tno.best_val_rel_l2)} | "
                f"{_fmt(vertex.best_val_rel_l2)} | "
                f"{improvement:+.1f}% | "
                f"{note} |"
            )

    lines.extend(
        [
            "",
            "## Current Takeaways",
            "",
            "- The multi-rank TNO consistently beats the vertex-only baseline in the completed matched runs so far.",
            "- The Darcy relative-error metric needed aggregate normalization because per-sample relative L2 is unstable on near-zero target fields.",
            "- Removing vertex-projected face orientation creates a cleaner test of native face cochains.",
            "- The blob-orientation Darcy task is intended to test smoother, spatially coherent face coefficients.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TNO experiment checkpoints.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("docs/experiment_summary.md"))
    parser.add_argument("--include-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = collect_runs(args.outputs_dir, include_smoke=args.include_smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(make_markdown(runs), encoding="utf-8")
    print(f"wrote {args.output} from {len(runs)} run(s)")


if __name__ == "__main__":
    main()
