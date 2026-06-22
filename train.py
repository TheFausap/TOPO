from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from topo.complex import (
    grid_complex,
    grid_complex_with_holes,
    triangulated_square_with_holes,
)
from topo.data import AnisotropicDarcyDataset, DarcyHolesDataset, PoissonDataset, poisson_collate
from topo.dec import DECOperators
from topo.tno import TopologicalNeuralOperator, VertexGraphOperator


def parse_csv_ints(value: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal rigid-DEC TNO.")
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model", choices=["tno", "vertex"], default="tno")
    parser.add_argument(
        "--tno-ablation",
        choices=["full", "no_face_to_edge", "no_vertex_to_edge", "no_edge_laplacian"],
        default="full",
        help="Disable selected TNO edge-update DEC routes.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--task",
        choices=[
            "poisson",
            "darcy",
            "darcy_holes",
            "darcy_holes_flux",
            "darcy_holes_tri",
            "darcy_holes_tri_flux",
            "darcy_holes_tri_meshes",
            "darcy_holes_tri_meshes_flux",
        ],
        default="poisson",
    )
    parser.add_argument(
        "--train-mesh-seeds",
        type=parse_csv_ints,
        default=[0, 1, 2, 3],
        help="Comma-separated triangulation seeds for variable-mesh training tasks.",
    )
    parser.add_argument(
        "--val-mesh-seeds",
        type=parse_csv_ints,
        default=[100, 101],
        help="Comma-separated held-out triangulation seeds for variable-mesh validation tasks.",
    )
    parser.add_argument(
        "--darcy-min-solution-norm",
        type=float,
        default=0.5,
        help="Reject generated Darcy samples whose solution norm is below this value.",
    )
    parser.add_argument(
        "--darcy-vertex-projection",
        choices=["mean", "none"],
        default=None,
        help=(
            "How face orientation channels are projected into Darcy vertex inputs. "
            "Defaults to 'mean' for plain Darcy and 'none' for holed Darcy tasks."
        ),
    )
    parser.add_argument(
        "--darcy-orientation",
        choices=["iid", "blobs"],
        default=None,
        help=(
            "How per-face Darcy conductivity orientations are generated. "
            "Defaults to 'iid' for plain Darcy and 'blobs' for holed Darcy tasks."
        ),
    )
    parser.add_argument(
        "--darcy-orientation-blobs",
        type=int,
        default=4,
        help="Number of smooth orientation blobs used when --darcy-orientation=blobs.",
    )
    parser.add_argument(
        "--darcy-orientation-sigma",
        type=float,
        default=0.25,
        help="Gaussian width for smooth orientation blobs.",
    )
    parser.add_argument("--save", type=Path, default=Path("outputs/tno_poisson.pt"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    result = vars(args).copy()
    result["save"] = str(args.save)
    return result


def apply_task_defaults(args: argparse.Namespace) -> None:
    holed_tasks = {
        "darcy_holes",
        "darcy_holes_flux",
        "darcy_holes_tri",
        "darcy_holes_tri_flux",
        "darcy_holes_tri_meshes",
        "darcy_holes_tri_meshes_flux",
    }
    if args.darcy_vertex_projection is None:
        args.darcy_vertex_projection = "none" if args.task in holed_tasks else "mean"
    if args.darcy_orientation is None:
        args.darcy_orientation = "blobs" if args.task in holed_tasks else "iid"


def is_flux_task(task: str) -> bool:
    return task in {
        "darcy_holes_flux",
        "darcy_holes_tri_flux",
        "darcy_holes_tri_meshes_flux",
    }


def is_multi_mesh_task(task: str) -> bool:
    return task in {"darcy_holes_tri_meshes", "darcy_holes_tri_meshes_flux"}


def split_sample_budget(total: int, parts: int) -> list[int]:
    if parts < 1:
        raise ValueError("parts must be positive")
    if total < parts:
        raise ValueError("sample budget must be at least the number of meshes")
    base = total // parts
    remainder = total % parts
    return [base + int(index < remainder) for index in range(parts)]


def sample_relative_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_edge: torch.Tensor | None = None,
    target_edge: torch.Tensor | None = None,
) -> torch.Tensor:
    numerator_sq = torch.sum((pred - target) ** 2, dim=(-2, -1))
    denominator_sq = torch.sum(target**2, dim=(-2, -1))
    if pred_edge is not None and target_edge is not None:
        numerator_sq = numerator_sq + torch.sum((pred_edge - target_edge) ** 2, dim=(-2, -1))
        denominator_sq = denominator_sq + torch.sum(target_edge**2, dim=(-2, -1))
    numerator = torch.sqrt(numerator_sq)
    denominator = torch.sqrt(denominator_sq).clamp_min(1e-8)
    return (numerator / denominator).mean()


def run_epoch(
    model: TopologicalNeuralOperator,
    op_loaders: Sequence[tuple[DECOperators, DataLoader]],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_sample_rel = 0.0
    total_error_sq = 0.0
    total_target_sq = 0.0
    total_items = 0

    for ops, loader in op_loaders:
        for x0, x1, x2, y0, y1 in loader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)
            y0 = y0.to(device)
            y1 = y1.to(device) if y1 is not None else None
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(ops, x0, x1, x2, return_edges=y1 is not None)
            if y1 is None:
                pred = output
                pred_edge = None
                loss = torch.nn.functional.mse_loss(pred, y0)
            else:
                pred, pred_edge = output
                loss = torch.nn.functional.mse_loss(pred, y0) + torch.nn.functional.mse_loss(
                    pred_edge, y1
                )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch = x0.shape[0]
            total_loss += float(loss.detach()) * batch
            total_sample_rel += float(
                sample_relative_l2(
                    pred.detach(),
                    y0,
                    pred_edge.detach() if pred_edge is not None else None,
                    y1,
                )
            ) * batch
            total_error_sq += float(torch.sum((pred.detach() - y0) ** 2))
            total_target_sq += float(torch.sum(y0**2))
            if pred_edge is not None and y1 is not None:
                total_error_sq += float(torch.sum((pred_edge.detach() - y1) ** 2))
                total_target_sq += float(torch.sum(y1**2))
            total_items += batch

    aggregate_rel = math.sqrt(total_error_sq / max(total_target_sq, 1e-12))
    return total_loss / total_items, total_sample_rel / total_items, aggregate_rel


def darcy_holes_dataset(
    args: argparse.Namespace,
    complex_,
    samples: int,
    seed: int,
) -> DarcyHolesDataset:
    return DarcyHolesDataset(
        complex_,
        samples,
        seed=seed,
        min_solution_norm=args.darcy_min_solution_norm,
        vertex_projection=args.darcy_vertex_projection,
        orientation_mode=args.darcy_orientation,
        orientation_blobs=args.darcy_orientation_blobs,
        orientation_sigma=args.darcy_orientation_sigma,
        predict_flux=is_flux_task(args.task),
    )


def loader_for_dataset(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=poisson_collate,
        generator=generator,
    )


def main() -> None:
    args = parse_args()
    apply_task_defaults(args)
    set_seed(args.seed)
    device = torch.device("cpu")
    run_args = jsonable_args(args)

    if is_multi_mesh_task(args.task):
        train_counts = split_sample_budget(args.train_samples, len(args.train_mesh_seeds))
        val_counts = split_sample_budget(args.val_samples, len(args.val_mesh_seeds))
        train_datasets = []
        val_datasets = []
        train_op_loaders = []
        val_op_loaders = []
        for index, (mesh_seed, samples) in enumerate(zip(args.train_mesh_seeds, train_counts)):
            complex_ = triangulated_square_with_holes(args.nx, args.ny, seed=mesh_seed)
            dataset = darcy_holes_dataset(args, complex_, samples, seed=args.seed + 1000 * index)
            train_datasets.append(dataset)
            train_op_loaders.append(
                (
                    DECOperators.from_complex(complex_, device=device),
                    loader_for_dataset(dataset, args.batch_size, True, args.seed + index),
                )
            )
        for index, (mesh_seed, samples) in enumerate(zip(args.val_mesh_seeds, val_counts)):
            complex_ = triangulated_square_with_holes(args.nx, args.ny, seed=mesh_seed)
            dataset = darcy_holes_dataset(
                args,
                complex_,
                samples,
                seed=args.seed + 1 + 1000 * index,
            )
            val_datasets.append(dataset)
            val_op_loaders.append(
                (
                    DECOperators.from_complex(complex_, device=device),
                    loader_for_dataset(dataset, args.batch_size, False, args.seed + 100 + index),
                )
            )
        sample = train_datasets[0][0]
    else:
        if args.task in {"darcy_holes", "darcy_holes_flux"}:
            complex_ = grid_complex_with_holes(args.nx, args.ny)
        elif args.task in {"darcy_holes_tri", "darcy_holes_tri_flux"}:
            complex_ = triangulated_square_with_holes(args.nx, args.ny)
        else:
            complex_ = grid_complex(args.nx, args.ny)
        ops = DECOperators.from_complex(complex_, device=device)
        if args.task == "poisson":
            train_data = PoissonDataset(complex_, args.train_samples, seed=args.seed)
            val_data = PoissonDataset(complex_, args.val_samples, seed=args.seed + 1)
        elif args.task == "darcy":
            train_data = AnisotropicDarcyDataset(
                complex_,
                args.train_samples,
                seed=args.seed,
                min_solution_norm=args.darcy_min_solution_norm,
                vertex_projection=args.darcy_vertex_projection,
                orientation_mode=args.darcy_orientation,
                orientation_blobs=args.darcy_orientation_blobs,
                orientation_sigma=args.darcy_orientation_sigma,
            )
            val_data = AnisotropicDarcyDataset(
                complex_,
                args.val_samples,
                seed=args.seed + 1,
                min_solution_norm=args.darcy_min_solution_norm,
                vertex_projection=args.darcy_vertex_projection,
                orientation_mode=args.darcy_orientation,
                orientation_blobs=args.darcy_orientation_blobs,
                orientation_sigma=args.darcy_orientation_sigma,
            )
        else:
            train_data = darcy_holes_dataset(args, complex_, args.train_samples, seed=args.seed)
            val_data = darcy_holes_dataset(args, complex_, args.val_samples, seed=args.seed + 1)
        train_op_loaders = [
            (
                ops,
                loader_for_dataset(train_data, args.batch_size, True, args.seed),
            )
        ]
        val_op_loaders = [
            (
                ops,
                loader_for_dataset(val_data, args.batch_size, False, args.seed + 1),
            )
        ]
        sample = train_data[0]
    vertex_in = sample.x0.shape[-1]
    edge_in = sample.x1.shape[-1]
    face_in = sample.x2.shape[-1]

    if args.model == "tno":
        model = TopologicalNeuralOperator(
            vertex_in=vertex_in,
            edge_in=edge_in,
            face_in=face_in,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            dropout=args.dropout,
            ablation=args.tno_ablation,
        ).to(device)
    else:
        model = VertexGraphOperator(
            vertex_in=vertex_in,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            dropout=args.dropout,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = math.inf
    args.save.parent.mkdir(parents=True, exist_ok=True)
    history_path = args.save.with_suffix(".history.json")
    history: list[dict[str, float | int | str]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_sample_rel, train_rel = run_epoch(
            model, train_op_loaders, optimizer, device
        )
        with torch.no_grad():
            val_loss, val_sample_rel, val_rel = run_epoch(
                model, val_op_loaders, None, device
            )
        is_best = val_rel < best_val
        record = {
            "epoch": epoch,
            "model": args.model,
            "task": args.task,
            "train_mse": train_loss,
            "train_rel_l2": train_rel,
            "train_sample_rel_l2": train_sample_rel,
            "val_mse": val_loss,
            "val_rel_l2": val_rel,
            "val_sample_rel_l2": val_sample_rel,
            "is_best": is_best,
        }
        history.append(record)
        history_path.write_text(
            json.dumps(
                {
                    "args": run_args,
                    "metric": "aggregate_relative_l2",
                    "best_val_relative_l2": min(best_val, val_rel),
                    "history": history,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if is_best:
            best_val = val_rel
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": run_args,
                    "best_val_relative_l2": best_val,
                    "metric": "aggregate_relative_l2",
                    "history_path": str(history_path),
                },
                args.save,
            )
        print(
            f"epoch={epoch:03d} "
            f"model={args.model} "
            f"task={args.task} "
            f"tno_ablation={args.tno_ablation} "
            f"train_mse={train_loss:.6e} train_rel_l2={train_rel:.4f} "
            f"train_sample_rel_l2={train_sample_rel:.4f} "
            f"val_mse={val_loss:.6e} val_rel_l2={val_rel:.4f} "
            f"val_sample_rel_l2={val_sample_rel:.4f}"
        )

    print(f"best_val_relative_l2={best_val:.4f}")
    print(f"saved={args.save}")


if __name__ == "__main__":
    main()
