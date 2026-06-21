from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from topo.complex import grid_complex
from topo.data import AnisotropicDarcyDataset, PoissonDataset, poisson_collate
from topo.dec import DECOperators
from topo.tno import TopologicalNeuralOperator, VertexGraphOperator


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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task", choices=["poisson", "darcy"], default="poisson")
    parser.add_argument(
        "--darcy-min-solution-norm",
        type=float,
        default=0.5,
        help="Reject generated Darcy samples whose solution norm is below this value.",
    )
    parser.add_argument(
        "--darcy-vertex-projection",
        choices=["mean", "none"],
        default="mean",
        help="How face orientation channels are projected into Darcy vertex inputs.",
    )
    parser.add_argument(
        "--darcy-orientation",
        choices=["iid", "blobs"],
        default="iid",
        help="How per-face Darcy conductivity orientations are generated.",
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


def sample_relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.vector_norm(pred - target, dim=(-2, -1))
    denominator = torch.linalg.vector_norm(target, dim=(-2, -1)).clamp_min(1e-8)
    return (numerator / denominator).mean()


def run_epoch(
    model: TopologicalNeuralOperator,
    ops: DECOperators,
    loader: DataLoader,
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

    for x0, x1, x2, y0 in loader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)
        y0 = y0.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        pred = model(ops, x0, x1, x2)
        loss = torch.nn.functional.mse_loss(pred, y0)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch = x0.shape[0]
        total_loss += float(loss.detach()) * batch
        total_sample_rel += float(sample_relative_l2(pred.detach(), y0)) * batch
        total_error_sq += float(torch.sum((pred.detach() - y0) ** 2))
        total_target_sq += float(torch.sum(y0**2))
        total_items += batch

    aggregate_rel = math.sqrt(total_error_sq / max(total_target_sq, 1e-12))
    return total_loss / total_items, total_sample_rel / total_items, aggregate_rel


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")
    run_args = jsonable_args(args)

    complex_ = grid_complex(args.nx, args.ny)
    ops = DECOperators.from_complex(complex_, device=device)
    if args.task == "poisson":
        train_data = PoissonDataset(complex_, args.train_samples, seed=args.seed)
        val_data = PoissonDataset(complex_, args.val_samples, seed=args.seed + 1)
    else:
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
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=poisson_collate,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=poisson_collate,
    )

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
            model, ops, train_loader, optimizer, device
        )
        with torch.no_grad():
            val_loss, val_sample_rel, val_rel = run_epoch(model, ops, val_loader, None, device)
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
            f"train_mse={train_loss:.6e} train_rel_l2={train_rel:.4f} "
            f"train_sample_rel_l2={train_sample_rel:.4f} "
            f"val_mse={val_loss:.6e} val_rel_l2={val_rel:.4f} "
            f"val_sample_rel_l2={val_sample_rel:.4f}"
        )

    print(f"best_val_relative_l2={best_val:.4f}")
    print(f"saved={args.save}")


if __name__ == "__main__":
    main()
