# Minimal Topological Neural Operator

This repository contains a small PyTorch prototype inspired by the paper
`2606.09806v1.pdf`, "Topological Neural Operators".

The goal is not to reproduce the full paper. This is the compact, rigid-DEC
version:

- build a 2D triangular cell complex,
- construct signed incidence matrices `B1` and `B2`,
- derive DEC routes `d0`, `d1`, `delta1`, `delta2`, and Hodge Laplacian pieces,
- train a residual neural operator over vertex, edge, and face cochains.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Smoke Train

```bash
.venv/bin/python train.py --epochs 20 --train-samples 128 --val-samples 32
```

The toy task learns a Poisson solution operator on a fixed triangulated square
with zero Dirichlet boundary conditions. Inputs live on vertices, edges, and
faces; the output is a vertex scalar field.

## Compare Against A Vertex Baseline

Train the multi-rank TNO:

```bash
.venv/bin/python train.py --model tno --epochs 100 --train-samples 512 --val-samples 128 --save outputs/tno_poisson.pt
```

Train a rank-0 vertex-only graph baseline:

```bash
.venv/bin/python train.py --model vertex --epochs 100 --train-samples 512 --val-samples 128 --save outputs/vertex_poisson.pt
```

The vertex baseline uses the same mesh and training loop, but only consumes
vertex features and the graph Laplacian induced by `B1`. It ignores edge and
face cochains.

Training reports two relative errors:

- `*_rel_l2`: aggregate relative L2, `sqrt(sum ||error||^2 / sum ||target||^2)`.
- `*_sample_rel_l2`: mean of per-sample relative L2 values.

Checkpoints are selected with aggregate relative L2 because per-sample relative
errors become unstable when a generated target field has a tiny norm.

## Face-Native Anisotropic Darcy Task

This task makes the coefficient field live naturally on faces. Each triangle gets
a random anisotropic conductivity tensor orientation, the finite-element system is
assembled with that face tensor, and the model predicts the vertex solution.

The TNO receives the face tensor channels directly. The vertex baseline receives
only a vertex-averaged projection of the same face orientation signal.

```bash
.venv/bin/python train.py --task darcy --model tno --epochs 100 --train-samples 512 --val-samples 128 --save outputs/tno_darcy.pt
.venv/bin/python train.py --task darcy --model vertex --epochs 100 --train-samples 512 --val-samples 128 --save outputs/vertex_darcy.pt
```

By default, Darcy samples with solution norm below `0.5` are rejected to avoid
near-zero targets dominating `sample_rel_l2`. You can change this with
`--darcy-min-solution-norm`.

To test whether native face cochains help when the vertex baseline does not get
a projected face-orientation shortcut, disable the Darcy vertex projection:

```bash
.venv/bin/python train.py --task darcy --darcy-vertex-projection none --model tno --epochs 100 --train-samples 512 --val-samples 128 --save outputs/tno_darcy_no_vertex_projection.pt
.venv/bin/python train.py --task darcy --darcy-vertex-projection none --model vertex --epochs 100 --train-samples 512 --val-samples 128 --save outputs/vertex_darcy_no_vertex_projection.pt
```

The default orientation field is iid per face. To test smoother, spatially
coherent conductivity structure, use Gaussian orientation blobs:

```bash
.venv/bin/python train.py --task darcy --darcy-orientation blobs --darcy-vertex-projection none --model tno --epochs 100 --train-samples 512 --val-samples 128 --save outputs/tno_darcy_blobs_no_vertex_projection.pt
.venv/bin/python train.py --task darcy --darcy-orientation blobs --darcy-vertex-projection none --model vertex --epochs 100 --train-samples 512 --val-samples 128 --save outputs/vertex_darcy_blobs_no_vertex_projection.pt
```

## Summarize Results

Each new training run writes a checkpoint plus a sidecar history file:

```text
outputs/name.pt
outputs/name.history.json
```

Generate or refresh the Markdown experiment report with:

```bash
.venv/bin/python scripts/summarize_results.py
```

The report is written to `docs/experiment_summary.md`. By default, smoke-test
checkpoints are ignored; add `--include-smoke` if you want to include them.

## Run Repeat-Seed Sweeps

Use the sweep runner to launch paired TNO/vertex experiments across seeds and
refresh the summary report automatically:

```bash
.venv/bin/python scripts/run_experiments.py --experiments darcy_blobs_none --seeds 7,8,9
```

Available experiment names are:

- `poisson`
- `darcy_iid_mean`
- `darcy_iid_none`
- `darcy_blobs_none`
- `darcy_holes_blobs_none`
- `darcy_holes_flux_blobs_none`

Check the runner on tiny settings first with:

```bash
.venv/bin/python scripts/run_experiments.py --quick --experiments darcy_blobs_none --seeds 1 --overwrite
```

## Topological Darcy With Holes

The first topology-aware dataset uses a fixed square mesh with two circular holes.
The outer boundary is fixed to `u=0`; each hole boundary receives a random
Dirichlet value per sample. The coefficient field remains face-native.

```bash
.venv/bin/python train.py --task darcy_holes --darcy-orientation blobs --darcy-vertex-projection none --model tno --epochs 100 --train-samples 512 --val-samples 128 --save outputs/tno_darcy_holes_blobs_no_vertex_projection.pt
.venv/bin/python train.py --task darcy_holes --darcy-orientation blobs --darcy-vertex-projection none --model vertex --epochs 100 --train-samples 512 --val-samples 128 --save outputs/vertex_darcy_holes_blobs_no_vertex_projection.pt
```

For repeated seeds:

```bash
.venv/bin/python scripts/run_experiments.py --experiments darcy_holes_blobs_none --seeds 7,8,9
```

## Multi-Rank Darcy Holes: Potential Plus Edge Flux

The next fixed-complex task predicts both the vertex potential `u0` and an
edge-supported flux/circulation target `u1`. The TNO decodes `u1` from hidden
edge cochains; the vertex baseline decodes it from graph-gradient edge features.

```bash
.venv/bin/python train.py --task darcy_holes_flux --darcy-orientation blobs --darcy-vertex-projection none --model tno --epochs 100 --train-samples 512 --val-samples 128 --save outputs/tno_darcy_holes_flux_blobs_no_vertex_projection.pt
.venv/bin/python train.py --task darcy_holes_flux --darcy-orientation blobs --darcy-vertex-projection none --model vertex --epochs 100 --train-samples 512 --val-samples 128 --save outputs/vertex_darcy_holes_flux_blobs_no_vertex_projection.pt
```

For repeated seeds:

```bash
.venv/bin/python scripts/run_experiments.py --experiments darcy_holes_flux_blobs_none --seeds 7,8,9
```

## Project Layout

```text
topo/
  complex.py  # mesh and incidence construction
  dec.py      # sparse DEC operators
  data.py     # synthetic Poisson dataset
  tno.py      # rigid DEC TNO model
train.py      # training entry point
```
