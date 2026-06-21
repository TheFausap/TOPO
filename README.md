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

## Project Layout

```text
topo/
  complex.py  # mesh and incidence construction
  dec.py      # sparse DEC operators
  data.py     # synthetic Poisson dataset
  tno.py      # rigid DEC TNO model
train.py      # training entry point
```
