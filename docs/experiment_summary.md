# Experiment Summary

Generated from checkpoint metadata and, when present, `*.history.json` files.
Lower relative L2 is better. Checkpoints are selected by aggregate validation relative L2.

## Runs

| Experiment | Model | Best val rel L2 | Final val rel L2 | Final sample rel L2 | Final val MSE | Artifact |
|---|---:|---:|---:|---:|---:|---|
| darcy/orientation=blobs/vertex_projection=none | tno | 0.1865 | 0.1870 | 0.2741 | 0.0736 | `outputs/tno_darcy_blobs_no_vertex_projection.pt` |
| darcy/orientation=blobs/vertex_projection=none | vertex | 0.2508 | 0.2594 | 0.3482 | 0.1417 | `outputs/vertex_darcy_blobs_no_vertex_projection.pt` |
| darcy/orientation=iid/vertex_projection=mean | tno | 0.2021 | n/a | n/a | n/a | `outputs/tno_darcy.pt` |
| darcy/orientation=iid/vertex_projection=mean | vertex | 0.2183 | n/a | n/a | n/a | `outputs/vertex_darcy.pt` |
| darcy/orientation=iid/vertex_projection=none | tno | 0.1847 | n/a | n/a | n/a | `outputs/tno_darcy_no_vertex_projection.pt` |
| darcy/orientation=iid/vertex_projection=none | vertex | 0.1968 | n/a | n/a | n/a | `outputs/vertex_darcy_no_vertex_projection.pt` |
| poisson | tno | 0.1792 | n/a | n/a | n/a | `outputs/tno_poisson.pt` |
| poisson | vertex | 0.1988 | n/a | n/a | n/a | `outputs/vertex_poisson.pt` |

## Matched Comparisons

| Experiment | TNO best | Vertex best | TNO improvement | Notes |
|---|---:|---:|---:|---|
| darcy/orientation=blobs/vertex_projection=none | 0.1865 | 0.2508 | +25.6% | TNO better |
| darcy/orientation=iid/vertex_projection=mean | 0.2021 | 0.2183 | +7.4% | TNO better |
| darcy/orientation=iid/vertex_projection=none | 0.1847 | 0.1968 | +6.1% | TNO better |
| poisson | 0.1792 | 0.1988 | +9.8% | TNO better |

## Current Takeaways

- The multi-rank TNO consistently beats the vertex-only baseline in the completed matched runs so far.
- The Darcy relative-error metric needed aggregate normalization because per-sample relative L2 is unstable on near-zero target fields.
- Removing vertex-projected face orientation creates a cleaner test of native face cochains.
- The blob-orientation Darcy task is intended to test smoother, spatially coherent face coefficients.
