# Experiment Summary

Generated from checkpoint metadata and, when present, `*.history.json` files.
Lower relative L2 is better. Checkpoints are selected by aggregate validation relative L2.
Equivalent artifacts are deduplicated by experiment, model, and seed.

## Runs

| Experiment | Model | Best val rel L2 | Final val rel L2 | Final sample rel L2 | Final val MSE | Artifact |
|---|---:|---:|---:|---:|---:|---|
| darcy/orientation=blobs/vertex_projection=none | tno | 0.1865 | 0.1870 | 0.2741 | 0.0736 | `outputs/darcy_blobs_none_tno_seed7.pt` |
| darcy/orientation=blobs/vertex_projection=none | tno | 0.1825 | 0.1834 | 0.2915 | 0.0709 | `outputs/darcy_blobs_none_tno_seed8.pt` |
| darcy/orientation=blobs/vertex_projection=none | tno | 0.1738 | 0.1824 | 0.3060 | 0.0661 | `outputs/darcy_blobs_none_tno_seed9.pt` |
| darcy/orientation=blobs/vertex_projection=none | vertex | 0.2508 | 0.2594 | 0.3482 | 0.1417 | `outputs/darcy_blobs_none_vertex_seed7.pt` |
| darcy/orientation=blobs/vertex_projection=none | vertex | 0.2459 | 0.2750 | 0.3895 | 0.1592 | `outputs/darcy_blobs_none_vertex_seed8.pt` |
| darcy/orientation=blobs/vertex_projection=none | vertex | 0.2332 | 0.2374 | 0.3225 | 0.1121 | `outputs/darcy_blobs_none_vertex_seed9.pt` |
| darcy/orientation=iid/vertex_projection=mean | tno | 0.2021 | n/a | n/a | n/a | `outputs/tno_darcy.pt` |
| darcy/orientation=iid/vertex_projection=mean | vertex | 0.2183 | n/a | n/a | n/a | `outputs/vertex_darcy.pt` |
| darcy/orientation=iid/vertex_projection=none | tno | 0.1847 | n/a | n/a | n/a | `outputs/tno_darcy_no_vertex_projection.pt` |
| darcy/orientation=iid/vertex_projection=none | vertex | 0.1968 | n/a | n/a | n/a | `outputs/vertex_darcy_no_vertex_projection.pt` |
| darcy_holes/orientation=blobs/vertex_projection=none | tno | 0.0894 | 0.0929 | 0.0895 | 0.0036 | `outputs/darcy_holes_blobs_none_tno_seed7.pt` |
| darcy_holes/orientation=blobs/vertex_projection=none | tno | 0.0882 | 0.0882 | 0.0853 | 0.0031 | `outputs/darcy_holes_blobs_none_tno_seed8.pt` |
| darcy_holes/orientation=blobs/vertex_projection=none | tno | 0.0972 | 0.0972 | 0.0866 | 0.0038 | `outputs/darcy_holes_blobs_none_tno_seed9.pt` |
| darcy_holes/orientation=blobs/vertex_projection=none | vertex | 0.1821 | 0.1870 | 0.1802 | 0.0145 | `outputs/darcy_holes_blobs_none_vertex_seed7.pt` |
| darcy_holes/orientation=blobs/vertex_projection=none | vertex | 0.1899 | 0.1905 | 0.1792 | 0.0144 | `outputs/darcy_holes_blobs_none_vertex_seed8.pt` |
| darcy_holes/orientation=blobs/vertex_projection=none | vertex | 0.1937 | 0.2045 | 0.1927 | 0.0169 | `outputs/darcy_holes_blobs_none_vertex_seed9.pt` |
| darcy_holes_flux/orientation=blobs/vertex_projection=none | tno | 0.1542 | 0.1559 | 0.1518 | 0.0069 | `outputs/darcy_holes_flux_blobs_none_tno_seed7.pt` |
| darcy_holes_flux/orientation=blobs/vertex_projection=none | tno | 0.1598 | 0.1598 | 0.1531 | 0.0068 | `outputs/darcy_holes_flux_blobs_none_tno_seed8.pt` |
| darcy_holes_flux/orientation=blobs/vertex_projection=none | tno | 0.1639 | 0.1674 | 0.1524 | 0.0078 | `outputs/darcy_holes_flux_blobs_none_tno_seed9.pt` |
| darcy_holes_flux/orientation=blobs/vertex_projection=none | vertex | 0.4309 | 0.4349 | 0.4238 | 0.0447 | `outputs/darcy_holes_flux_blobs_none_vertex_seed7.pt` |
| darcy_holes_flux/orientation=blobs/vertex_projection=none | vertex | 0.4518 | 0.4570 | 0.4394 | 0.0478 | `outputs/darcy_holes_flux_blobs_none_vertex_seed8.pt` |
| darcy_holes_flux/orientation=blobs/vertex_projection=none | vertex | 0.4450 | 0.4456 | 0.4222 | 0.0462 | `outputs/darcy_holes_flux_blobs_none_vertex_seed9.pt` |
| poisson | tno | 0.1792 | n/a | n/a | n/a | `outputs/tno_poisson.pt` |
| poisson | vertex | 0.1988 | n/a | n/a | n/a | `outputs/vertex_poisson.pt` |

## Seed Aggregates

| Experiment | Seeds | TNO best mean | Vertex best mean | TNO improvement |
|---|---:|---:|---:|---:|
| darcy/orientation=blobs/vertex_projection=none | 7, 8, 9 | 0.1809 +/- 0.0065 | 0.2433 +/- 0.0091 | +25.6% |
| darcy/orientation=iid/vertex_projection=mean | 7 | 0.2021 | 0.2183 | +7.4% |
| darcy/orientation=iid/vertex_projection=none | 7 | 0.1847 | 0.1968 | +6.1% |
| darcy_holes/orientation=blobs/vertex_projection=none | 7, 8, 9 | 0.0916 +/- 0.0049 | 0.1886 +/- 0.0059 | +51.4% |
| darcy_holes_flux/orientation=blobs/vertex_projection=none | 7, 8, 9 | 0.1593 +/- 0.0049 | 0.4426 +/- 0.0107 | +64.0% |
| poisson | 7 | 0.1792 | 0.1988 | +9.8% |

## Matched Comparisons

| Experiment | TNO best | Vertex best | TNO improvement | Notes |
|---|---:|---:|---:|---|
| darcy/orientation=blobs/vertex_projection=none | 0.1865 | 0.2508 | +25.6% | TNO better |
| darcy/orientation=blobs/vertex_projection=none | 0.1825 | 0.2459 | +25.8% | TNO better |
| darcy/orientation=blobs/vertex_projection=none | 0.1738 | 0.2332 | +25.5% | TNO better |
| darcy/orientation=iid/vertex_projection=mean | 0.2021 | 0.2183 | +7.4% | TNO better |
| darcy/orientation=iid/vertex_projection=none | 0.1847 | 0.1968 | +6.1% | TNO better |
| darcy_holes/orientation=blobs/vertex_projection=none | 0.0894 | 0.1821 | +50.9% | TNO better |
| darcy_holes/orientation=blobs/vertex_projection=none | 0.0882 | 0.1899 | +53.5% | TNO better |
| darcy_holes/orientation=blobs/vertex_projection=none | 0.0972 | 0.1937 | +49.8% | TNO better |
| darcy_holes_flux/orientation=blobs/vertex_projection=none | 0.1542 | 0.4309 | +64.2% | TNO better |
| darcy_holes_flux/orientation=blobs/vertex_projection=none | 0.1598 | 0.4518 | +64.6% | TNO better |
| darcy_holes_flux/orientation=blobs/vertex_projection=none | 0.1639 | 0.4450 | +63.2% | TNO better |
| poisson | 0.1792 | 0.1988 | +9.8% | TNO better |

## Current Takeaways

- The multi-rank TNO consistently beats the vertex-only baseline in the completed matched runs so far.
- The Darcy relative-error metric needed aggregate normalization because per-sample relative L2 is unstable on near-zero target fields.
- Removing vertex-projected face orientation creates a cleaner test of native face cochains.
- The holed Darcy task shows a stronger topology-aware signal than the simply connected square.
- The multi-rank holed Darcy task, predicting vertex potential plus edge flux, shows the strongest result so far.
