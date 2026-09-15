# FTDiff status

Independent QuickVina / QED / SA on held-out test pockets.
Do not treat training-reward improvement as success by itself.

| Metric | Unguided DiffSBDD | FTDiff |
|---|---:|---:|
| Mean Vina | -4.051 | -1.409 |
| Validity | 39.4% | 94.4% |
| QED | 0.459 | 0.472 |
| SA | 0.675 | 0.852 |
| Diversity | 0.919 | 0.794 |
| Mean reward | 0.573 | 1.383 |

Paired mean Vina delta (FTDiff - unguided): 2.642. Negative is better.

Independent Vina did not improve. Do not claim FTDiff success.
