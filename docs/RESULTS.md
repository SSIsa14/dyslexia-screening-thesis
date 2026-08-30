# Results summary

## Principal configurations

| Configuration | Threshold | Precision | Recall | F2 | AUPRC | Referral rate |
|---|---:|---:|---:|---:|---:|---:|
| RF without oversampling | 0.250 ± 0.012 | 0.306 ± 0.013 | 0.774 ± 0.041 | **0.592 ± 0.015** | **0.464 ± 0.020** | 0.272 ± 0.024 |
| RF with oversampling | 0.216 ± 0.005 | 0.279 ± 0.012 | 0.791 ± 0.025 | 0.579 ± 0.021 | 0.456 ± 0.029 | 0.304 ± 0.006 |
| LR without oversampling | 0.506 ± 0.029 | 0.287 ± 0.013 | 0.704 ± 0.038 | 0.545 ± 0.020 | 0.388 ± 0.027 | 0.263 ± 0.019 |
| LR with oversampling | 0.448 ± 0.072 | 0.259 ± 0.029 | 0.750 ± 0.046 | 0.543 ± 0.040 | 0.378 ± 0.031 | 0.313 ± 0.036 |
| NN with oversampling | 0.374 ± 0.121 | 0.296 ± 0.043 | 0.669 ± 0.041 | 0.531 ± 0.030 | 0.412 ± 0.064 | 0.246 ± 0.035 |
| Flag all | — | 0.107 | 1.000 | 0.375 | 0.107 | 1.000 |

## Interpretation

RF without oversampling is the main recommendation because it achieved the highest mean F2 and AUPRC under the declared protocol. NN without oversampling achieved higher Recall but referred approximately 73.1% of all records and remained close to the flag-all F2 baseline, illustrating why Recall alone is not an adequate selection criterion.

The exact paired McNemar comparison between RF and LR without oversampling produced `p = 0.276`. The repository therefore does not claim universal statistical superiority; the recommendation is conditional on the F2 objective and referral-workload profile.

Complete results are available under `results/tables/`, and record-level out-of-fold predictions are under `results/predictions/`.

