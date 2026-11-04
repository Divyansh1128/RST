Implemented all eight changes and regenerated the [notebook](notebooks/RST_train_association_model.ipynb).

150-step smoke — validation:

| Metric | Baseline | Trained |
|---|---:|---:|
| Judge AUC | 0.7289 | 0.7249 |
| Recall@10 | 0.0896 | 0.1168 |
| MRR | 0.5304 | 0.6350 |

Single 1,500-step trend run — validation:

| Metric | Baseline | Trained |
|---|---:|---:|
| Judge AUC | 0.7289 | **0.8334** |
| Recall@10 | 0.0896 | **0.1367** |
| MRR | 0.5304 | **0.6853** |

Both runs passed artifact checks and matched the current server loader exactly. Maximum NumPy/PyTorch error was `0.000201`, below `0.001`. The lookup contains **482,053** pairs.

Updated [verification report](notebooks/SMOKE_VERIFICATION.md) includes both validation/test tables, timings and limitations. Full T4 timing remains unverified.

Temporary data and artifacts were deleted. I did not edit `server/` or `web/`, and made no commit.