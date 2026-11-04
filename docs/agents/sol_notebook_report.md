Created the [Colab notebook](notebooks/RST_train_association_model.ipynb) and [generator](notebooks/build_notebook.py).

Ran all cells twice in the project venv, CPU smoke mode, 150 steps. Cached execution took **57 seconds**.

| Test metric | GloVe | Trained |
|---|---:|---:|
| Judge ROC-AUC | 0.7296 | 0.7404 |
| Judge PR-AUC | 0.6364 | 0.6524 |
| Player Recall@10 | 0.0909 | 0.0635 |
| Player MRR | 0.5530 | 0.4163 |

Export checks passed: maximum NumPy/PyTorch probability error **0.000126**. `server.nlp.AssociationModel` loaded successfully with exact NumPy score parity. ConceptNet parser checks also passed.

The smoke player underperformed GloVe. Full 300d/T4 timing and CUDA AMP remain unverified; licensing caveats are documented. See the [verification report](notebooks/SMOKE_VERIFICATION.md) for complete results.

Temporary downloads and smoke artifacts were deleted. No `server/` or `web/` source changes; no commit.