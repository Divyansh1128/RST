"""Execute every notebook code cell locally with the requested CPU smoke overrides.

Run from the repository root: .venv/bin/python notebooks/run_smoke.py
Downloads/artifacts go to /private/tmp/rst_smoke by default. This does not modify
notebook outputs or configuration. Dependencies must already be installed.
"""
import os, sys, time, json
from pathlib import Path
RUN_START = time.perf_counter()
sys.dont_write_bytecode = True
import nbformat

root = Path(__file__).resolve().parents[1]
os.environ.update(RST_SMOKE="1", RST_DEVICE="cpu", RST_SKIP_INSTALL="1")
os.environ.setdefault("RST_DATA_DIR", "/private/tmp/rst_smoke")
os.environ.setdefault("RST_ARTIFACT_DIR", "/private/tmp/rst_smoke/rst_artifacts")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rst_smoke/matplotlib")
namespace = {"__name__": "__main__"}
nb = nbformat.read(root / "notebooks/RST_train_association_model.ipynb", as_version=4)
nbformat.validate(nb)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        print(f"\nExecuting cell {i} ...", flush=True)
        exec(compile(cell.source, f"RST_train_association_model.ipynb:cell-{i}", "exec"), namespace)

# This check follows the notebook's schema/numerical contract validation.
sys.path.insert(0, str(root))
from server.nlp import AssociationModel
import numpy as np
model = AssociationModel(os.environ["RST_ARTIFACT_DIR"])
a, b = namespace["check_a"], namespace["check_b"]
actual = model._pair_prob(a, b)
expected = namespace["numpy_reference"]
error = float(np.max(np.abs(actual - expected)))
assert len(model.candidates("cat", k=10)) == 10
assert model.random_seed_word()[0] not in "rst"
if error < 1e-6:
    status = "loaded_with_model_parity"
    print("PASS: server.nlp.AssociationModel loads artifacts, 1000-pair parity max error", error)
else:
    status = "loaded_but_model_parity_failed"
    print("SERVER COMPATIBILITY: artifacts load, but current server scores differ; max error", error)
    print("Required formula: mlp(x) + w_cos*cos before Platt calibration; known-pair serving policy is separate.")
print("cat candidates:", model.candidates("cat", k=10))
metrics = namespace["metrics"]
metrics["verification"] = dict(device="cpu", server_loader=status,
    server_max_probability_error=error, runner_seconds=time.perf_counter()-RUN_START)
label = os.environ.get("RST_RUN_LABEL", "smoke" if namespace["CFG"]["STEPS"] == 150 else "trend")
assert label in {"smoke", "trend"}
result = root / "notebooks" / (label + "_metrics.json")
result.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
if label == "trend":
    val = metrics["val"]
    checks = {
        "val_recall10_beats_baseline": val["player"]["trained"]["recall@10"] > val["player"]["baseline"]["recall@10"],
        "val_mrr_beats_baseline": val["player"]["trained"]["mrr"] > val["player"]["baseline"]["mrr"],
        "val_judge_auc_beats_baseline": val["judge"]["trained"]["roc_auc"] > val["judge"]["baseline"]["roc_auc"],
    }
    metrics["verification"]["trend_checks"] = checks
    result.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print("Trend checks (reported honestly; no retries):", checks)
print("Retained metrics:", result, "; runner seconds:", metrics["verification"]["runner_seconds"])
