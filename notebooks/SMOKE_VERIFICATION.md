# Review 1 — local verification

Verified on 2026-11-04 with the project `.venv`, CPU, Python 3.12.7.
All code cells of the generated notebook executed successfully for the 150-step smoke
run, followed by **one** 1,500-step trend run. Both used the same real GloVe 100d,
EAT/USF data, vocabulary, splits, legal-word masks, seeds and fixed 400-cue selection
subset. ConceptNet is disabled in smoke mode. No artifacts or zip files are retained.
I did not edit `server/` or `web/`, and made no commit.

Commands, from the repository root (existing venv dependencies were sufficient):

```sh
.venv/bin/python notebooks/build_notebook.py
.venv/bin/python -u notebooks/run_smoke.py
RST_STEPS=1500 RST_PATIENCE=100 RST_RUN_LABEL=trend \
  RST_ARTIFACT_DIR=/private/tmp/rst_smoke/trend_artifacts \
  .venv/bin/python -u notebooks/run_smoke.py
```

The runner forces CPU and smoke mode, skips redundant pip installation, and places
all downloads, NLTK corpora, plotting caches and artifacts under
`/private/tmp/rst_smoke/`. Numeric overrides are applied after smoke presets.
Trend patience was raised to 100 solely to complete all 1,500 steps; checkpoint
selection still used the prescribed criterion. The generated notebook defaults
remain **glove300, VOCAB=30000, STEPS=4000, PATIENCE=8, fixed tau=0.07**.
Its opening markdown gives the T4/Run all instructions; its final cell shows a
validation/test summary and offers the verified zip for download.

## 150-step smoke results

| Split | Model | Judge ROC-AUC | Judge PR-AUC | Brier | Player Recall@10 | Player MRR |
|---|---|---:|---:|---:|---:|---:|
| val | baseline | 0.728850 | 0.644666 | 0.183465 | 0.089613 | 0.530384 |
| val | trained | 0.724868 | 0.650569 | 0.183439 | 0.116802 | 0.634996 |
| test | baseline | 0.729567 | 0.636419 | 0.184022 | 0.092134 | 0.548843 |
| test | trained | 0.728123 | 0.642771 | 0.183539 | 0.119349 | 0.636002 |

All 150 steps ran; the selected checkpoint was step **100**.
Runner wall time was **72.98 s**, with cached data;
the pipeline timer after imports measured **62.84 s**,
and training plus periodic validation took **9.61 s**.
The player beats the baseline at this short checkpoint, while the judge AUC is
slightly lower. This is a pipeline check, not the final training budget.

## Single 1,500-step trend check

| Split | Model | Judge ROC-AUC | Judge PR-AUC | Brier | Player Recall@10 | Player MRR |
|---|---|---:|---:|---:|---:|---:|
| val | baseline | 0.728850 | 0.644666 | 0.183465 | 0.089613 | 0.530384 |
| val | trained | 0.833412 | 0.775005 | 0.143799 | 0.136736 | 0.685294 |
| test | baseline | 0.729567 | 0.636419 | 0.184022 | 0.092134 | 0.548843 |
| test | trained | 0.843054 | 0.777501 | 0.141522 | 0.142482 | 0.686239 |

All 1,500 steps ran; the selected checkpoint was step **1400**.
Runner wall time was **92.81 s**; the pipeline timer
measured **84.11 s**, and training plus validation
took **52.71 s**. There were no additional trend runs or retries.
The selected model beats raw GloVe on **all three requested validation metrics**:
Recall@10, MRR and judge AUC. Test metrics also improve, without selecting on test.

## Model selection and vocabulary

The player has zero-initialized `Linear(dim_z, dim_e, bias=False)` maps and computes
`q=normalize(e+Wq(z))`, `k=normalize(e+Wk(z))`. The step-zero q/k vectors matched
normalized GloVe to maximum error **8.94069672e-08**.
Step-zero validation-subset Recall@10 and MRR matched the GloVe baseline.
A separate untrained 300d shape check passed (`E/Q/K=300`, `Z=128`) and confirmed
zero map weights and a trainable cosine scalar initialized to 5.0.

Every 50 smoke steps, model selection computes validation judge AUC and player
Recall@10 on the same 400 randomly selected cues against the complete legal vocab.
The score is **val_auc + val_recall10**, with the initial checkpoint eligible too;
patience counts evaluations without a score improvement greater than 0.0001.
The subset cue indices and score history are retained in each metrics file.
Final reported player metrics cover **all 723 validation and 712 test cues**.
Judge metrics use 1,000 positives, 1,000 random negatives and 1,000 hard negatives
per split. All metrics are **model-only**, with no attested-pair acceptance lookup.
PR-AUC is average precision. Full Recall@5/50, Brier, false-reject rates at
validation-selected 5%/10% FAR, sanity tables and threshold rates are in the JSON files.

The semantic vocabulary contains **14,000 words** (4,000 base + 10,000 extensions),
with **12,796 playable**, **9,871 legal output words**, and **1,500 seed nouns**.
The extension cap excludes 6,719 otherwise eligible pair words. Both the base and
appended extension are sorted by descending Zipf, and the entire vocabulary,
including the boundary, passed an independent order check. Index zero is `the`.
Manual function-word exclusions and adverb-only/common mixed-POS restrictions
apply only to bot playability; the words remain in semantic vocabulary.

Train/validation/test pairs remain **176,770 / 25,419 / 24,792**. Strict splitting
removed **40,428** training edges incident to held-out cue families.

## Revised export contract

Both runs passed shape, dtype, normalization and judge-symmetry checks:

- `emb.npy`, `q.npy`, `k.npy`: float16 `[14000, 100]`; `z.npy`: float16 `[14000, 128]`.
  Metadata records `dim_e=100`, `dim_z=128`, **`dim_q=100`**.
- `judge.npz` retains transposed float32 MLP weights and adds **`w_cos [1]`**.
  Metadata records **`judge_formula: "mlp(x) + w_cos*cos"`**. The learned scalar
  was 4.944992 for smoke and 4.981092 for trend.
- `known_pairs.npy`: int32 **[482053, 2]**, deduplicated, both directions; metadata
  records **`known_pairs_count=482053`**. An independent audit against the raw
  EAT/USF JSON confirmed exact coverage across all sources/splits, including
  strict-split removals, inflected same-lemma pairs and **445 self-pairs**.
  Training still excludes same-lemma pairs; gameplay must apply repeat rules before
  any attested-pair acceptance. ConceptNet uses the same all-source export path
  in full mode. The lookup is never used in model evaluation.

The NumPy-only judge matched PyTorch on 1,000 random pairs with maximum probability
error **0.000118673** (smoke) and
**0.000201344** (trend), both below **0.001**.
After these contract checks, the current `server.nlp.AssociationModel` loaded both
exports and matched the notebook NumPy scores exactly: maximum error **0.0** in
both cases. The server now implements the cosine skip and optional attested-pair
lookup; I only read/tested it. Seed generation and ten `cat` candidates also passed.

The exact notebook ConceptNet parser additionally passed synthetic checks for
weighted and absent-weight schemas, both-end English filtering, multiword and
`HasContext` rejection, and self-response preservation. This was not a full
ConceptNet-data run.

## Remaining limits and cleanup

Full 300d/T4 training, CUDA AMP and MPS remain unmeasured; the <25-minute full T4
target is not established by these CPU smoke timings. Negative labels remain
unreviewed proxies for human judgments. USF deployment licensing and EAT original
provenance remain unresolved, with source/license caveats preserved in metadata.
The expected Matplotlib Agg display warning did not affect saved plots or exports.

Notebook JSON/schema and all Python cell syntax passed validation; regenerating
from the generator produced identical notebook bytes. The temporary download
directory, both exported artifact directories and both zip files were deleted
after verification. Retained files are the notebook/generator/runner, this report,
[smoke metrics](smoke_metrics.json), [smoke log](smoke_run.log),
[trend metrics](trend_metrics.json) and [trend log](trend_run.log).
