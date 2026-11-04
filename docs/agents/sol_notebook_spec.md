# Sol task — Colab training notebook for the RST association model

Repo: . (read `docs/agents/astra_research_report.md` first — it is the research basis; `server/nlp.py` is the serving code your artifacts must load into).

## Deliverable
`notebooks/RST_train_association_model.ipynb` — runs top-to-bottom on Google Colab (T4 GPU, ~12 GB RAM; must also work on CPU runtime, just slower). Build the .ipynb programmatically (e.g. write a small generator script `notebooks/build_notebook.py` using `nbformat`, then run it) so it is valid JSON. Markdown cells should explain each step briefly (this is a student research project; the notebook is shown to evaluators).

## Pipeline (one section per step)
1. **Setup**: `pip install -q wordfreq nltk`; nltk download wordnet, omw-1.4. A `CFG` dict at the top: `EMB="glove300"` (alt `"glove100"`), `VOCAB=30000`, `SEED=13`, `SMOKE=False`, batch sizes, lrs, steps. Fix all seeds. Detect device (cuda/mps/cpu).
2. **Downloads** (to `/content/data`, cached, skip if present), using Astra's URLs:
   - GloVe 6B 300d text (`https://huggingface.co/kcz358/glove/resolve/main/glove.6B/glove.6B.300d.txt`) or 100d gz (`https://huggingface.co/datasets/SLU-CSCI4750/glove.6B.100d.txt/resolve/main/glove.6B.100d.txt.gz`). Stream the file and keep only vocabulary rows.
   - EAT json `https://raw.githubusercontent.com/dariusk/ea-thesaurus/master/ea-thesaurus.json`
   - USF json `https://raw.githubusercontent.com/jmichaelov/easy-fas/main/usffan_dict.json`
   - ConceptNet English essentials CSV `https://huggingface.co/datasets/openworld-domains/conceptnet-full-en-essentials/resolve/main/conceptnet-full-en-essentials.csv` (stream/chunk with pandas; inspect columns at runtime and write the parser defensively — print the head). Keep single-word, alphabetic, both-ends-English edges with relations RelatedTo, Synonym, Antonym, IsA, PartOf, HasA, AtLocation, UsedFor, CapableOf, HasProperty, MadeOf, Causes, HasPrerequisite, Desires, DistinctFrom, SimilarTo, LocatedNear, HasContext(skip), etc.
   - Bad words list `https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en`
   - Inspect the actual JSON structures of EAT/USF at runtime (print a sample) and parse robustly; handle case/whitespace, drop multiword responses.
3. **Vocabulary**: wordfreq `top_n_list('en', 80000)` ∩ GloVe ∩ alphabetic ascii, len ≥ 2 → keep first `VOCAB`. Make sure every word that appears in a cue/response pair and is in GloVe + wordfreq top 80k is included even if beyond the cut (cap total at VOCAB+10000). Lemma map with WordNetLemmatizer (noun then verb then adj; only map if lemma is itself in vocab). `playable` = in vocab, len ≥ 3, not stopword (nltk stopwords + a few: also, would, could...), not in bad-word list, has a WordNet synset, wordfreq zipf ≥ 2.5. `seeds` = ~1500 playable common concrete-ish nouns (WordNet noun as first sense, zipf ≥ 3.5) not starting with r/s/t. Blocked words (bad list) remain in vocab for semantics but are never playable.
4. **Pairs + split**: build directed weighted pairs (cue → response, weight = normalized count per cue; ConceptNet edges weight from its weight column if present else 1, down-weighted ×0.5 vs human norms; add reverse edge at ×0.5 for symmetric relations). Split **by cue lemma** 80/10/10 *before* augmentation; for val/test cues remove all training edges incident to them (strict). Print dataset stats table.
5. **Model** (PyTorch), exactly per Astra §C but with 300d init:
   - `E`: nn.Embedding init from GloVe (L2-normalised rows), trainable at lr 1e-4, drift penalty 0.01·mean((E−E0)²).
   - `enc`: MLP d→256→128 (ReLU, dropout 0.1), output `z = normalize(.)`. **Use dim names so serving matches.**
   - Judge: features `[z_a*z_b, |z_a−z_b|, cos(E_a,E_b)]` → Linear(2·Z+1, 64) → ReLU → Linear(64,32) → ReLU → Linear(32,1). Symmetric.
   - Player: `q = normalize(Wq z)`, `k = normalize(Wk z)` (Linear Z→Z, no bias), score = q·k / τ, τ = 0.07 learnable-or-fixed (state which).
   - Loss: multi-positive InfoNCE over in-batch + sampled negatives with known-positive masking (weights √count), + BCE for judge (positives = pairs; negatives = 1:1 random-vocab pairs + 1:1 "hard" = rank 300–3000 by raw GloVe cosine of the cue that are not known positives) + drift. AdamW, grad clip 1.0, cosine LR, early stopping on val judge AUC. Use AMP on cuda.
6. **Evaluation** on test cues (also report val): 
   - Judge: ROC-AUC, PR-AUC, Brier, and false-reject rate at 5% / 10% false-accept, for (a) baseline raw GloVe cosine (logistic-calibrated on val), (b) trained judge. 
   - Player: Recall@5/10/50, MRR against held-out responses over the full legal vocab (mask r/s/t initials, mask the cue's lemma), for baseline cosine vs trained q·k.
   - Sanity table: judge prob for ~30 hand-picked pairs (cat dog, cat purr, coffee morning, ocean wave, doctor hospital, bank river, bank money, hot cold, cat piano, apple galaxy, …) and top-10 bot candidates for ~8 cues.
   - Matplotlib: ROC curves baseline vs trained; training curves. Save a `metrics.json`.
7. **Calibration + thresholds**: fit 2-param logistic (Platt) on val judge logits. Thresholds: choose on val so false-accept rate (on random+hard negatives) is ≈ 15% (lenient), 8% (normal), 3% (strict); store the calibrated-prob thresholds.
8. **Export** to `/content/rst_artifacts/` then zip as `rst_artifacts.zip` and offer `google.colab.files.download` (guarded with try/except when not on Colab). Contract (server/nlp.py will be updated to this — keep it EXACT):
   - `meta.json`: `{"version": "rst-assoc-v1-<EMB>-<date>", "judge": "mlp", "dim_e": d, "dim_z": Z, "platt": [a, b], "thresholds": {"lenient": x, "normal": y, "strict": z}, "metrics": {...test metrics...}, "data": {sources + licenses}, "config": CFG}`
   - `vocab.json`: `{"words": [...], "lemma": {w: l}, "playable": [0/1...], "seeds": [...]}`
   - `emb.npy` float16 [N, d] — L2-normalised *fine-tuned* E (for the cosine feature)
   - `z.npy` float16 [N, Z], `q.npy` float16 [N, Z], `k.npy` float16 [N, Z] — all precomputed in eval mode, normalised
   - `judge.npz` float32: `W1 [2Z+1,64], b1 [64], W2 [64,32], b2 [32], W3 [32,1], b3 [1]` (transpose torch Linear weights so serving does `x @ W1 + b1`)
   - Judge serving formula = `sigmoid(platt_a * logit + platt_b)`.
   - Include a final cell that reloads the exported files **with numpy only** and reproduces torch judge probs to 1e-3 (fp16 tolerance) on 1000 random pairs — this guards the contract.
9. Runtime target: < 25 min on T4 with full config. `SMOKE=True` = glove100, VOCAB 4000, no ConceptNet, 150 steps — must finish in a few minutes on a laptop CPU.

## Verification you must do
- Run the generator, then execute the notebook locally in **SMOKE** mode (e.g. `jupyter nbconvert --execute` or by extracting code cells into a script) using the project venv `.venv` (install only what you need with `--no-cache-dir`; disk is tight: ~5 GB free — put downloads in `/private/tmp/rst_smoke/` and delete them afterwards). Data dir must be configurable (`CFG["DATA_DIR"]`, default `/content/data` on Colab else `./data`).
- Confirm the smoke artifacts load with `server.nlp.AssociationModel` **after** you check the contract above (if nlp.py doesn't yet match the contract, don't edit server/ — just report).
- Do not commit. Do not touch `web/` or `server/`. Report: what you ran, smoke metrics, anything that didn't work, open issues.
