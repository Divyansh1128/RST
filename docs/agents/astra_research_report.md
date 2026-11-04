**Recommendation:** start with **GloVe 6B 100d, a trainable embedding table, a small neural relatedness classifier, and a directional retrieval head**. Restrict vocabulary to 30,000 words. Estimated serving artifacts: **~30 MB**. Training within 30 minutes and CPU judgment below 20 ms are plausible, but require measurement on your M2.

**Key finding:** SWOW currently requires a download agreement and has restrictive licensing; it does **not** satisfy your no-form requirement.

**A. Association datasets**

Sizes below are publisher/repository figures, rounded. I checked public listings and download targets; I did not download every complete payload. Some Hugging Face CDN downloads could not be completed by the research tool.

| Dataset | Direct download and format | Size | Public hobby/student use |
|---|---|---:|---|
| **EAT** | [JSON download](https://raw.githubusercontent.com/dariusk/ea-thesaurus/master/ea-thesaurus.json): cue → responses with counts | GitHub reports **4.62 MB** | Mirror declares **Open Government Licence**, permitting commercial/noncommercial reuse with attribution. However, this declaration comes from the mirror maintainer; original licensor provenance was not independently confirmed. |
| **USF Free Association Norms** | [JSON mirror](https://raw.githubusercontent.com/jmichaelov/easy-fas/main/usffan_dict.json): cue/response counts | GitHub reports **1.83 MB** | Downloadable, but **original dataset deployment license unresolved**. The wrapper’s MIT license should not automatically be treated as relicensing the underlying norms. |
| **SWOW-EN18** | [Official release page](https://smallworldofwords.org/en/project/research): ZIP containing CSV responses, statistics and association strengths | **80 MB ZIP** | **CC BY-NC-ND 3.0**; research-oriented. Current downloads require an agreement form. Public deployment/distribution of derived artifacts needs clarification from the authors; being a student project does not automatically resolve this. |
| **ConceptNet, English essentials mirror** | [Direct CSV](https://huggingface.co/datasets/openworld-domains/conceptnet-full-en-essentials/resolve/main/conceptnet-full-en-essentials.csv), approximately 4.71 million rows | **562 MB** | Upstream **CC BY-SA 4.0** permits public use with attribution/share-alike obligations. This mirror drops some original metadata, including edge weights/provenance fields. |
| **WordNet 3.0** | [NLTK ZIP](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip): native WordNet index/data files | **10,775,600 bytes ZIP**, **36,353,991 bytes extracted** | **Yes.** Princeton’s permissive WordNet license allows use/modification/distribution, retaining required notices. |

Evidence: [EAT format/license](https://github.com/dariusk/ea-thesaurus), [USF mirror provenance](https://github.com/jmichaelov/easy-fas), [ConceptNet license](https://github.com/commonsense/conceptnet5/wiki/Copying-and-sharing-ConceptNet), [WordNet size/license manifest](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml).

Practical selection:

- **Best association supervision:** SWOW, followed by EAT/USF—but the access/licensing distinctions above matter.
- **Clearest public-release starting point:** filtered ConceptNet + WordNet + your own labeled moves.
- **EAT:** useful supplemental human associations, with dated British cultural references.
- **WordNet:** supplemental synonyms, antonyms, hypernyms and part–whole relations; insufficient alone for `coffee → morning`.

For ConceptNet, retain single-word English endpoints and relations such as `RelatedTo`, `Synonym`, `Antonym`, `IsA`, `PartOf`, `AtLocation`, `UsedFor`, `CapableOf`, and `HasProperty`. Exclude `ExternalURL`, etymology and spelling-only edges. Check **both** endpoint language codes: the “English” mirror includes some cross-language edges.

**B. Pretrained embeddings**

| Choice | Exact download | Download size / format | License and verdict |
|---|---|---|---|
| **GloVe 6B 100d — default** | [Hugging Face gzip](https://huggingface.co/datasets/SLU-CSCI4750/glove.6B.100d.txt/resolve/main/glove.6B.100d.txt.gz) | **134 MB gzip**, approximately **347 MB text**; 400k words | **PDDL 1.0**. Small, straightforward initialization. |
| GloVe 6B 100d alternative | [Zenodo ZIP](https://zenodo.org/records/4925376/files/glove.6B.100d.zip?download=1) | **115.1 MB**, ZIP-LZMA containing word2vec binary | Same vectors/license; different serialization. |
| **GloVe 6B 300d** | [HF text](https://huggingface.co/kcz358/glove/resolve/main/glove.6B/glove.6B.300d.txt); [smaller Zenodo ZIP](https://zenodo.org/records/4925376/files/glove.6B.300d.zip?download=1) | **1.04 GB text**, or **339,432,218 bytes ZIP** containing word2vec binary | **PDDL 1.0**. Feasible after vocabulary filtering, but start with 100d. |
| **Numberbatch 19.08, filtered English** | [Zenodo ZIP](https://zenodo.org/records/4911598/files/numberbatch-19.08-en.zip?download=1) | **188,185,146 bytes**; ZIP-LZMA, word2vec binary, 300d | **CC BY-SA 4.0**. Strong alternative for associations. Repack removes nonalphabetic tokens. |
| fastText Wiki-News 300d, 1M | [Official ZIP](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip) | Approximately **683 MB ZIP**; text vectors; full float32 matrix alone is **1.2 GB** | **CC BY-SA 3.0**. Can be filtered, but less attractive under these constraints. |
| Numberbatch `mini.h5` | [Official HDF5 target](https://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/19.08/mini.h5) | Exact disk size **unverified**; documentation says **<150 MB RAM**, which is a different measurement | Research fetch returned **403**. Prefer the English Zenodo repack. |

Sources: [GloVe data license](https://www-nlp.stanford.edu/projects/glove/), [GloVe repacks](https://zenodo.org/records/4925376), [Numberbatch repack](https://zenodo.org/records/4911598), [Numberbatch license/formats](https://github.com/commonsense/conceptnet-numberbatch), [fastText license](https://fasttext.cc/docs/en/english-vectors.html).

**Reported benchmark evidence — Spearman correlation, not move-acceptance accuracy:**

| Embedding | MEN | WordSim-353 **relatedness subset** | SimLex-999 |
|---|---:|---:|---:|
| GloVe 6B 100d | **0.693** | **0.496** | **0.2975** |
| GloVe 6B 300d | **0.749** | **0.573** | **0.3705** |

MEN/WS353-rel values are from [Lee et al., Table 1](https://www.cs.nccu.edu.tw/~hhhuang/docs/www2016a.pdf); SimLex values from [Jo & Choi, Table 2](https://aclanthology.org/W18-3003.pdf). These are reported experimental results; preprocessing changes scores.

Numberbatch **16.09** reports **MEN test 0.866** and **full WS353 0.828**—the latter is **not WS353-rel**. A separate study reports Numberbatch **SimLex 0.63**, without clearly identifying the release. Do not attribute these numbers to `mini.h5` or the filtered 19.08 file. [Original Numberbatch paper](https://arxiv.org/pdf/1612.03975), [independent evaluation, Table 6](https://aclanthology.org/2020.lincr-1.1.pdf).

I did not verify matching MEN/WS353-rel/SimLex results for the exact Wiki-News fastText download. Its release paper reports other benchmarks. [fastText paper](https://arxiv.org/pdf/1712.09405)

**C. Trainable architecture and objectives**

Use one shared model for all players.

```text
E[word]: trainable 100d embedding, initialized from GloVe
z(word) = normalize(MLP 100 → 128 → 64, ReLU)

JUDGE:
features = [z(a) * z(b), abs(z(a) - z(b)), cosine(E[a], E[b])]
logit = MLP 129 → 64 → 32 → 1, ReLU
probability = sigmoid(calibrated_logit)

PLAYER:
q(a) = normalize(Wq z(a))
k(b) = normalize(Wk z(b))
retrieval_score(a,b) = q(a) · k(b) / temperature
```

The judge is symmetric, matching a forgiving “these words are associated” rule. The player is directional because response frequencies differ between `a → b` and `b → a`.

**Training:**

- **Player loss:** multi-positive InfoNCE, temperature **0.1**. Weight observed responses by `sqrt(count)` and normalize within cue; this preserves weaker associations.
- Mask other known positives from negative slots, including in-batch negatives.
- **Judge loss:** binary cross-entropy. Association pairs provide positive evidence; use random negatives conservatively and **human-confirmed hard negatives**.
- Nearest neighbors missing from a dataset are **unlabeled**, not automatically negative. Antonyms are often valid associations.
- Starting combined objective:  
  `L = L_InfoNCE + L_BCE + 0.01 × mean_squared_embedding_drift`.
- Add **2,000–5,000 human-rated pairs**, emphasizing plausible associations, ambiguous cases and misleading high-cosine pairs. Split these by cue too.
- Fit a two-parameter logistic calibration on held-out, realistically sampled moves. Association frequency and contrastive softmax are **not** acceptance probabilities.

| Hyperparameter | Starting value |
|---|---|
| Vocabulary | 30,000 |
| Positive training pairs | Cap at 200,000, sampled across cues |
| Batch | 256 |
| Negatives | In-batch; judge batch mixes random and reviewed hard negatives |
| Optimizer | AdamW |
| Learning rates | Heads **1e-3**; embeddings **1e-4** |
| Regularization | Dropout **0.1**, gradient clipping **1.0** |
| Duration | **3–5 epochs**, maximum **2,000–3,000 steps**, validation early stopping |
| Device | CPU first; compare MPS using a short timing run |

This clearly constitutes **a neural model you trained**: both nonlinear layers and embeddings learn from task supervision.

**Serving and budget:**

- Precompute `z` and candidate `k` after training. Retaining `E`, `z`, and `k` costs about **27.4 MB float32**, plus vocabulary and small heads: approximately **30 MB total**.
- Player: mask illegal/used words, score all candidates with one matrix–vector product, then judge/rerank the best **256**. No vector database needed.
- Judge: two lookups and a small MLP. **<20 ms should be achievable**, but verify warm p95 CPU latency including normalization and rule checks.
- Stream embeddings and retain only selected vocabulary rows. Never load text vectors into a giant Python dictionary.
- Use a native ARM64 environment and `pip install --no-cache-dir torch numpy`. Avoid unnecessary TorchVision, TorchText, Transformers and duplicate download caches.
- Planning allowance: **1–1.5 GB environment**, **<1 GB selected downloads**, **<300 MB processed data/checkpoints**, plus temporary space. Measure actual installation size; wheel size is smaller than installed size.
- The **30-minute target covers training**, not annotation or downloads. Time 200 steps before committing to an epoch budget.

**Difficulty starting settings:** sample from judge-approved candidates after every legality mask:

| Level | Rank range | Mistake rate |
|---|---:|---:|
| Rookie | 20–100 | 15% |
| Casual | 5–40 | 7% |
| Pro | 1–10 | 2% |
| Grandmaster | 1–3 | 0% |

Use available ranks when fewer candidates remain. Model mistakes as a timeout/pass or a deliberately rejected association; retain R/S/T and no-repeat masks. These settings control fluency, not strategic mastery.

**D. Evaluation and vocabulary**

- **Split:** 80/10/10 by normalized cue/lemma, before augmentation. Deduplicate across datasets and remove reverse-pair leakage. For a strict test, remove every training edge incident to validation/test cues; pretrained embeddings may still contain those words.
- **Judge:** ROC-AUC, PR-AUC, Brier score/calibration, and **false rejection rate at a fixed false acceptance rate**. Select the threshold on validation, targeting high recall for genuine associations.
- **Player:** macro Recall@5/10/50, MRR of the first valid reference, and frequency-weighted nDCG@10. Evaluate against the full legal vocabulary, with used-word masks of several sizes. Report vocabulary coverage and cues with no legal reference answers.
- **Baseline:** untouched GloVe cosine, with its threshold/calibration tuned on the same validation set. Also compare frozen embeddings + trained heads versus embedding fine-tuning.
- **Human check:** association datasets omit valid answers. Blind-rate top bot outputs and judge disagreements; automatic retrieval metrics alone undercount valid novel associations.
- **Performance:** warmed batch-one p50/p95/p99 CPU latency, peak RAM, artifact bytes and wall-clock training time. Test concurrent game requests using one shared model.

Vocabulary source: [wordfreq](https://github.com/rspeer/wordfreq), with [English frequency data](https://raw.githubusercontent.com/rspeer/wordfreq/master/wordfreq/data/large_en.msgpack.gz), approximately **1.43 MB**. Iterate by frequency, filter to dictionary-recognized common words, then retain 30k. Its data license is **CC BY-SA 4.0**; preserve attribution and notices.

Keep R/S/T words in the **semantic vocabulary and training data**. Exclude them from playable outputs through game rules.

**E. Pitfalls**

- **Inflections:** define whether `cat/cats` count as repeats. Recommended: retain surface spelling but use a curated lemma-family repeat key. Avoid crude suffix stripping.
- **Proper nouns:** capitalization alone is unreliable, especially with uncased embeddings. Use a reviewed vocabulary; preserve ordinary senses of words such as `apple`.
- **OOV:** return “word not recognized” separately from “unrelated.” A shared unknown vector cannot judge arbitrary words reliably. Static fastText `.vec` files do **not** provide arbitrary-word subword inference.
- **Polysemy:** `bank → money` and `bank → river` can both be valid. One vector per word will still miss some senses.
- **Offensive words:** start with [Shutterstock’s English list](https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en), **CC BY 4.0**. Match whole normalized words, manually review exclusions and allow exceptions; substring filtering causes false positives. [License/context](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words)
- **Game validity:** reject repeats and prohibited initials deterministically before scoring. Relatedness and legality are separate.
- **Seven players:** share weights and vocabulary; maintain a separate used-word set per game and update turns atomically.