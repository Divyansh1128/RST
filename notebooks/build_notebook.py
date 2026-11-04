"""Build the self-contained Colab notebook. Run with the project's Python/nbformat."""
from pathlib import Path
import textwrap
import nbformat as nbf

cells = []
def md(s):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))

md('''
# RST: train a word association judge and player

Student research pipeline based on `docs/agents/astra_research_report.md`, section C,
with **300-dimensional GloVe → 256 → 128** and temperature **0.07 (fixed)**.
The symmetric judge estimates association; the directional player retrieves responses.
**Runtime → Change runtime type → T4 GPU, then Runtime → Run all.**
The notebook also works on CPU. Set `SMOKE=True` for a short
100d / 4,000-base-word / 150-step run with no ConceptNet. Vocabulary extensions may
add up to 10,000 association words. Smoke results test the pipeline, not model quality.

Full defaults target <25 minutes on T4 including preprocessing and evaluation, subject
to network speed; this is a target, not a measured guarantee. The notebook reports
wall time and projects training time after 100 steps. Downloads are cached atomically.
''')
md('''
## 1. Setup and reproducibility

Edit `CFG` here. `DATA_DIR` defaults to `/content/data` on Colab, `./data` elsewhere.
Environment overrides support local verification without editing the notebook.
CUDA uses AMP; MPS and CPU use float32. Seeds and deterministic kernels are fixed
where supported (different devices/library versions need not be bit-identical).
''')
code('''
import os, sys, subprocess, importlib.util, time
# Colab already provides torch/numpy/pandas/sklearn/matplotlib; install missing ones locally.
# Equivalent to: pip install -q wordfreq nltk
if os.environ.get("RST_SKIP_INSTALL") != "1":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "wordfreq", "nltk"])
    missing = [p for p, m in [("torch", "torch"), ("numpy", "numpy"), ("pandas", "pandas"),
               ("scikit-learn", "sklearn"), ("matplotlib", "matplotlib")] if importlib.util.find_spec(m) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *missing])

IN_COLAB = importlib.util.find_spec("google.colab") is not None if importlib.util.find_spec("google") else False
CFG = dict(EMB="glove300", VOCAB=30000, SEED=13, SMOKE=False,
           DATA_DIR="/content/data" if IN_COLAB else "./data",
           ARTIFACT_DIR="/content/rst_artifacts" if IN_COLAB else "./rst_artifacts",
           BATCH=256, NEGATIVES=256, STEPS=4000, LR_HEAD=1e-3, LR_EMB=1e-4,
           Z=128, HIDDEN=256, TAU=0.07, DRIFT=0.01, DROPOUT=0.1,
           EVAL_EVERY=100, PATIENCE=8, JUDGE_EVAL_MAX=12000, SELECTION_CUES=400,
           HARD_POOL=64, HARD_BATCH=128, ENCODE_BATCH=2048, EVAL_BATCH=128,
           CN_CHUNK=100000, EXTRA_VOCAB=10000, CPU_THREADS=4)
CFG["SMOKE"] = CFG["SMOKE"] or os.environ.get("RST_SMOKE") == "1"
if CFG["SMOKE"]:
    CFG.update(EMB="glove100", VOCAB=4000, BATCH=64, NEGATIVES=128, STEPS=150,
               EVAL_EVERY=50, PATIENCE=10, JUDGE_EVAL_MAX=1000, HARD_BATCH=64)
for key in ("DATA_DIR", "ARTIFACT_DIR"):
    CFG[key] = os.environ.get("RST_" + key, CFG[key])
# Apply numeric overrides after the smoke preset, e.g. RST_STEPS=1500.
for key in ("STEPS", "PATIENCE", "EVAL_EVERY", "SELECTION_CUES"):
    CFG[key] = int(os.environ.get("RST_" + key, CFG[key]))
assert CFG["STEPS"] > 0 and CFG["EVAL_EVERY"] > 0 and CFG["PATIENCE"] > 0
assert 1 <= CFG["SELECTION_CUES"] <= 400

import json, gzip, random, math, shutil, urllib.request, contextlib, copy, gc
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import nltk
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem import WordNetLemmatizer
from wordfreq import top_n_list, zipf_frequency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    display = print

START = time.perf_counter()
DATA = Path(CFG["DATA_DIR"]); DATA.mkdir(parents=True, exist_ok=True)
# Keep corpora and plotting caches in the configurable temporary data directory.
nltk.data.path.insert(0, str(DATA / "nltk_data"))
for corpus in ("wordnet", "omw-1.4", "stopwords"):
    if not nltk.download(corpus, download_dir=str(DATA / "nltk_data"), quiet=True):
        raise RuntimeError("NLTK download failed: " + corpus)
random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"]); torch.manual_seed(CFG["SEED"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["SEED"])
torch.set_num_threads(CFG["CPU_THREADS"])
threadpool_limits(limits=CFG["CPU_THREADS"])
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
device = torch.device(os.environ.get("RST_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
rng = np.random.default_rng(CFG["SEED"])
print("Device:", device, "\\nConfig:", json.dumps(CFG, indent=2))
def clean(w):
    w = str(w).strip().lower()
    return w if w.isascii() and w.isalpha() and len(w) >= 2 else None
freq_words = list(dict.fromkeys(w for x in top_n_list("en", 80000) if (w := clean(x))))
freq_set = set(freq_words)
''')
md('''
## 2. Download and inspect sources

The exact URLs are from Astra's report. GloVe is read line by line, retaining only
frequency-vocabulary rows; gzip is never expanded to disk. EAT currently stores
lists of one-entry dictionaries; USF uses nested `_count` fields. We print actual
samples and accept either layout. ConceptNet is chunked, detects column names,
and requires `/c/en/` on **both** endpoints. `HasContext`, spelling and etymology
relations are excluded. Missing ConceptNet weights default to 1.

Source notices: [GloVe](https://nlp.stanford.edu/projects/glove/) PDDL 1.0;
[EAT mirror](https://github.com/dariusk/ea-thesaurus) declares OGL (original provenance
unconfirmed); [USF mirror](https://github.com/jmichaelov/easy-fas) has MIT wrapper
code but the norms' deployment license is unresolved;
[ConceptNet](https://github.com/commonsense/conceptnet5/wiki/Copying-and-sharing-ConceptNet)
CC BY-SA 4.0; [bad-word list](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words)
CC BY 4.0; [wordfreq](https://github.com/rspeer/wordfreq) data CC BY-SA 4.0;
[WordNet](https://wordnet.princeton.edu/license-and-commercial-use) Princeton license.
These caveats are preserved in `meta.json`; public redistribution needs a license review.
''')
code('''
URLS = {
    "glove300": "https://huggingface.co/kcz358/glove/resolve/main/glove.6B/glove.6B.300d.txt",
    "glove100": "https://huggingface.co/datasets/SLU-CSCI4750/glove.6B.100d.txt/resolve/main/glove.6B.100d.txt.gz",
    "eat": "https://raw.githubusercontent.com/dariusk/ea-thesaurus/master/ea-thesaurus.json",
    "usf": "https://raw.githubusercontent.com/jmichaelov/easy-fas/main/usffan_dict.json",
    "conceptnet": "https://huggingface.co/datasets/openworld-domains/conceptnet-full-en-essentials/resolve/main/conceptnet-full-en-essentials.csv",
    "bad_words": "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en",
}
def download(key, filename):
    dest = DATA / filename
    if dest.exists() and dest.stat().st_size > 0:
        print("Cached:", dest.name); return dest
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(3):
        try:
            req = urllib.request.Request(URLS[key], headers={"User-Agent": "RST-student-research/1.0"})
            with urllib.request.urlopen(req, timeout=180) as src, part.open("wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            part.replace(dest)
            print("Downloaded:", dest.name, round(dest.stat().st_size / 1e6, 1), "MB")
            return dest
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == 2: raise
            time.sleep(2)

# source, cue, response, count, relation (empty relation for human norms)
raw_edges = []
def positive_count(value):
    if isinstance(value, dict):
        value = value.get("_count", value.get("count", value.get("weight", 1)))
    try:
        x = float(value)
        return x if np.isfinite(x) and x > 0 else 0.
    except (ValueError, TypeError):
        return 0.

def responses(value):
    if isinstance(value, dict):
        # A structured response record, or response -> count mapping.
        if "response" in value or "word" in value:
            yield value.get("response", value.get("word")), positive_count(value)
        else:
            for word, count in value.items():
                if not str(word).startswith("_"):
                    yield word, positive_count(count)
    elif isinstance(value, list):
        for entry in value:
            if isinstance(entry, (dict, list)):
                if isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], str):
                    yield entry[0], positive_count(entry[1])
                else:
                    yield from responses(entry)
            elif isinstance(entry, str):
                yield entry, 1.
    else:
        raise ValueError("Unsupported response container: " + str(type(value)))

for source in ("eat", "usf"):
    obj = json.loads(download(source, source + ".json").read_text())
    print(source, "root:", type(obj).__name__, "sample:", str(list(obj.items())[:1])[:1200])
    if not isinstance(obj, dict):
        raise ValueError("Expected cue-keyed JSON; inspect sample and update parser")
    for cue, value in obj.items():
        a = clean(cue)
        if a not in freq_set: continue
        for response, count in responses(value):
            b = clean(response)
            # Preserve self-responses for the all-source lookup; training drops same-lemma pairs.
            if b in freq_set and count > 0:
                raw_edges.append((source, a, b, count, ""))
print("Human edges:", Counter(e[0] for e in raw_edges))
bad_words = {w for line in download("bad_words", "bad_words.txt").read_text().splitlines() if (w := clean(line))}

ALLOWED_REL = set("RelatedTo Synonym Antonym IsA PartOf HasA AtLocation UsedFor CapableOf HasProperty MadeOf Causes HasPrerequisite Desires DistinctFrom SimilarTo LocatedNear ReceivesAction CausesDesire MotivatedByGoal Entails MannerOf".split())
SYMMETRIC_REL = set("RelatedTo Synonym Antonym DistinctFrom SimilarTo LocatedNear".split())
def column_name(columns, aliases, required=True):
    lookup = {str(c).strip().lower().replace("_", ""): c for c in columns}
    for alias in aliases:
        if alias.replace("_", "").lower() in lookup:
            return lookup[alias.replace("_", "").lower()]
    if required: raise ValueError("Missing column " + str(aliases) + "; actual: " + str(list(columns)))
    return None

def parse_conceptnet(path):
    kept = Counter()
    for chunk_no, chunk in enumerate(pd.read_csv(path, chunksize=CFG["CN_CHUNK"], low_memory=False)):
        if chunk_no == 0:
            print("ConceptNet columns:", list(chunk.columns)); display(chunk.head())
        cols = chunk.columns
        ac = column_name(cols, ["start_uri", "start", "source_uri", "head_uri"])
        bc = column_name(cols, ["end_uri", "end", "target_uri", "tail_uri"])
        rc = column_name(cols, ["relation_uri", "relation_text", "relation", "rel"])
        wc = column_name(cols, ["weight", "edge_weight", "score"], required=False)
        if chunk_no == 0: print("ConceptNet weight column:", wc or "none (using 1)")
        # Vectorized relation/language filtering before a small Python token loop.
        rel = chunk[rc].fillna("").astype(str).str.rsplit("/").str[-1]
        au, bu = chunk[ac].fillna("").astype(str), chunk[bc].fillna("").astype(str)
        ok = rel.isin(ALLOWED_REL) & au.str.startswith("/c/en/") & bu.str.startswith("/c/en/")
        sub = chunk.loc[ok]
        weights = sub[wc] if wc else pd.Series(1., index=sub.index)
        for a_uri, b_uri, relation, weight in zip(au[ok], bu[ok], rel[ok], weights):
            a = clean(a_uri.split("/")[3]); b = clean(b_uri.split("/")[3])
            weight = positive_count(weight)
            if a in freq_set and b in freq_set and weight > 0:
                kept[(a, b, relation)] += weight
        if chunk_no % 10 == 0: print("ConceptNet chunks:", chunk_no + 1, "unique eligible edges:", len(kept))
    return [("conceptnet", a, b, count, relation) for (a, b, relation), count in kept.items()]

if not CFG["SMOKE"]:
    raw_edges.extend(parse_conceptnet(download("conceptnet", "conceptnet.csv")))
else:
    print("SMOKE: ConceptNet disabled")

dim_e = 300 if CFG["EMB"] == "glove300" else 100
assert CFG["EMB"] in ("glove300", "glove100")
glove_path = download(CFG["EMB"], "glove.6B." + str(dim_e) + ("d.txt.gz" if dim_e == 100 else "d.txt"))
glove_rows = {}
open_glove = gzip.open if str(glove_path).endswith(".gz") else open
with open_glove(glove_path, "rt", encoding="utf-8") as handle:
    for line in handle:
        word, sep, vector = line.partition(" ")
        if word in freq_set:
            row = np.fromstring(vector, sep=" ", dtype=np.float32)
            if len(row) == dim_e and np.isfinite(row).all() and np.linalg.norm(row) > 0:
                glove_rows[word] = row
print("GloVe rows retained:", len(glove_rows), "; raw source edges:", len(raw_edges))
''')
md('''
## 3. Vocabulary, lemma families and legal words

Sort the wordfreq/GloVe intersection by descending Zipf frequency (stable ties),
then append eligible association words beyond the base cut, also sorted by Zipf,
up to the 10,000-word extension cap. Assert frequency order including the boundary.
Report any cap exclusions. Index zero is the most frequent retained word.
Blocked words remain semantic inputs. Playable words need WordNet coverage, frequency,
length and stopword/offensive-word checks; R/S/T are masked at retrieval time.
Adverb-only words are excluded. Words with an adverb sense are allowed only if
Zipf ≥ 4.5 and they also have a noun/adjective sense. A manual function-word list
further filters dull bot outputs. All these words remain semantic vocabulary inputs.
Lemma families apply noun, then verb, then adjective lemmatization, only retaining
results in the vocabulary. Seeds prefer common nouns with a physical-entity sense.
''')
code('''
zipf = {w: zipf_frequency(w, "en") for w in freq_words}
eligible = sorted((w for w in freq_words if w in glove_rows), key=lambda w: -zipf[w])
pair_words = {w for _, a, b, _, _ in raw_edges for w in (a, b)}
words = eligible[:CFG["VOCAB"]]
base = set(words)
extra = [w for w in eligible if w in pair_words and w not in base]
words += extra[:CFG["EXTRA_VOCAB"]]
assert all(zipf[a] >= zipf[b] for a, b in zip(words, words[1:]))
assert words[:len(base)] == eligible[:CFG["VOCAB"]]
assert words[len(base):] == extra[:CFG["EXTRA_VOCAB"]]
index = {w: i for i, w in enumerate(words)}
print("Base:", len(base), "extensions:", len(words) - len(base),
      "eligible pair words excluded by cap:", max(0, len(extra) - CFG["EXTRA_VOCAB"]))
print("PASS: base + appended extensions in descending Zipf order; first words:", words[:10])
N = len(words)
E0 = np.stack([glove_rows[w] for w in words])
E0 /= np.linalg.norm(E0, axis=1, keepdims=True).clip(1e-12)
del glove_rows; gc.collect()
lemmatizer = WordNetLemmatizer()
def lemma_word(w):
    result = w
    for pos in ("n", "v", "a"):
        candidate = lemmatizer.lemmatize(result, pos=pos)
        if candidate in index:
            result = candidate
    return result
lemma = {w: lemma_word(w) for w in words}
# Collapse any chains to a stable family key.
for w in words:
    seen = {w}; root = lemma[w]
    while lemma[root] != root and root not in seen:
        seen.add(root); root = lemma[root]
    lemma[w] = min(seen | {root}) if root in seen and lemma[root] != root else root
lemma_ids = np.array([index[lemma[w]] for w in words])
manual_stop = set("also would could should may might must shall can get got even much many one two really very something anything everything nothing someone anyone thing things maybe perhaps either another well indeed certainly definitely quite rather actually just yet ever never always often already instead however therefore thus hence etc".split())
stop = set(stopwords.words("english")) | manual_stop
playable = np.zeros(N, dtype=bool)
seed_pool = []
physical = wn.synset("physical_entity.n.01")
for i, w in enumerate(words):
    senses = wn.synsets(w)
    pos = {sense.pos() for sense in senses}
    adverb_ok = "r" not in pos or (zipf[w] >= 4.5 and bool(pos & {"n", "a", "s"}))
    playable[i] = len(w) >= 3 and w not in stop and w not in bad_words and bool(senses) and zipf[w] >= 2.5 and adverb_ok
    if playable[i] and w[0] not in "rst" and senses[0].pos() == "n" and zipf[w] >= 3.5:
        first = senses[0]
        # Skip proper nouns (e.g. "martin" -> St. Martin): capitalised lemma or an instance synset.
        proper = first.lemmas()[0].name()[0].isupper() or bool(first.instance_hypernyms())
        if not proper and any(physical in path for path in first.hypernym_paths()):
            seed_pool.append(w)
seeds = seed_pool[:1500]
legal = playable & np.array([w[0] not in "rst" for w in words])
assert not any(playable[index[w]] for w in bad_words if w in index)
assert not any(playable[index[w]] for w in stop if w in index)
assert seeds and legal.any()
print("Vocabulary:", N, "dimension:", dim_e, "playable:", int(playable.sum()),
      "legal RST outputs:", int(legal.sum()), "seed nouns:", len(seeds))
print("Seed sample:", seeds[:30])
''')
md('''
## 4. Weighted pairs and strict split by cue lemma

Normalize human counts per source/cue, and scale ConceptNet weights by 0.5.
Split original cue lemma families 80/10/10 **before** adding reverse ConceptNet edges.
Training removes every edge incident to validation/test cue families, including
response-side inflections. Reverse edges (0.5 weight) are added only for symmetric
ConceptNet relations within the training partition. Evaluation uses original edges.
Known-positive masks include either direction and lemma families to reduce false negatives.
Separately, preserve every vocabulary-index source pair (including strict-split
removals and same-lemma surface forms) in both directions for `known_pairs.npy`.
This all-data lookup is exported only; model evaluation never consults it.
''')
code('''
# Attested surface pairs from ALL sources, before lemma/split filtering or augmentation.
attested = np.array([(index[a], index[b]) for _, a, b, _, _ in raw_edges
                     if a in index and b in index], dtype=np.int32).reshape(-1, 2)
known_pairs_export = np.unique(np.concatenate((attested, attested[:, ::-1])), axis=0)
del attested
print("Attested lookup pairs (deduplicated, both directions):", len(known_pairs_export))
filtered = [(s, index[a], index[b], float(c), r) for s, a, b, c, r in raw_edges
            if a in index and b in index and lemma[a] != lemma[b]]
source_totals = Counter()
for s, a, b, count, rel in filtered:
    source_totals[(s, a)] += count
cue_families = np.array(sorted({int(lemma_ids[a]) for _, a, _, _, _ in filtered}))
split_rng = np.random.default_rng(CFG["SEED"])
split_rng.shuffle(cue_families)
n1, n2 = int(.8 * len(cue_families)), int(.9 * len(cue_families))
family_sets = {"train": set(cue_families[:n1]), "val": set(cue_families[n1:n2]), "test": set(cue_families[n2:])}
heldout = family_sets["val"] | family_sets["test"]
split_pairs = {s: Counter() for s in family_sets}
# Lookup in lemma space; symmetric judge positives are not asserted negative in either direction.
known = defaultdict(set)
source_stats = Counter(); removed = 0; reverse_added = 0
for source, a, b, count, rel in filtered:
    la, lb = int(lemma_ids[a]), int(lemma_ids[b])
    known[la].add(lb); known[lb].add(la)
    split = "val" if la in family_sets["val"] else "test" if la in family_sets["test"] else "train"
    if split == "train" and (la in heldout or lb in heldout):
        removed += 1; continue
    # ConceptNet retains its edge weight, whereas norms use conditional response counts.
    weight = .5 * count if source == "conceptnet" else count / source_totals[(source, a)]
    split_pairs[split][(a, b)] += weight
    source_stats[(split, source)] += 1
    if split == "train" and rel in SYMMETRIC_REL:
        split_pairs[split][(b, a)] += .5 * weight
        reverse_added += 1
assert all(lemma_ids[a] not in heldout and lemma_ids[b] not in heldout for a, b in split_pairs["train"])
assert not (family_sets["train"] & heldout) and not (family_sets["val"] & family_sets["test"])
train_pos = defaultdict(dict)
for (a, b), weight in split_pairs["train"].items():
    train_pos[a][b] = weight
train_cues = np.array(sorted(train_pos), dtype=np.int64)
# No held-out lemma endpoints in training negatives or contrastive candidates either.
train_pool = np.array([i for i in range(N) if lemma_ids[i] not in heldout], dtype=np.int64)
train_choices = {}
for a, neighbors in train_pos.items():
    ids = np.array(list(neighbors), dtype=np.int64)
    weights = np.sqrt(np.array(list(neighbors.values()), dtype=np.float64))
    train_choices[a] = (ids, weights / weights.sum())
rows = []
for split, pairs in split_pairs.items():
    cues = {a for a, b in pairs}
    rows.append(dict(split=split, cue_lemmas=len(family_sets[split]), retained_cues=len(cues),
                     pairs=len(pairs), **{s: source_stats[(split, s)] for s in ("eat", "usf", "conceptnet")}))
display(pd.DataFrame(rows))
print("Strictly removed edges:", removed, "; reverse training additions:", reverse_added)
del raw_edges, filtered; gc.collect()
''')
md('''
## 5. Neural model, negatives and training

`E` is a trainable normalized-GloVe initialization; `enc` maps `dim_e → 256 → dim_z`.
The judge uses `[z_a*z_b, abs(z_a-z_b), cosine(E_a,E_b)]`, plus a learnable raw-cosine
skip: `logit = mlp(features) + w_cos*cos`, initialized with `w_cos=5`.
The player uses `q = normalize(e + Wq(z))`, `k = normalize(e + Wk(z))`.
The bias-free maps go from `dim_z` to `dim_e` and start at zero, so the initial
player exactly reproduces normalized GloVe cosine. Export q/k in embedding dimensions.
Temperature is fixed at 0.07, so it is unnecessary for ranking exports.
AdamW uses separate head/embedding learning rates, cosine decay, 0.01 embedding drift
and gradient clipping at 1.0. Every evaluation checks judge AUC and player Recall@10
on a fixed random subset of at most 400 validation cues against the full legal vocab.
Select and early-stop on **val_auc + val_recall10** (including the initial checkpoint).

Multi-positive InfoNCE is weighted cross-entropy over all observed responses present
in a shared in-batch + random candidate set, with √count weights normalized per cue.
Other known positives and same-lemma candidates are masked from negative slots.
Judge batches have one positive, one random negative and one hard negative per cue.
Hard pools sample raw-GloVe cosine ranks 300–3000 (1-based), excluding known positives.
Pools are constructed in bounded batches and cached in memory, never an N×N matrix.
These are **unverified negative labels**: missing associations can be valid, so the
reported false-accept/reject rates describe this proxy task, not human game acceptance.
''')
code('''
class AssociationNet(nn.Module):
    def __init__(self, initial, dim_e, dim_z):
        super().__init__()
        self.dim_e, self.dim_z = dim_e, dim_z
        self.E = nn.Embedding.from_pretrained(torch.as_tensor(initial).clone(), freeze=False)
        self.register_buffer("E0", torch.as_tensor(initial).clone())
        self.enc = nn.Sequential(nn.Linear(dim_e, CFG["HIDDEN"]), nn.ReLU(),
                                 nn.Dropout(CFG["DROPOUT"]), nn.Linear(CFG["HIDDEN"], dim_z))
        self.judge = nn.Sequential(nn.Linear(2 * dim_z + 1, 64), nn.ReLU(),
                                   nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.Wq = nn.Linear(dim_z, dim_e, bias=False)
        self.Wk = nn.Linear(dim_z, dim_e, bias=False)
        nn.init.zeros_(self.Wq.weight); nn.init.zeros_(self.Wk.weight)
        self.w_cos = nn.Parameter(torch.tensor([5.0], dtype=torch.float32))
        self.tau = CFG["TAU"]  # fixed, explicitly recorded in config
    def encode(self, ids):
        e = F.normalize(self.E(ids), dim=-1)
        z = F.normalize(self.enc(e), dim=-1)
        return e, z
    def judge_logits(self, a, b):
        ea, za = self.encode(a); eb, zb = self.encode(b)
        features = torch.cat((za * zb, torch.abs(za - zb), (ea * eb).sum(-1, keepdim=True)), dim=-1)
        cosine = (ea * eb).sum(-1)
        return self.judge(features).squeeze(-1) + self.w_cos[0] * cosine

model = AssociationNet(E0, dim_e, CFG["Z"]).to(device)
def tensor(ids):
    return torch.as_tensor(ids, dtype=torch.long, device=device)

# Generate hard pools for each cue used in any split, never across all N^2 words.
all_cues = np.array(sorted({a for pairs in split_pairs.values() for a, b in pairs}), dtype=np.int64)
raw_t = torch.as_tensor(E0, device=device)
hard_pools = {}; hard_rng = np.random.default_rng(CFG["SEED"] + 17)
with torch.no_grad():
    for start in range(0, len(all_cues), CFG["HARD_BATCH"]):
        batch = all_cues[start:start + CFG["HARD_BATCH"]]
        scores = raw_t[tensor(batch)] @ raw_t.T
        ranked = scores.topk(min(3000, N), dim=1).indices.cpu().numpy()[:, 299:]
        for a, ranks in zip(batch, ranked):
            la = int(lemma_ids[a])
            pool = [int(b) for b in ranks if lemma_ids[b] != la and lemma_ids[b] not in known[la]]
            if not pool: raise RuntimeError("No hard negatives for " + words[a])
            hard_rng.shuffle(pool)
            # Separate train pools must also exclude heldout endpoints.
            if a in train_pos:
                pool = [b for b in pool if lemma_ids[b] not in heldout]
            if not pool: raise RuntimeError("No strict training hard negatives")
            hard_pools[int(a)] = np.array(pool[:CFG["HARD_POOL"]], dtype=np.int64)
        if start % (20 * CFG["HARD_BATCH"]) == 0:
            print("Hard pools:", min(start + len(batch), len(all_cues)), "/", len(all_cues))
del raw_t, scores; gc.collect()

def random_negative(a, local_rng, pool):
    la = int(lemma_ids[a])
    for _ in range(10000):
        b = int(local_rng.choice(pool))
        if lemma_ids[b] != la and lemma_ids[b] not in known[la]:
            return b
    raise RuntimeError("Could not sample a negative")

def make_judge_eval(split, seed):
    local = np.random.default_rng(seed)
    pairs = np.array(list(split_pairs[split]), dtype=np.int64)
    # A fixed, uniformly sampled pair subset bounds validation/test cost.
    if len(pairs) > CFG["JUDGE_EVAL_MAX"]:
        pairs = pairs[local.choice(len(pairs), CFG["JUDGE_EVAL_MAX"], replace=False)]
    a, b = pairs.T
    random_b = np.array([random_negative(int(x), local, np.arange(N)) for x in a])
    hard_b = np.array([local.choice(hard_pools[int(x)]) for x in a])
    return dict(a=np.tile(a, 3), b=np.concatenate((b, random_b, hard_b)),
                y=np.concatenate((np.ones(len(a)), np.zeros(2 * len(a)))),
                kind=np.repeat(["positive", "random", "hard"], len(a)))
eval_sets = {s: make_judge_eval(s, CFG["SEED"] + i + 1) for i, s in enumerate(("val", "test"))}
@torch.no_grad()
def judge_scores(ds):
    model.eval(); out = []
    for start in range(0, len(ds["a"]), CFG["ENCODE_BATCH"]):
        out.append(model.judge_logits(tensor(ds["a"][start:start+CFG["ENCODE_BATCH"]]),
                                     tensor(ds["b"][start:start+CFG["ENCODE_BATCH"]])).float().cpu().numpy())
    return np.concatenate(out)

player_references = {split: defaultdict(set) for split in ("val", "test")}
for split, references in player_references.items():
    for a, b in split_pairs[split]: references[a].add(b)
selection_rng = np.random.default_rng(CFG["SEED"] + 123)
selection_cues = np.sort(selection_rng.choice(
    sorted(player_references["val"]),
    min(CFG["SELECTION_CUES"], len(player_references["val"])), replace=False))

@torch.no_grad()
def encode_all():
    result = {name: [] for name in ("emb", "z", "q", "k")}
    model.eval()
    for start in range(0, N, CFG["ENCODE_BATCH"]):
        e, z = model.encode(torch.arange(start, min(start + CFG["ENCODE_BATCH"], N), device=device))
        values = (e, z, F.normalize(e + model.Wq(z), dim=-1), F.normalize(e + model.Wk(z), dim=-1))
        for name, value in zip(result, values): result[name].append(value.float().cpu().numpy())
    return {name: np.concatenate(values) for name, values in result.items()}
@torch.no_grad()
def player_eval(split, query, keys, cue_subset=None):
    references = player_references[split]
    cue_ids = np.array(sorted(references) if cue_subset is None else cue_subset, dtype=np.int64)
    kt = torch.as_tensor(keys, device=device)
    result = []; no_legal = 0; total_refs = 0; excluded_refs = 0
    for start in range(0, len(cue_ids), CFG["EVAL_BATCH"]):
        batch = cue_ids[start:start + CFG["EVAL_BATCH"]]
        scores = (torch.as_tensor(query[batch], device=device) @ kt.T).float().cpu().numpy()
        for a, score in zip(batch, scores):
            mask = legal & (lemma_ids != lemma_ids[a])
            refs = np.array(sorted(b for b in references[int(a)] if mask[b]), dtype=np.int64)
            total_refs += len(references[int(a)]); excluded_refs += len(references[int(a)]) - len(refs)
            if not len(refs): no_legal += 1; continue
            score[~mask] = -np.inf
            top = np.argpartition(-score, min(50, N - 1))[:50]
            top = top[np.argsort(-score[top], kind="stable")]
            rank = 1 + int(np.sum(score > np.max(score[refs])))
            result.append([len(set(top[:k]) & set(refs)) / len(refs) for k in (5, 10, 50)] + [1. / rank])
    mean = np.mean(result, axis=0)
    return dict(zip(("recall@5", "recall@10", "recall@50", "mrr"), map(float, mean)),
                cues=len(cue_ids), evaluated_cues=len(result), no_legal_reference_cues=no_legal,
                total_references=total_refs, excluded_illegal_references=excluded_refs)
initial_arrays = encode_all()
initial_player_error = max(float(np.max(np.abs(initial_arrays[name] - E0))) for name in ("q", "k"))
assert initial_player_error < 2e-7, initial_player_error
baseline_selection = player_eval("val", E0, E0, selection_cues)
initial_selection = player_eval("val", initial_arrays["q"], initial_arrays["k"], selection_cues)
assert abs(initial_selection["recall@10"] - baseline_selection["recall@10"]) < 1e-6
assert abs(initial_selection["mrr"] - baseline_selection["mrr"]) < 1e-6
print("PASS: zero-initialized residual player equals GloVe; max vector error:", initial_player_error)
print("Initial validation subset:", initial_selection)
del initial_arrays

head_params = [p for name, p in model.named_parameters() if name != "E.weight"]
optimizer = torch.optim.AdamW([{"params": [model.E.weight], "lr": CFG["LR_EMB"], "weight_decay": 0.},
                               {"params": head_params, "lr": CFG["LR_HEAD"], "weight_decay": .01}])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["STEPS"])
amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp)
initial_auc = float(roc_auc_score(eval_sets["val"]["y"], judge_scores(eval_sets["val"])))
best_score = initial_auc + initial_selection["recall@10"]
best_auc = initial_auc; best_recall10 = initial_selection["recall@10"]; best_step = 0
best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
history = [dict(step=0, val_auc=initial_auc, val_recall10=best_recall10, selection_score=best_score)]
stale = 0
TRAIN_START = time.perf_counter()
for step in range(1, CFG["STEPS"] + 1):
    model.train()
    a = rng.choice(train_cues, size=CFG["BATCH"], replace=True)
    b = np.array([rng.choice(train_choices[int(x)][0], p=train_choices[int(x)][1]) for x in a])
    # Unique shared candidates avoid duplicated in-batch words changing their effective weight.
    candidates = np.unique(np.concatenate((b, rng.choice(train_pool, CFG["NEGATIVES"]))))
    positions = {int(x): j for j, x in enumerate(candidates)}
    target = np.zeros((len(a), len(candidates)), dtype=np.float32)
    allowed = np.ones_like(target, dtype=bool)
    candidate_lemmas = lemma_ids[candidates]
    for row, cue in enumerate(a):
        la = int(lemma_ids[cue])
        allowed[row] = (candidate_lemmas != la) & ~np.isin(candidate_lemmas, list(known[la]))
        for response, weight in train_pos[int(cue)].items():
            col = positions.get(response)
            if col is not None:
                target[row, col] = math.sqrt(weight)
                allowed[row, col] = True
    target /= target.sum(1, keepdims=True)
    random_b = np.array([random_negative(int(x), rng, train_pool) for x in a])
    hard_b = np.array([rng.choice(hard_pools[int(x)]) for x in a])
    ja = np.tile(a, 3); jb = np.concatenate((b, random_b, hard_b))
    labels = torch.cat((torch.ones(len(a), device=device), torch.zeros(2 * len(a), device=device)))
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        ea, za = model.encode(tensor(a)); ec, zc = model.encode(tensor(candidates))
        q = F.normalize(ea + model.Wq(za), dim=-1); k = F.normalize(ec + model.Wk(zc), dim=-1)
        logits = (q @ k.T).float() / model.tau
        logits = logits.masked_fill(~torch.as_tensor(allowed, device=device), -1e4)
        retrieval_loss = -(torch.as_tensor(target, device=device) * F.log_softmax(logits, dim=1)).sum(1).mean()
        judge_loss = F.binary_cross_entropy_with_logits(model.judge_logits(tensor(ja), tensor(jb)), labels)
        drift = (model.E.weight - model.E0).square().mean()
        loss = retrieval_loss + judge_loss + CFG["DRIFT"] * drift
    if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    scaler.step(optimizer); scaler.update(); scheduler.step()
    record = dict(step=step, loss=float(loss.detach()), retrieval=float(retrieval_loss.detach()),
                  judge=float(judge_loss.detach()), drift=float(drift.detach()))
    if step % CFG["EVAL_EVERY"] == 0 or step == CFG["STEPS"]:
        val_auc = roc_auc_score(eval_sets["val"]["y"], judge_scores(eval_sets["val"]))
        current_arrays = encode_all()
        subset_metrics = player_eval("val", current_arrays["q"], current_arrays["k"], selection_cues)
        del current_arrays
        val_recall10 = subset_metrics["recall@10"]
        selection_score = float(val_auc + val_recall10)
        record.update(val_auc=float(val_auc), val_recall10=val_recall10, selection_score=selection_score)
        print(f"Step {step}: loss={record['loss']:.4f}, val AUC={val_auc:.4f}, "
              f"val R@10={val_recall10:.4f}, sum={selection_score:.4f}, elapsed={time.perf_counter()-TRAIN_START:.1f}s")
        if selection_score > best_score + 1e-4:
            best_score = selection_score; best_auc = float(val_auc); best_recall10 = val_recall10
            best_step = step; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
    history.append(record)
    if step == 100:
        projected = (time.perf_counter() - TRAIN_START) / step * CFG["STEPS"]
        print(f"Projected training wall time: {projected/60:.1f} minutes (excludes downloads/preprocessing).")
    if stale >= CFG["PATIENCE"]:
        print("Early stopping; restore best val_auc + val_recall10 checkpoint."); break
model.load_state_dict(best_state); model.eval()
training_seconds = time.perf_counter() - TRAIN_START
print("Selected step:", best_step, "; val AUC + R@10:", best_score,
      "; training seconds:", round(training_seconds, 2))
''')
md('''
## 6. Held-out evaluation and qualitative checks

Judge evaluation uses a fixed uniform sample of at most `JUDGE_EVAL_MAX` positive
pairs per split, each with random and hard negatives (1:1:1). We report sample sizes.
Baseline cosine and trained logits each receive a two-parameter logistic calibrator
fitted **only on validation**. False-reject rates use validation-selected thresholds
at 5%/10% false-accept targets and also report achieved false-accept rates on each split.
PR-AUC is average precision; Brier uses calibrated probabilities.

Player evaluation covers **all retained validation/test cues**, over the full legal
vocabulary. It excludes R/S/T and the cue's entire lemma family, counts legal held-out
references only, and reports excluded references/cues. Recall is macro fraction of
references recovered; MRR uses the first reference. Multiple valid unlisted responses
mean these scores understate human judgments. No test labels tune the model.
''')
code('''
arrays = encode_all()
raw_scores = {s: np.sum(E0[d["a"]] * E0[d["b"]], axis=1) for s, d in eval_sets.items()}
trained_scores = {s: judge_scores(d) for s, d in eval_sets.items()}
def fit_platt(scores, labels):
    fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=CFG["SEED"])
    fit.fit(np.asarray(scores).reshape(-1, 1), labels)
    return [float(fit.coef_[0, 0]), float(fit.intercept_[0])]
def sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(x, -80, 80)))
def calibrated(scores, ab):
    return sigmoid(ab[0] * scores + ab[1])
platt = fit_platt(trained_scores["val"], eval_sets["val"]["y"])
baseline_platt = fit_platt(raw_scores["val"], eval_sets["val"]["y"])
def far_threshold(prob, y, far):
    neg = np.asarray(prob)[np.asarray(y) == 0]
    # > quantile / nextafter avoids accepting all ties for an uninformative scorer.
    return float(np.nextafter(np.quantile(neg, 1 - far, method="higher"), np.inf))
judge_metrics = {}; judge_rows = []
for name, scores, ab in (("baseline", raw_scores, baseline_platt), ("trained", trained_scores, platt)):
    val_prob = calibrated(scores["val"], ab)
    fixed = {far: far_threshold(val_prob, eval_sets["val"]["y"], far) for far in (.05, .10)}
    for split in ("val", "test"):
        y = eval_sets[split]["y"]; prob = calibrated(scores[split], ab)
        m = dict(roc_auc=float(roc_auc_score(y, prob)), pr_auc=float(average_precision_score(y, prob)),
                 brier=float(brier_score_loss(y, prob)), n=int(len(y)), positives=int(y.sum()))
        for far, threshold in fixed.items():
            suffix = str(int(far * 100))
            m["false_reject_at_" + suffix + "pct_val_far"] = float((prob[y == 1] < threshold).mean())
            m["achieved_far_at_" + suffix + "pct_val_far"] = float((prob[y == 0] >= threshold).mean())
        judge_metrics.setdefault(split, {})[name] = m
        judge_rows.append(dict(split=split, model=name, **m))
display(pd.DataFrame(judge_rows))
print("Trained Platt [a, b]:", platt, "baseline:", baseline_platt)

player_metrics = {}; player_rows = []
for split in ("val", "test"):
    for name, queries, keys in (("baseline", E0, E0), ("trained", arrays["q"], arrays["k"])):
        m = player_eval(split, queries, keys)
        player_metrics.setdefault(split, {})[name] = m
        player_rows.append(dict(split=split, model=name, **m))
display(pd.DataFrame(player_rows))

sanity_pairs = "cat dog,cat purr,coffee morning,ocean wave,doctor hospital,bank river,bank money,hot cold,cat piano,apple galaxy,dog bark,bird fly,car road,food hunger,book read,night moon,shoe foot,fish water,tree leaf,milk cow,rain umbrella,fire smoke,ice cold,sun warm,king queen,chair table,music guitar,bread butter,school teacher,phone call,apple fruit,cat kitten".split(",")
sanity = []
for pair in sanity_pairs:
    a, b = pair.split()
    if a in index and b in index:
        ia, ib = index[a], index[b]
        logit = float(judge_scores(dict(a=np.array([ia]), b=np.array([ib])))[0])
        sanity.append(dict(pair=pair, baseline=float(calibrated(E0[ia] @ E0[ib], baseline_platt)),
                           trained=float(calibrated(logit, platt)), status="in vocabulary"))
    else:
        sanity.append(dict(pair=pair, baseline=None, trained=None, status="OOV"))
display(pd.DataFrame(sanity))
candidate_rows = []
for cue in ("cat", "coffee", "ocean", "doctor", "bank", "hot", "apple", "music"):
    if cue not in index:
        candidate_rows.append(dict(cue=cue, model="both", candidates="OOV")); continue
    a = index[cue]; mask = legal & (lemma_ids != lemma_ids[a])
    for name, score in (("baseline", E0 @ E0[a]), ("trained", arrays["k"] @ arrays["q"][a])):
        score = np.where(mask, score, -np.inf)
        top = np.argsort(-score)[:10]
        candidate_rows.append(dict(cue=cue, model=name, candidates=", ".join(words[i] for i in top)))
display(pd.DataFrame(candidate_rows))
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for name, scores, ab in (("baseline", raw_scores, baseline_platt), ("trained", trained_scores, platt)):
    fpr, tpr, _ = roc_curve(eval_sets["test"]["y"], calibrated(scores["test"], ab))
    axes[0].plot(fpr, tpr, label=name)
axes[0].plot([0, 1], [0, 1], "k--", alpha=.4)
axes[0].set(xlabel="False accept rate", ylabel="True accept rate", title="Test judge ROC")
axes[0].legend()
h = pd.DataFrame(history)
for name in ("loss", "retrieval", "judge"): axes[1].plot(h.step, h[name], label=name)
axes[1].set(xlabel="Step", ylabel="Loss", title="Training losses"); axes[1].legend()
for name in ("val_auc", "val_recall10", "selection_score"):
    points = h.dropna(subset=[name])
    axes[2].plot(points.step, points[name], "o-", label=name)
axes[2].set(xlabel="Step", ylabel="Metric", title="Validation / model selection"); axes[2].legend()
plt.tight_layout(); plt.show()
metrics = {s: dict(judge=judge_metrics[s], player=player_metrics[s]) for s in ("val", "test")}
metrics["training"] = dict(seconds=training_seconds, steps=step, selected_step=best_step,
    selection_criterion="val_auc + val_recall10", best_selection_score=float(best_score),
    selected_val_auc=float(best_auc), selected_subset_recall10=float(best_recall10),
    selection_cues=selection_cues.tolist(), baseline_selection=baseline_selection,
    initial_player_max_vector_error=initial_player_error)
metrics["vocabulary"] = dict(words=N, base=len(base), extensions=N-len(base),
    playable=int(playable.sum()), legal=int(legal.sum()), seeds=len(seeds),
    descending_zipf=True, known_pairs_count=int(len(known_pairs_export)))
print("Elapsed before export:", round(time.perf_counter() - START, 2), "seconds")
''')
md('''
## 7. Calibration and acceptance thresholds

The two-parameter Platt fits above used validation logits only. Choose probability
thresholds from the same validation random+hard negative mixture to target false
acceptance of 15% (lenient), 8% (normal), and 3% (strict). Show achieved rates and
positive rejection; empirical quantiles/ties can prevent an exact target. Calibration
reflects this constructed 1:2 positive/negative mixture, not gameplay prevalence.
''')
code('''
val_prob = calibrated(trained_scores["val"], platt)
thresholds = {}; threshold_rows = []
for level, target_far in (("lenient", .15), ("normal", .08), ("strict", .03)):
    threshold = far_threshold(val_prob, eval_sets["val"]["y"], target_far)
    thresholds[level] = threshold
    row = dict(level=level, threshold=threshold, target_far=target_far)
    for split in ("val", "test"):
        prob = calibrated(trained_scores[split], platt); y = eval_sets[split]["y"]
        row[split + "_far"] = float((prob[y == 0] >= threshold).mean())
        row[split + "_false_reject"] = float((prob[y == 1] < threshold).mean())
    threshold_rows.append(row)
assert thresholds["lenient"] <= thresholds["normal"] <= thresholds["strict"]
display(pd.DataFrame(threshold_rows))
metrics["threshold_operating_points"] = threshold_rows
''')
md('''
## 8. Export, download and verify the serving contract

Precompute all vectors in eval mode and normalize them; export float16 arrays and
float32 judge weights, transposed for NumPy `x @ W + b`. `meta.json` stores dimensions,
calibration, thresholds, test metrics, source/license notes and effective configuration.
`metrics.json` additionally retains validation results, training history and diagnostics.
`q.npy`/`k.npy` are `[N, dim_e]`, with `dim_q=dim_e` in metadata. `judge.npz`
includes float32 `w_cos [1]`; metadata records `judge_formula="mlp(x) + w_cos*cos"`.
The judge serving formula is `sigmoid(platt_a * (mlp(x) + w_cos*cos) + platt_b)`.
`known_pairs.npy` is int32 `[M, 2]`, deduplicated, both directions, from every
eligible source pair across all splits. `known_pairs_count` records M. The lookup
includes held-out supervision: it is for serving only and never used in these metrics.
The serving loader must implement the cosine skip and optional attested-pair policy.

The last cell reloads the artifacts, uses NumPy alone for inference, and compares
1,000 random pair probabilities against PyTorch at tolerance 1e-3. It also checks
schema, shapes, dtypes, normalization and symmetric judge behavior.
''')
code('''
OUT = Path(CFG["ARTIFACT_DIR"]); OUT.mkdir(parents=True, exist_ok=True)
licenses = {
    "glove": "PDDL 1.0; Stanford GloVe 6B",
    "eat": "Mirror declares Open Government Licence; original licensor provenance unconfirmed",
    "usf": "Original norms deployment license unresolved; mirror code MIT does not relicense data",
    "conceptnet": "CC BY-SA 4.0; ConceptNet contributors; mirror may omit provenance/weights",
    "bad_words": "CC BY 4.0; LDNOOBW / Shutterstock contributors",
    "wordfreq": "Data CC BY-SA 4.0; software Apache 2.0; Robyn Speer and contributors",
    "wordnet": "Princeton WordNet license; retain Princeton copyright/permission notices",
}
sources = {"glove": dict(url=URLS[CFG["EMB"]], license=licenses["glove"])}
for source in ("eat", "usf", "bad_words"):
    sources[source] = dict(url=URLS[source], license=licenses[source])
if not CFG["SMOKE"]:
    sources["conceptnet"] = dict(url=URLS["conceptnet"], license=licenses["conceptnet"])
sources["wordfreq"] = dict(url="https://github.com/rspeer/wordfreq", license=licenses["wordfreq"])
sources["wordnet"] = dict(url="https://wordnet.princeton.edu/license-and-commercial-use", license=licenses["wordnet"])
meta = dict(version="rst-assoc-v1-" + CFG["EMB"] + "-" + datetime.now(timezone.utc).strftime("%Y%m%d"),
            judge="mlp", judge_formula="mlp(x) + w_cos*cos", dim_e=dim_e,
            dim_z=CFG["Z"], dim_q=dim_e, known_pairs_count=int(len(known_pairs_export)),
            platt=platt, thresholds=thresholds,
            metrics=metrics["test"], data=sources, config=CFG)
def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
write_json(OUT / "meta.json", meta)
write_json(OUT / "vocab.json", dict(words=words, lemma=lemma, playable=playable.astype(int).tolist(), seeds=seeds))
np.save(OUT / "known_pairs.npy", known_pairs_export.astype(np.int32, copy=False))
for name, value in arrays.items():
    value = value / np.linalg.norm(value, axis=1, keepdims=True).clip(1e-12)
    np.save(OUT / (name + ".npy"), value.astype(np.float16))
judge_weights = {}
judge_weights["w_cos"] = model.w_cos.detach().cpu().numpy().copy().astype(np.float32)
for n, layer in enumerate((model.judge[0], model.judge[2], model.judge[4]), 1):
    judge_weights["W" + str(n)] = layer.weight.detach().cpu().numpy().T.copy().astype(np.float32)
    judge_weights["b" + str(n)] = layer.bias.detach().cpu().numpy().copy().astype(np.float32)
np.savez(OUT / "judge.npz", **judge_weights)
fig.savefig(OUT / "evaluation.png", dpi=140, bbox_inches="tight")
metrics["training"]["total_pipeline_seconds"] = time.perf_counter() - START
metrics["config"] = CFG
metrics["history"] = history
metrics["sanity_pairs"] = sanity
metrics["bot_candidates"] = candidate_rows
write_json(OUT / "metrics.json", metrics)
print("Artifacts:", OUT.resolve())
print("Artifact MB:", round(sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) / 1e6, 2))
''')
code('''
# Freeze a torch reference first. The reload/inference block below requires only numpy/json/pathlib.
check_rng = np.random.default_rng(CFG["SEED"] + 99)
check_a, check_b = check_rng.integers(0, N, size=(2, 1000))
torch_reference = calibrated(judge_scores(dict(a=check_a, b=check_b)), platt)

# ---- NumPy-only artifact reload + serving formula ----
import numpy as np
import json
from pathlib import Path
export_path = Path(CFG["ARTIFACT_DIR"])
m = json.loads((export_path / "meta.json").read_text())
v = json.loads((export_path / "vocab.json").read_text())
loaded = {name: np.load(export_path / (name + ".npy")) for name in ("emb", "z", "q", "k")}
with np.load(export_path / "judge.npz") as f:
    J = {key: f[key] for key in f.files}
n, de, dz, dq = len(v["words"]), m["dim_e"], m["dim_z"], m["dim_q"]
assert set(m) == {"version", "judge", "judge_formula", "dim_e", "dim_z", "dim_q", "known_pairs_count",
                  "platt", "thresholds", "metrics", "data", "config"}
assert dq == de and m["judge_formula"] == "mlp(x) + w_cos*cos"
assert set(v) == {"words", "lemma", "playable", "seeds"}
assert m["judge"] == "mlp" and len(v["playable"]) == n and len(set(v["words"])) == n
assert set(m["thresholds"]) == {"lenient", "normal", "strict"} and len(m["platt"]) == 2
for name, array in loaded.items():
    expected_dim = dz if name == "z" else de if name == "emb" else dq
    assert array.dtype == np.float16 and array.shape == (n, expected_dim)
    assert np.isfinite(array).all()
    assert np.max(np.abs(np.linalg.norm(array.astype(np.float32), axis=1) - 1)) < 1e-3
expected_shapes = dict(W1=(2*dz+1,64), b1=(64,), W2=(64,32), b2=(32,), W3=(32,1), b3=(1,), w_cos=(1,))
assert set(J) == set(expected_shapes)
for key, shape in expected_shapes.items():
    assert J[key].shape == shape and J[key].dtype == np.float32 and np.isfinite(J[key]).all()
loaded = {key: value.astype(np.float32) for key, value in loaded.items()}
attested_loaded = np.load(export_path / "known_pairs.npy")
assert attested_loaded.dtype == np.int32 and attested_loaded.shape == (m["known_pairs_count"], 2)
assert np.all((attested_loaded >= 0) & (attested_loaded < n))
codes = attested_loaded[:, 0].astype(np.int64) * n + attested_loaded[:, 1]
reverse_codes = attested_loaded[:, 1].astype(np.int64) * n + attested_loaded[:, 0]
assert np.all(np.diff(codes) > 0)  # lexicographically sorted and deduplicated
assert np.array_equal(np.sort(reverse_codes), codes)
assert np.array_equal(attested_loaded, known_pairs_export)  # no source/split pairs lost in export
def numpy_judge(ia, ib):
    ea, eb = loaded["emb"][ia], loaded["emb"][ib]
    za, zb = loaded["z"][ia], loaded["z"][ib]
    cosine = np.sum(ea * eb, axis=1, keepdims=True)
    x = np.concatenate((za * zb, np.abs(za - zb), cosine), axis=1)
    x = np.maximum(0., x @ J["W1"] + J["b1"])
    x = np.maximum(0., x @ J["W2"] + J["b2"])
    logit = (x @ J["W3"] + J["b3"])[:, 0] + J["w_cos"][0] * cosine[:, 0]
    return 1. / (1. + np.exp(-np.clip(m["platt"][0] * logit + m["platt"][1], -80, 80)))
numpy_reference = numpy_judge(check_a, check_b)
max_error = float(np.max(np.abs(numpy_reference - torch_reference)))
assert np.allclose(numpy_reference, torch_reference, atol=1e-3, rtol=0), max_error
assert np.allclose(numpy_reference, numpy_judge(check_b, check_a), atol=1e-7, rtol=0)
print(f"PASS: NumPy-only serving contract, 1000 pairs, max probability error={max_error:.8f}")
# Only package and offer download after the contract check has passed.
metrics["contract"] = dict(pairs=1000, max_probability_error=max_error, tolerance=1e-3)
write_json(OUT / "metrics.json", metrics)
zip_path = Path(shutil.make_archive(str(OUT), "zip", root_dir=OUT))
print("Zip:", zip_path.resolve(), "; pipeline minutes:", round((time.perf_counter()-START)/60, 2))
summary_rows = []
for split in ("val", "test"):
    for name in ("baseline", "trained"):
        j, p = metrics[split]["judge"][name], metrics[split]["player"][name]
        summary_rows.append(dict(split=split, model=name, judge_auc=j["roc_auc"],
            player_recall10=p["recall@10"], player_mrr=p["mrr"]))
print("Final model-only results (attested lookup excluded):")
display(pd.DataFrame(summary_rows))
print("Selected checkpoint:", best_step, "; known pairs:", m["known_pairs_count"],
      "; dimensions E/Z/Q:", de, dz, dq, "; max parity error:", max_error)
try:
    from google.colab import files
    files.download(str(zip_path))
except ImportError:
    print("Outside Colab: download the zip from the printed path.")
except Exception as exc:
    print("Automatic Colab download unavailable; use the Files sidebar:", exc)
''')

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "colab": {"name": "RST_train_association_model.ipynb", "provenance": []},
})
# Stable IDs keep regenerated notebook diffs readable.
for i, cell in enumerate(nb.cells):
    cell.id = f"rst-{i:02d}"
nbf.validate(nb)
target = Path(__file__).with_name("RST_train_association_model.ipynb")
nbf.write(nb, target)
print(f"Wrote {target} ({len(cells)} cells)")
