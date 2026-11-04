"""Build *baseline* serving artifacts from raw GloVe-100d cosine (no training).

This exists so the game runs locally before the Colab-trained model is dropped in,
and it is the baseline the trained model must beat.

Usage:
  python scripts/build_baseline.py GLOVE_DIR OUT_DIR
GLOVE_DIR must hold the HuggingFace fse/glove-wiki-gigaword-100 files
(glove-wiki-gigaword-100.model renamed g.model, ...vectors.npy renamed g.npy).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})


class SafeUnpickler(pickle.Unpickler):
    """Reads the gensim KeyedVectors pickle without importing/executing gensim code."""
    def find_class(self, module, name):
        if module.startswith("numpy") or module in ("builtins", "collections", "copyreg"):
            if module == "builtins" and name not in ("dict", "list", "set", "tuple", "int", "str", "object"):
                raise pickle.UnpicklingError(f"blocked {module}.{name}")
            return super().find_class(module, name)
        return _Stub


def main(glove_dir, out_dir, n_vocab=50_000, n_playable=25_000):
    glove_dir, out = Path(glove_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kv = SafeUnpickler(open(glove_dir / "g.model", "rb")).load()
    keys = kv.__dict__.get("index_to_key") or kv.__dict__["index2word"]
    vecs = np.load(glove_dir / "g.npy", mmap_mode="r")

    stop = set(stopwords.words("english"))
    lem = WordNetLemmatizer()
    words, rows = [], []
    for i, w in enumerate(keys):
        if w.isalpha() and w.isascii() and len(w) >= 2:
            words.append(w)
            rows.append(i)
        if len(words) >= n_vocab:
            break
    E = np.asarray(vecs[rows], dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8

    lemma = {}
    for w in words:
        l = lem.lemmatize(lem.lemmatize(w, "n"), "v")
        if l != w:
            lemma[w] = l
    playable = [int(i < n_playable and w not in stop and len(w) >= 3 and bool(wn.synsets(w)))
                for i, w in enumerate(words)]
    seeds = [w for w in words[:4000] if w not in stop and w[0] not in "rst" and len(w) >= 3
             and any(s.pos() == "n" for s in wn.synsets(w)) and lemma.get(w, w) == w][:1500]

    np.save(out / "emb.npy", E.astype(np.float16))
    (out / "vocab.json").write_text(json.dumps({"words": words, "lemma": lemma,
                                                "playable": playable, "seeds": seeds}))
    # Raw cosine -> pseudo-probability. Cosine ~0.45 between associates maps to ~0.5.
    meta = {"version": "baseline-glove100-cosine", "dim": int(E.shape[1]), "judge": "cosine",
            "platt": [9.0, -4.0],
            "thresholds": {"lenient": 0.25, "normal": 0.4, "strict": 0.55},
            "metrics": {}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(words)} words, {sum(playable)} playable, {len(seeds)} seeds -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
