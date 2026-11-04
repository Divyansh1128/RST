"""Serve the trained association model with numpy only (no torch at runtime).

Artifact directory layout (produced by notebooks/RST_train_association_model.ipynb):

  meta.json    {"version", "judge": "mlp" | "cosine", "platt": [a, b],
                "thresholds": {"lenient","normal","strict"}, "metrics": {...}}
  vocab.json   {"words": [...], "lemma": {word: lemma}, "playable": [0/1 per word],
                "seeds": [...]}
  emb.npy      float16 [N, d]  fine-tuned word embeddings (L2-normalised)
  z.npy        float16 [N, Z]  judge space          (optional; defaults to emb)
  q.npy, k.npy float16 [N, Z]  directional player space: score(a->b) = q[a].k[b]
  judge.npz    MLP over [z_a*z_b, |z_a-z_b|, cos(e_a,e_b)]: W1, b1, W2, b2, W3, b3,
               plus optional w_cos: logit = mlp(x) + w_cos * cos(e_a, e_b)
  known_pairs.npy  (optional) int32 [M, 2] attested human/ConceptNet associations;
               these are accepted outright, the neural judge decides everything else.
  hub.npy      (optional) float16 [N] hubness from scripts/compute_hubness.py; a novel
               pair scores p - meta.hub.beta * hub[word] so generic words can't be spammed.

The judge is symmetric. Its probability is sigmoid(platt_a * logit + platt_b);
the cosine fallback (baseline artifacts) is sigmoid(a * cos + b) with a, b = platt.
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np

FORBIDDEN = ("r", "s", "t")
KNOWN_PAIR_PROB = 0.95  # attested associations pass even the strict threshold


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class AssociationModel:
    def __init__(self, artifact_dir: str | Path):
        d = Path(artifact_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        vocab = json.loads((d / "vocab.json").read_text())
        self.words: list[str] = vocab["words"]
        self.index = {w: i for i, w in enumerate(self.words)}
        self.lemma: dict[str, str] = vocab.get("lemma", {})
        self.playable = np.asarray(vocab.get("playable", [1] * len(self.words)), dtype=bool)
        self.seeds = [w for w in vocab.get("seeds", []) if w in self.index and w[0] not in FORBIDDEN
                      and self.lemma.get(w, w) == w]  # "hat", not "hats"
        self.E = np.load(d / "emb.npy").astype(np.float32)
        load = lambda name: np.load(d / name).astype(np.float32) if (d / name).exists() else self.E
        self.Z, self.Q, self.K = load("z.npy"), load("q.npy"), load("k.npy")
        self.forbidden_mask = np.array([w[0] in FORBIDDEN for w in self.words])
        self.judge = None
        if self.meta.get("judge") == "mlp" and (d / "judge.npz").exists():
            self.judge = dict(np.load(d / "judge.npz"))
        # Retrieval only ever returns playable words, so search just those rows.
        self.play_idx = np.flatnonzero(self.playable)
        self.K_play = np.ascontiguousarray(self.K[self.play_idx])
        del self.K  # only the playable rows are ever searched; frees ~50 MB
        self.play_forbidden = self.forbidden_mask[self.play_idx]
        # Known pairs as sorted int64 keys a*N+b (both directions) for vectorised lookup.
        n = len(self.words)
        self.known = np.zeros(0, dtype=np.int64)
        if (d / "known_pairs.npy").exists():
            kp = np.load(d / "known_pairs.npy").astype(np.int64)
            self.known = np.unique(np.concatenate([kp[:, 0] * n + kp[:, 1], kp[:, 1] * n + kp[:, 0]]))
        self.hub = np.load(d / "hub.npy").astype(np.float32) if (d / "hub.npy").exists() else None
        self.hub_beta = float(self.meta.get("hub", {}).get("beta", 0.0))
        self.platt = self.meta.get("platt") or self.meta.get("cosine_calib", [8.0, -3.0])
        self._thr = self.meta["thresholds"]

    # ------------------------------------------------------------ lexicon
    def in_vocab(self, w: str) -> bool:
        return w in self.index

    def normalize(self, w: str) -> str:
        return self.lemma.get(w, w)

    def threshold(self, strictness: str) -> float:
        return float(self._thr.get(strictness, self._thr["normal"]))

    def random_seed_word(self) -> str:
        if self.seeds:
            return random.choice(self.seeds)
        pool = np.flatnonzero(self.playable[:3000] & ~self.forbidden_mask[:3000])
        return self.words[int(random.choice(pool))]

    # ------------------------------------------------------------- scoring
    def _score(self, i: int, js: np.ndarray) -> np.ndarray:
        """Game score for prev word i -> each candidate j: attested pairs pass outright,
        novel pairs pay a hubness penalty for the played word."""
        js = np.asarray(js)
        p = self._pair_prob(np.full(len(js), i), js)
        if self.hub is not None:
            p = np.clip(p - self.hub_beta * self.hub[js], 0.0, 1.0)
        return np.where(self._is_known(i, js), np.maximum(p, KNOWN_PAIR_PROB), p)

    def _is_known(self, i: int, js: np.ndarray) -> np.ndarray:
        if not len(self.known):
            return np.zeros(len(js), dtype=bool)
        keys = i * len(self.words) + np.asarray(js, dtype=np.int64)
        pos = np.minimum(np.searchsorted(self.known, keys), len(self.known) - 1)
        return self.known[pos] == keys

    def _pair_prob(self, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
        """Index arrays [n] -> [n] judge probabilities."""
        a, b = self.platt
        cos = np.sum(self.E[ia] * self.E[ib], axis=1, keepdims=True)
        if self.judge is None:
            return _sigmoid(a * cos[:, 0] + b)
        J, za, zb = self.judge, self.Z[ia], self.Z[ib]
        x = np.concatenate([za * zb, np.abs(za - zb), cos], axis=1)
        h = np.maximum(0, x @ J["W1"] + J["b1"])
        h = np.maximum(0, h @ J["W2"] + J["b2"])
        logit = (h @ J["W3"] + J["b3"])[:, 0]
        if "w_cos" in J:
            logit = logit + float(J["w_cos"].reshape(-1)[0]) * cos[:, 0]
        return _sigmoid(a * logit + b)

    @lru_cache(maxsize=50_000)
    def relatedness(self, a: str, b: str) -> float:
        if a not in self.index or b not in self.index:
            return 0.0
        if self.normalize(a) == self.normalize(b):
            return 0.0
        return float(self._score(self.index[a], np.array([self.index[b]]))[0])

    def candidates(self, prev: str, exclude: set[str] = frozenset(), k: int = 50,
                   allow_forbidden: bool = False, retrieve: int = 200) -> list[tuple[str, float]]:
        """Top-k playable next words for `prev`, in the order a person would think of
        them (player head), each with its judge probability."""
        if prev not in self.index:
            return []
        i = self.index[prev]
        sims = self.K_play @ self.Q[i]
        if not allow_forbidden:
            sims[self.play_forbidden] = -np.inf
        retrieve = min(retrieve, len(sims) - 1)
        top = np.argpartition(-sims, retrieve)[:retrieve]
        top = top[np.isfinite(sims[top])]
        top = self.play_idx[top[np.argsort(-sims[top])]]
        top = top[top != i]
        probs = self._score(i, top)
        prev_norm = self.normalize(prev)
        out, seen = [], set()
        for j, pr in zip(top, probs):
            w = self.words[int(j)]
            n = self.normalize(w)
            if n in exclude or n in seen or n == prev_norm or prev_norm in w or w in prev:
                continue
            seen.add(n)
            out.append((w, float(pr)))
            if len(out) >= k:
                break
        return out

    def forbidden_pressure_many(self, words: list[str], used: set[str], k: int = 20) -> np.ndarray:
        """For each word, the share of its k most natural follow-ups (player head) that the
        next player cannot use: banned initial letter or already played. One matmul."""
        idx = np.array([self.index[w] for w in words])
        sims = self.K_play @ self.Q[idx].T                      # [P, n]
        top = np.argpartition(-sims, k + 4, axis=0)[: k + 4]    # a few spare for self/lemma hits
        out = np.zeros(len(words))
        for c, i in enumerate(idx):
            col = top[:, c][np.argsort(-sims[top[:, c], c])]
            norm_i = self.normalize(self.words[i])
            picks = [self.words[j] for j in self.play_idx[col]
                     if self.normalize(self.words[j]) != norm_i][:k]
            out[c] = sum(w[0] in FORBIDDEN or self.normalize(w) in used for w in picks) / max(1, len(picks))
        return out

    def forbidden_pressure(self, w: str, used: set[str], k: int = 20) -> float:
        return float(self.forbidden_pressure_many([w], used, k)[0])
