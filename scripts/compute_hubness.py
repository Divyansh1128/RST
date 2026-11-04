"""Measure judge hubness and pick the hub penalty, then save hub.npy into the artifacts.

Some very common words ("film", "school", "man") are judged related to a big share of
random cue words, so a player could spam them. For every playable word we measure
hub[w] = share of random cues c with judge(c, w) >= the Normal threshold. At serving
time a *novel* pair needs  p - beta * hub[w] >= threshold  (attested pairs are exempt).

beta is chosen on a tradeoff curve: random-pair false accepts on the 500 worst hubs vs.
model-only recall on known association pairs.

Usage: python scripts/compute_hubness.py artifacts/trained
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.nlp import AssociationModel  # noqa: E402

N_CUES, SEED = 200, 7


def main(art_dir):
    art = Path(art_dir)
    m = AssociationModel(art)
    thr = m.threshold("normal")
    rng = np.random.default_rng(SEED)
    P = m.play_idx
    cues = rng.choice(P, N_CUES, replace=False)

    hub = np.zeros(len(m.words), dtype=np.float32)
    for start in range(0, len(P), 500):
        bs = P[start:start + 500]
        probs = m._pair_prob(np.repeat(cues, len(bs)), np.tile(bs, len(cues)))
        hub[bs] = (probs >= thr).reshape(len(cues), len(bs)).mean(0)
    print("hub percentiles 50/90/99/99.9:", np.percentile(hub[P], [50, 90, 99, 99.9]).round(3))

    # Tradeoff: fresh random cues against the worst hubs vs. known pairs (model only).
    worst = P[np.argsort(-hub[P])[:500]]
    test_cues = rng.choice(P, 300, replace=False)
    a_neg = np.repeat(test_cues, len(worst)); b_neg = np.tile(worst, len(test_cues))
    p_neg = m._pair_prob(a_neg, b_neg)
    kp = np.load(art / "known_pairs.npy").astype(np.int64)
    kp = kp[m.playable[kp[:, 1]]]
    kp = kp[rng.choice(len(kp), min(20000, len(kp)), replace=False)]
    p_pos = m._pair_prob(kp[:, 0], kp[:, 1])
    rows = []
    for beta in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        fa = float(((p_neg - beta * hub[b_neg]) >= thr).mean())
        rec = float(((p_pos - beta * hub[kp[:, 1]]) >= thr).mean())
        rows.append((beta, fa, rec))
        print(f"beta={beta:.1f}  hub false-accept={fa:6.1%}  known-pair recall(model only)={rec:6.1%}")
    base_rec = rows[0][2]
    # Largest false-accept cut that costs at most 3 recall points.
    beta = max((r for r in rows if base_rec - r[2] <= 0.03), key=lambda r: -r[1])[0]
    print("chosen beta:", beta)

    np.save(art / "hub.npy", hub.astype(np.float16))
    meta = json.loads((art / "meta.json").read_text())
    meta["hub"] = {"beta": beta, "n_cues": N_CUES, "threshold": "normal",
                   "tradeoff": [dict(beta=b, hub_false_accept=f, known_recall=r) for b, f, r in rows]}
    (art / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
