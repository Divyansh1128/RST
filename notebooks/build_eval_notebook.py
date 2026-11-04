"""Generate notebooks/RST_evaluate_bots.ipynb.

The notebook embeds the exact serving code (server/nlp.py, server/bots.py) so the
numbers it reports are for the code the game actually runs.
Run:  python notebooks/build_eval_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# RST: evaluate the judge and calibrate the AI difficulty levels

Run this **after** `RST_train_association_model.ipynb`. It needs `rst_artifacts.zip`
(the notebook's download), and it runs fine on a CPU runtime.

1. Load the artifacts with the **same numpy code the game server uses**
2. Try the judge interactively
3. Simulate thousands of bot-vs-bot games to measure, for each difficulty level,
   the mistake rate, link strength, and head-to-head win rate
""")
md("## 1. Serving code (copied verbatim from the repo)")
code("%%writefile rst_nlp.py\n" + (ROOT / "server/nlp.py").read_text())
code("%%writefile rst_bots.py\n" + (ROOT / "server/bots.py").read_text())
md("## 2. Load artifacts\nUpload `rst_artifacts.zip` when prompted (or set `ZIP` to a Drive path).")
code("""
import os, zipfile, json, random, time, itertools
from pathlib import Path
import numpy as np
ZIP = "rst_artifacts.zip"
if not Path(ZIP).exists():
    try:
        from google.colab import files
        ZIP = next(iter(files.upload()))
    except ImportError:
        raise SystemExit("Put rst_artifacts.zip next to this notebook.")
with zipfile.ZipFile(ZIP) as z:
    z.extractall("artifacts")
ART = next(p.parent for p in Path("artifacts").rglob("meta.json"))
from rst_nlp import AssociationModel
from rst_bots import PROFILES, choose_word
model = AssociationModel(ART)
print(model.meta["version"], "|", len(model.words), "words |", int(model.playable.sum()), "playable |",
      len(model.known) // 2, "known pairs")
print("thresholds:", model.meta["thresholds"])
t = time.perf_counter(); [model.candidates("coffee", k=60) for _ in range(50)]
print(f"bot candidate search: {(time.perf_counter()-t)/50*1000:.1f} ms")
""")
md("## 3. Try the judge")
code("""
PAIRS = [("cat","dog"),("cat","purr"),("coffee","morning"),("ocean","wave"),("doctor","nurse"),
         ("bank","river"),("bank","money"),("hot","cold"),("bread","butter"),("moon","night"),
         ("cat","piano"),("apple","galaxy"),("chair","volcano"),("guitar","banana")]
thr = model.meta["thresholds"]
for a, b in PAIRS:
    p = model.relatedness(a, b)
    verdict = "strict" if p >= thr["strict"] else "normal" if p >= thr["normal"] else "lenient" if p >= thr["lenient"] else "reject"
    print(f"{a:>8} -> {b:<8} {p:.3f}  accepted at: {verdict}")
""")
code("""
def judge(a, b):
    a, b = a.lower().strip(), b.lower().strip()
    if not model.in_vocab(b): return f"'{b}' is not in the dictionary"
    if b[0] in "rst": return f"'{b}' starts with {b[0].upper()} - banned"
    return f"{a} -> {b}: {model.relatedness(a, b):.3f}"
for cue in ["ocean", "coffee", "winter", "music", "doctor"]:
    print(cue, "->", [w for w, _ in model.candidates(cue, k=12)])
print(judge("winter", "cold"))
""")
md("""
## 4. Difficulty calibration by simulation

This mirrors the server rules: 3 lives, a miss costs a life and passes the same word on,
no repeats, and a clock that shrinks 4% per accepted word (15 s start, 4 s floor). A bot
"freezes" if it returns no word, and its think time is compared with the clock.
""")
code("""
BASE, MIN_TURN, SPEEDUP = 15.0, 4.0, 0.96
def simulate(levels, lives=3, strictness="normal", seed=0, max_turns=400):
    rng = random.Random(seed)
    thr = model.threshold(strictness)
    seedw = rng.choice(model.seeds)
    chain, used = [seedw], {model.normalize(seedw)}
    life = [lives] * len(levels); stats = [dict(turns=0, ok=0, miss=0, probs=[]) for _ in levels]
    turn = rng.randrange(len(levels)); n = 0
    while sum(l > 0 for l in life) > 1 and n < max_turns:
        n += 1
        length = max(MIN_TURN, BASE * SPEEDUP ** (len(chain) - 1))
        scale = min(1.0, length / BASE)
        prof = PROFILES[levels[turn]]
        think = rng.uniform(*prof.think) * (0.5 + 0.5 * scale)
        w = choose_word(model, chain[-1], used, levels[turn], thr, rng=rng, pressure=1 - scale)
        st = stats[turn]; st["turns"] += 1
        ok = (w is not None and think <= length and w[0] not in "rst" and model.in_vocab(w)
              and model.normalize(w) not in used and model.relatedness(chain[-1], w) >= thr)
        if ok:
            st["ok"] += 1; st["probs"].append(model.relatedness(chain[-1], w))
            chain.append(w); used.add(model.normalize(w))
        else:
            st["miss"] += 1; life[turn] -= 1
        for step in range(1, len(levels) + 1):
            j = (turn + step) % len(levels)
            if life[j] > 0: turn = j; break
    alive = [i for i, l in enumerate(life) if l > 0]
    return (alive[0] if len(alive) == 1 else None), stats, chain

LEVELS = list(PROFILES)
agg = {l: dict(turns=0, ok=0, probs=[]) for l in LEVELS}
for g in range(300):
    _, stats, _ = simulate(LEVELS, seed=g)
    for l, s in zip(LEVELS, stats):
        agg[l]["turns"] += s["turns"]; agg[l]["ok"] += s["ok"]; agg[l]["probs"] += s["probs"]
print(f"{'level':<12}{'success rate':>14}{'mean link prob':>16}")
for l in LEVELS:
    a = agg[l]
    print(f"{l:<12}{a['ok']/max(1,a['turns']):>14.1%}{np.mean(a['probs']):>16.3f}")
""")
code("""
# Head-to-head: row level vs column level, 200 one-on-one games each.
import pandas as pd
wins = pd.DataFrame(0.0, index=LEVELS, columns=LEVELS)
for a, b in itertools.permutations(LEVELS, 2):
    w = sum(simulate([a, b], seed=1000 + g)[0] == 0 for g in range(200))
    wins.loc[a, b] = w / 200
print("Win rate of ROW against COLUMN (1v1):")
display(wins.style.format("{:.0%}").background_gradient(cmap="RdYlGn", vmin=0, vmax=1))
""")
code("""
winner, stats, chain = simulate(["casual", "pro", "grandmaster"], seed=7)
print("sample game chain:", " -> ".join(chain[:40]))
""")
md("""
A well-calibrated ladder has each level beating the one below it more often than not, and
Grandmaster beating Rookie almost always. If two adjacent levels come out near 50%, adjust
`PROFILES` in `server/bots.py` (rank window, blunder rate, think time) and re-run.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}
out = ROOT / "notebooks" / "RST_evaluate_bots.ipynb"
nbf.write(nb, out)
print("Wrote", out)
