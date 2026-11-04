# RST — the word game

Players take turns saying a word **related to the previous word**. The catch: the word may
**not start with R, S or T**, and no word can be used twice (`cat` and `cats` count as the
same word). Fail, repeat, or run out of time and you lose a life; the last player standing wins.
Every accepted word shortens the clock by 4%, so even perfect players eventually crack.

* Online rooms with a 5-letter code (2–7 players), invite links, reconnect on refresh
* AI players at five levels: **Rookie, Casual, Sharp, Pro, Grandmaster**
  (Grandmaster deliberately plays words whose best answers start with R/S/T)
* A trained neural association model judges every move and powers the bots

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# put model artifacts in artifacts/current (see below), then:
.venv/bin/uvicorn server.app:app --port 8000
# open http://localhost:8000
```

## The model

Training runs in Google Colab: open `notebooks/RST_train_association_model.ipynb`, choose a
GPU runtime, **Run all**. It downloads the data, trains, evaluates against a raw-GloVe baseline,
and downloads `rst_artifacts.zip`. Then:

```bash
mkdir -p artifacts/trained && unzip rst_artifacts.zip -d artifacts/trained
.venv/bin/python scripts/compute_hubness.py artifacts/trained   # ~1.5 min, see below
ln -sfn trained artifacts/current
```

`compute_hubness.py` fixes a known weakness of embedding judges: very common words
("film", "school", "man") look related to everything, so a player could spam them. It
measures each word's hubness (share of random cues it gets accepted against), picks a
penalty on a recall/false-accept tradeoff curve, and saves `hub.npy`. On the v1 model it
cut hub-word false accepts from 15.4% to 10.3% for 2.9 points of novel-pair recall;
attested associations are unaffected.

Then run `notebooks/RST_evaluate_bots.ipynb` (CPU is fine; upload the same zip). It runs the
exact serving code, lets you query the judge, and simulates bot-vs-bot games to report each AI
level's success rate and a head-to-head win-rate matrix.

Both notebooks are generated: edit `notebooks/build_notebook.py` / `build_eval_notebook.py`
and re-run them rather than editing the `.ipynb` by hand.

Architecture (see `docs/agents/astra_research_report.md` for the research behind it):

* GloVe-initialised word embeddings, fine-tuned with a drift penalty
* encoder MLP → normalised association space `z`
* **judge**: MLP over `[z_a·z_b, |z_a−z_b|, cos(e_a,e_b)]`, Platt-calibrated; thresholds for
  Lenient / Normal / Strict picked on validation false-accept rates
* **player**: directional retrieval `q(a)·k(b)` trained with multi-positive InfoNCE on human
  cue→response data; bots sample from this ranking according to their skill level

Data: USF Free Association Norms, Edinburgh Associative Thesaurus, ConceptNet 5 (CC BY-SA 4.0),
WordNet, wordfreq (CC BY-SA 4.0), GloVe (PDDL). Split 80/10/10 **by cue**, so test cues are
never seen in training.

Serving is numpy-only (`server/nlp.py`): ~2 ms per bot move on a laptop CPU.

`scripts/build_baseline.py` builds untrained GloVe-cosine artifacts, used both as a dev fallback
and as the baseline the trained model is compared against.

## Layout

```
server/   FastAPI + WebSocket rooms (game.py rules, bots.py AI players, nlp.py model)
web/      the site — plain HTML/CSS/JS, no build step
notebooks/ Colab training notebook (+ its generator script)
tests/    pytest for the game rules
```

## Deploy

Any host that supports WebSockets (Render, Railway, Fly.io…) can run the `Dockerfile`.
Rooms live in memory, so run a single instance.
