Model artifacts live here (git-ignored).

* `baseline/` — raw GloVe cosine, built with `scripts/build_baseline.py`
* `trained/`  — unzip `rst_artifacts.zip` from the Colab notebook here
* `current`   — symlink to whichever one the server should load
                (`ln -sfn trained artifacts/current`), or set `RST_ARTIFACTS=/path`.
