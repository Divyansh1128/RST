# Astra research brief — RST word game NLP model

Game: players take turns saying a word RELATED to the previous word. The new word may NOT start with R, S, or T. No repeats. Up to 7 players, some may be AI bots with difficulty levels (e.g. Rookie / Casual / Pro / Grandmaster).

We need a deep-learning NLP model that does two jobs:
1. JUDGE: given (prev_word, new_word) -> relatedness probability, used to accept/reject a human's move. Must be fast (<20 ms on CPU), forgiving of genuine associations (cat->dog, cat->purr, ocean->wave, coffee->morning) and reject unrelated ones (cat->piano).
2. PLAYER: given prev_word + set of used words, rank candidate next words (excluding R/S/T-initial and used words). Difficulty = which part of the ranked distribution to sample from + mistake rate.

HARD CONSTRAINTS: Apple M2, 8 GB RAM, only ~5 GB free disk TOTAL (including PyTorch install). CPU/MPS training, ideally < 30 min. Final served model artifacts should be < 150 MB. English only.

Please research (use web search) and answer concisely, with exact URLs:
A. Best freely downloadable word-association datasets (USF Free Association Norms, Small World of Words SWOW-EN, EAT, ConceptNet, WordNet etc.): exact download URL that works without a login/form, file size, format, license (can we use it in a public hobby/student web project?).
B. Best compact pretrained word embeddings to initialise from (GloVe 6B 100d/300d, fastText, ConceptNet Numberbatch mini, etc.): exact URL (HuggingFace mirrors preferred), size, license, reported performance on relatedness benchmarks (MEN, WordSim-353-rel, SimLex).
C. Recommend a model architecture + training objective for JUDGE and PLAYER that genuinely counts as "a DL model we trained" (not just cosine of off-the-shelf vectors), e.g. contrastive fine-tuning of embeddings (InfoNCE) on association pairs + small MLP/bilinear scorer with hard negatives. Include hyperparameters that fit the constraints.
D. Evaluation plan: held-out cue split, metrics (AUC for judge, recall@k / MRR for player), a baseline (raw cosine) to beat, and a vocabulary of common English words (~20-40k) to restrict to (source URL).
E. Pitfalls (e.g. inflections cat/cats, proper nouns, offensive words filter list URL).

Write your final answer as markdown. Be concrete, no fluff.
