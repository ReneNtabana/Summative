"""Static word embeddings for Kirundi: Word2Vec and FastText (gensim).

Experiments
-----------
Corpus:
  * "kir"      : KIRNEWS training split only (~280k tokens; very small for embeddings)
  * "kir+kin"  : the same plus 11k KINNEWS articles in Kinyarwanda. Kinyarwanda and
                 Kirundi are closely related Bantu languages (largely mutually
                 intelligible, many shared roots) and are often treated as dialects of
                 one language. We test whether this extra text improves the vectors.
                 KINNEWS articles that also appear in KIRNEWS were removed in data.py,
                 so no Kirundi validation/test text is ever seen.
Algorithm:
  * Word2Vec skip-gram (Mikolov et al., 2013)
  * FastText skip-gram (Bojanowski et al., 2017). Each word is a bag of character
    n-grams, so unseen inflected forms still get a vector. That suits Kirundi's
    rich morphology (11% test OOV).

Downstream probe: a document is represented by the mean of its word vectors and
classified with logistic regression. The same vectors also initialise the BiLSTM (src/lstm.py).

Run:  python -m src.embeddings
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
from gensim.models import FastText, Word2Vec
from gensim.models.fasttext import FastTextKeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.data import PROC_DIR, ROOT, load_splits
from src.evaluate import compute_metrics, log_experiment

EMB_DIR = ROOT / "models" / "embeddings"
DIM = 200
SEED = 42
TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")


def tokenize(text: str) -> list[str]:
    """Word tokeniser shared by embeddings and LSTM (keeps elisions like rw'abatavuga)."""
    return TOKEN_RE.findall(text)


def corpus(name: str) -> list[list[str]]:
    sents = [tokenize(t) for t in load_splits()["train"].text]
    if name == "kir+kin":
        kin = pd.read_csv(PROC_DIR / "kinnews_unique.csv")
        sents += [tokenize(t) for t in kin.text]
    return sents


def train_embeddings(algo: str, corpus_name: str):
    sents = corpus(corpus_name)
    # sg=1 skip-gram: better than CBOW for rare words, which dominate here.
    # workers=4: much faster, but gensim is then only approximately reproducible
    # (thread scheduling changes the update order). Use workers=1 for exact repeats.
    # Note: gensim 4.4 may print "Exception ignored in ...our_dot_float" a few
    # times on macOS. We checked that it is benign: nearest neighbours are
    # sensible and the downstream probe works (see results/embeddings_log.txt).
    common = dict(sentences=sents, vector_size=DIM, window=5, min_count=2, sg=1,
                  epochs=10, seed=SEED, workers=4)
    if algo == "fasttext":
        # 200k hash buckets for character n-grams (default 2M = 1.6 GB on disk;
        # far more than a corpus of this size needs).
        model = FastText(bucket=200_000, min_n=3, max_n=6, **common)
    else:
        model = Word2Vec(**common)
    return model.wv


def path_for(algo: str, corpus_name: str):
    return EMB_DIR / f"{algo}_{corpus_name.replace('+', '_')}.kv"


def doc_vectors(wv, texts) -> np.ndarray:
    """Mean of word vectors. For Word2Vec, OOV words are skipped. FastText builds them from n-grams."""
    out = np.zeros((len(texts), wv.vector_size), dtype=np.float32)
    is_ft = isinstance(wv, FastTextKeyedVectors)
    for i, t in enumerate(texts):
        toks = tokenize(t)
        vecs = [wv[w] for w in toks if is_ft or w in wv.key_to_index]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


def main() -> None:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    s = load_splits()
    for algo in ("w2v", "fasttext"):
        for corpus_name in ("kir", "kir+kin"):
            t0 = time.time()
            if path_for(algo, corpus_name).exists():  # reuse saved vectors
                from gensim.models import KeyedVectors
                wv = KeyedVectors.load(str(path_for(algo, corpus_name)))
            else:
                wv = train_embeddings(algo, corpus_name)
                wv.save(str(path_for(algo, corpus_name)))
            secs = time.time() - t0

            # Probe: mean-pooled vectors + logistic regression (C tuned on val).
            Xtr, Xva, Xte = (doc_vectors(wv, s[k].text) for k in ("train", "val", "test"))
            best = None
            for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
                clf.fit(Xtr, s["train"].label_name)
                m = compute_metrics(s["val"].label_name, clf.predict(Xva))
                if best is None or m["macro_f1"] > best[1]["macro_f1"]:
                    best = (C, m, clf)
            C, val_m, clf = best
            test_m = compute_metrics(s["test"].label_name, clf.predict(Xte))
            name = f"avg-{algo}-{corpus_name}"
            cfg = {"dim": DIM, "C": C, "vocab": len(wv.key_to_index)}
            log_experiment("embeddings", name, cfg, "val", val_m, train_seconds=secs)
            log_experiment("embeddings", name, cfg, "test", test_m, train_seconds=secs)
            print(f"{name:22s} vocab={len(wv.key_to_index):6d} C={C:<4} val F1={val_m['macro_f1']:.3f} "
                  f"test acc={test_m['accuracy']:.3f} F1={test_m['macro_f1']:.3f}")

    # Qualitative check for the report: nearest neighbours of a few topic words.
    for algo, corpus_name in (("w2v", "kir"), ("w2v", "kir+kin"), ("fasttext", "kir+kin")):
        from gensim.models import KeyedVectors
        wv = KeyedVectors.load(str(path_for(algo, corpus_name)))
        print(f"\nNearest neighbours ({algo}, {corpus_name}):")
        for w in ("umupira", "amafaranga", "abanyeshure", "indwara", "leta"):
            try:
                print(f"  {w:12s} ->", ", ".join(x for x, _ in wv.most_similar(w, topn=6)))
            except KeyError:
                print(f"  {w:12s} -> (OOV)")


if __name__ == "__main__":
    main()
