"""Classical baselines: majority class and TF-IDF + linear models.

Why these baselines
-------------------
* Majority class: the floor. It shows what accuracy you get by always predicting "politics".
* TF-IDF + linear model: the standard strong baseline for topic classification
  (Joachims, 1998). It is also the baseline used in the original KINNEWS/KIRNEWS paper
  (Niyongabo et al., 2020).
* Word vs. character n-grams: Kirundi is agglutinative, so one root appears in many
  surface forms (e.g. abanyeshure / umunyeshure / ishure). EDA found that 61% of the
  training vocabulary are hapaxes and that the test OOV rate is 11%. Character n-grams
  share information across these forms, so we test whether they help.

Hyper-parameters (the regularisation C) are tuned on the validation set only; the
test set is used once per final configuration. We also report 5-fold stratified
cross-validation on train+val, because the smallest classes have only 4 test
items and a single split is noisy.

Run:  python -m src.baselines
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.data import ROOT, label_list, load_splits
from src.evaluate import compute_metrics, log_experiment, save_report

MODEL_DIR = ROOT / "models"
SEED = 42

# Tokeniser for Kirundi words: letters plus the elision apostrophe ("rw'abatavuga"
# stays one token). Tokens are lower-case already (see data.normalise).
WORD_PATTERN = r"(?u)\b[^\W\d_][^\W\d_']*(?:'[^\W\d_]+)*\b"


def word_tfidf(ngrams=(1, 2)):
    return TfidfVectorizer(token_pattern=WORD_PATTERN, ngram_range=ngrams, min_df=2,
                           sublinear_tf=True, max_features=100_000)


def char_tfidf():
    # char_wb = n-grams inside word boundaries; captures roots/affixes.
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, max_features=200_000)


def build(name: str, C: float = 1.0) -> Pipeline:
    """Factory for every baseline configuration we compare."""
    if name == "majority":
        return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
    feats = {
        "word": word_tfidf(),
        "char": char_tfidf(),
        "word+char": FeatureUnion([("w", word_tfidf()), ("c", char_tfidf())]),
    }
    feat_name, clf_name = name.split("|")
    clf = {
        "nb": ComplementNB(alpha=C),  # ComplementNB is designed for imbalanced text data
        "logreg": LogisticRegression(C=C, max_iter=3000, class_weight="balanced"),
        "svm": LinearSVC(C=C, class_weight="balanced"),
    }[clf_name]
    return Pipeline([("tfidf", feats[feat_name]), ("clf", clf)])


CONFIGS = [
    "word|nb", "word|logreg", "word|svm",
    "char|logreg", "char|svm",
    "word+char|logreg", "word+char|svm",
]
C_GRID = {"nb": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3], "logreg": [0.1, 0.3, 1, 3, 10, 30], "svm": [0.03, 0.1, 0.3, 1, 3]}


def main() -> None:
    s = load_splits()
    labels = label_list()
    Xtr, ytr = s["train"].text, s["train"].label_name
    Xva, yva = s["val"].text, s["val"].label_name
    Xte, yte = s["test"].text, s["test"].label_name
    Xall = pd.concat([Xtr, Xva]); yall = pd.concat([ytr, yva])
    rows = []

    # --- Majority class floor -------------------------------------------- #
    m = build("majority").fit(Xtr, ytr)
    for split, X, y in (("val", Xva, yva), ("test", Xte, yte)):
        met = compute_metrics(y, m.predict(X))
        log_experiment("baseline", "majority", {}, split, met)
    rows.append({"model": "majority", "C": "-", **met})

    # --- TF-IDF models ----------------------------------------------------- #
    best = (None, -1, None)
    for cfg in CONFIGS:
        clf_name = cfg.split("|")[1]
        # 1) tune C on validation
        val_scores = {}
        for C in C_GRID[clf_name]:
            t0 = time.time()
            pipe = build(cfg, C).fit(Xtr, ytr)
            val_scores[C] = compute_metrics(yva, pipe.predict(Xva))
            log_experiment("baseline", f"tfidf-{cfg}", {"C": C}, "val", val_scores[C],
                           train_seconds=time.time() - t0)
        C = max(val_scores, key=lambda c: val_scores[c]["macro_f1"])

        # 2) evaluate the selected C once on test
        t0 = time.time()
        pipe = build(cfg, C).fit(Xtr, ytr)
        secs = time.time() - t0
        test_met = compute_metrics(yte, pipe.predict(Xte))
        log_experiment("baseline", f"tfidf-{cfg}", {"C": C}, "test", test_met, train_seconds=secs)

        # 3) 5-fold CV on train+val (robustness check for the small classes)
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        cv_pred = cross_val_predict(build(cfg, C), Xall, yall, cv=cv, n_jobs=-1)
        cv_met = compute_metrics(yall, cv_pred)
        log_experiment("baseline", f"tfidf-{cfg}", {"C": C}, "cv5", cv_met)

        rows.append({"model": f"tfidf-{cfg}", "C": C, "val_macro_f1": val_scores[C]["macro_f1"],
                     "cv5_macro_f1": cv_met["macro_f1"], **test_met})
        print(f"{cfg:18s} C={C:<5} val F1={val_scores[C]['macro_f1']:.3f}  "
              f"cv5 F1={cv_met['macro_f1']:.3f}  test acc={test_met['accuracy']:.3f} F1={test_met['macro_f1']:.3f}")
        # Model selection uses validation macro-F1 only (never test).
        if val_scores[C]["macro_f1"] > best[1]:
            best = (cfg, val_scores[C]["macro_f1"], (pipe, C))

    cfg, _, (pipe, C) = best
    print(f"\nBest baseline by validation macro-F1: {cfg} (C={C})")
    save_report("baseline_best", yte, pipe.predict(Xte), labels)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "baseline_best.joblib")
    pd.DataFrame(rows).to_csv(Path("results") / "baselines_summary.csv", index=False)


if __name__ == "__main__":
    main()
