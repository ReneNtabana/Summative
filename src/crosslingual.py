"""Extra TF-IDF experiments (CPU only):

1. OOD evaluation (MasakhaNEWS-run) of EVERY TF-IDF configuration at its validation-
   selected C, not only the one picked by validation. This makes the in-domain vs.
   out-of-domain comparison with the Transformers fair.
2. Cross-lingual transfer from Kinyarwanda (KINNEWS) to Kirundi, replicating the
   setting of Niyongabo et al. (2020, Table 7):
     a) zero-shot  : train on Kinyarwanda only, test on our Kirundi test split
     b) augmented  : train on Kirundi train + Kinyarwanda, test on Kirundi test
   Only KINNEWS classes that exist in our 10-class label set are used.

Run:  python -m src.crosslingual
"""
import json

import pandas as pd
from sklearn.metrics import f1_score

from src.baselines import CONFIGS, build
from src.data import PROC_DIR, RESULTS_DIR, label_list, load_splits
from src.evaluate import EXPERIMENTS_CSV, compute_metrics, log_experiment
from src.ood import LABEL_MAP, load_masakhanews_run


def selected_C() -> dict:
    """C chosen on validation for each config (read back from the experiment log)."""
    d = pd.read_csv(EXPERIMENTS_CSV)
    d = d[(d.family == "baseline") & (d.split == "test") & d.model.str.startswith("tfidf-")]
    return {m.replace("tfidf-", ""): json.loads(c)["C"] for m, c in zip(d.model, d.config)}


def main() -> None:
    s = load_splits()
    labels = label_list()
    Xtr, ytr = s["train"].text, s["train"].label_name
    Xte, yte = s["test"].text, s["test"].label_name
    ood = load_masakhanews_run()
    ood_labels = sorted(LABEL_MAP.values())
    Cs = selected_C()
    out = {"ood": {}, "crosslingual": {}}

    # 1. OOD for every TF-IDF configuration
    for cfg in CONFIGS:
        pipe = build(cfg, Cs[cfg]).fit(Xtr, ytr)
        pred = pipe.predict(ood.text)
        m = compute_metrics(ood.label_name, pred)
        m["macro_f1"] = round(f1_score(ood.label_name, pred, labels=ood_labels, average="macro", zero_division=0), 4)
        log_experiment("ood", f"tfidf-{cfg}", {"C": Cs[cfg], "dataset": "masakhanews-run"}, "ood", m)
        out["ood"][cfg] = m
        print(f"OOD  {cfg:18s} acc={m['accuracy']:.3f} macro-F1={m['macro_f1']:.3f}")

    # 2. Kinyarwanda -> Kirundi
    kin = pd.read_csv(PROC_DIR / "kinnews_unique.csv")
    kin = kin[kin.label_name.isin(labels)]

    # Leakage check: a Kinyarwanda version of the same story as a Kirundi test article
    # would not be caught by exact de-duplication. Measure the max char-n-gram cosine
    # similarity of each Kirundi test article to any KINNEWS article.
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit(
        pd.concat([kin.text, Xte]))
    sim = (vec.transform(Xte) @ vec.transform(kin.text).T).max(axis=1).toarray().ravel()
    out["kin_to_test_max_similarity"] = round(float(sim.max()), 3)
    out["test_articles_with_kin_neighbour_above_0.6"] = int((sim > 0.6).sum())
    print(f"Leakage check: max similarity of a Kirundi test article to KINNEWS = {sim.max():.3f}")
    cfg = "char|svm"  # best TF-IDF family on 5-fold CV
    for name, (X, y) in {
        "zero-shot (train: Kinyarwanda only)": (kin.text, kin.label_name),
        "augmented (train: Kirundi + Kinyarwanda)": (pd.concat([Xtr, kin.text]), pd.concat([ytr, kin.label_name])),
        "monolingual (train: Kirundi only)": (Xtr, ytr),
    }.items():
        pipe = build(cfg, Cs[cfg]).fit(X, y)
        m = compute_metrics(yte, pipe.predict(Xte))
        log_experiment("crosslingual", f"tfidf-{cfg} {name}", {"C": Cs[cfg], "n_train": len(X)}, "test", m)
        po = pipe.predict(ood.text)
        m_ood = compute_metrics(ood.label_name, po)
        m_ood["macro_f1"] = round(f1_score(ood.label_name, po, labels=ood_labels, average="macro", zero_division=0), 4)
        log_experiment("crosslingual", f"tfidf-{cfg} {name}", {"C": Cs[cfg], "n_train": len(X)}, "ood", m_ood)
        out["crosslingual"][name] = {**m, "n_train": len(X), "ood_accuracy": m_ood["accuracy"],
                                     "ood_macro_f1": m_ood["macro_f1"]}
        print(f"XLING {name:42s} n={len(X):6d} acc={m['accuracy']:.3f} macro-F1={m['macro_f1']:.3f} "
              f"| OOD acc={m_ood['accuracy']:.3f} macro-F1={m_ood['macro_f1']:.3f}")

    with open(RESULTS_DIR / "crosslingual_and_ood_baselines.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
