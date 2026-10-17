"""Error analysis for the final model (and comparison with the TF-IDF baseline).

Outputs (results/error_analysis/):
  * misclassified.csv        : every wrong test prediction with confidence and an excerpt
  * confused_pairs.csv       : most frequent (true -> predicted) confusions
  * by_length.csv / by_oov.csv / by_source.csv : error rate by article length, OOV rate and news site
  * model_agreement.json     : how often baseline and transformer are wrong on the same items
  * selective.csv            : accuracy vs coverage for confidence thresholds (validation set)
  * unseen_inputs.csv        : confidence on inputs outside the label set (unseen topics,
                               Kinyarwanda-only topics, English/French text)

Run:  python -m src.error_analysis --model models/ft-afro-xlmr-base
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import plotstyle
from src.data import LABELS, PROC_DIR, RESULTS_DIR, audit_and_clean, download, load_splits
from src.embeddings import tokenize
from src.predict import BaselinePredictor, TransformerPredictor

OUT = RESULTS_DIR / "error_analysis"

# Short hand-written probes in other languages. These are NOT training data;
# they only test how the model behaves on text in the wrong language.
FOREIGN_PROBES = [
    ("english", "The central bank raised interest rates to fight inflation, the finance minister said on Monday."),
    ("english", "The national football team won the match two goals to one in the final minutes."),
    ("french", "Le président a rencontré les chefs des partis politiques pour discuter des élections."),
    ("french", "Les élèves ont repris le chemin de l'école après les vacances scolaires."),
    ("swahili", "Timu ya taifa ya mpira wa miguu imeshinda mechi ya jana kwa mabao mawili."),
]


def predict_df(pred, texts):
    labels, probs = pred.predict_proba(list(texts))
    return np.array(labels)[probs.argmax(1)], probs.max(1), probs, labels


def bucket_errors(df: pd.DataFrame, col: str, bins, labels) -> pd.DataFrame:
    b = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return (df.assign(bucket=b).groupby("bucket", observed=False)
              .agg(n=("correct", "size"), accuracy=("correct", "mean")).round(3).reset_index())


def unseen_inputs() -> pd.DataFrame:
    """Naturally occurring inputs whose correct label is NOT in our label set."""
    rows = []
    # (a) Kirundi articles of the classes we dropped (history, religion). They were
    #     removed from the training data, so they are unseen topics for the model.
    kir, _ = audit_and_clean_keep_all()
    for _, r in kir[kir.label_name.isin(["history", "religion"])].iterrows():
        rows.append(("kirundi-unseen-topic:" + r.label_name, r.text))
    # (b) Kinyarwanda articles from KINNEWS categories that do not exist in KIRNEWS.
    kin = pd.read_csv(PROC_DIR / "kinnews_unique.csv")
    for lab in ("tourism", "fashion"):
        for t in kin[kin.label_name == lab].text.head(15):
            rows.append(("kinyarwanda-unseen-topic:" + lab, t))
    # (c) Kinyarwanda articles of known topics, to test the closely related language.
    for lab in ("sport", "politics", "health"):
        for t in kin[kin.label_name == lab].text.head(10):
            rows.append(("kinyarwanda-known-topic:" + lab, t))
    for lang, t in FOREIGN_PROBES:
        rows.append(("foreign:" + lang, t))
    return pd.DataFrame(rows, columns=["kind", "text"])


def audit_and_clean_keep_all():
    """Re-run the cleaning but keep the small classes (needed for the unseen-topic test)."""
    import src.data as d
    old = d.MIN_CLASS_SIZE
    d.MIN_CLASS_SIZE = 0
    try:
        return audit_and_clean(download("kirnews_raw"))
    finally:
        d.MIN_CLASS_SIZE = old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plotstyle.apply()

    s = load_splits()
    test = s["test"].copy()
    tf = TransformerPredictor(a.model)
    bl = BaselinePredictor()

    test["pred"], test["conf"], _, _ = predict_df(tf, test.text)
    test["pred_baseline"], _, _, _ = predict_df(bl, test.text)
    test["correct"] = test.pred == test.label_name
    test["correct_baseline"] = test.pred_baseline == test.label_name

    # Article length and OOV rate (words never seen in the training split)
    train_vocab = Counter(w for t in s["train"].text for w in tokenize(t))
    toks = test.text.map(tokenize)
    test["n_words"] = toks.map(len)
    test["oov_rate"] = toks.map(lambda ws: np.mean([w not in train_vocab for w in ws]) if ws else 0)

    # 1. Misclassified examples
    wrong = test[~test.correct].copy()
    wrong["excerpt"] = wrong.text.str[:300]
    wrong[["label_name", "pred", "conf", "pred_baseline", "source", "n_words", "oov_rate", "excerpt", "url"]] \
        .sort_values("conf", ascending=False).to_csv(OUT / "misclassified.csv", index=False)

    # 2. Most confused pairs
    pairs = Counter(zip(wrong.label_name, wrong.pred))
    pd.DataFrame([(t, p, n) for (t, p), n in pairs.most_common()], columns=["true", "pred", "count"]) \
        .to_csv(OUT / "confused_pairs.csv", index=False)

    # 3. Error rate by length / OOV / source
    by_len = bucket_errors(test, "n_words", [0, 120, 200, 300, 450, 10_000],
                           ["<120", "120-200", "200-300", "300-450", ">450"])
    by_oov = bucket_errors(test, "oov_rate", [0, 0.06, 0.09, 0.12, 0.16, 1],
                           ["<6%", "6-9%", "9-12%", "12-16%", ">16%"])
    by_src = test.groupby("source").agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                        baseline_accuracy=("correct_baseline", "mean")).round(3).reset_index()
    by_len.to_csv(OUT / "by_length.csv", index=False)
    by_oov.to_csv(OUT / "by_oov.csv", index=False)
    by_src.to_csv(OUT / "by_source.csv", index=False)

    # 4. Agreement between models
    agree = {
        "both_correct": int((test.correct & test.correct_baseline).sum()),
        "only_transformer_correct": int((test.correct & ~test.correct_baseline).sum()),
        "only_baseline_correct": int((~test.correct & test.correct_baseline).sum()),
        "both_wrong": int((~test.correct & ~test.correct_baseline).sum()),
        "both_wrong_same_label": int((~test.correct & (test.pred == test.pred_baseline)).sum()),
    }
    json.dump(agree, open(OUT / "model_agreement.json", "w"), indent=2)

    # 5. Selective prediction on the VALIDATION set: choose the "uncertain" threshold
    val = s["val"].copy()
    val["pred"], val["conf"], _, _ = predict_df(tf, val.text)
    val["correct"] = val.pred == val.label_name
    sel = []
    for th in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        keep = val.conf >= th
        sel.append({"threshold": th, "coverage": round(keep.mean(), 3),
                    "accuracy_on_kept": round(val[keep].correct.mean(), 3) if keep.any() else None,
                    "share_of_errors_flagged": round((~keep & ~val.correct).sum() / max((~val.correct).sum(), 1), 3)})
    pd.DataFrame(sel).to_csv(OUT / "selective.csv", index=False)

    # Confidence histogram for correct vs wrong test predictions
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bins = np.linspace(0, 1, 21)
    ax.hist(test[test.correct].conf, bins=bins, color=plotstyle.BLUE, label="correct", edgecolor=plotstyle.SURFACE)
    ax.hist(test[~test.correct].conf, bins=bins, color=plotstyle.ORANGE, label="wrong", edgecolor=plotstyle.SURFACE)
    ax.set_xlabel("Top-1 probability"); ax.set_ylabel("Test articles"); ax.set_yscale("log")
    ax.legend(frameon=False); ax.set_title("Model confidence: correct vs. wrong predictions", loc="left")
    fig.savefig(RESULTS_DIR / "figures" / "confidence_hist.png"); plt.close(fig)

    # 6. Inputs outside the label set
    un = unseen_inputs()
    un["pred"], un["conf"], _, _ = predict_df(tf, un.text)
    un["excerpt"] = un.text.str[:120]
    un.drop(columns="text").to_csv(OUT / "unseen_inputs.csv", index=False)
    summary_unseen = un.groupby(un.kind.str.split(":").str[0]).agg(
        n=("conf", "size"), mean_conf=("conf", "mean"),
        flagged_below_0_5=("conf", lambda c: (c < 0.5).mean())).round(3)
    pred_dist = un.groupby("kind").pred.agg(lambda p: dict(Counter(p).most_common(3)))

    summary = {
        "test_accuracy": round(test.correct.mean(), 4),
        "n_errors": int((~test.correct).sum()),
        "confused_pairs_top5": [f"{t}->{p}: {n}" for (t, p), n in pairs.most_common(5)],
        "agreement": agree,
        "mean_conf_correct": round(test[test.correct].conf.mean(), 3),
        "mean_conf_wrong": round(test[~test.correct].conf.mean(), 3),
        "by_length": by_len.to_dict(orient="records"),
        "by_oov": by_oov.to_dict(orient="records"),
        "by_source": by_src.to_dict(orient="records"),
        "selective_val": sel,
        "unseen_summary": summary_unseen.reset_index().to_dict(orient="records"),
        "unseen_pred_dist": pred_dist.to_dict(),
    }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
