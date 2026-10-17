"""Out-of-distribution test on MasakhaNEWS Kirundi (Adelani et al., 2023).

Why: KIRNEWS comes from 8 Burundian news websites. A model can score well on the
in-domain split while relying on site-specific style. MasakhaNEWS-run is written
by a different publisher (BBC Gahuza), in a different period and style, and has
zero overlap with KIRNEWS. Testing on it therefore shows how well each model
generalises to new Kirundi news.

None of these articles are used for training. Only the 5 categories that also
exist in our label set are kept (religion is not one of our classes):
    business -> economy, sports -> sport, politics, health, entertainment.

Run:  python -m src.ood
"""
from __future__ import annotations

import json

import pandas as pd
from huggingface_hub import hf_hub_download

from src.data import RAW_DIR, RESULTS_DIR, ROOT
from src.evaluate import compute_metrics, log_experiment, plot_confusion
from src.predict import BaselinePredictor, TransformerPredictor

LABEL_MAP = {"business": "economy", "sports": "sport", "politics": "politics",
             "health": "health", "entertainment": "entertainment"}


def load_masakhanews_run() -> pd.DataFrame:
    frames = []
    for split in ("train", "dev", "test"):
        p = hf_hub_download("masakhane/masakhanews", f"data/run/{split}.tsv", repo_type="dataset",
                            local_dir=RAW_DIR / "masakhanews")
        frames.append(pd.read_csv(p, sep="\t"))
    df = pd.concat(frames, ignore_index=True)
    df = df[df.category.isin(LABEL_MAP)].copy()
    df["label_name"] = df.category.map(LABEL_MAP)
    # Same input format as training: "title . content" (normalised by the predictor).
    df["text"] = df.headline.fillna("") + " . " + df.text.fillna("")
    return df.drop_duplicates("text").reset_index(drop=True)


def evaluate(name: str, predictor, df: pd.DataFrame) -> dict:
    labels, probs = predictor.predict_proba(df.text.tolist())
    pred = [labels[i] for i in probs.argmax(1)]
    m = compute_metrics(df.label_name, pred)
    # Macro-F1 is computed over the 5 true classes. Predictions of a class that
    # does not exist in MasakhaNEWS (e.g. "education") simply count as errors.
    from sklearn.metrics import f1_score
    m["macro_f1"] = round(f1_score(df.label_name, pred, labels=sorted(LABEL_MAP.values()),
                                   average="macro", zero_division=0), 4)
    log_experiment("ood", name, {"dataset": "masakhanews-run", "n": len(df)}, "ood", m)
    plot_confusion(df.label_name, pred, sorted(set(labels)), RESULTS_DIR / "figures" / f"cm_ood_{name}.png",
                   title=f"{name} on MasakhaNEWS-run")
    df = df.assign(pred=pred, conf=probs.max(1))
    df[df.pred != df.label_name][["headline", "label_name", "pred", "conf", "url"]].to_csv(
        RESULTS_DIR / f"ood_errors_{name}.csv", index=False)
    print(f"{name:20s} n={len(df)} acc={m['accuracy']:.3f} macro-F1={m['macro_f1']:.3f}")
    return m


def main() -> None:
    df = load_masakhanews_run()
    print("MasakhaNEWS-run (mapped classes):", df.label_name.value_counts().to_dict())
    results = {"baseline_svm": evaluate("baseline_svm", BaselinePredictor(), df)}
    for d in sorted((ROOT / "models").glob("ft-*")):
        results[d.name] = evaluate(d.name, TransformerPredictor(d), df)
    with open(RESULTS_DIR / "ood_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
