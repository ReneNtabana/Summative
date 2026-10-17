"""Aggregate results/experiments.csv into the tables and figure used in the report.

* For multi-seed models (BiLSTM, Transformers): mean ± std over seeds.
* For classical baselines: the configuration selected on validation (one row per model).

Outputs: results/summary_table.csv, results/summary_table.md, results/figures/model_comparison.png
Run:  python -m src.summarize
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import plotstyle
from src.data import RESULTS_DIR
from src.evaluate import EXPERIMENTS_CSV

FAMILY_ORDER = ["baseline", "embeddings", "rnn", "transformer"]


def main() -> None:
    df = pd.read_csv(EXPERIMENTS_CSV)
    test = df[(df.split == "test") & df.family.isin(FAMILY_ORDER)].copy()
    # For baselines, only the C selected on validation is evaluated on test, so the
    # test rows already correspond to the selected configuration.
    test["config_short"] = test.config.map(lambda c: ", ".join(f"{k}={v}" for k, v in json.loads(c).items()
                                                               if k in ("C", "lr", "max_len")) if c.startswith("{") else "")
    g = test.groupby(["family", "model"])
    tab = g.agg(n_seeds=("seed", "nunique"),
                acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                wf1_mean=("weighted_f1", "mean"),
                train_s=("train_seconds", "mean"),
                config=("config_short", "first")).reset_index()

    # Add 5-fold CV and OOD scores where they exist
    cv = df[df.split == "cv5"].groupby("model").macro_f1.mean().rename("cv5_f1")
    ood = df[df.split == "ood"].copy()
    ood["model_key"] = ood.model.str.replace("ft-", "", regex=False)
    ood_f1 = ood.groupby("model_key").macro_f1.mean().rename("ood_f1")
    ood_acc = ood.groupby("model_key").accuracy.mean().rename("ood_acc")
    tab = tab.join(cv, on="model").join(ood_f1, on="model").join(ood_acc, on="model")
    # (TF-IDF OOD rows are logged per configuration by src/crosslingual.py; the
    # "baseline_svm" row from src/ood.py is the same word+char SVM and is not needed here.)

    tab["family"] = pd.Categorical(tab.family, FAMILY_ORDER, ordered=True)
    tab = tab.sort_values(["family", "f1_mean"]).round(4)
    tab.to_csv(RESULTS_DIR / "summary_table.csv", index=False)

    def pm(m, s):
        return f"{m:.3f}" if pd.isna(s) else f"{m:.3f} ± {s:.3f}"

    lines = ["| Family | Model | Config | Seeds | Accuracy | Macro-F1 | 5-fold CV F1 | OOD Acc | OOD Macro-F1 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        lines.append(f"| {r.family} | {r.model} | {r.config} | {r.n_seeds} | {pm(r.acc_mean, r.acc_std)} | "
                     f"{pm(r.f1_mean, r.f1_std)} | {'' if pd.isna(r.cv5_f1) else f'{r.cv5_f1:.3f}'} | "
                     f"{'' if pd.isna(r.ood_acc) else f'{r.ood_acc:.3f}'} | {'' if pd.isna(r.ood_f1) else f'{r.ood_f1:.3f}'} |")
    lines.append("")
    lines.append("OOD scores for Transformers are from the seed-42 model.")
    xl = df[df.family == "crosslingual"]
    if len(xl):
        lines += ["", "| Cross-lingual (TF-IDF char SVM) | n train | Test acc | Test macro-F1 | OOD acc | OOD macro-F1 |",
                  "|---|---|---|---|---|---|"]
        for model, g in xl.groupby("model", sort=False):
            t, o = g[g.split == "test"].iloc[-1], g[g.split == "ood"].iloc[-1]
            lines.append(f"| {model.split(' ', 1)[1]} | {json.loads(t.config)['n_train']} | {t.accuracy:.3f} | "
                         f"{t.macro_f1:.3f} | {o.accuracy:.3f} | {o.macro_f1:.3f} |")
    (RESULTS_DIR / "summary_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Figure: test macro-F1 per model (mean ± std), grouped by family
    plotstyle.apply()
    t = tab[tab.model != "majority"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(t) + 1.2))
    y = np.arange(len(t))
    best = t.f1_mean.idxmax()
    colors = [plotstyle.ORANGE if i == best else plotstyle.BLUE for i in range(len(t))]
    ax.barh(y, t.f1_mean, xerr=t.f1_std.fillna(0), color=colors, height=0.65,
            error_kw={"ecolor": plotstyle.INK_2, "elinewidth": 1, "capsize": 2})
    ax.set_yticks(y, [f"{m}  [{f}]" for m, f in zip(t.model, t.family)], fontsize=8)
    for yi, v in zip(y, t.f1_mean):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=8, color=plotstyle.INK_2)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Test macro-F1 (mean ± std over seeds)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Model comparison on the held-out KIRNEWS test set", loc="left")
    fig.savefig(RESULTS_DIR / "figures" / "model_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
