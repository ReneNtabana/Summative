"""Exploratory data analysis of the cleaned KIRNEWS splits.

Produces figures in results/figures/ and summary tables in results/eda.json.
Run:  python -m src.eda
"""
import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import plotstyle
from src.data import RESULTS_DIR, load_splits

FIG = RESULTS_DIR / "figures"
TOKEN_RE = re.compile(r"[a-z']+")

# A small hand-made list of very frequent Kirundi function words (conjunctions,
# prepositions, demonstratives). It is only used to make "top words per class"
# readable and is NOT used by any model.
KIRUNDI_STOP = set("""
na mu ku kuri ko ya wa ba ca za rya twa ka bwa kwa mw kw rw cy y n
ari ari ivyo iyo uwo aho ngo kandi ariko canke nk nkuko uko kubera
ico ici aba uyo izo ubu ubwo ibi iki ivyo vyo bose cane muri hari
avuga yavuze yagize ati uno uru iyi ino ryo rwo bo we yo ho kuko
""".split())


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def main() -> None:
    plotstyle.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    s = load_splits()
    df = pd.concat([d.assign(split=k) for k, d in s.items()], ignore_index=True)
    df["n_tokens"] = df.text.map(lambda t: len(tokens(t)))
    out = {}

    # 1. Class distribution (all splits pooled), sorted by size.
    counts = df.label_name.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(counts.index, counts.values, color=plotstyle.BLUE, height=0.7)
    for y, v in enumerate(counts.values):
        ax.text(v + 6, y, str(v), va="center", color=plotstyle.INK_2, fontsize=9)
    ax.set_xlabel("Articles (after cleaning)")
    ax.set_title("KIRNEWS class distribution (n = %d)" % len(df), loc="left")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "class_distribution.png"); plt.close(fig)
    out["class_counts"] = counts.sort_values(ascending=False).to_dict()
    out["imbalance_ratio_max_min"] = round(counts.max() / counts.min(), 1)

    # 2. Article length distribution (tokens). Shows why max_len matters for transformers.
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    clipped = df.n_tokens.clip(upper=1000)
    ax.hist(clipped, bins=50, color=plotstyle.BLUE, edgecolor=plotstyle.SURFACE, linewidth=1)
    med = df.n_tokens.median()
    ax.axvline(med, color=plotstyle.INK_2, linestyle="--", linewidth=1)
    ax.text(med + 10, ax.get_ylim()[1] * 0.9, f"median = {med:.0f}", color=plotstyle.INK_2)
    ax.set_xlabel("Words per article (clipped at 1000)")
    ax.set_ylabel("Articles")
    ax.set_title("Article length", loc="left")
    fig.savefig(FIG / "length_hist.png"); plt.close(fig)
    out["length_tokens"] = df.n_tokens.describe().round(1).to_dict()
    out["length_by_class_median"] = df.groupby("label_name").n_tokens.median().to_dict()

    # 3. Vocabulary statistics.
    tr_vocab = Counter(t for x in s["train"].text for t in tokens(x))
    te_tokens = [t for x in s["test"].text for t in tokens(x)]
    out["train_vocab_size"] = len(tr_vocab)
    out["train_hapax_ratio"] = round(sum(1 for c in tr_vocab.values() if c == 1) / len(tr_vocab), 3)
    out["test_oov_token_rate"] = round(sum(t not in tr_vocab for t in te_tokens) / len(te_tokens), 3)

    # 4. Top words per class (train only), as a table for the report.
    top = {}
    for lab, g in s["train"].groupby("label_name"):
        c = Counter(t for x in g.text for t in tokens(x) if t not in KIRUNDI_STOP and len(t) > 2)
        top[lab] = [w for w, _ in c.most_common(12)]
    out["top_words_per_class"] = top

    # 5. News source by class. Some topics come mostly from one site, which a
    # model could exploit as a shortcut (discussed in error analysis).
    src = pd.crosstab(df.label_name, df.source)
    out["source_by_class"] = src.to_dict(orient="index")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    frac = src.div(src.sum(axis=1), axis=0)
    im = ax.imshow(frac.values, cmap=plotstyle.SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(frac.columns)), frac.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(frac.index)), frac.index)
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            v = frac.values[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else plotstyle.INK)
    ax.grid(False)
    ax.set_title("Share of each class coming from each news site", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.savefig(FIG / "source_by_class.png"); plt.close(fig)

    with open(RESULTS_DIR / "eda.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print(json.dumps({k: v for k, v in out.items() if k not in ("source_by_class",)}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
