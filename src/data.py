"""Data preparation for Kirundi news topic classification.

Pipeline
--------
1. Download the KIRNEWS (Kirundi) and KINNEWS (Kinyarwanda) corpora
   (Niyongabo et al., 2020) from the Hugging Face Hub parquet export.
2. Audit and clean KIRNEWS:
   * The official release has severe duplication: every one of the 923 test
     articles also appears in the training set, and about half of the training set
     consists of repeated articles. Evaluating on the official split would measure
     memorisation, not generalisation. We therefore pool train+test,
     de-duplicate on normalised article text and build a fresh split.
   * Articles that occur several times with *different* labels are dropped,
     because there is no way to tell which label is correct.
   * Classes with fewer than MIN_CLASS_SIZE unique articles are dropped. With
     fewer than about 25 examples there would be only 3–4 test items, which is
     too few for a meaningful per-class score.
3. Create a stratified 70/15/15 train/val/test split (seed 42).
4. Save the KINNEWS unique articles (excluding anything that also appears in
   KIRNEWS) for Word2Vec pre-training and cross-lingual transfer experiments.

Run:  python -m src.data
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

HF_REPO = "andreniyongabo/kinnews_kirnews"
HF_REVISION = "refs/convert/parquet"  # auto-converted parquet (the repo itself is a legacy loading script)

SEED = 42
MIN_CLASS_SIZE = 25

# Original KIRNEWS label ids -> (Kirundi name, English name). Ids 7 and 9
# (tourism, fashion) exist in KINNEWS only.
LABELS = {
    0: ("poritike", "politics"),
    1: ("inkino", "sport"),
    2: ("ubutunzi", "economy"),
    3: ("amagara", "health"),
    4: ("kwidagadura", "entertainment"),
    5: ("akahise", "history"),
    6: ("ubuhinga", "technology"),
    7: ("ubukerarugendo", "tourism"),
    8: ("akaranga", "culture"),
    9: ("imideri", "fashion"),
    10: ("ukwemera", "religion"),
    11: ("ibidukikije", "environment"),
    12: ("indero", "education"),
    13: ("urukundo", "relationship"),
}


# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")
# Kirundi orthography uses the apostrophe for elision (e.g. "rw’abatavuga").
# Sources mix the typographic ’ ‘ and the ASCII ', so we unify them.
_APOS_RE = re.compile(r"[’‘`´]")
# Web-page boilerplate found during EDA (share widgets in French, contact
# details). It is rare (<1% of articles) but site-specific, so a model could use
# it as a shortcut to the news source instead of learning the topic.
_BOILER_RE = re.compile(
    r"(cliquez pour [^()]*|imprimer)\(ouvre dans une nouvelle fen[eê]tre\)"
    r"|sur le m[eê]me th[eè]me"
    r"|\S+@\S+\.\w+"                      # e-mail addresses
    r"|tel\s*:\s*\+?[\d\s-]{7,}",         # phone numbers
    flags=re.IGNORECASE,
)


def normalise(text: str) -> str:
    """Light normalisation shared by every model (and by the web app).

    Lower-case, unify apostrophes and quotes, remove URLs, collapse whitespace.
    Punctuation is kept here. Each model's tokeniser decides what to do with it.
    """
    text = str(text or "")
    text = _APOS_RE.sub("'", text)
    text = text.replace("“", '"').replace("”", '"')
    text = _URL_RE.sub(" ", text)
    text = _BOILER_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip().lower()
    return text


def dedup_key(text: str) -> str:
    """Key used to detect duplicate articles (normalised content)."""
    return normalise(text)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download(config: str) -> pd.DataFrame:
    """Download the raw train+test parquet files of one config, tagged by split."""
    frames = []
    for split in ("train", "test"):
        path = hf_hub_download(
            HF_REPO,
            f"{config}/{split}/0000.parquet",
            repo_type="dataset",
            revision=HF_REVISION,
            local_dir=RAW_DIR,
        )
        df = pd.read_parquet(path)
        df["orig_split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
NEAR_DUP_THRESHOLD = 0.9


def drop_near_duplicates(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Drop articles whose char 3-5-gram TF-IDF cosine similarity to an earlier
    article is above NEAR_DUP_THRESHOLD. Only the first occurrence is kept."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(df["key"])
    sim = (X @ X.T).tocoo()  # sparse; rows are L2-normalised so this is cosine
    drop = set()
    for i, j, v in zip(sim.row, sim.col, sim.data):
        if j > i and v > NEAR_DUP_THRESHOLD and i not in drop:
            drop.add(j)
    stats["near_duplicates_removed"] = len(drop)
    return df.drop(index=sorted(drop)).reset_index(drop=True)


def audit_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """De-duplicate, drop label conflicts and tiny classes. Returns (df, stats)."""
    stats: dict = {"raw_total": len(df), "raw_by_split": df.orig_split.value_counts().to_dict()}

    df = df.copy()
    df["key"] = df["content"].map(dedup_key)
    tr_keys = set(df.loc[df.orig_split == "train", "key"])
    te = df[df.orig_split == "test"]
    stats["official_test_in_train"] = int(te["key"].isin(tr_keys).sum())
    stats["duplicate_rows"] = int(df["key"].duplicated().sum())

    # Articles that appear with more than one label are unreliable, so drop all copies.
    n_labels = df.groupby("key")["label"].nunique()
    conflicting = set(n_labels[n_labels > 1].index)
    stats["conflicting_label_articles"] = len(conflicting)
    df = df[~df["key"].isin(conflicting)]

    df = df.drop_duplicates("key").reset_index(drop=True)
    stats["exact_unique_articles"] = len(df)

    # Near-duplicates: the same story lightly edited or re-posted on another
    # site. Exact matching misses these, and they would leak across splits.
    df = drop_near_duplicates(df, stats)
    stats["unique_articles"] = len(df)
    stats["unique_by_label"] = {LABELS[k][1]: int(v) for k, v in df.label.value_counts().sort_index().items()}

    counts = df.label.value_counts()
    keep = sorted(counts[counts >= MIN_CLASS_SIZE].index)
    dropped = sorted(set(counts.index) - set(keep))
    stats["dropped_classes"] = {LABELS[k][1]: int(counts[k]) for k in dropped}
    df = df[df.label.isin(keep)].reset_index(drop=True)

    # Model input = title + content. The title is often the most informative part.
    df["text"] = (df["title"].fillna("").map(normalise) + " . " + df["content"].map(normalise)).str.strip(" .")
    df["label_name"] = df["label"].map(lambda i: LABELS[i][1])
    df["kir_label"] = df["label"].map(lambda i: LABELS[i][0])
    df["source"] = df["url"].str.extract(r"https?://(?:www\.)?([^/]+)", expand=False)
    stats["final_articles"] = len(df)
    stats["final_classes"] = len(keep)
    return df, stats


def make_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Stratified 70 / 15 / 15 split."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label_name"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label_name"], random_state=SEED)
    return {"train": train, "val": val, "test": test}


# --------------------------------------------------------------------------- #
# Loading helpers used by every training script
# --------------------------------------------------------------------------- #
COLUMNS = ["text", "label_name", "kir_label", "title", "source", "url"]


def load_splits() -> dict[str, pd.DataFrame]:
    """Load the processed splits (run `python -m src.data` first)."""
    return {s: pd.read_csv(PROC_DIR / f"{s}.csv") for s in ("train", "val", "test")}


def label_list() -> list[str]:
    """Sorted list of label names used across all models (fixed ordering)."""
    return sorted(pd.read_csv(PROC_DIR / "train.csv")["label_name"].unique())


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading KIRNEWS (raw) ...")
    kir = download("kirnews_raw")
    kir, stats = audit_and_clean(kir)
    splits = make_splits(kir)
    for name, part in splits.items():
        part[COLUMNS].to_csv(PROC_DIR / f"{name}.csv", index=False)
        stats[f"{name}_size"] = len(part)
        stats[f"{name}_by_label"] = part.label_name.value_counts().sort_index().to_dict()

    # Leakage guard: no article may appear in two splits.
    keys = {n: set(p["text"]) for n, p in splits.items()}
    assert not (keys["train"] & keys["val"] or keys["train"] & keys["test"] or keys["val"] & keys["test"])

    print("Downloading KINNEWS (raw, Kinyarwanda) ...")
    kin = download("kinnews_raw")
    kin["key"] = kin["content"].map(dedup_key)
    kin = kin.drop_duplicates("key")
    kin = kin[~kin["key"].isin(set(kir["key"]))]  # never let Kirundi eval text leak in
    kin["text"] = (kin["title"].fillna("").map(normalise) + " . " + kin["content"].map(normalise)).str.strip(" .")
    kin["label_name"] = kin["label"].map(lambda i: LABELS[i][1])
    kin[["text", "label_name"]].to_csv(PROC_DIR / "kinnews_unique.csv", index=False)
    stats["kinnews_unique_articles"] = len(kin)

    with open(RESULTS_DIR / "data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
