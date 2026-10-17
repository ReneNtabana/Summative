"""Sanity checks on the processed splits (run `python -m src.data` first)."""
import json

from src.data import PROC_DIR, RESULTS_DIR, load_splits, normalise


def test_splits_do_not_overlap():
    s = load_splits()
    tr, va, te = (set(s[k]["text"]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_same_label_set_in_every_split():
    s = load_splits()
    labels = [set(s[k]["label_name"]) for k in ("train", "val", "test")]
    assert labels[0] == labels[1] == labels[2]
    assert len(labels[0]) == json.load(open(RESULTS_DIR / "data_stats.json"))["final_classes"]


def test_no_empty_text():
    for df in load_splits().values():
        assert df["text"].str.len().min() > 20


def test_normalise():
    assert normalise("  Rw’Abatavuga   https://x.bi/a  ") == "rw'abatavuga"
