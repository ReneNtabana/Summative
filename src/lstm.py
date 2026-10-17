"""Bidirectional LSTM classifier (PyTorch) with optional attention pooling.

Architecture
------------
tokens -> Embedding(200d) -> BiLSTM(hidden 128 per direction) -> pooling -> dropout -> Linear(10)

Pooling options:
  * "maxmean": concatenate max- and mean-pooling over time (a strong, simple default)
  * "attn": additive self-attention (Bahdanau et al., 2015; Yang et al., 2016). A
    learned vector scores each time step, a softmax turns the scores into weights,
    and the document vector is the weighted sum of the hidden states. The weights
    also show which words the model relied on, which we use in error analysis.

Embedding initialisation:
  * "random": learned from scratch on 1.2k articles
  * "w2v-kir+kin" / "fasttext-kir+kin": pretrained vectors from src/embeddings.py
    (fine-tuned during training)

Training details: class-weighted cross-entropy (inverse frequency) for the
imbalance, Adam, gradient clipping, early stopping on validation macro-F1
(patience 4), max 400 tokens per article (the 75th length percentile is ~290).

Run:  python -m src.lstm            (all configurations x 3 seeds)
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors

from src.data import ROOT, label_list, load_splits
from src.embeddings import DIM, path_for, tokenize
from src.evaluate import compute_metrics, log_experiment, save_report

MAX_LEN = 400
PAD, UNK = 0, 1
DEVICE = torch.device("cpu")  # small model; CPU is fast enough and fully deterministic


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_vocab(texts, min_freq: int = 2) -> dict[str, int]:
    c = Counter(w for t in texts for w in tokenize(t))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w, n in c.most_common():
        if n >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def encode(texts, vocab) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [[vocab.get(w, UNK) for w in tokenize(t)][:MAX_LEN] or [UNK] for t in texts]
    lengths = torch.tensor([len(x) for x in ids])
    out = torch.full((len(ids), int(lengths.max())), PAD, dtype=torch.long)
    for i, x in enumerate(ids):
        out[i, : len(x)] = torch.tensor(x)
    return out, lengths


def pretrained_matrix(vocab, init: str) -> tuple[torch.Tensor, float]:
    """Embedding matrix from saved gensim vectors. Returns (matrix, coverage)."""
    algo, corpus_name = init.split("-", 1)
    wv = KeyedVectors.load(str(path_for(algo, corpus_name)))
    is_ft = algo == "fasttext"
    mat = np.random.default_rng(0).normal(0, 0.1, (len(vocab), DIM)).astype(np.float32)
    mat[PAD] = 0
    hit = 0
    for w, i in vocab.items():
        if i <= UNK:
            continue
        if is_ft or w in wv.key_to_index:
            mat[i] = wv[w]
            hit += w in wv.key_to_index
    return torch.from_numpy(mat), hit / (len(vocab) - 2)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes, pooling="maxmean", emb=None, hidden=128, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, DIM, padding_idx=PAD)
        if emb is not None:
            self.emb.weight.data.copy_(emb)
        self.emb_drop = nn.Dropout(0.25)
        self.lstm = nn.LSTM(DIM, hidden, batch_first=True, bidirectional=True)
        self.pooling = pooling
        h2 = 2 * hidden
        if pooling == "attn":
            # additive attention: score_t = v^T tanh(W h_t)
            self.att_W = nn.Linear(h2, h2)
            self.att_v = nn.Linear(h2, 1, bias=False)
            out_dim = h2
        else:
            out_dim = 2 * h2
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, n_classes)

    def forward(self, x, lengths, return_attention=False):
        mask = x != PAD                                        # (B, T)
        e = self.emb_drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.size(1))  # (B, T, 2H)
        att = None
        if self.pooling == "attn":
            scores = self.att_v(torch.tanh(self.att_W(h))).squeeze(-1)    # (B, T)
            scores = scores.masked_fill(~mask, float("-inf"))
            att = torch.softmax(scores, dim=1)
            doc = (att.unsqueeze(-1) * h).sum(1)
        else:
            m = mask.unsqueeze(-1).float()
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
            mx = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
            doc = torch.cat([mean, mx], dim=-1)
        logits = self.fc(self.drop(doc))
        return (logits, att) if return_attention else logits


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def predict(model, X, L, bs=64) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(X[i:i + bs], L[i:i + bs]).argmax(-1).numpy())
    return np.concatenate(out)


def run(init: str, pooling: str, seed: int, epochs: int = 30, patience: int = 4, lr: float = 2e-3, bs: int = 32,
        save_name: str | None = None):
    torch.manual_seed(seed); np.random.seed(seed)
    s = load_splits()
    labels = label_list(); lab2id = {l: i for i, l in enumerate(labels)}
    vocab = build_vocab(s["train"].text)
    data = {k: (*encode(s[k].text, vocab), torch.tensor(s[k].label_name.map(lab2id).values)) for k in s}

    emb, coverage = (None, None) if init == "random" else pretrained_matrix(vocab, init)
    model = BiLSTMClassifier(len(vocab), len(labels), pooling, emb).to(DEVICE)

    ytr = data["train"][2]
    weights = len(ytr) / (len(labels) * torch.bincount(ytr, minlength=len(labels)).float())
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xtr, Ltr, _ = data["train"]
    best_f1, best_state, bad, t0 = -1, None, 0, time.time()
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx], Ltr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_pred = predict(model, data["val"][0], data["val"][1])
        f1 = compute_metrics(data["val"][2].numpy(), val_pred)["macro_f1"]
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    secs = time.time() - t0
    model.load_state_dict(best_state)

    id2lab = np.array(labels)
    res = {}
    for split in ("val", "test"):
        X, L, y = data[split]
        res[split] = compute_metrics(id2lab[y.numpy()], id2lab[predict(model, X, L)])
    cfg = {"init": init, "pooling": pooling, "epochs_run": ep + 1, "lr": lr, "max_len": MAX_LEN,
           "vocab": len(vocab), "emb_coverage": None if coverage is None else round(coverage, 3)}
    name = f"bilstm-{pooling}-{init}"
    for split in ("val", "test"):
        log_experiment("rnn", name, cfg, split, res[split], seed=seed, train_seconds=secs)
    if save_name:
        save_report(save_name, id2lab[data["test"][2].numpy()], id2lab[predict(model, *data["test"][:2])], labels)
        (ROOT / "models").mkdir(exist_ok=True)
        torch.save({"state": best_state, "vocab": vocab, "labels": labels, "cfg": cfg}, ROOT / "models" / f"{save_name}.pt")
    print(f"{name:32s} seed={seed} ep={ep + 1:2d} cov={cfg['emb_coverage']} "
          f"val F1={res['val']['macro_f1']:.3f} test acc={res['test']['accuracy']:.3f} "
          f"F1={res['test']['macro_f1']:.3f} ({secs:.0f}s)")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()
    torch.set_num_threads(8)
    configs = [
        ("random", "maxmean"),
        ("w2v-kir", "maxmean"),
        ("w2v-kir+kin", "maxmean"),
        ("fasttext-kir+kin", "maxmean"),
        ("fasttext-kir+kin", "attn"),
        ("random", "attn"),
    ]
    summary = {}
    for init, pooling in configs:
        f1s = [run(init, pooling, seed)["test"]["macro_f1"] for seed in args.seeds]
        summary[(init, pooling)] = (np.mean(f1s), np.std(f1s))
    print("\nTest macro-F1 mean ± std over seeds:")
    for k, (m, sd) in summary.items():
        print(f"  {k[1]:8s} {k[0]:18s} {m:.3f} ± {sd:.3f}")


if __name__ == "__main__":
    main()
