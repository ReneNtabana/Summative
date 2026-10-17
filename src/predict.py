"""Inference for the trained classifiers. Shared by the CLI, the OOD test and the web app.

Two predictor types share one interface, `predict_proba(texts) -> (labels, probs)`:
  * TransformerPredictor: a fine-tuned HF model (local folder or Hub id)
  * BaselinePredictor   : the saved TF-IDF + linear SVM pipeline (scikit-learn)

Usage:
  python -m src.predict "umukino w'umupira w'amaguru wahuje intamba mu rugamba na ..."
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.data import ROOT, normalise

# Below this top-1 probability we tell the user the prediction is uncertain.
# Chosen on the validation set (results/error_analysis/selective.csv): 0.9 flags 42%
# of validation errors while keeping 93.5% of predictions.
CONFIDENCE_THRESHOLD = 0.9


class TransformerPredictor:
    def __init__(self, path_or_id: str | Path, max_len: int = 512, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else
                                 "cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(str(path_or_id))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path_or_id)).to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.max_len = max_len

    def predict_proba(self, texts: list[str], bs: int = 16) -> tuple[list[str], np.ndarray]:
        texts = [normalise(t) for t in texts]   # same normalisation as training
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = self.tok(texts[i:i + bs], truncation=True, max_length=self.max_len,
                               padding=True, return_tensors="pt").to(self.device)
                out.append(self.torch.softmax(self.model(**enc).logits, -1).cpu().numpy())
        return self.labels, np.concatenate(out)


class BaselinePredictor:
    """TF-IDF + LinearSVC. SVMs give margins, not probabilities, so we apply a
    softmax over the decision scores. This is only a rough confidence proxy."""

    def __init__(self, path: str | Path = ROOT / "models" / "baseline_best.joblib"):
        import joblib
        self.pipe = joblib.load(path)
        self.labels = list(self.pipe.classes_)

    def predict_proba(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        scores = self.pipe.decision_function([normalise(t) for t in texts])
        e = np.exp(scores - scores.max(1, keepdims=True))
        return self.labels, e / e.sum(1, keepdims=True)


def top_k(predictor, text: str, k: int = 3) -> list[tuple[str, float]]:
    labels, probs = predictor.predict_proba([text])
    order = np.argsort(-probs[0])[:k]
    return [(labels[i], float(probs[0][i])) for i in order]


if __name__ == "__main__":
    model_path = ROOT / "models" / "final"
    pred = TransformerPredictor(model_path) if model_path.exists() else BaselinePredictor()
    text = " ".join(sys.argv[1:]) or "Intamba mu Rugamba yatsinze umukino w'umupira w'amaguru"
    for lab, p in top_k(pred, text):
        print(f"{lab:15s} {p:.3f}")
