"""Export the final model to ONNX (+ int8 quantisation) for in-browser inference.

Why: free Hugging Face hosting no longer runs Python (Gradio) apps, but static Spaces
are free. The web app therefore runs the model in the visitor's browser with
transformers.js (ONNX Runtime Web). To make the download practical we apply dynamic
int8 weight quantisation (1.1 GB -> ~280 MB) and check how much it changes
predictions against the original PyTorch model.

Output layout follows the transformers.js convention:
    <out>/config.json, tokenizer.json, tokenizer_config.json, onnx/model.onnx, onnx/model_quantized.onnx

Run:  python -m src.export_onnx --model models/ft-afro-xlmr-base --out models/onnx-afro-xlmr-base
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import RESULTS_DIR, load_splits, normalise
from src.evaluate import compute_metrics


def export(model_dir: Path, out: Path) -> None:
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    dummy = tok(["umukino w'umupira", "poritike"], padding=True, return_tensors="pt")
    with torch.no_grad():
        torch.onnx.export(
            model, (dummy["input_ids"], dummy["attention_mask"]), str(out / "onnx" / "model.onnx"),
            input_names=["input_ids", "attention_mask"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                          "logits": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
    # Weight-only dynamic int8 quantisation of the linear/embedding weights.
    quantize_dynamic(str(out / "onnx" / "model.onnx"), str(out / "onnx" / "model_quantized.onnx"),
                     weight_type=QuantType.QInt8)
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(model_dir / f, out / f)


def onnx_predict(path: Path, tok, texts, bs=1) -> np.ndarray:
    """bs=1 by default: dynamic int8 quantisation computes activation scales over the
    whole input tensor INCLUDING padding, so padded batches give slightly different
    outputs. The web app classifies one article at a time, so we verify with bs=1."""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=512, padding=True, return_tensors="np")
        out.append(sess.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                         "attention_mask": enc["attention_mask"].astype(np.int64)})[0])
    return np.concatenate(out)


def verify(model_dir: Path, out: Path) -> dict:
    """Compare PyTorch vs ONNX fp32 vs ONNX int8 on the held-out test set."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    test = load_splits()["test"]
    texts = [normalise(t) for t in test.text]
    pt = []
    with torch.no_grad():
        for i in range(0, len(texts), 8):
            enc = tok(texts[i:i + 8], truncation=True, max_length=512, padding=True, return_tensors="pt")
            pt.append(model(**enc).logits.numpy())
    logits = {"pytorch": np.concatenate(pt),
              "onnx_fp32": onnx_predict(out / "onnx" / "model.onnx", tok, texts),
              "onnx_int8": onnx_predict(out / "onnx" / "model_quantized.onnx", tok, texts)}
    ref = logits["pytorch"].argmax(1)
    report = {}
    for name, lg in logits.items():
        pred = lg.argmax(1)
        report[name] = {**compute_metrics(test.label_name, np.array(labels)[pred]),
                        "agreement_with_pytorch": round(float((pred == ref).mean()), 4),
                        "max_abs_logit_diff": round(float(np.abs(lg - logits["pytorch"]).max()), 4)}
    for f in ("model.onnx", "model_quantized.onnx"):
        report[f + "_MB"] = round((out / "onnx" / f).stat().st_size / 1e6, 1)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ft-afro-xlmr-base")
    ap.add_argument("--out", default="models/onnx-afro-xlmr-base")
    ap.add_argument("--verify_only", action="store_true")
    a = ap.parse_args()
    model_dir, out = Path(a.model), Path(a.out)
    if not a.verify_only:
        export(model_dir, out)
    rep = verify(model_dir, out)
    with open(RESULTS_DIR / "onnx_verification.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
