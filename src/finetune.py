"""Fine-tune pretrained Transformer encoders for Kirundi topic classification.

Models compared (same data, same hyper-parameter budget):
  * FacebookAI/xlm-roberta-base : XLM-R (Conneau et al., 2020). Kirundi is NOT in its
                                  CC-100 pretraining data, so it is our "general
                                  multilingual model" control.
  * Davlan/afro-xlmr-base       : XLM-R-base after multilingual adaptive fine-tuning
                                  (MAFT) on 17 African languages including Kirundi
                                  (Alabi et al., 2022). Comparing it with XLM-R isolates
                                  the effect of African-language adaptation.
  * castorini/afriberta_base    : AfriBERTa (Ogueji et al., 2021), ~111M params,
                                  pretrained from scratch on <1GB of African text
                                  including "Gahuza" (Kinyarwanda + Kirundi).

Fine-tuning: a classification head (linear layer on the <s>/[CLS] representation)
is added on top of the encoder and all weights are updated with AdamW. We use
class-weighted cross-entropy, linear warm-up and decay, and keep the epoch with the
best validation macro-F1. Articles are truncated to the first MAX_LEN subword tokens;
the title is at the front, so it is always kept.

Run:  python -m src.finetune --model Davlan/afro-xlmr-base --seeds 42 43 44
"""
from __future__ import annotations

import os

# Environment must be configured BEFORE torch is imported.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # fall back to CPU for any op MPS lacks
# Apple-silicon memory management. MPS shares RAM with the OS. Without these limits
# PyTorch's caching allocator keeps growing (we measured 11 GB after the first
# evaluation) and macOS starts swapping, which slowed training ~40x. The low
# watermark makes PyTorch release cached blocks early; the high watermark caps the total.
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.9")

import argparse
import json
import time

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainerCallback, TrainingArguments, set_seed)

from src.data import ROOT, label_list, load_splits
from src.evaluate import compute_metrics, log_experiment, save_report



class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy (inverse class frequency)."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        # Our compute_loss returns a per-batch MEAN, so the Trainer must divide it by
        # the gradient-accumulation steps. Because XLM-R's forward() takes **kwargs,
        # the Trainer otherwise assumes the loss is already normalised and skips that
        # division, which would double the gradients (bug found via train_loss ≈ 2×ln(10)).
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


class FreeMPSCache(TrainerCallback):
    """Release PyTorch's cached MPS memory after each evaluation / epoch."""

    def _free(self, *args, **kwargs):
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    on_evaluate = _free
    on_epoch_end = _free


def short(name: str) -> str:
    return name.split("/")[-1]


def run(model_name: str, seed: int, lr: float, epochs: int, max_len: int, bs: int,
        save_dir: str | None = None, freeze_embeddings: bool = True, tag: str = "") -> dict:
    set_seed(seed)
    s = load_splits()
    labels = label_list()
    lab2id = {l: i for i, l in enumerate(labels)}

    tok = AutoTokenizer.from_pretrained(model_name)
    # AfriBERTa's saved tokenizer config omits model_max_length; set it explicitly.
    tok.model_max_length = max_len

    def to_ds(df):
        ds = Dataset.from_dict({"text": df.text.tolist(), "label": df.label_name.map(lab2id).tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len), batched=True,
                      remove_columns=["text"])

    ds = {k: to_ds(v) for k, v in s.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=lab2id)
    if freeze_embeddings:
        # The word-embedding matrix is 69% of XLM-R/AfroXLMR's parameters (250k x 768)
        # and 48% of AfriBERTa's. With 1.2k training articles, updating it mainly risks
        # overfitting (only tokens present in the data would move). Freezing it also cuts
        # gradient + AdamW memory by about two-thirds, which is what lets the 278M models
        # train within 16 GB on an M1. Position embeddings and all Transformer layers
        # are still fine-tuned. The same setting is used for every model.
        model.base_model.embeddings.word_embeddings.weight.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_train / 1e6:.1f}M of {model.num_parameters() / 1e6:.1f}M", flush=True)

    counts = np.bincount(ds["train"]["label"], minlength=len(labels))
    weights = torch.tensor(len(ds["train"]) / (len(labels) * counts), dtype=torch.float)

    def metrics_fn(p):
        return compute_metrics(p.label_ids, p.predictions.argmax(-1))

    out_dir = ROOT / "checkpoints" / f"{short(model_name)}-s{seed}"
    accum = max(1, 16 // bs)                                   # effective batch size 16
    total_steps = int(np.ceil(len(ds["train"]) / (bs * accum))) * epochs
    args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=4,   # small eval batches keep MPS memory low
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        warmup_steps=int(0.1 * total_steps),                   # 10% linear warm-up
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=25,
        report_to=[],
        seed=seed,
        dataloader_pin_memory=False,
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
        processing_class=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics_fn, class_weights=weights,
        callbacks=[FreeMPSCache()],
    )
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0

    res = {}
    preds = {}
    for split in ("val", "test"):
        p = trainer.predict(ds[split])
        preds[split] = p.predictions.argmax(-1)
        res[split] = compute_metrics(p.label_ids, preds[split])
    cfg = {"lr": lr, "epochs": epochs, "max_len": max_len, "eff_bs": 16,
           "freeze_emb": freeze_embeddings, "trainable_M": round(n_train / 1e6, 1),
           "best_epoch": _best_epoch(trainer)}
    for split in ("val", "test"):
        log_experiment("transformer", short(model_name) + tag, cfg, split, res[split], seed=seed, train_seconds=secs)
    print(f"{short(model_name) + tag} seed={seed} lr={lr} val F1={res['val']['macro_f1']:.3f} "
          f"test acc={res['test']['accuracy']:.3f} F1={res['test']['macro_f1']:.3f} ({secs / 60:.1f} min)", flush=True)

    if save_dir:
        id2lab = np.array(labels)
        y_true = id2lab[np.array(ds["test"]["label"])]
        save_report(f"{short(model_name)}_s{seed}", y_true, id2lab[preds["test"]], labels)
        trainer.save_model(save_dir)
        tok.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "training_summary.json"), "w") as f:
            json.dump({"model": model_name, "seed": seed, **cfg, "results": res}, f, indent=2)

    # Free memory between runs (MPS does not always release it promptly).
    del trainer, model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)
    return res


def _best_epoch(trainer) -> float | None:
    best = trainer.state.best_metric
    for h in trainer.state.log_history:
        if h.get("eval_macro_f1") == best:
            return h.get("epoch")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--save_dir", default=None, help="save the model of the FIRST seed here")
    ap.add_argument("--no_freeze_embeddings", action="store_true",
                    help="also fine-tune the word-embedding matrix (needs much more memory)")
    ap.add_argument("--tag", default="", help="suffix for the model name in results (e.g. -lr5e-5)")
    a = ap.parse_args()
    for i, seed in enumerate(a.seeds):
        run(a.model, seed, a.lr, a.epochs, a.max_len, a.bs, save_dir=a.save_dir if i == 0 else None,
            freeze_embeddings=not a.no_freeze_embeddings, tag=a.tag)


if __name__ == "__main__":
    main()
