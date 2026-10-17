"""Gradio web app: Kirundi news topic classifier.

Deployed as a Hugging Face Space. The fine-tuned model is downloaded from the
Hugging Face Hub (MODEL_ID). This file is self-contained so the Space does not
need the rest of the repository.
"""
from __future__ import annotations

import os
import re

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "ReneNtabana/kirundi-news-classifier")
MAX_LEN = 512
# Below this top-1 probability we warn that the prediction is uncertain.
# The fine-tuned model is over-confident (wrong test predictions still average 0.86),
# so a high threshold is needed. On the VALIDATION set, 0.9 flags 42% of errors while
# keeping 93.5% of predictions (see results/error_analysis/selective.csv).
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.9))

# Very frequent function words, used to spot input that is clearly English or French.
# The model itself cannot tell: it is multilingual and confidently classifies English/French
# text too (see error analysis). Kirundi news contains French/English names and quotes,
# so we only warn when foreign function words clearly dominate.
KIRUNDI_FUNCTION_WORDS = set("""na mu ku kuri ko ya wa ba ca za rya twa ka bwa kwa ry vy kandi ariko
    canke ivyo iyo uwo aho ngo nkuko uko kubera ico ici aba ubu ibi iki muri hari ari kugira hamwe
    rero cane mbere""".split())
FOREIGN_FUNCTION_WORDS = set("""the of and to in is that for on with was are by from le la les des et
    de du pour est une un dans sur qui que avec""".split())


def looks_foreign(text: str) -> bool:
    words = re.findall(r"[^\W\d_]+", text)
    k = sum(w in KIRUNDI_FUNCTION_WORDS for w in words)
    f = sum(w in FOREIGN_FUNCTION_WORDS for w in words)
    return f >= 2 and f > k

# Kirundi / English names of the 10 categories
KIRUNDI = {
    "politics": "Poritike", "sport": "Inkino", "economy": "Ubutunzi", "health": "Amagara",
    "entertainment": "Kwidagadura", "technology": "Ubuhinga", "culture": "Akaranga",
    "environment": "Ibidukikije", "education": "Indero", "relationship": "Urukundo",
}

# ---- Text normalisation: MUST match src/data.py::normalise (checked by tests) ----
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")
_APOS_RE = re.compile(r"[’‘`´]")
_BOILER_RE = re.compile(
    r"(cliquez pour [^()]*|imprimer)\(ouvre dans une nouvelle fen[eê]tre\)"
    r"|sur le m[eê]me th[eè]me"
    r"|\S+@\S+\.\w+"
    r"|tel\s*:\s*\+?[\d\s-]{7,}",
    flags=re.IGNORECASE,
)


def normalise(text: str) -> str:
    text = str(text or "")
    text = _APOS_RE.sub("'", text)
    text = text.replace("“", '"').replace("”", '"')
    text = _URL_RE.sub(" ", text)
    text = _BOILER_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip().lower()


# ---- Model ----
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()


def classify(title: str, body: str):
    text = normalise(f"{title} . {body}".strip(" ."))
    if len(text.split()) < 3:
        return {}, "⚠️ Please enter at least a few words of Kirundi text."
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, -1)[0].cpu()
    scores = {f"{lab} ({KIRUNDI.get(lab, '')})": float(probs[i])
              for i, lab in model.config.id2label.items()}
    top = float(probs.max())
    n_tokens = len(enc["input_ids"][0])
    msg = f"Top-1 confidence: {top:.0%}"
    if looks_foreign(text):
        msg = ("⚠️ This text looks like English or French. The model was trained on Kirundi news; "
               "it may still guess the topic, but results are not validated for other languages.")
    elif top < CONFIDENCE_THRESHOLD:
        msg = (f"⚠️ Low confidence ({top:.0%}). The text may be outside the 10 news topics, "
               "not in Kirundi, or about several topics at once. Treat the prediction with caution.")
    if n_tokens >= MAX_LEN:
        msg += f"\n\nℹ️ Long article: only the first {MAX_LEN} subword tokens were used."
    return scores, msg


EXAMPLES_FILE = os.path.join(os.path.dirname(__file__), "examples.tsv")
examples = []
if os.path.exists(EXAMPLES_FILE):
    with open(EXAMPLES_FILE, encoding="utf-8") as f:
        examples = [line.rstrip("\n").split("\t")[:2] for line in f if "\t" in line]

with gr.Blocks(title="Kirundi News Topic Classifier") as demo:
    gr.Markdown(
        "# Kirundi News Topic Classifier · *Gutondora amakuru mu Kirundi*\n"
        "Paste a Kirundi news article (title and/or text). A Transformer fine-tuned on the "
        "KIRNEWS corpus predicts one of 10 topics: politics, sport, economy, health, entertainment, "
        "technology, culture, environment, education, relationship.\n\n"
        "**Limitations:** trained on ~1.2k articles from eight Burundian news websites. "
        "Rare topics (culture, environment, technology) are less reliable, and the model only "
        "understands Kirundi (closely related Kinyarwanda partly works)."
    )
    with gr.Row():
        with gr.Column():
            title = gr.Textbox(label="Title / Umutwe", lines=1)
            body = gr.Textbox(label="Article text / Inkuru", lines=10)
            btn = gr.Button("Classify / Tondora", variant="primary")
        with gr.Column():
            out = gr.Label(num_top_classes=3, label="Predicted topic")
            note = gr.Markdown()
    btn.click(classify, [title, body], [out, note])
    if examples:
        gr.Examples(examples, [title, body], label="Examples from the held-out test set and MasakhaNEWS")
    gr.Markdown("Code & report: [github.com/ReneNtabana/Summative](https://github.com/ReneNtabana/Summative)")

if __name__ == "__main__":
    demo.launch()
