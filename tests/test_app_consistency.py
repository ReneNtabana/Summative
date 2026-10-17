"""The web app has its own copy of the text normalisation (it is deployed alone).
This test makes sure the copy stays identical to the one used in training."""
import ast
import re
from pathlib import Path

from src.data import load_splits, normalise

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


def _load_app_normalise():
    # Execute only the normalisation part of app.py (avoids downloading the model).
    src = APP.read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if (isinstance(n, ast.Assign) and any(getattr(t, "id", "").startswith("_") for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name == "normalise")]
    ns = {"re": re}
    exec(compile(ast.Module(keep, []), str(APP), "exec"), ns)
    return ns["normalise"]


def test_app_normalise_matches_training():
    app_norm = _load_app_normalise()
    samples = load_splits()["test"].text.tolist()[:50] + [
        "Rw’Abatavuga https://x.bi  “quote” cliquez pour partager sur twitter(ouvre dans une nouvelle fenêtre)",
        "Tel: +257 22 33 44 55 email: a@b.com",
    ]
    for s in samples:
        assert app_norm(s) == normalise(s)
