"""The browser app (web/) re-implements preprocessing in JavaScript.
This test runs web/normalise.js with Node and checks it matches the Python version."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.data import load_splits, normalise

WEB = Path(__file__).resolve().parents[1] / "web"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_js_normalise_matches_python(tmp_path):
    samples = load_splits()["test"].text.tolist()[:50] + [
        "Rw’Abatavuga https://x.bi  “quote” cliquez pour partager sur twitter(ouvre dans une nouvelle fenêtre)",
        "Tel: +257 22 33 44 55 email: a@b.com  Ubutunzi ‘bw’igihugu’",
    ]
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    script = (f"import {{ normalise }} from '{(WEB / 'normalise.js').as_uri()}';"
              f"import fs from 'fs';"
              f"const s = JSON.parse(fs.readFileSync('{inp}', 'utf8'));"
              f"process.stdout.write(JSON.stringify(s.map(normalise)));")
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    for raw, j in zip(samples, js):
        assert j == normalise(raw), raw[:80]
