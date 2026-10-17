// Text preprocessing for the web app. MUST match src/data.py::normalise
// (checked by tests/test_web_normalise.py, which runs this file with Node).

const URL_RE = /https?:\/\/\S+|www\.\S+/g;
const WS_RE = /\s+/g;
const APOS_RE = /[’‘`´]/g;
const BOILER_RE = new RegExp(
  "(cliquez pour [^()]*|imprimer)\\(ouvre dans une nouvelle fen[eê]tre\\)" +
  "|sur le m[eê]me th[eè]me" +
  "|\\S+@\\S+\\.\\w+" +
  "|tel\\s*:\\s*\\+?[\\d\\s-]{7,}",
  "gi"
);

export function normalise(text) {
  let t = String(text ?? "");
  t = t.replace(APOS_RE, "'");
  t = t.replace(/“/g, '"').replace(/”/g, '"');
  t = t.replace(URL_RE, " ");
  t = t.replace(BOILER_RE, " ");
  return t.replace(WS_RE, " ").trim().toLowerCase();
}

// Function words used to spot input that is clearly English/French (same lists as app/app.py).
const KIRUNDI_FW = new Set(("na mu ku kuri ko ya wa ba ca za rya twa ka bwa kwa ry vy kandi ariko " +
  "canke ivyo iyo uwo aho ngo nkuko uko kubera ico ici aba ubu ibi iki muri hari ari kugira hamwe " +
  "rero cane mbere").split(" "));
const FOREIGN_FW = new Set(("the of and to in is that for on with was are by from le la les des et " +
  "de du pour est une un dans sur qui que avec").split(" "));

export function looksForeign(text) {
  const words = text.match(/[\p{L}]+/gu) || [];
  let k = 0, f = 0;
  for (const w of words) { if (KIRUNDI_FW.has(w)) k++; if (FOREIGN_FW.has(w)) f++; }
  return f >= 2 && f > k;
}
