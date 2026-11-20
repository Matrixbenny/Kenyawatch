"""Does the Beck-informed street-language layer help? A controlled comparison.

Beck (2015) found that Nairobi Sheng keeps the Swahili verbal frame while the
stem varies: English stems in Swahili verbs ("a-si-contract"), the repetitive
extension -ang- ("wa-na-fany-ang-a"), class 6 "ma-" plurals of borrowed nouns
("ma-chali") and lengthened vowels ("kolee", "stoori"). This script rewrites the
held-out fold of each incident-grouped split in those patterns and compares
three ways of preparing text for the same model:

  A  raw text (how the model was trained before)
  B  Kenya-lingo glossary only (training and the API now prepare text alike)
  C  glossary + street-language morphology (ml/src/text_prep.py)

Only test folds are rewritten; nothing rewritten is ever trained on. The
rewriting rules were written with the layer in mind, so this measures whether
the layer handles the patterns Beck documents, not accuracy on real reports.

    python -m ml.lingo.evaluate_street_language
"""

from __future__ import annotations

import csv
import json
import random
import re
import statistics
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from ml.src.text_prep import prepare_batch
from .common import ROOT

warnings.filterwarnings("ignore")
DATA = ROOT / "ml" / "data" / "synthetic" / "kenya_watch_synthetic_combined.csv"
OUT = ROOT / "ml" / "artifacts" / "street_language_evaluation.json"
SEED = 2015

# English past-tense verbs from the corpus, re-voiced as an English stem inside a
# Swahili past frame, with or without the first-person object marker -ni-.
ENGLISH_VERBS = {
    "threatened": "threaten", "asked": "ask", "demanded": "demand", "requested": "request",
    "forced": "force", "changed": "change", "altered": "alter", "diverted": "divert",
    "forged": "forge", "falsified": "falsify", "awarded": "award", "hired": "hire",
    "collected": "collect", "received": "receive", "used": "use", "approved": "approve",
    "ordered": "order", "required": "require", "issued": "issue", "favored": "favour",
    "allocated": "allocate", "removed": "remove", "fabricated": "fabricate", "directed": "direct",
    "took": "take", "stole": "steal", "paid": "pay", "refused": "refuse", "wanted": "want",
}
PLURAL_NOUNS = {
    "officers": "officer", "documents": "document", "tenders": "tender", "contracts": "contract",
    "payments": "payment", "receipts": "receipt", "records": "record", "suppliers": "supplier",
    "funds": "fund", "permits": "permit", "bribes": "bribe", "relatives": "relative",
    "officials": "official", "certificates": "certificate", "invoices": "invoice",
}
SWAHILI_VERB = re.compile(r"\b((?:a|wa|ni|tu)(?:li|na|me))([a-z]{3,}?)(a)\b")


def beck_rewrite(text: str, rng: random.Random) -> str:
    out = []
    for word in text.split():
        core = re.sub(r"[^\w]", "", word.lower())
        tail = word[len(word.rstrip(".,;:!?")):]
        if core in ENGLISH_VERBS:
            word = ("alini" if rng.random() < 0.5 else "ali") + ENGLISH_VERBS[core] + tail
        elif core in PLURAL_NOUNS:
            word = "ma" + PLURAL_NOUNS[core] + tail
        else:
            match = SWAHILI_VERB.fullmatch(core)
            if match and not core.endswith("wa"):
                # Repetitive/habitual -ang- before the final vowel (aliomba -> aliombanga).
                word = match.group(1) + match.group(2) + "anga" + tail
            elif len(core) >= 4 and core[-1] in "aeiou" and rng.random() < 0.15:
                # Lengthened final vowel, as in clipped Sheng words and SMS emphasis.
                word = core + core[-1] * 2 + tail
        out.append(word)
    return " ".join(out)


def build(mode: str) -> Pipeline:
    steps = []
    if mode != "raw":
        steps.append(("prep", FunctionTransformer(prepare_batch, kw_args={"morphology": mode == "morphology", "concepts": False})))
    steps.append(("features", FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
                                 min_df=2, max_features=50000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", lowercase=True, ngram_range=(3, 5),
                                 min_df=2, max_features=50000, sublinear_tf=True)),
    ])))
    steps.append(("classifier", LinearSVC(class_weight="balanced", random_state=42)))
    return Pipeline(steps)


def main() -> int:
    rows = list(csv.DictReader(DATA.open(encoding="utf-8")))
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    groups = [r["group_id"] for r in rows]
    rng = random.Random(SEED)
    rewritten = [beck_rewrite(t, rng) for t in texts]
    changed = sum(a != b for a, b in zip(texts, rewritten))

    modes = {"A_raw": "raw", "B_glossary": "glossary", "C_glossary_morphology": "morphology"}
    results = {name: {"clean": [], "street": []} for name in modes}
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in splitter.split(texts, labels, groups):
        for name, mode in modes.items():
            model = build(mode).fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
            truth = [labels[i] for i in test_idx]
            for key, source in (("clean", texts), ("street", rewritten)):
                pred = model.predict([source[i] for i in test_idx])
                results[name][key].append(f1_score(truth, pred, average="macro", zero_division=0))

    summary = {
        name: {key: {"mean": round(statistics.mean(v), 4), "std": round(statistics.pstdev(v), 4),
                     "folds": [round(x, 4) for x in v]} for key, v in scores.items()}
        for name, scores in results.items()
    }
    report = {
        "dataset": str(DATA.relative_to(ROOT)),
        "rows": len(rows),
        "rows_changed_by_rewrite": changed,
        "validation": "5-fold StratifiedGroupKFold (random_state=42); only test folds are rewritten",
        "model": "word (1-2) + char_wb (3-5) TF-IDF, LinearSVC (class_weight=balanced)",
        "macro_f1": summary,
        "examples": [{"original": texts[i], "rewritten": rewritten[i]} for i in range(len(texts))
                     if texts[i] != rewritten[i]][:6],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"rows rewritten in test folds: {changed}/{len(rows)}")
    print(f"{'setup':26s} {'clean macro-F1':>16s} {'street macro-F1':>17s}")
    for name, scores in summary.items():
        print(f"{name:26s} {scores['clean']['mean']:>9.4f} ±{scores['clean']['std']:.3f} "
              f"{scores['street']['mean']:>10.4f} ±{scores['street']['std']:.3f}")
    print(f"written: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
