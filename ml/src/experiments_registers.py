"""Does the 10,000-report register corpus help on text it was not generated from?

The register corpus (ml/src/generate_register_dataset.py) writes 2,500
fictional scenarios in English, Kiswahili, mixed English-Kiswahili and Sheng.
Because it comes from templates, its own cross-validation score says little.
It is therefore only ever added to the training side of the existing bench,
and every score below is measured on text written separately:

  in_domain     5-fold grouped CV on the 756-row combined corpus; the register
                rows are added to each training fold, never to a test fold
  street        the same test folds rewritten in Beck (2015) Sheng patterns
  cross         trained on the 540-row formal seed corpus (+ register rows),
                tested on the separately written 216-row colloquial corpus
  registers     the reverse check: the current model (756 rows only) scored on
                the register corpus, by register

Register rows are given a training weight, so that 10,000 template rows do
not drown out the 540 to 756 hand-written ones.

    python -m ml.src.experiments_registers
"""

from __future__ import annotations

import json
import os
import random
import statistics
import warnings
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest
from sklearn.model_selection import StratifiedGroupKFold

from ml.lingo.evaluate_street_language import beck_rewrite
from ml.src.experiments import load, macro
from ml.src.train import _build_model

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("KW_OUT") or ROOT / "ml" / "artifacts" / "register_experiments.json")
# 0.0 = no register corpus. KW_WEIGHTS="0,0.25" runs a subset.
WEIGHTS = tuple(float(w) for w in os.environ.get("KW_WEIGHTS", "0,0.25,1").split(","))


def fit(texts, labels, extra, weight):
    model = _build_model("word_char_linear_svc")
    if weight == 0.0:
        return model.fit(texts, labels)
    x = texts + [r["text"] for r in extra]
    y = labels + [r["label"] for r in extra]
    w = [1.0] * len(texts) + [weight] * len(extra)
    return model.fit(x, y, classifier__sample_weight=w)


def mcnemar(truth, base, new):
    fixed = sum(b != t and n == t for t, b, n in zip(truth, base, new))
    broke = sum(b == t and n != t for t, b, n in zip(truth, base, new))
    p = binomtest(fixed, fixed + broke).pvalue if fixed + broke else 1.0
    return {"fixed": fixed, "broke": broke, "p": round(p, 4)}


def by_language(rows, pred):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["language"]].append(i)
    return {lang: round(macro([rows[i]["label"] for i in idx], [pred[i] for i in idx]), 4)
            for lang, idx in sorted(groups.items())}


def main() -> int:
    comb, seed, col = load("kenya_watch_synthetic_combined.csv"), load("kenya_watch_synthetic_seed.csv"), \
        load("kenya_watch_synthetic_colloquial.csv")
    reg = load("kenya_watch_synthetic_registers.csv")
    os.environ["KW_SUPPLEMENT"] = "0"   # this script adds the register rows itself
    texts, labels, groups = [r["text"] for r in comb], [r["label"] for r in comb], [r["group_id"] for r in comb]
    street = [beck_rewrite(t, random.Random(2015 + i)) for i, t in enumerate(texts)]
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(texts, labels, groups))
    results, preds = {}, {}
    for weight in WEIGHTS:
        name = "current (no register corpus)" if weight == 0.0 else f"+ register corpus, weight {weight:g}"
        in_dom, st = [], []
        pooled_in, pooled_st = [None] * len(texts), [None] * len(texts)
        for tr, te in folds:
            m = fit([texts[i] for i in tr], [labels[i] for i in tr], reg, weight)
            p_in, p_st = m.predict([texts[i] for i in te]), m.predict([street[i] for i in te])
            y = [labels[i] for i in te]
            in_dom.append(macro(y, p_in))
            st.append(macro(y, p_st))
            for j, i in enumerate(te):
                pooled_in[i], pooled_st[i] = p_in[j], p_st[j]
        cm = fit([r["text"] for r in seed], [r["label"] for r in seed], reg, weight)
        p_cross = list(cm.predict([r["text"] for r in col]))
        preds[weight] = {"in": pooled_in, "street": pooled_st, "cross": p_cross}
        results[name] = {
            "in_domain": round(statistics.mean(in_dom), 4), "in_domain_std": round(statistics.pstdev(in_dom), 4),
            "in_domain_by_language": by_language(comb, pooled_in),
            "street": round(statistics.mean(st), 4),
            "cross_corpus": round(macro([r["label"] for r in col], p_cross), 4),
            "cross_corpus_by_language": by_language(col, p_cross),
        }
        if weight and 0.0 in preds:
            base = preds[0.0]
            results[name]["mcnemar_vs_current"] = {
                "in_domain": mcnemar(labels, base["in"], pooled_in),
                "street": mcnemar(labels, base["street"], pooled_st),
                "cross_corpus": mcnemar([r["label"] for r in col], base["cross"], p_cross),
            }
        r = results[name]
        print(f"{name:34s} in-domain {r['in_domain']:.3f}±{r['in_domain_std']:.3f}  street {r['street']:.3f}  "
              f"cross {r['cross_corpus']:.3f}  {r['cross_corpus_by_language']}", flush=True)
        if weight:
            print("   McNemar vs current:", r["mcnemar_vs_current"], flush=True)
    current = _build_model("word_char_linear_svc").fit(texts, labels)
    p_reg = list(current.predict([r["text"] for r in reg]))
    results["current model scored on the register corpus"] = {
        "macro_f1": round(macro([r["label"] for r in reg], p_reg), 4), "by_register": by_language(reg, p_reg)}
    print("current model on the register corpus:", results["current model scored on the register corpus"], flush=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
