"""Do the new Kenyan corpora help? Four text preparations on the three-part bench.

  v3            current preparation (glossary, street-language stems, concepts)
  + lexicon     v3 plus English meanings of Sheng words, learned with IBM Model 1
                from the English-Sheng parallel corpus (ml/lingo/sheng_lexicon.json)
  + clusters    v3 plus word-cluster tokens learned from unlabelled Kenyan text
                (ml/lingo/word_clusters.json)
  + both        v3 plus lexicon and clusters

The bench is the one in ml/src/experiments.py: in-domain grouped CV, Beck-style
street rewrites of the test folds, and formal-to-colloquial cross-corpus.

    python -m ml.src.experiments_corpora
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from ml.lingo.evaluate_street_language import beck_rewrite
from ml.src import text_prep as tp
from ml.src.experiments import features, load, macro

ROOT = Path(__file__).resolve().parents[2]
LEXICON = json.loads((ROOT / "ml" / "lingo" / "sheng_lexicon.json").read_text(encoding="utf-8"))["entries"]
CLUSTERS = json.loads((ROOT / "ml" / "lingo" / "word_clusters.json").read_text(encoding="utf-8"))["clusters"]


def prepare(texts, lexicon=False, clusters=False):
    out = []
    for value in texts:
        normalized = tp.normalize_text(value)
        text = tp.expand_lingo_terms(normalized)
        extras = tp.morphology_extras(normalized)
        if extras:
            text = f"{text} {tp.expand_lingo_terms(' '.join(extras))}"
        words = normalized.split() + extras
        seen = set(text.split())
        added = []
        if lexicon:
            for w in words:
                meaning = LEXICON.get(w)
                if meaning and meaning not in seen:
                    added.append(meaning)
                    seen.add(meaning)
        if clusters:
            for w in words:
                c = CLUSTERS.get(w)
                if c is not None and f"wc{c}" not in seen:
                    added.append(f"wc{c}")
                    seen.add(f"wc{c}")
        if added:
            text = f"{text} {' '.join(added)}"
        tokens = tp.concept_tokens(text)
        out.append(f"{text} {' '.join(tokens)}" if tokens else text)
    return out


def main() -> int:
    comb, seed, col = load("kenya_watch_synthetic_combined.csv"), load("kenya_watch_synthetic_seed.csv"), \
        load("kenya_watch_synthetic_colloquial.csv")
    texts, labels, groups = [r["text"] for r in comb], [r["label"] for r in comb], [r["group_id"] for r in comb]
    street = [beck_rewrite(t, random.Random(2015 + i)) for i, t in enumerate(texts)]
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(texts, labels, groups))
    sheng = [i for i, r in enumerate(col) if r["language"] == "Sheng"]
    results = {}
    for name, kw in {"v3 (current)": {}, "+ Sheng lexicon": {"lexicon": True},
                     "+ word clusters": {"clusters": True}, "+ lexicon and clusters": {"lexicon": True, "clusters": True}}.items():
        make = lambda: Pipeline([("prep", FunctionTransformer(prepare, kw_args=kw)), ("features", features(1, (2, 6))),
                                 ("classifier", LinearSVC(class_weight="balanced", random_state=42))])
        in_dom, st = [], []
        for tr, te in folds:
            m = make().fit([texts[i] for i in tr], [labels[i] for i in tr])
            y = [labels[i] for i in te]
            in_dom.append(macro(y, m.predict([texts[i] for i in te])))
            st.append(macro(y, m.predict([street[i] for i in te])))
        cm = make().fit([r["text"] for r in seed], [r["label"] for r in seed])
        pred = cm.predict([r["text"] for r in col])
        results[name] = {
            "in_domain": round(statistics.mean(in_dom), 4), "in_domain_std": round(statistics.pstdev(in_dom), 4),
            "street": round(statistics.mean(st), 4),
            "cross_corpus": round(macro([r["label"] for r in col], pred), 4),
            "cross_corpus_sheng": round(macro([col[i]["label"] for i in sheng], [pred[i] for i in sheng]), 4),
        }
        r = results[name]
        print(f"{name:26s} in-domain {r['in_domain']:.3f}±{r['in_domain_std']:.3f}  street {r['street']:.3f}  "
              f"cross {r['cross_corpus']:.3f} (Sheng rows {r['cross_corpus_sheng']:.3f})", flush=True)
    (ROOT / "ml" / "artifacts" / "corpora_experiments.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
