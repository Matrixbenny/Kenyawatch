"""Learn a Sheng-to-English word list from parallel sentences with IBM Model 1.

The parallel English-Sheng corpus (ml/sheng.md, used with the owner's
permission and kept out of the repository) is aligned word by word with the
classic IBM Model 1 expectation-maximisation algorithm (Brown et al. 1993).
For each Sheng word the most probable English translation is kept when the
evidence is strong enough. Only the resulting word pairs are written out
(ml/lingo/sheng_lexicon.json); no sentence from the corpus is stored.

    python -m ml.lingo.learn_sheng_lexicon
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from ml.src.text_prep import normalize_text

ROOT = Path(__file__).resolve().parents[2]
PARALLEL = ROOT / "ml" / "sheng.md"
OUT = ROOT / "ml" / "lingo" / "sheng_lexicon.json"

ENGLISH_STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being am do does did
have has had i you he she it we they me him her us them my your his its our their this that these those there here
not no so then than too very can could will would shall should may might must just also into out up down over about
who whom which what when where why how all any each some such only own same other more most both few one two
said says say saying unto upon thee thou thy ye hath shalt let lord god jesus""".split())


def tokens(text: str) -> list[str]:
    return [t for t in normalize_text(text).split() if t.isalpha() and len(t) >= 2]


def load_pairs():
    pairs = []
    with PARALLEL.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            en, sh = tokens(row.get("English", "")), tokens(row.get("Sheng", ""))
            if en and sh and len(en) <= 80 and len(sh) <= 80:
                pairs.append((en, sh))
    return pairs


def ibm_model1(pairs, iterations: int = 10):
    """t[s][e] = P(english word e | sheng word s), with a NULL source word."""
    t: dict[str, dict[str, float]] = defaultdict(dict)
    english_vocab = {e for en, _ in pairs for e in en}
    uniform = 1.0 / len(english_vocab)
    for _ in range(iterations):
        count: dict[str, Counter] = defaultdict(Counter)
        total: Counter = Counter()
        for en, sh in pairs:
            sources = sh + ["<null>"]
            for e in en:
                norm = sum(t[s].get(e, uniform) for s in sources)
                for s in sources:
                    c = t[s].get(e, uniform) / norm
                    count[s][e] += c
                    total[s] += c
        t = defaultdict(dict, {s: {e: c / total[s] for e, c in es.items()} for s, es in count.items()})
    return t


def main() -> int:
    pairs = load_pairs()
    sheng_freq = Counter(s for _, sh in pairs for s in sh)
    english_freq = Counter(e for en, _ in pairs for e in en)
    t = ibm_model1(pairs)
    lexicon = {}
    for s, translations in t.items():
        if s == "<null>" or sheng_freq[s] < 3 or len(s) < 3 or s in ENGLISH_STOP:
            continue
        # A Sheng-side word that is common in the English side is English code-switching, not Sheng.
        if english_freq[s] >= sheng_freq[s] * 0.5:
            continue
        best = sorted(((p, e) for e, p in translations.items() if e not in ENGLISH_STOP and len(e) >= 3),
                      reverse=True)
        if best and best[0][0] >= 0.25 and best[0][1] != s:
            lexicon[s] = best[0][1]
    OUT.write_text(json.dumps({
        "version": 1,
        "source": "Learned with IBM Model 1 from an English-Sheng parallel corpus used with the owner's permission; "
                  "only word pairs are stored, no corpus text.",
        "pairs": len(pairs),
        "entries": dict(sorted(lexicon.items())),
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"sentence pairs: {len(pairs)} | Sheng words seen 3+ times: {sum(1 for c in sheng_freq.values() if c >= 3)} | lexicon entries: {len(lexicon)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
