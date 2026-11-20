"""Text preparation shared by training and the live service.

The live API prepares report text with backend/utils/kenyaLingo.js (normalise,
then expand the Sheng/Kiswahili glossary) and backend/utils/shengMorphology.js
(expose the stem inside Swahili verb frames, after Beck 2015). This module is
the Python twin, so the model is trained on text prepared exactly as it will be
scored. ml/tests/test_text_prep.py checks the two implementations against each
other and checks that the glossary snapshot matches the live glossary.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

LINGO_DIR = Path(__file__).resolve().parents[1] / "lingo"
GLOSSARY_PATH = LINGO_DIR / "glossary_snapshot.json"
MORPHOLOGY_PATH = LINGO_DIR / "sheng_morphology.json"
CONCEPTS_PATH = LINGO_DIR / "concepts.json"
VOWELS = set("aeiou")
LETTERS_ONLY = re.compile(r"^[a-z]+$")


def normalize_text(value: str) -> str:
    """NFKD, drop accents, lower case, and turn anything but letters and digits into spaces."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return re.sub(r"[\W_]+", " ", text).strip()


@lru_cache(maxsize=None)
def load_glossary() -> dict[str, str]:
    return json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def load_morphology_rules() -> dict:
    return json.loads(MORPHOLOGY_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def load_concepts() -> dict:
    return json.loads(CONCEPTS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _concept_patterns(concepts_json: str) -> tuple:
    concepts = json.loads(concepts_json)["concepts"]
    patterns = []
    for name in sorted(concepts):
        phrases = sorted(set(concepts[name]), key=lambda item: (-len(item), item))
        alternation = "|".join(re.escape(phrase) for phrase in phrases)
        patterns.append((name, re.compile(r"(?<![^\W_])(?:" + alternation + r")(?![^\W_])")))
    return tuple(patterns)


def concept_tokens(text: str, concepts: dict | None = None) -> list[str]:
    """One token per concept present, e.g. concept_bribe, in alphabetical order of concept."""
    patterns = _concept_patterns(json.dumps(concepts or load_concepts(), sort_keys=True))
    return [f"concept_{name}" for name, pattern in patterns if pattern.search(text)]


@lru_cache(maxsize=None)
def _glossary_patterns() -> tuple:
    # Longest phrase first, whole words only, as in kenyaLingo.buildPhrasePatterns.
    ordered = sorted(load_glossary().items(), key=lambda item: -len(item[0]))
    return tuple(
        (re.compile(r"(?<![^\W_])" + re.escape(term) + r"(?![^\W_])"), f" {canonical} ")
        for term, canonical in ordered
    )


def expand_lingo_terms(normalized: str) -> str:
    text = normalized
    for pattern, replacement in _glossary_patterns():
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_for_matching(value: str) -> str:
    return expand_lingo_terms(normalize_text(value))


def _longest_first(items) -> list[str]:
    return sorted(items, key=lambda item: (-len(item), item))


@lru_cache(maxsize=None)
def _compiled(rules_json: str) -> dict:
    rules = json.loads(rules_json)
    return {
        **rules,
        "subject": _longest_first(rules["subject"]),
        "tam": _longest_first(rules["tam"]),
        "relative": _longest_first([item for item in rules["relative"] if len(item) >= 2]),
        "object": _longest_first(rules["object"]),
        "negativeSubject": _longest_first(rules["negativeSubject"]),
        "negativeTam": _longest_first(rules["negativeTam"]),
        "habitual": _longest_first(rules["habitual"]),
    }


def _stem_variants(rest: str, negative_present: bool) -> list[str]:
    out = [rest]
    if re.search(r"(anga|angi|ange)$", rest) and len(rest) - 4 >= 2:
        out.append(rest[:-4] + "a")
    if rest.endswith("wa") and len(rest) >= 5:
        base = rest[:-2]
        out.append(base if base[-1] in VOWELS else base + "a")
    if negative_present and rest.endswith("i") and len(rest) >= 4 and rest[-2] not in VOWELS:
        out.append(rest[:-1] + "a")
    return out


def _stripped_ok(rest: str, rules: dict) -> bool:
    return len(rest) >= rules["minStem"] + 1 or (len(rest) == rules["minStem"] and rest.endswith("i"))


def _remainders(after_tam: str, rules: dict) -> list[str]:
    found: list[str] = []
    rel = next((r for r in rules["relative"] if after_tam.startswith(r)), None)
    for rel_part in (["", rel] if rel else [""]):
        after_rel = after_tam[len(rel_part):]
        om = next((o for o in rules["object"] if after_rel.startswith(o)), None)
        for om_part in (["", om] if om else [""]):
            rest = after_rel[len(om_part):]
            stripped = rel_part != "" or om_part != ""
            long_enough = _stripped_ok(rest, rules) if stripped else len(rest) >= rules["minStem"]
            if long_enough and any(ch in VOWELS for ch in rest) and rest not in found:
                found.append(rest)
    return found


def _parse_verb(token: str, rules: dict) -> list[str]:
    for sc in rules["subject"]:
        if not token.startswith(sc):
            continue
        after_sc = token[len(sc):]
        for tam in rules["tam"]:
            if not after_sc.startswith(tam):
                continue
            rests = _remainders(after_sc[len(tam):], rules)
            if rests:
                return [v for rest in rests for v in _stem_variants(rest, False)]
    for neg in rules["negativeSubject"]:
        if not token.startswith(neg):
            continue
        after_neg = token[len(neg):]
        for tam in rules["negativeTam"]:
            if not after_neg.startswith(tam):
                continue
            rests = _remainders(after_neg[len(tam):], rules)
            if rests:
                return [v for rest in rests for v in _stem_variants(rest, False)]
        if token.endswith("i"):
            rests = _remainders(after_neg, rules)
            if rests:
                return [v for rest in rests for v in _stem_variants(rest, True)]
    for hab in rules["habitual"]:
        if not token.startswith(hab):
            continue
        rests = _remainders(token[len(hab):], rules)
        if rests:
            return [v for rest in rests for v in _stem_variants(rest, False)]
    return []


def analyse_token(token: str, rules: dict) -> list[str]:
    if len(token) < rules["minToken"] or not LETTERS_ONLY.match(token):
        return []
    extras: list[str] = []
    word = re.sub(r"(.)\1{2,}", r"\1", token)
    if word != token:
        extras.append(word)
    if len(word) >= 5 and word[-1] in VOWELS and word[-1] == word[-2]:
        word = word[:-1]
        extras.append(word)
    plural = rules["nounPlural"]
    if word.startswith(plural) and len(word) - len(plural) >= rules["minNounRest"]:
        extras.append(word[len(plural):])
    extras.extend(_parse_verb(word, rules))
    return [extra for extra in extras if extra != token and len(extra) >= 3]


def morphology_extras(normalized: str, rules: dict | None = None) -> list[str]:
    rules = _compiled(json.dumps(rules or load_morphology_rules(), sort_keys=True))
    seen: set[str] = set()
    extras: list[str] = []
    for token in normalized.split(" "):
        for extra in analyse_token(token, rules):
            if extra not in seen:
                seen.add(extra)
                extras.append(extra)
    return extras


def prepare_model_text(value: str, morphology: bool = True, concepts: bool = True) -> str:
    """The text the category model sees: the glossary-expanded report, then the
    street-language stems (expanded on their own so no phrase spans the join),
    then one token for each concept the text expresses."""
    normalized = normalize_text(value)
    text = expand_lingo_terms(normalized)
    if morphology:
        extras = morphology_extras(normalized)
        if extras:
            text = f"{text} {expand_lingo_terms(' '.join(extras))}".strip()
    if concepts:
        tokens = concept_tokens(text)
        if tokens:
            text = f"{text} {' '.join(tokens)}"
    return text


def prepare_batch(texts, morphology: bool = True, concepts: bool = True) -> list[str]:
    return [prepare_model_text(text, morphology, concepts) for text in texts]
