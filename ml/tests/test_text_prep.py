"""The training text pipeline must match the live API's, word for word."""

from __future__ import annotations

import csv
import json
import shutil
import unittest

from ml.src.text_prep import (
    analyse_token,
    load_morphology_rules,
    normalise_for_matching,
    prepare_model_text,
    _compiled,
)
from ml.lingo.common import ROOT, node_tool

NODE = shutil.which("node") is not None and (ROOT / "backend" / "node_modules").is_dir()
RULES = _compiled(json.dumps(load_morphology_rules(), sort_keys=True))
SAMPLES = [
    "Anamwitishanga pesa kila siku", "Alinibribe ndio apige stamp", "Asicontract", "Mnathink nini",
    "Wanafanyanga hivyo", "Alinidai kitu kidogo", "Walikulanga pesa ya CDF",
    "Maofficer wa kanjo walinithreaten", "Pesaaa yote imeliwa", "Sitaki kulipa hongo", "Hatupendi",
    "Huitisha chai kila wiki", "Afisa aliyependekezwa anadaiwa kuomba malipo", "Mkuchoma moto",
    "dey askd me 4 500 2 fix ma permit", "Café résumé – naïve", "", "   ",
]


class StreetLanguageTests(unittest.TestCase):
    def test_reads_beck_style_verb_frames(self) -> None:
        def stems(word):
            return analyse_token(word, RULES)
        self.assertIn("itisha", stems("anamwitishanga"))
        self.assertIn("dai", stems("alinidai"))
        self.assertIn("contract", stems("asicontract"))
        self.assertIn("threaten", stems("walinithreaten"))
        self.assertIn("fanya", stems("wanafanyanga"))
        self.assertIn("penda", stems("hatupendi"))
        self.assertIn("officer", stems("maofficer"))
        self.assertIn("pesa", stems("pesaaa"))
        self.assertNotIn("mia", stems("angetumia"))
        self.assertEqual(stems("mkuchoma"), [])

    def test_glossary_matches_whole_words_only(self) -> None:
        self.assertEqual(normalise_for_matching("mkuchoma moto"), "mkuchoma moto")
        self.assertIn("rushwa bribe", normalise_for_matching("Walitaka kitu kidogo"))

    def test_original_words_are_kept(self) -> None:
        prepared = prepare_model_text("Walinidai pesa")
        self.assertTrue(prepared.startswith("walinidai pesa "))
        self.assertEqual(prepare_model_text("Walinidai pesa", morphology=False, concepts=False), normalise_for_matching("Walinidai pesa"))
        self.assertTrue(prepare_model_text("Walinidai pesa").endswith("concept_money"))

    @unittest.skipUnless(NODE, "Node.js and the backend packages are needed for the parity check")
    def test_python_and_live_api_prepare_text_identically(self) -> None:
        corpus = ROOT / "ml" / "data" / "synthetic" / "kenya_watch_synthetic_combined.csv"
        texts = SAMPLES + [row["text"] for row in csv.DictReader(corpus.open(encoding="utf-8"))]
        self.assertEqual([prepare_model_text(text) for text in texts], node_tool("prepare", texts))

    @unittest.skipUnless(NODE, "Node.js and the backend packages are needed for the snapshot check")
    def test_glossary_snapshot_matches_the_live_glossary(self) -> None:
        snapshot = json.loads((ROOT / "ml" / "lingo" / "glossary_snapshot.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot, node_tool("export-lexicon"),
                         "Regenerate it: node backend/scripts/lingo-tools.js export-lexicon > ml/lingo/glossary_snapshot.json")


if __name__ == "__main__":
    unittest.main()
