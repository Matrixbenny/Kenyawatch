"""Build the native-speaker Sheng review sheet.

The register and colloquial corpora were written by the project author, so their
Sheng is unverified. This script writes the sheet a Sheng speaker corrects: one row
per Sheng sentence, with the alleged category, the register, and empty columns for
the reviewer's verdict and rewritten sentence.

Nothing is changed by this script, and no correction is applied until the sheet is
filled in and `apply_sheng_review.py` has checked it.

Usage:
    python -m ml.lingo.build_sheng_review_sheet
    python -m ml.lingo.build_sheng_review_sheet --max-rows 120
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "ml" / "data" / "synthetic"
OUT_DIR = ROOT / "ml" / "data" / "lingo"
OUT_CSV = OUT_DIR / "sheng_native_review.csv"

SOURCES = (
    ("kenya_watch_synthetic_colloquial.csv", "colloquial"),
    ("kenya_watch_synthetic_registers.csv", "registers"),
)

# Rows are sampled evenly across categories so the sheet is not one category deep.
PER_CATEGORY = {
    "colloquial": 3,
    "registers": 8,
}

FIELDS = [
    "row_id", "corpus", "alleged_category", "sheng_text",
    "verdict", "corrected_text", "is_it_natural_sheng", "notes",
]

# The three verdicts the reviewer may use. Anything else is rejected on ingest.
VERDICTS = ("keep", "rewrite", "drop")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for filename, corpus in SOURCES:
        path = SYN / filename
        if not path.exists():
            print(f"missing corpus: {path.name}")
            continue
        with path.open(encoding="utf-8") as handle:
            content = list(csv.DictReader(handle))
        by_category: dict[str, list[dict[str, str]]] = {}
        for row in content:
            if (row.get("language") or "").strip().lower() != "sheng":
                continue
            by_category.setdefault(row["label"], []).append(row)
        for category in sorted(by_category):
            # Deterministic, evenly spaced sample; no randomness, so the sheet is reproducible.
            pool = by_category[category]
            take = PER_CATEGORY.get(corpus, 3)
            step = max(len(pool) // take, 1)
            for row in pool[::step][:take]:
                rows.append({
                    "row_id": f"{corpus}-{len(rows) + 1:03d}",
                    "corpus": corpus,
                    "alleged_category": category,
                    "sheng_text": row["text"],
                    "verdict": "",
                    "corrected_text": "",
                    "is_it_natural_sheng": "",
                    "notes": "",
                })

    rows = rows[: args.max_rows]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} Sheng sentences to {OUT_CSV.relative_to(ROOT)}")
    print("Reviewer: set verdict to keep / rewrite / drop, rewrite the sentence in "
          "corrected_text, and answer is_it_natural_sheng with yes or no.")
    print("Then run: python -m ml.lingo.apply_sheng_review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
