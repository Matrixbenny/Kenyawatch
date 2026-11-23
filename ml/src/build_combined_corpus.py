"""Combine the seed and colloquial corpora into one training file, with metadata.

The combined file is the corpus most experiments use. Building it by hand in each
experiment script meant it had no `.metadata.json`, so `ml.src.train` classified it
as `unverified_local_dataset`. That is technically safe but misleading: it is a
synthetic corpus, and the readiness report should say so.

Usage:
    python -m ml.src.build_combined_corpus
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

SYN = Path(__file__).resolve().parents[1] / "data" / "synthetic"
SEED = SYN / "kenya_watch_synthetic_seed.csv"
COLLOQUIAL = SYN / "kenya_watch_synthetic_colloquial.csv"
REALISTIC = SYN / "kenya_watch_synthetic_realistic.csv"
OUT = SYN / "kenya_watch_synthetic_combined.csv"
META = SYN / "kenya_watch_synthetic_combined.metadata.json"

STATUS = "synthetic_only_not_for_real_world_validation_or_production_training"


def main() -> int:
    rows: list[dict[str, str]] = []
    for path in (SEED, COLLOQUIAL, REALISTIC):
        with path.open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("text", "label", "group_id", "language", "county"))
        writer.writeheader()
        writer.writerows(rows)

    languages: dict[str, int] = {}
    for row in rows:
        languages[row["language"]] = languages.get(row["language"], 0) + 1

    META.write_text(json.dumps({
        "dataset_name": "Kenya Watch synthetic combined corpus (seed + colloquial + realistic report-form)",
        "status": STATUS,
        "generation_method": "Concatenation of kenya_watch_synthetic_seed.csv, kenya_watch_synthetic_colloquial.csv and kenya_watch_synthetic_realistic.csv",
        "rows": len(rows),
        "scenario_groups": len({row["group_id"] for row in rows}),
        "language_rows": languages,
        "inputs": [SEED.name, COLLOQUIAL.name, REALISTIC.name],
        "limitations": [
            "Synthetic text, not observations of real Kenyan reports.",
            "The Sheng passages are written by the project author and are unverified until a Sheng speaker reviews them.",
            "Variants within an incident group must stay in the same validation fold.",
            "Scores are not a real-world accuracy estimate and do not satisfy ml/src/readiness.py.",
        ],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"rows": len(rows), "groups": len({r["group_id"] for r in rows}),
                      "by_language": languages, "output": str(OUT), "metadata": str(META)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
