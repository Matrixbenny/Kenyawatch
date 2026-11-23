"""Apply a completed native-speaker Sheng review sheet, with checks.

The review sheet is filled in by a Sheng speaker. This script refuses to apply it
unless every row carries a valid verdict, and it records who reviewed the sheet and
when, in the same permissioned-source pattern as ml/lingo/permissions.json.

It writes a NEW verified corpus rather than editing the originals, so the untested
text is still available and the provenance of the corrected text is explicit:

    ml/data/synthetic/kenya_watch_synthetic_sheng_verified.csv

Usage:
    python -m ml.lingo.apply_sheng_review --reviewer "Name" --reviewed-on 2026-11-22
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "ml" / "data" / "synthetic"
LINGO = ROOT / "ml" / "data" / "lingo"
REVIEW_CSV = LINGO / "sheng_native_review.csv"
OUT_CSV = SYN / "kenya_watch_synthetic_sheng_verified.csv"
ATTESTATION = LINGO / "sheng_native_review.attestation.json"

VERDICTS = {"keep", "rewrite", "drop"}
SOURCES = {
    "colloquial": "kenya_watch_synthetic_colloquial.csv",
    "registers": "kenya_watch_synthetic_registers.csv",
}


def load_review() -> list[dict[str, str]]:
    if not REVIEW_CSV.exists():
        raise SystemExit(
            "No review sheet found. Run: python -m ml.lingo.build_sheng_review_sheet"
        )
    with REVIEW_CSV.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    problems = []
    for index, row in enumerate(rows, start=2):  # row 1 is the header
        verdict = (row.get("verdict") or "").strip().lower()
        if verdict not in VERDICTS:
            problems.append(f"line {index} ({row.get('row_id')}): verdict must be keep/rewrite/drop")
            continue
        if verdict == "rewrite" and not (row.get("corrected_text") or "").strip():
            problems.append(f"line {index} ({row.get('row_id')}): verdict rewrite needs corrected_text")
        natural = (row.get("is_it_natural_sheng") or "").strip().lower()
        if natural not in {"yes", "no"}:
            problems.append(f"line {index} ({row.get('row_id')}): is_it_natural_sheng must be yes or no")
    if problems:
        raise SystemExit(
            "The review sheet is not complete; nothing was applied.\n  " + "\n  ".join(problems[:20])
            + (f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else "")
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer", required=True, help="name or initials of the Sheng speaker")
    parser.add_argument("--reviewed-on", default=date.today().isoformat())
    args = parser.parse_args()

    review = load_review()
    by_id = {row["row_id"]: row for row in review}

    kept, rewritten, dropped = 0, 0, 0
    out_rows: list[dict[str, str]] = []
    for corpus, filename in SOURCES.items():
        with (SYN / filename).open(encoding="utf-8") as handle:
            source = list(csv.DictReader(handle))
        for row in source:
            if (row.get("language") or "").strip().lower() != "sheng":
                continue
            # Find the review row for this exact sentence in this corpus.
            match = next(
                (r for r in review
                 if r["corpus"] == corpus and r["sheng_text"].strip() == row["text"].strip()),
                None,
            )
            if match is None:
                # Not sampled for review: keep as-is, but mark it unverified.
                out_rows.append({**row, "sheng_status": "unreviewed"})
                continue
            verdict = match["verdict"].strip().lower()
            if verdict == "drop":
                dropped += 1
                continue
            text = match["corrected_text"].strip() if verdict == "rewrite" else row["text"]
            if verdict == "rewrite":
                rewritten += 1
            else:
                kept += 1
            out_rows.append({**row, "text": text, "sheng_status": f"verified-{verdict}"})

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("text", "label", "group_id", "language", "county", "sheng_status"),
        )
        writer.writeheader()
        writer.writerows(out_rows)

    ATTESTATION.write_text(json.dumps({
        "review_sheet": str(REVIEW_CSV.relative_to(ROOT)),
        "reviewer": args.reviewer,
        "reviewed_on": args.reviewed_on,
        "rows_in_sheet": len(review),
        "verdicts": {
            "keep": kept,
            "rewrite": rewritten,
            "drop": dropped,
        },
        "statement": (
            "The reviewer named above checked each Sheng sentence in the review sheet and "
            "recorded whether it is natural Sheng, rewriting it where needed. Only rewritten "
            "and kept sentences are carried into the verified corpus."
        ),
        "output": str(OUT_CSV.relative_to(ROOT)),
        "note": "This verifies language, not labels. The alleged category is unchanged.",
    }, indent=2) + "\n", encoding="utf-8")

    print(f"kept {kept}, rewrote {rewritten}, dropped {dropped}; "
          f"{len(out_rows)} Sheng rows written to {OUT_CSV.relative_to(ROOT)}")
    print(f"attestation written to {ATTESTATION.relative_to(ROOT)}")
    by_id.clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
