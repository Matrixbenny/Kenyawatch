from __future__ import annotations

import io
import csv
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ml.src.audit_dataset import main as audit_main
from ml.src.dataset import CATEGORIES, DatasetError, audit_rows, validate_for_training
from ml.src.generate_synthetic_dataset import (
    OUTPUT_CSV,
    OUTPUT_METADATA,
    rows_for_dataset,
)
from ml.src.generate_realistic_synthetic_dataset import (
    OUT_CSV,
    OUT_METADATA,
    rows_for_dataset as realistic_rows,
)
from ml.src.train import MODEL_CANDIDATES, train


class DatasetAuditTests(unittest.TestCase):
    def test_realistic_synthetic_corpus_is_deterministic_balanced_and_grouped(self) -> None:
        rows = realistic_rows()
        self.assertEqual(rows, realistic_rows())
        self.assertEqual(len(rows), 2160)
        self.assertEqual(Counter(row["label"] for row in rows), {category: 240 for category in CATEGORIES})
        self.assertEqual(Counter(row["language"] for row in rows), {
            "English": 540,
            "Kiswahili": 540,
            "Mixed": 540,
            "Sheng": 540,
        })
        groups_by_category = {
            category: {row["group_id"] for row in rows if row["label"] == category}
            for category in CATEGORIES
        }
        self.assertTrue(all(len(groups) == 60 for groups in groups_by_category.values()))
        for category, group_ids in groups_by_category.items():
            for group_id in group_ids:
                group_rows = [row for row in rows if row["group_id"] == group_id]
                self.assertEqual(len(group_rows), 4, category)
                self.assertEqual(Counter(row["language"] for row in group_rows), {
                    "English": 1,
                    "Kiswahili": 1,
                    "Mixed": 1,
                    "Sheng": 1,
                })
        self.assertEqual(len({row["text"].strip().casefold() for row in rows}), len(rows))

        with OUT_CSV.open(encoding="utf-8", newline="") as source:
            generated_csv = list(csv.DictReader(source))
        metadata = json.loads(OUT_METADATA.read_text(encoding="utf-8"))
        self.assertEqual(generated_csv, rows)
        self.assertEqual(metadata["status"], "synthetic_only_not_for_real_world_validation_or_production_training")
        self.assertIn("not a real-world accuracy estimate", " ".join(metadata["limitations"]))

    def test_synthetic_corpus_is_deterministic_balanced_and_grouped(self) -> None:
        rows = rows_for_dataset()
        self.assertEqual(rows, rows_for_dataset())
        self.assertEqual(len(rows), 540)
        self.assertEqual(Counter(row["label"] for row in rows), {category: 60 for category in CATEGORIES})

        groups_by_category = {
            category: {
                row["group_id"] for row in rows if row["label"] == category
            }
            for category in CATEGORIES
        }
        self.assertTrue(all(len(groups) == 15 for groups in groups_by_category.values()))
        for category, group_ids in groups_by_category.items():
            for group_id in group_ids:
                group_rows = [row for row in rows if row["group_id"] == group_id]
                self.assertEqual(len(group_rows), 4, category)
                self.assertEqual(Counter(row["language"] for row in group_rows), {
                    "English": 2,
                    "Kiswahili": 2,
                })
        self.assertEqual(len({row["text"] for row in rows}), len(rows))

        with OUTPUT_CSV.open(encoding="utf-8", newline="") as source:
            generated_csv = list(csv.DictReader(source))
        metadata = json.loads(OUTPUT_METADATA.read_text(encoding="utf-8"))
        self.assertEqual(generated_csv, rows)
        self.assertEqual(metadata["status"], "synthetic_only_not_for_real_world_validation_or_production_training")
        self.assertIn("not a real-world accuracy estimate", " ".join(metadata["limitations"]))

    def test_audit_reports_aggregate_quality_without_echoing_text(self) -> None:
        rows = [
            {"text": "Private report text", "label": "Bribery", "group_id": "incident-1"},
            {"text": "  PRIVATE   REPORT TEXT ", "label": "bribery", "group_id": ""},
            {"text": "", "label": "", "group_id": ""},
        ]

        report = audit_rows(rows, {"text", "label", "group_id"})

        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["empty_text"], 1)
        self.assertEqual(report["empty_label"], 1)
        self.assertEqual(report["invalid_label_values"], 1)
        self.assertEqual(report["exact_duplicate_text_rows"], 1)
        self.assertNotIn("Private report text", str(report))

    def test_training_rejects_empty_or_unreviewed_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty text and labels"):
            validate_for_training(
                [{"text": "", "label": "Bribery", "group_id": "incident-1"}],
                {"text", "label", "group_id"},
            )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_for_training(
                [{"text": "Report narrative", "label": "bribery", "group_id": "incident-1"}],
                {"text", "label", "group_id"},
            )

    def test_training_rejects_too_few_independent_incident_groups(self) -> None:
        rows = [
            {
                "text": f"{category} reviewed conduct example number {index}",
                "label": category,
                "group_id": f"g-{index % 9}",
            }
            for category in CATEGORIES
            for index in range(50)
        ]
        with self.assertRaisesRegex(DatasetError, "at least 10 distinct incident groups"):
            validate_for_training(rows, {"text", "label", "group_id"})

    def test_audit_cli_does_not_print_report_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory) / "reports.csv"
            dataset.write_text(
                "text,label,group_id\nPrivate report narrative,Bribery,incident-1\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch("sys.argv", ["audit_dataset", str(dataset)]), redirect_stdout(output):
                result = audit_main()

        self.assertEqual(result, 0)
        self.assertIn('"rows": 1', output.getvalue())
        self.assertNotIn("Private report narrative", output.getvalue())

    def test_category_taxonomy_matches_backend(self) -> None:
        categories_file = Path(__file__).parents[2] / "backend" / "utils" / "categories.js"
        source = categories_file.read_text(encoding="utf-8")
        for category in CATEGORIES:
            self.assertIn(f"'{category}'", source)

    def test_training_pipeline_writes_only_aggregate_evaluation_and_model(self) -> None:
        rows = [
            {
                "text": (
                    f"{category.lower().replace('/', ' ')} alleged conduct report "
                    f"sample {group_index} item {item_index}"
                ),
                "label": category,
                "group_id": f"{category}-{group_index}",
                "language": "English",
                "county": "Nairobi",
            }
            for category in CATEGORIES
            for group_index in range(12)
            for item_index in range(5)
        ]
        # The register supplement is switched off: this test checks the pipeline's
        # outputs on a tiny corpus, not the deployed training data.
        with tempfile.TemporaryDirectory() as temporary_directory, patch.dict("os.environ", {"KW_SUPPLEMENT": "0"}):
            result = train(
                rows,
                Path(temporary_directory),
                data_status="synthetic_only_not_for_real_world_validation_or_production_training",
            )
            self.assertEqual(result["rows"], len(rows))
            self.assertEqual(
                result["data_status"],
                "synthetic_only_not_for_real_world_validation_or_production_training",
            )
            self.assertFalse(result["real_world_performance_claim"])
            self.assertIn("fold_local_majority_baseline", result)
            self.assertEqual(
                result["nested_out_of_fold"]["confusion_matrix"]["labels"],
                list(CATEGORIES),
            )
            self.assertEqual(
                result["validation"]["candidate_models"],
                list(MODEL_CANDIDATES),
            )
            self.assertEqual(
                len(result["outer_fold_selected_models"]),
                5,
            )
            self.assertEqual(len(result["outer_fold_metrics"]), 5)
            self.assertGreaterEqual(result["outer_fold_macro_f1_std"], 0)
            self.assertTrue((Path(temporary_directory) / "category_model.joblib").is_file())
            self.assertTrue((Path(temporary_directory) / "category_model.runtime.json").is_file())
            self.assertTrue((Path(temporary_directory) / "evaluation.json").is_file())
            self.assertTrue((Path(temporary_directory) / "model_metadata.json").is_file())
            metadata = json.loads(
                (Path(temporary_directory) / "model_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["data_status"], result["data_status"])
            self.assertFalse(metadata["real_world_performance_claim"])
            self.assertIn("Synthetic-only", metadata["note"])
            model_bytes = (Path(temporary_directory) / "category_model.joblib").read_bytes()
            self.assertEqual(metadata["artifact_sha256"], hashlib.sha256(model_bytes).hexdigest())
            runtime_model = json.loads(
                (Path(temporary_directory) / "category_model.runtime.json").read_text(encoding="utf-8")
            )
            self.assertEqual(runtime_model["modelVersion"], metadata["model_version"])
            self.assertEqual(runtime_model["trainingDataStatus"], result["data_status"])
            self.assertFalse(runtime_model["realWorldPerformanceClaim"])
            self.assertCountEqual(runtime_model["categories"], CATEGORIES)
            self.assertIn("word", runtime_model["features"])
            # The artefact records how text was prepared, so the API prepares it the same way.
            self.assertEqual(runtime_model["preprocessing"]["normaliser"], "kenya-lingo")
            self.assertIn("subject", runtime_model["preprocessing"]["morphology"])
            self.assertIn("bribe", runtime_model["preprocessing"]["concepts"]["concepts"])
            # Confidence is calibrated on grouped out-of-fold scores and travels with the model.
            self.assertGreater(runtime_model["calibration"]["temperature"], 0)
            self.assertEqual(runtime_model["calibration"]["confidentAt"], 0.6)
            self.assertIn("accuracy_when_confident", result["calibration"])


class RegisterSupplementTests(unittest.TestCase):
    def test_register_corpus_is_valid_grouped_and_separate_from_the_test_corpora(self) -> None:
        from ml.src.dataset import read_rows
        from ml.src.train import SUPPLEMENT_PATH

        rows, headers = read_rows(SUPPLEMENT_PATH)
        validate_for_training(rows, headers)
        self.assertEqual(len(rows), 10000)
        self.assertEqual(len({row["text"] for row in rows}), 10000)
        self.assertEqual(Counter(row["language"] for row in rows),
                         {"English": 2500, "Kiswahili": 2500, "Mixed": 2500, "Sheng": 2500})
        by_group: dict[str, set[str]] = {}
        for row in rows:
            by_group.setdefault(row["group_id"], set()).add(row["language"])
        self.assertEqual(len(by_group), 2500)
        self.assertTrue(all(len(languages) == 4 for languages in by_group.values()))
        synthetic = Path(__file__).parents[1] / "data" / "synthetic"
        scored = {
            row["text"].strip().lower()
            for name in ("kenya_watch_synthetic_seed.csv", "kenya_watch_synthetic_colloquial.csv")
            for row in csv.DictReader((synthetic / name).open(encoding="utf-8"))
        }
        self.assertFalse(scored & {row["text"].strip().lower() for row in rows})

    def test_supplement_is_added_to_training_only_with_its_weight(self) -> None:
        from ml.src import train as train_module

        class Recorder:
            def fit(self, texts, labels, **params):
                self.texts, self.labels, self.params = texts, labels, params
                return self

        extra = [{"text": "kanjo alinidai thao", "label": "Bribery", "group_id": "g"}]
        with patch.object(train_module, "supplement_rows", return_value=extra):
            model = train_module._fit(Recorder(), ["main report"], ["Other"])
        self.assertEqual(model.texts, ["main report", "kanjo alinidai thao"])
        self.assertEqual(model.params["classifier__sample_weight"], [1.0, train_module.SUPPLEMENT_WEIGHT])
        with patch.dict("os.environ", {"KW_SUPPLEMENT": "0"}):
            self.assertEqual(train_module.supplement_rows(), [])



if __name__ == "__main__":
    unittest.main()
