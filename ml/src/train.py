"""Train and evaluate an experimental, offline category model."""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ml.src.dataset import CATEGORIES, DatasetError, read_rows, validate_for_training
from ml.src.runtime_export import export_runtime_model
from ml.src.text_prep import TextPrep

MODEL_CANDIDATES = ("word_logistic", "word_char_linear_svc")


def _metrics(y_true: list[str], y_pred: list[str], labels: tuple[str, ...]) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "confusion_matrix": {
            "labels": list(labels),
            "counts": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
        "per_category": {
            category: {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
            for index, category in enumerate(labels)
        },
    }


def street_language_enabled() -> bool:
    """The Beck-informed morphology layer is on unless KW_MORPHOLOGY=0 (used for comparisons)."""
    return os.environ.get("KW_MORPHOLOGY", "1") != "0"


def lexicon_enabled() -> bool:
    """The learned Sheng lexicon (unknown words only) is on unless KW_LEXICON=0."""
    return os.environ.get("KW_LEXICON", "1") != "0"


def concepts_enabled() -> bool:
    """The concept layer (ml/lingo/concepts.json) is on unless KW_CONCEPTS=0."""
    return os.environ.get("KW_CONCEPTS", "1") != "0"


# Above this calibrated probability the suggestion is shown as a single
# category; below it the reviewer sees the two most likely categories instead.
CONFIDENT_AT = 0.6


def _build_model(model_name: str, morphology: bool | None = None) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline

    # Text is prepared exactly as the live API prepares it (glossary, then the
    # street-language stems). The step has no fitted state, so it cannot leak
    # information from validation folds.
    prep = ("prep", TextPrep(
        morphology=street_language_enabled() if morphology is None else morphology,
        concepts=concepts_enabled(),
        lexicon=lexicon_enabled(),
    ))

    word_features = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
    )
    if model_name == "word_logistic":
        from sklearn.linear_model import LogisticRegression

        return Pipeline([
            prep,
            ("features", word_features),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            )),
        ])
    if model_name == "word_char_linear_svc":
        from sklearn.svm import LinearSVC

        features = FeatureUnion([
            ("word", word_features),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                lowercase=True,
                ngram_range=(2, 6),
                min_df=2,
                max_features=50000,
                sublinear_tf=True,
            )),
        ])
        return Pipeline([
            prep,
            ("features", features),
            ("classifier", LinearSVC(
                class_weight="balanced",
                random_state=42,
            )),
        ])
    raise DatasetError(f"Unknown model candidate: {model_name}")


def _validate_split(y: list[str], groups: list[str], train_indices: Any, test_indices: Any) -> None:
    train_labels = {y[index] for index in train_indices}
    test_labels = {y[index] for index in test_indices}
    if train_labels != set(CATEGORIES) or test_labels != set(CATEGORIES):
        raise DatasetError(
            "A grouped validation fold lacks one or more categories. Add independently reviewed "
            "examples from more incident groups in each category; no score can be reported."
        )
    if set(groups[index] for index in train_indices) & set(groups[index] for index in test_indices):
        raise DatasetError("Incident groups overlap between training and validation folds.")


def _cross_validate_candidate(
    rows: list[dict[str, str]],
    model_name: str,
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedGroupKFold

    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    groups = [row["group_id"] for row in rows]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    predictions: list[str | None] = [None] * len(rows)
    for train_indices, test_indices in splitter.split(texts, labels, groups):
        _validate_split(labels, groups, train_indices, test_indices)
        model = _build_model(model_name)
        model.fit([texts[index] for index in train_indices], [labels[index] for index in train_indices])
        for index, prediction in zip(test_indices, model.predict([texts[index] for index in test_indices])):
            predictions[index] = str(prediction)
    if any(prediction is None for prediction in predictions):
        raise DatasetError("Grouped validation did not produce one out-of-fold prediction for every row.")
    predicted_labels = [prediction for prediction in predictions if prediction is not None]
    return {
        "predictions": predicted_labels,
        "metrics": _metrics(labels, predicted_labels, CATEGORIES),
    }


def _select_model(rows: list[dict[str, str]], n_splits: int, random_state: int) -> tuple[str, dict[str, float]]:
    scores = {
        model_name: _cross_validate_candidate(rows, model_name, n_splits, random_state)["metrics"]["macro_f1"]
        for model_name in MODEL_CANDIDATES
    }
    winner = max(MODEL_CANDIDATES, key=lambda name: scores[name])
    return winner, scores


def evaluate(rows: list[dict[str, str]]) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedGroupKFold

    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    groups = [row["group_id"] for row in rows]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    predictions: list[str | None] = [None] * len(rows)
    majority_predictions: list[str | None] = [None] * len(rows)
    selected_candidates: list[str] = []
    selection_scores: list[dict[str, float]] = []
    outer_fold_metrics: list[dict[str, Any]] = []

    for fold_index, (train_indices, test_indices) in enumerate(splitter.split(texts, labels, groups)):
        _validate_split(labels, groups, train_indices, test_indices)
        outer_train_rows = [rows[index] for index in train_indices]
        selected_model, fold_selection_scores = _select_model(
            outer_train_rows,
            n_splits=3,
            random_state=100 + fold_index,
        )
        selected_candidates.append(selected_model)
        selection_scores.append(fold_selection_scores)
        majority_label = Counter(labels[index] for index in train_indices).most_common(1)[0][0]
        for index in test_indices:
            majority_predictions[index] = majority_label
        model = _build_model(selected_model)
        model.fit([texts[index] for index in train_indices], [labels[index] for index in train_indices])
        fold_predictions: list[str] = []
        for index, prediction in zip(test_indices, model.predict([texts[i] for i in test_indices])):
            predictions[index] = str(prediction)
            fold_predictions.append(str(prediction))
        fold_truth = [labels[index] for index in test_indices]
        outer_fold_metrics.append({
            "selected_model": selected_model,
            "test_rows": len(test_indices),
            "test_incident_groups": len({groups[index] for index in test_indices}),
            "macro_f1": _metrics(fold_truth, fold_predictions, CATEGORIES)["macro_f1"],
        })

    if any(prediction is None for prediction in predictions):
        raise DatasetError("Group validation did not produce one out-of-fold prediction for every row.")
    if any(prediction is None for prediction in majority_predictions):
        raise DatasetError("Grouped validation did not produce a majority-baseline prediction for every row.")
    predicted_labels = [prediction for prediction in predictions if prediction is not None]
    baseline_labels = [prediction for prediction in majority_predictions if prediction is not None]
    final_model_name, final_selection_scores = _select_model(rows, n_splits=5, random_state=42)
    result: dict[str, Any] = {
        "validation": {
            "outer": "5-fold StratifiedGroupKFold; incident groups are disjoint",
            "model_selection": "3-fold stratified group validation inside each outer training fold",
            "selection_metric": "macro F1",
            "candidate_models": list(MODEL_CANDIDATES),
        },
        "rows": len(rows),
        "category_counts": dict(sorted(Counter(labels).items())),
        "fold_local_majority_baseline": _metrics(labels, baseline_labels, CATEGORIES),
        "nested_out_of_fold": _metrics(labels, predicted_labels, CATEGORIES),
        "outer_fold_selected_models": selected_candidates,
        "outer_fold_inner_selection_macro_f1": selection_scores,
        "outer_fold_metrics": outer_fold_metrics,
        "outer_fold_macro_f1_mean": round(
            statistics.mean(fold["macro_f1"] for fold in outer_fold_metrics),
            4,
        ),
        "outer_fold_macro_f1_std": round(
            statistics.pstdev(fold["macro_f1"] for fold in outer_fold_metrics),
            4,
        ),
        "final_model_selection": {
            "method": "5-fold stratified group CV on all approved data; selection scores are not an unbiased final estimate",
            "candidate_macro_f1": final_selection_scores,
            "selected_model": final_model_name,
        },
        "subgroups": {},
    }
    for column in ("language", "county"):
        values = sorted({row.get(column, "").strip() for row in rows if row.get(column, "").strip()})
        result["subgroups"][column] = {}
        for value in values:
            indices = [
                index for index, row in enumerate(rows)
                if row.get(column, "").strip() == value
            ]
            group_labels = [labels[index] for index in indices]
            if len(indices) < 20 or len(set(group_labels)) < 2:
                result["subgroups"][column][value] = {
                    "status": "insufficient_sample",
                    "rows": len(indices),
                    "represented_categories": len(set(group_labels)),
                }
                continue
            subgroup_metrics = _metrics(
                group_labels,
                [predicted_labels[index] for index in indices],
                tuple(category for category in CATEGORIES if category in group_labels),
            )
            result["subgroups"][column][value] = {
                "status": "aggregate_only",
                "rows": len(indices),
                **subgroup_metrics,
            }
    return result


def calibrate(rows: list[dict[str, str]], model_name: str) -> dict[str, Any]:
    """Fit a softmax temperature on grouped out-of-fold decision values, so the
    probabilities the API derives from the linear scores are calibrated, and
    record how the confidence threshold behaves on those out-of-fold reports."""
    import numpy as np
    from scipy.optimize import minimize_scalar
    from sklearn.model_selection import StratifiedGroupKFold

    texts = [row["text"] for row in rows]
    labels = np.array([row["label"] for row in rows])
    groups = [row["group_id"] for row in rows]
    decisions = np.zeros((len(rows), len(CATEGORIES)))
    classes = None
    for train_idx, test_idx in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(texts, labels, groups):
        model = _build_model(model_name).fit([texts[i] for i in train_idx], labels[train_idx])
        classes = model.classes_
        decisions[test_idx] = model.decision_function([texts[i] for i in test_idx])
    truth = np.searchsorted(classes, labels)

    def probabilities(temperature: float):
        z = decisions / temperature
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def nll(temperature: float) -> float:
        return float(-np.mean(np.log(probabilities(temperature)[np.arange(len(truth)), truth] + 1e-12)))

    def ece(p) -> float:
        confidence, correct, total = p.max(axis=1), p.argmax(axis=1) == truth, 0.0
        for low in np.linspace(0, 1, 10, endpoint=False):
            in_bin = (confidence > low) & (confidence <= low + 0.1)
            if in_bin.any():
                total += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
        return float(total)

    temperature = float(minimize_scalar(nll, bounds=(0.01, 5.0), method="bounded").x)
    p = probabilities(temperature)
    confident = p.max(axis=1) >= CONFIDENT_AT
    correct = p.argmax(axis=1) == truth
    top2 = np.argsort(-p, axis=1)[:, :2]
    return {
        "method": "temperature scaling on 5-fold grouped out-of-fold decision values",
        "temperature": round(temperature, 4),
        "confidentAt": CONFIDENT_AT,
        "expected_calibration_error_before": round(ece(probabilities(1.0)), 4),
        "expected_calibration_error_after": round(ece(p), 4),
        "confident_share": round(float(confident.mean()), 4),
        "accuracy_when_confident": round(float(correct[confident].mean()), 4) if confident.any() else None,
        "top2_accuracy_when_unsure": round(float(np.mean([truth[i] in top2[i] for i in np.where(~confident)[0]])), 4)
        if (~confident).any() else None,
    }


def train(
    rows: list[dict[str, str]],
    output_dir: Path,
    data_status: str = "unverified_local_dataset",
) -> dict[str, Any]:
    import joblib

    headers = {"text", "label", "group_id"} | {
        column for column in ("language", "county")
        if any(column in row for row in rows)
    }
    validate_for_training(rows, headers)
    if os.environ.get("KW_DEPLOY_BUILD") == "1":
        # Deployment build: the nested evaluation only reports how good the model
        # is and does not change it, so it is skipped to keep the image build
        # short on small build machines. Model selection and calibration still run.
        selected, scores = _select_model(rows, n_splits=5, random_state=42)
        evaluation = {
            "deploy_build": True,
            "note": "Nested evaluation skipped in the deployment build; run ml.src.train locally for the full evaluation.",
            "rows": len(rows),
            "final_model_selection": {"candidate_macro_f1": scores, "selected_model": selected},
        }
    else:
        evaluation = evaluate(rows)
    evaluation["data_status"] = data_status
    evaluation["real_world_performance_claim"] = False
    final_model = _build_model(evaluation["final_model_selection"]["selected_model"])
    final_model.fit([row["text"] for row in rows], [row["label"] for row in rows])
    evaluation["calibration"] = calibrate(rows, evaluation["final_model_selection"]["selected_model"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "category_model.joblib"
    joblib.dump(final_model, model_path)
    runtime_metadata = export_runtime_model(
        final_model,
        model_path,
        output_dir / "category_model.runtime.json",
        data_status,
        calibration={key: evaluation["calibration"][key] for key in ("temperature", "confidentAt")},
    )
    (output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "model_metadata.json").write_text(
        json.dumps({
            "model": evaluation["final_model_selection"]["selected_model"],
            **runtime_metadata,
            "categories": list(CATEGORIES),
            "training_rows": len(rows),
            "training_group_count": len({row["group_id"] for row in rows}),
            "data_status": data_status,
            "real_world_performance_claim": False,
            "note": (
                "Synthetic-only pipeline demonstration; not valid for real-world validation or production."
                if data_status == "synthetic_only_not_for_real_world_validation_or_production_training"
                else "Experimental local artifact. Selection scores are not an unbiased final performance estimate."
            ),
        }, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return evaluation


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m ml.src.train <private-csv-path>", file=sys.stderr)
        return 2
    try:
        input_path = Path(sys.argv[1])
        rows, headers = read_rows(input_path)
        validate_for_training(rows, headers)
        metadata_path = input_path.with_suffix(".metadata.json")
        data_status = "unverified_local_dataset"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") == "synthetic_only_not_for_real_world_validation_or_production_training":
                data_status = metadata["status"]
            elif metadata.get("status") == "adjudicated_real_reports" and metadata.get("approval_reference"):
                # Real, double-labelled reports exported under a recorded approval (see README).
                data_status = metadata["status"]
        evaluation = train(
            rows,
            Path(__file__).resolve().parents[1] / "artifacts",
            data_status=data_status,
        )
        print(json.dumps(evaluation, indent=2, ensure_ascii=True))
        print("Experimental artifacts saved locally. The runtime model is available to reviewer suggestions only.")
        return 0
    except DatasetError as error:
        print(f"Training stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
