"""Model experiments for the category suggestion, on a fixed three-part test bench.

  in_domain   5-fold StratifiedGroupKFold on the combined corpus (756 rows)
  cross       train on the formal seed corpus only (540 rows), test on the
              separately written colloquial corpus (216 rows): a real shift in
              register, spelling, Sheng and code-switching
  street      the in-domain folds with the test fold rewritten in the Sheng
              patterns Beck (2015) documents (ml/lingo/evaluate_street_language.py)

Every candidate is a linear model over the same kind of TF-IDF features, so the
winner can be exported to the existing JavaScript scorer without new services.

    python -m ml.src.experiments
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from ml.lingo.evaluate_street_language import beck_rewrite
from ml.src.text_prep import prepare_batch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "ml" / "data" / "synthetic"
OUT = ROOT / "ml" / "artifacts" / "model_experiments.json"


def load(name):
    return list(csv.DictReader((SYN / name).open(encoding="utf-8")))


def features(word_min_df=2, char_range=(3, 5), char_min_df=2):
    return FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
                                 min_df=word_min_df, max_features=50000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", lowercase=True, ngram_range=char_range,
                                 min_df=char_min_df, max_features=50000, sublinear_tf=True)),
    ])


class AveragedLinear(BaseEstimator, ClassifierMixin):
    """Average of the standardised decision functions of several linear models.
    Because every member is linear over the same features, the average is itself a
    single linear model (one weight matrix), so it exports like any other."""

    def __init__(self, members=None):
        self.members = members

    def fit(self, X, y):
        self.fitted_ = [m.fit(X, y) for m in self.members]
        self.classes_ = self.fitted_[0].classes_
        scales = [np.std(m.decision_function(X)) or 1.0 for m in self.fitted_]
        self.coef_ = sum(m.coef_ / s for m, s in zip(self.fitted_, scales)) / len(self.fitted_)
        self.intercept_ = sum(m.intercept_ / s for m, s in zip(self.fitted_, scales)) / len(self.fitted_)
        return self

    def decision_function(self, X):
        return X @ self.coef_.T + self.intercept_

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


class NBSVM(BaseEstimator, ClassifierMixin):
    """Wang & Manning's NB-SVM, one-vs-rest: features are scaled by the naive Bayes
    log-count ratio before a linear SVM. The scaling folds into the weights, so the
    result is still one linear model over the original features."""

    def __init__(self, C=1.0, alpha=1.0):
        self.C = C
        self.alpha = alpha

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        X = sparse.csr_matrix(X)
        coefs, intercepts = [], []
        for c in self.classes_:
            pos = (y == c)
            p = self.alpha + np.asarray(X[pos].sum(axis=0)).ravel()
            q = self.alpha + np.asarray(X[~pos].sum(axis=0)).ravel()
            r = np.log((p / p.sum()) / (q / q.sum()))
            svm = LinearSVC(C=self.C, class_weight="balanced", random_state=42).fit(X.multiply(r).tocsr(), pos.astype(int))
            coefs.append(svm.coef_.ravel() * r)
            intercepts.append(svm.intercept_[0])
        self.coef_ = np.vstack(coefs)
        self.intercept_ = np.array(intercepts)
        return self

    def decision_function(self, X):
        return np.asarray(X @ self.coef_.T) + self.intercept_

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


def candidates():
    def prep(concepts):
        return ("prep", FunctionTransformer(prepare_batch, kw_args={"morphology": True, "concepts": concepts}))
    svm = lambda C=1.0: LinearSVC(C=C, class_weight="balanced", random_state=42)
    lr = lambda C=10.0: LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=42)
    rich = lambda: features(1, (2, 6))
    return {
        "M0 current: SVM, char 3-5, no concepts": lambda: Pipeline([prep(False), ("features", features()), ("classifier", svm())]),
        "M3 SVM, richer n-grams, no concepts": lambda: Pipeline([prep(False), ("features", rich()), ("classifier", svm())]),
        "C1 SVM, richer n-grams + concepts": lambda: Pipeline([prep(True), ("features", rich()), ("classifier", svm())]),
        "C2 SVM C=0.5, richer n-grams + concepts": lambda: Pipeline([prep(True), ("features", rich()), ("classifier", svm(0.5))]),
        "C3 logistic regression + concepts": lambda: Pipeline([prep(True), ("features", rich()), ("classifier", lr())]),
        "C4 averaged SVM + LR + concepts": lambda: Pipeline([prep(True), ("features", rich()),
                                                             ("classifier", AveragedLinear([svm(), lr()]))]),
    }


def macro(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def main() -> int:
    combined, seed, colloquial = load("kenya_watch_synthetic_combined.csv"), load("kenya_watch_synthetic_seed.csv"), \
        load("kenya_watch_synthetic_colloquial.csv")
    texts = [r["text"] for r in combined]
    labels = [r["label"] for r in combined]
    groups = [r["group_id"] for r in combined]
    street = [beck_rewrite(t, random.Random(2015 + i)) for i, t in enumerate(texts)]
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(texts, labels, groups))

    results = {}
    for name, make in candidates().items():
        in_dom, st = [], []
        for tr, te in folds:
            model = make().fit([texts[i] for i in tr], [labels[i] for i in tr])
            truth = [labels[i] for i in te]
            in_dom.append(macro(truth, model.predict([texts[i] for i in te])))
            st.append(macro(truth, model.predict([street[i] for i in te])))
        cross_model = make().fit([r["text"] for r in seed], [r["label"] for r in seed])
        cross = macro([r["label"] for r in colloquial], cross_model.predict([r["text"] for r in colloquial]))
        results[name] = {
            "in_domain": round(statistics.mean(in_dom), 4), "in_domain_std": round(statistics.pstdev(in_dom), 4),
            "street": round(statistics.mean(st), 4), "cross_corpus": round(cross, 4),
        }
        r = results[name]
        print(f"{name:40s} in-domain {r['in_domain']:.3f}±{r['in_domain_std']:.3f}  "
              f"street {r['street']:.3f}  cross-corpus {r['cross_corpus']:.3f}", flush=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
