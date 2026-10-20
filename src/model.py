"""The MATRIX classifier: TF-IDF + Logistic Regression with calibrated confidence.

Why this design
---------------
* Logistic Regression is linear in the TF-IDF features, so every prediction can be split
  exactly into one contribution per word (that is what Linear SHAP computes).
* Raw Logistic Regression probabilities are often over- or under-confident. We wrap the
  model in Platt (sigmoid) calibration so that "90% confident" really means roughly
  9 out of 10 such messages are suspicious. Platt scaling is a monotonic function of the
  logit, so it never changes which words matter, only how confident the final number is.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .preprocess import normalise, tokenize


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=str.split, token_pattern=None, lowercase=False,
        ngram_range=(1, 2), min_df=2, sublinear_tf=True,
    )


def build_word_vocab(texts, min_count: int = 2) -> set[str]:
    """Single words that appear in at least `min_count` of the given (training) messages."""
    counts = Counter()
    for t in texts:
        counts.update({u.token for u in tokenize(t, True, None) if " " not in u.token})
    return {w for w, c in counts.items() if c >= min_count}


class MatrixModel:
    """Vectoriser + base Logistic Regression + Platt calibration."""

    def __init__(self, C: float = 10.0, threshold: float = 0.5, robust: bool = True,
                 random_state: int = 42):
        self.C = C
        self.threshold = threshold
        self.robust = robust
        self.random_state = random_state

    # ------------------------------------------------------------------ training
    def fit(self, texts, labels):
        y = np.asarray(labels)
        # Pass 1: single words seen in 2+ training messages. This vocabulary decides how
        # disguised words are read ("wooon" -> "won", "f r e e" -> "free") and is reused
        # unchanged at prediction time, so training and the app read text the same way.
        self.word_vocab = build_word_vocab(texts) if self.robust else None
        # Pass 2: the documents the model is trained on
        docs = [self.normalise(t) for t in texts]
        base = make_pipeline(
            make_vectorizer(),
            LogisticRegression(C=self.C, max_iter=5000, solver="liblinear",
                               random_state=self.random_state))
        # The vectoriser sits inside the pipeline, so each calibration fold builds its own
        # vocabulary and IDF weights from its own training part (no leakage into the fold
        # used to fit the calibrator). ensemble=False: the calibrator is fit on 5-fold
        # out-of-fold scores, then one final pipeline is trained on all training data.
        # One model = one set of coefficients to explain.
        self.calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=5,
                                                 ensemble=False)
        self.calibrated.fit(docs, y)
        cc = self.calibrated.calibrated_classifiers_[0]
        self.vectorizer, self.lr = cc.estimator[0], cc.estimator[-1]
        self._cal = cc.calibrators[0]
        # Background (average training document) for Linear SHAP
        X = self.vectorizer.transform(docs)
        self.mu = np.asarray(X.mean(axis=0)).ravel()
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.sklearn_version = sklearn.__version__
        self.coef = self.lr.coef_.ravel()
        self.intercept = float(self.lr.intercept_[0])
        return self

    # --------------------------------------------------------------- prediction
    def _vocab(self):
        return getattr(self, "word_vocab", None) if self.robust else None

    def tokenize(self, text):
        """Explainable units, normalised exactly as the model sees them."""
        return tokenize(text, self.robust, self._vocab())

    def normalise(self, text):
        return normalise(text, self.robust, self._vocab())

    def vectorize_docs(self, docs):
        return self.vectorizer.transform(docs)

    def logit_docs(self, docs) -> np.ndarray:
        """Raw (uncalibrated) log-odds of 'suspicious' for normalised documents."""
        return self.lr.decision_function(self.vectorize_docs(docs))

    def calibrate_logit(self, z) -> np.ndarray:
        return self._cal.predict(np.atleast_1d(z))

    def proba_docs(self, docs) -> np.ndarray:
        return self.calibrate_logit(self.logit_docs(docs))

    def proba(self, texts) -> np.ndarray:
        """Calibrated P(suspicious) for raw texts."""
        return self.proba_docs([self.normalise(t) for t in texts])

    def raw_proba(self, texts) -> np.ndarray:
        """Uncalibrated Logistic Regression probability (for the calibration report)."""
        docs = [self.normalise(t) for t in texts]
        return self.lr.predict_proba(self.vectorize_docs(docs))[:, 1]

    def predict(self, texts) -> np.ndarray:
        return (self.proba(texts) >= self.threshold).astype(int)
