"""Checks that the published results are honest: no leakage, numbers that match the saved
predictions, and a model that can be reproduced exactly from the code and the seed.
Skipped when the dataset or the trained model is not present."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
HAS_DATA = (ROOT / "data" / "SMSSpamCollection").exists() or (ROOT / "data" / "spam.csv").exists()
HAS_MODEL = (MODELS / "matrix.joblib").exists() and (MODELS / "report.json").exists()
pytestmark = pytest.mark.skipif(not (HAS_DATA and HAS_MODEL),
                                reason="needs the dataset and a trained model")


@pytest.fixture(scope="module")
def split():
    from src.data import load_sms
    from src.train import SEED
    df = load_sms()
    tr, te = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    return df, tr, te


@pytest.fixture(scope="module")
def report():
    return json.loads((MODELS / "report.json").read_text())


def test_no_duplicates_and_no_overlap(split):
    df, tr, te = split
    assert df.text.is_unique                       # duplicates removed before splitting
    assert not set(tr.text) & set(te.text)         # no message in both train and test


def test_saved_test_set_is_the_held_out_split(split):
    _, _, te = split
    tp = pd.read_csv(MODELS / "test_predictions.csv")
    assert tp.text.tolist() == te.text.tolist()
    assert tp.label.tolist() == te.label.tolist()


def test_report_matches_saved_predictions(report):
    tp = pd.read_csv(MODELS / "test_predictions.csv")
    y, yhat = tp.label.to_numpy(), (tp.prob >= report["threshold"]).astype(int).to_numpy()
    tp_, fp = int(((y == 1) & (yhat == 1)).sum()), int(((y == 0) & (yhat == 1)).sum())
    fn, tn = int(((y == 1) & (yhat == 0)).sum()), int(((y == 0) & (yhat == 0)).sum())
    t = report["test"]
    assert (t["tp"], t["fp"], t["fn"], t["tn"]) == (tp_, fp, fn, tn)
    assert abs(t["precision"] - tp_ / (tp_ + fp)) < 1e-12
    assert abs(t["recall"] - tp_ / (tp_ + fn)) < 1e-12
    assert abs(t["false_positive_rate"] - fp / (fp + tn)) < 1e-12


def test_saved_model_reproduces_saved_predictions():
    m = joblib.load(MODELS / "matrix.joblib")
    tp = pd.read_csv(MODELS / "test_predictions.csv")
    assert np.allclose(m.proba(tp.text.tolist()), tp.prob.to_numpy(), atol=1e-12)


def test_training_is_reproducible(split):
    """Retraining from the code and the seed gives the saved model back, exactly."""
    from src.model import MatrixModel
    _, tr, te = split
    fresh = MatrixModel(robust=True).fit(tr.text.tolist(), tr.label.to_numpy())
    saved = joblib.load(MODELS / "matrix.joblib")
    assert list(fresh.feature_names) == list(saved.feature_names)
    assert np.allclose(fresh.coef, saved.coef, atol=1e-10)
    assert np.allclose(fresh.proba(te.text.tolist()), saved.proba(te.text.tolist()), atol=1e-10)
