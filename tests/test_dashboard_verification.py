"""Verifies that the Analyst Dashboard shows real, internally consistent numbers, and that the
app stays stable across navigation and repeated analyses. Tests that need the dataset or the
trained model are skipped when those are missing."""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix

from src.train import ece

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "models"
HAS_MODEL = (M / "matrix.joblib").exists() and (M / "report.json").exists()
needs_model = pytest.mark.skipif(not HAS_MODEL, reason="needs a trained model: python -m src.train")
CSV_OPTS = dict(keep_default_na=False, na_values=[""])   # same options as app.load_artifacts


@pytest.fixture(scope="module")
def report():
    return json.loads((M / "report.json").read_text())


@pytest.fixture(scope="module")
def preds():
    return pd.read_csv(M / "test_predictions.csv", **CSV_OPTS)


# --------------------------------------------------------------------------- ECE by hand
def test_ece_hand_computed():
    # bins (10 equal-width): [0.0,0.1): p=0.05 (y=0) ; [0.9,1.0]: p=0.95 (y=1), p=0.95 (y=0)
    y = np.array([0, 1, 0])
    p = np.array([0.05, 0.95, 0.95])
    expected = (1 / 3) * abs(0.05 - 0) + (2 / 3) * abs(0.95 - 0.5)
    assert abs(ece(y, p) - expected) < 1e-12
    # empty bins contribute nothing; a probability of exactly 1.0 lands in the last bin
    assert abs(ece(np.array([1]), np.array([1.0]))) < 1e-12


# --------------------------------------------------------------------------- confusion matrix
@needs_model
def test_labels_and_probabilities(preds, report):
    assert set(preds.label.unique()) <= {0, 1}                  # ham = 0, spam = 1 (positive)
    assert preds.prob.between(0, 1).all()
    assert (preds.pred == (preds.prob >= report["threshold"]).astype(int)).all()


@needs_model
def test_confusion_matrix_and_metrics(preds, report):
    y, yhat = preds.label.to_numpy(), preds.pred.to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()  # rows actual, cols predicted
    t = report["test"]
    assert (t["tn"], t["fp"], t["fn"], t["tp"]) == (tn, fp, fn, tp)
    assert tn + fp + fn + tp == len(preds) == report["dataset"]["test"]
    assert abs(t["precision"] - tp / (tp + fp)) < 1e-12
    assert abs(t["recall"] - tp / (tp + fn)) < 1e-12
    assert abs(t["f1"] - 2 * tp / (2 * tp + fp + fn)) < 1e-12
    assert abs(t["false_positive_rate"] - fp / (fp + tn)) < 1e-12
    counts = preds.outcome.value_counts()
    assert counts.get("false positive", 0) == fp and counts.get("false negative", 0) == fn


@needs_model
def test_model_comparison_uses_same_test_set(report):
    mc = pd.read_csv(M / "model_comparison.csv", **CSV_OPTS)
    t = report["test"]
    assert ((mc.test_tp + mc.test_fp + mc.test_tn + mc.test_fn) == report["dataset"]["test"]).all()
    row = mc[mc.model.str.startswith("MATRIX")].iloc[0]
    assert (row.test_tp, row.test_fp, row.test_tn, row.test_fn) == (t["tp"], t["fp"], t["tn"], t["fn"])


# --------------------------------------------------------------------------- calibration
@needs_model
def test_calibration_numbers(preds, report):
    model = joblib.load(M / "matrix.joblib")
    y, p = preds.label.to_numpy(), preds.prob.to_numpy()
    raw = model.raw_proba(preds.text.tolist())                  # same test examples, same order
    c = report["calibration"]
    assert abs(ece(y, p) - c["ece_calibrated"]) < 1e-12
    assert abs(ece(y, raw) - c["ece_raw"]) < 1e-12
    assert abs(brier_score_loss(y, p) - c["brier_calibrated"]) < 1e-12
    assert abs(brier_score_loss(y, raw) - c["brier_raw"]) < 1e-12
    curve = pd.read_csv(M / "calibration.csv", **CSV_OPTS)
    for name, probs in [("raw", raw), ("calibrated", p)]:
        frac, mean_pred = calibration_curve(y, probs, n_bins=8, strategy="quantile")
        st = curve[curve.model == name]
        assert np.allclose(st.mean_predicted, mean_pred) and np.allclose(st.fraction_spam, frac)


# --------------------------------------------------------------------------- other tabs
@needs_model
def test_explanation_quality_and_robustness_tables(report):
    ag = pd.read_csv(M / "agreement.csv", **CSV_OPTS)
    a = report["lime_shap_agreement"]
    assert a["messages"] == len(ag)
    assert abs(a["top5_overlap_mean"] - ag.topk_overlap.mean()) < 1e-12
    fa = pd.read_csv(M / "faithfulness.csv", **CSV_OPTS)
    assert abs(report["faithfulness"]["drop_logodds_shap"] - fa.drop_logodds_shap.mean()) < 1e-12
    cf = pd.read_csv(M / "counterfactuals.csv", **CSV_OPTS)
    fl = cf[cf.flagged]
    t = report["test"]
    assert len(fl) == report["counterfactuals"]["flagged_messages"] == t["tp"] + t["fp"]
    rb = pd.read_csv(M / "robustness.csv", **CSV_OPTS)
    for r in report["robustness"]:
        row = rb[(rb.attack == r["attack"]) & (rb.model == r["model"])
                 & (rb.message_class == r["message_class"])].iloc[0]
        assert row.n_tested == r["n_tested"]


@needs_model
def test_signals_match_model():
    model = joblib.load(M / "matrix.joblib")
    gi = pd.read_csv(M / "global_importance.csv", **CSV_OPTS)
    assert not gi.feature.isna().any()                          # e.g. the word "nan" survives
    w = dict(zip(model.feature_names, model.coef))
    assert all(abs(w[f] - v) < 1e-12 for f, v in zip(gi.feature, gi.weight))
    assert ((gi.weight > 0) == (gi.direction == "suspicious")).all()


# --------------------------------------------------------------------------- model + counterfactual
@needs_model
def test_model_artifact_compatible():
    model = joblib.load(M / "matrix.joblib")
    assert model.sklearn_version == sklearn.__version__
    assert len(model.coef) == len(model.feature_names) == len(model.mu)


@needs_model
def test_counterfactual_rescore_on_real_model():
    from src.explain import counterfactual
    from src.preprocess import units_to_doc
    model = joblib.load(M / "matrix.joblib")
    text = "C0ngratulati0ns!! U have been selected 4 a fr3e iPh0ne. Reply Y to 85023 to cl@im"
    units = model.tokenize(text)
    cf = counterfactual(model, units)
    keep = [True] * len(units)
    for s in cf.steps:
        for i in s.unit_ids:
            keep[i] = False
    assert abs(model.proba_docs([units_to_doc(units, keep)])[0] - cf.probs[-1]) < 1e-12
    assert cf.probs[0] >= model.threshold and cf.flipped == (cf.probs[-1] < model.threshold)


# --------------------------------------------------------------------------- data loading
def test_csv_loading_keeps_words_like_nan(tmp_path):
    f = tmp_path / "x.csv"
    pd.DataFrame({"feature": ["nan", "NA", "null", "free"], "v": [1.0, None, 2.0, 3.0]}).to_csv(f, index=False)
    a = pd.read_csv(f, **CSV_OPTS)
    b = pd.read_csv(f, **CSV_OPTS)
    assert a.feature.tolist() == ["nan", "NA", "null", "free"]
    assert a["v"].isna().tolist() == [False, True, False, False]   # real missing values stay missing
    pd.testing.assert_frame_equal(a, b)                              # stable across repeated loads


# --------------------------------------------------------------------------- app behaviour
def _app(root=ROOT):
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(root / "app.py"), default_timeout=300)


def _page_text(at):
    return " ".join(str(e.value) for kind in ("markdown", "html", "info") for e in at.get(kind))


@needs_model
def test_dashboard_values_stable_across_navigation(report):
    at = _app().run()
    seen = []
    for page in ["Dashboard", "Overview", "Analyzer", "Dashboard"]:
        at.session_state["page"] = page
        at.run()
        assert not at.exception, page
        if page == "Dashboard":
            seen.append(_page_text(at))
    t = report["test"]
    accuracy = (t["tp"] + t["tn"]) / (t["tp"] + t["tn"] + t["fp"] + t["fn"])
    for txt in seen:   # the same verified numbers every time the dashboard renders
        for value in (f"{t['precision']:.3f}", f"{t['recall']:.3f}", f"{t['f1']:.3f}",
                      f"{t['false_positive_rate']:.2%}", f"{accuracy:.2%}"):
            assert value in txt, value
    assert seen[0] == seen[1]


@needs_model
def test_repeated_analyses():
    at = _app().run()
    at.session_state["page"] = "Analyzer"
    at.run()
    results = []
    for msg in ["URGENT! You have WON a £1,000 cash prize. Call 09061701461 now",
                "Hey, are we still on for dinner tonight? I can pick you up at 7."]:
        at.session_state["msg"] = msg
        at.button(key="analyze").click().run()
        assert not at.exception
        r = at.session_state["result"]
        assert r["text"] == msg
        results.append(r["suspicious"])
        assert "What would change the decision" in _page_text(at)
    assert results == [True, False]
    at.session_state["page"] = "Dashboard"
    at.run()
    assert not at.exception


def test_missing_evaluation_artifacts(tmp_path):
    """No model and no results: every page explains what to do instead of crashing."""
    for sub in ["src", "ui", ".streamlit"]:
        shutil.copytree(ROOT / sub, tmp_path / sub)
    shutil.copy(ROOT / "app.py", tmp_path / "app.py")
    (tmp_path / "models").mkdir()
    (tmp_path / "data").mkdir()
    at = _app(tmp_path).run()
    for page in ["Overview", "Analyzer", "Dashboard", "About"]:
        at.session_state["page"] = page
        at.run()
        assert not at.exception, page
    at.session_state["page"] = "Dashboard"
    at.run()
    assert "No trained model" in _page_text(at)
