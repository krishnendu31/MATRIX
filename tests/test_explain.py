import itertools
from math import factorial

import numpy as np
import pytest

from src.explain import STOPWORDS, counterfactual, explain_message, lime_units, shap_units
from src.preprocess import units_to_doc


def test_shap_is_additive(model, spam_text, ham_text):
    for text in [spam_text, ham_text, "x"]:
        s = shap_units(model, model.tokenize(text))
        assert abs(s.base_value + s.values.sum() + s.rest - s.logit) < 1e-9
        # and it explains the real raw score of the model
        z = model.logit_docs([model.normalise(text)])[0]
        assert abs(s.logit - z) < 1e-9


def test_shap_matches_brute_force_shapley(model, ham_text):
    """Enumerate every coalition of the message's features (interventional value function,
    background = average training document) and compare with w_j * (x_j - mu_j)."""
    units = model.tokenize(ham_text)
    x = model.vectorize_docs([units_to_doc(units)]).toarray().ravel()
    present = list(np.flatnonzero(x))[:8]            # keep the enumeration small
    w, mu, b = model.coef, model.mu, model.intercept

    def v(S):
        z = x.copy()
        out = [j for j in present if j not in S]
        z[out] = mu[out]
        return b + w @ z

    n = len(present)
    assert n >= 3
    for j in present:
        rest = [k for k in present if k != j]
        phi = 0.0
        for r in range(n):
            for S in itertools.combinations(rest, r):
                weight = factorial(r) * factorial(n - r - 1) / factorial(n)
                phi += weight * (v(set(S) | {j}) - v(set(S)))
        assert abs(phi - w[j] * (x[j] - mu[j])) < 1e-9


def test_lime_is_reproducible(model, spam_text):
    units = model.tokenize(spam_text)
    a = lime_units(model, units, n_samples=300, seed=7)
    b = lime_units(model, units, n_samples=300, seed=7)
    assert np.allclose(a.values, b.values)
    assert len(a.values) == len(units)


def test_counterfactual_is_real_and_pruned(model, spam_text):
    units = model.tokenize(spam_text)
    cf = counterfactual(model, units)
    assert cf.direction == "to_safe"
    # every reported probability is a real model output for that edited message
    keep = [True] * len(units)
    assert abs(cf.probs[0] - model.proba_docs([units_to_doc(units)])[0]) < 1e-12
    for step, p in zip(cf.steps, cf.probs[1:]):
        for i in step.unit_ids:
            keep[i] = False
        assert abs(p - model.proba_docs([units_to_doc(units, keep)])[0]) < 1e-12
        assert step.token not in STOPWORDS
    # The toy model may not flip (its filler words are spam-only); the flip rate on real
    # data is measured by src/train.py. Score must at least move the right way.
    assert cf.probs[-1] <= cf.probs[0]
    if not cf.flipped:
        return
    assert cf.probs[-1] < model.threshold
    # pruned: putting back any one removed word makes it suspicious again
    removed = {s.token for s in cf.steps}
    if len(removed) > 1:
        for t in removed:
            k = [u.token not in removed - {t} for u in units]
            assert model.proba_docs([units_to_doc(units, k)])[0] >= model.threshold


def test_counterfactual_for_normal_message(model, ham_text):
    cf = counterfactual(model, model.tokenize(ham_text))
    assert cf.direction == "to_suspicious"
    assert cf.probs[0] < model.threshold


def test_explain_message_outputs(model, spam_text):
    e = explain_message(model, spam_text, lime_samples=200)
    assert e["suspicious"] is True
    assert 0 <= e["base_prob"] <= 1
    assert set(e["agreement"]) >= {"topk_overlap", "spearman", "sign_agreement"}
    assert e["uncertainty"]["level"] in {"low", "medium", "high"}


@pytest.mark.parametrize("text, expect", [
    ("", "no words"),
    ("Ich habe heute keine Zeit mehr, tschüss", "were seen in training"),
])
def test_uncertainty_notes(model, text, expect):
    u = explain_message(model, text, lime_samples=100)["uncertainty"]
    assert u["level"] == "high"
    assert any(expect in n["text"] for n in u["notes"])
