"""Regression tests for the "Delivery link" review: the model output is real and consistent,
and the rule-based phishing check is separate from (and never changes) the ML prediction."""
from pathlib import Path

import joblib
import pytest

from src.indicators import phishing_caution, phishing_indicators
from src.preprocess import normalise
from src.trust_lab import compare
from ui.analysis import analyze

ROOT = Path(__file__).resolve().parent.parent
needs_model = pytest.mark.skipif(not (ROOT / "models" / "matrix.joblib").exists(),
                                 reason="needs a trained model")
DELIVERY = ("Your parcel could not be delivered. Confirm your address within 24 hours at "
            "www.parcel-redelivery.com")


@pytest.fixture(scope="module")
def real_model():
    return joblib.load(ROOT / "models" / "matrix.joblib")


@needs_model
def test_delivery_link_score_is_the_real_model_output(real_model):
    m = real_model
    doc = m.normalise(DELIVERY)
    assert doc == normalise(DELIVERY, True, m.word_vocab)          # same as the training path
    assert "__url__" in doc.split() and "parcel" not in m.vectorizer.vocabulary_
    p = m.proba([DELIVERY])[0]
    assert abs(p - 0.445167) < 1e-6 and p < m.threshold == 0.5
    r = analyze(m, DELIVERY, with_probe=False)                     # Message Analyzer path
    c = compare(r, analyze(m, DELIVERY, with_probe=False))          # Trust Lab path
    assert abs(r["prob"] - p) < 1e-12 and abs(c["prob_before"] - p) < 1e-12
    assert not r["suspicious"]


def test_indicators_flag_delivery_phishing_patterns():
    keys = {i["key"] for i in phishing_indicators(DELIVERY)}
    assert {"link", "urgency", "verify", "personal", "delivery"} <= keys
    assert phishing_caution(phishing_indicators(DELIVERY), suspicious=False)


@pytest.mark.parametrize("text", [
    "Hey, are we still on for dinner tonight? I can pick you up at 7.",
    "Your Amazon order has been delivered. Thanks for shopping with us.",
    "Mum, can you call me when you get home? Need to ask about Sunday",
])
def test_no_caution_for_normal_messages(text):
    assert not phishing_caution(phishing_indicators(text), suspicious=False)


def test_caution_never_shown_for_flagged_messages():
    assert not phishing_caution(phishing_indicators(DELIVERY), suspicious=True)


def test_defanged_links_detected():
    assert any(i["key"] == "link" for i in phishing_indicators("go to site[.]com now"))


@needs_model
def test_analyzer_shows_caution_without_changing_verdict():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    at.session_state["page"] = "Analyzer"
    at.run()
    at.session_state["msg"] = DELIVERY
    at.button(key="analyze").click().run()
    assert not at.exception
    page = " ".join(str(e.value) for k in ("markdown", "html") for e in at.get(k))
    assert "Phishing patterns" in page and "Treat with caution" in page
    assert "Leans legitimate" in page                              # ML verdict unchanged
    assert abs(at.session_state["result"]["prob"] - 0.445167) < 1e-6
