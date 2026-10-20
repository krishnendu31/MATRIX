"""Regression tests: Trust Lab prediction-instability warning and phishing cautions."""
from pathlib import Path

import joblib
import pytest

from src.indicators import phishing_caution, phishing_indicators
from src.trust_lab import compare, edit_summary, instability
from ui.analysis import analyze

ROOT = Path(__file__).resolve().parent.parent
needs_model = pytest.mark.skipif(not (ROOT / "models" / "matrix.joblib").exists(),
                                 reason="needs a trained model")
ORIGINAL = "URGENT! You won a cash prize. Reply now to claim your reward."
REWORDED = "URGENT! You got a cash prize. Reply today to claim your reward."
REWORDED_TYPO = "URGENT! You got a cash prize. Repyl today to clami your reward."


@pytest.fixture(scope="module")
def real_model():
    return joblib.load(ROOT / "models" / "matrix.joblib")


def test_edit_summary_ignores_case_and_counts_words():
    assert edit_summary(ORIGINAL, ORIGINAL.upper())["words_changed"] == 0
    e = edit_summary(ORIGINAL, REWORDED_TYPO)
    assert e["words_changed"] == 4 and e["minor"]
    assert not edit_summary(ORIGINAL, "Hey, are we still on for dinner tonight?")["minor"]


@needs_model
def test_reported_probabilities_are_real(real_model):
    m = real_model
    for text, expected in [(ORIGINAL, 0.990135), (REWORDED, 0.894341), (REWORDED_TYPO, 0.309211)]:
        p = m.proba([text])[0]
        assert abs(p - expected) < 1e-5
        assert abs(analyze(m, text, with_probe=False)["prob"] - p) < 1e-12   # Analyzer = model


@needs_model
def test_instability_flagged_only_when_rule_applies(real_model):
    m = real_model
    r0 = analyze(m, ORIGINAL, with_probe=False)
    small = compare(r0, analyze(m, REWORDED, with_probe=False))          # -9.6 points, still flagged
    assert not instability(small, edit_summary(ORIGINAL, REWORDED))
    big = compare(r0, analyze(m, REWORDED_TYPO, with_probe=False))       # flips to ~31%
    assert big["label_changed"] and instability(big, edit_summary(ORIGINAL, REWORDED_TYPO))


def test_phishing_caution_for_prize_claim_without_link():
    found = phishing_indicators(REWORDED_TYPO)
    assert {"prize", "urgency"} <= {i["key"] for i in found}
    assert phishing_caution(found, suspicious=False)
    assert not phishing_caution(found, suspicious=True)


@needs_model
def test_trust_lab_and_analyzer_show_warnings():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    at.session_state["page"] = "Trust"
    at.run()
    at.session_state["tl_msg"] = ORIGINAL
    at.button(key="tl_analyze").click().run()
    at.session_state["tl_mod"] = REWORDED_TYPO
    at.button(key="tl_compare").click().run()
    assert not at.exception
    page = " ".join(str(e.value) for k in ("markdown", "html") for e in at.get(k))
    assert "Prediction instability detected" in page and "Treat with caution" in page
    assert abs(at.session_state["tl_r1"]["prob"] - 0.309211) < 1e-5
    at.session_state["page"] = "Analyzer"
    at.run()
    at.session_state["msg"] = REWORDED_TYPO
    at.button(key="analyze").click().run()
    assert not at.exception
    assert abs(at.session_state["result"]["prob"] - at.session_state["tl_r1"]["prob"]) < 1e-12
    page = " ".join(str(e.value) for k in ("markdown", "html") for e in at.get(k))
    assert "Treat with caution" in page
