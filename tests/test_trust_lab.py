"""Trust Lab: transformations, comparison maths and the page flow."""
from pathlib import Path

import pytest

from src.trust_lab import TRANSFORMS, compare, transform
from ui.analysis import analyze

ROOT = Path(__file__).resolve().parent.parent
needs_model = pytest.mark.skipif(not (ROOT / "models" / "matrix.joblib").exists(),
                                 reason="needs a trained model")


def test_transformations_are_real_edits(model, spam_text):
    assert transform(model, spam_text, "capitalization") == spam_text.upper()
    assert transform(model, spam_text, "greeting").endswith(spam_text)
    for kind in TRANSFORMS:
        out = transform(model, spam_text, kind)
        assert isinstance(out, str) and out          # applies, or returns the text unchanged
    with pytest.raises(ValueError):
        transform(model, spam_text, "not-a-transform")


def test_compare_uses_the_same_model(model, spam_text):
    edited = transform(model, spam_text, "leetspeak")
    r0, r1 = analyze(model, spam_text, with_probe=False), analyze(model, edited, with_probe=False)
    c = compare(r0, r1)
    p0, p1 = model.proba([spam_text])[0], model.proba([edited])[0]
    assert abs(c["prob_before"] - p0) < 1e-12 and abs(c["prob_after"] - p1) < 1e-12
    assert abs(c["delta"] - (p1 - p0)) < 1e-12
    assert c["label_changed"] == ((p0 >= model.threshold) != (p1 >= model.threshold))


def test_insights_follow_the_numbers(model, spam_text, ham_text):
    r_spam, r_ham = analyze(model, spam_text, with_probe=False), analyze(model, ham_text, with_probe=False)
    same = compare(r_spam, r_spam)
    assert not same["label_changed"] and same["delta"] == 0 and same["explanation_overlap"] == 1
    assert same["insights"][0][1].startswith("Prediction remained stable")
    flip = compare(r_spam, r_ham)
    assert flip["label_changed"] and flip["insights"][0][1].startswith("Prediction changed")
    if flip["explanation_overlap"] is not None and flip["explanation_overlap"] < 0.6:
        assert any("Explanation changed significantly" in s for _, s in flip["insights"])


@needs_model
def test_trust_lab_page_flow_and_navigation():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    for page in ["Overview", "Analyzer", "Trust", "Dashboard", "About"]:
        at.session_state["page"] = page
        at.run()
        assert not at.exception, page
    at.session_state["page"] = "Trust"
    at.run()
    at.button(key="tl_analyze").click().run()                       # empty input
    assert at.session_state["tl_r0"] is None and not at.exception
    at.session_state["tl_msg"] = "URGENT! You have WON a £1,000 cash prize. Call 09061701461 now"
    at.button(key="tl_analyze").click().run()
    assert at.session_state["tl_r0"]["suspicious"]
    at.button(key="tl_t_capitalization").click().run()
    assert at.session_state["tl_mod"] == at.session_state["tl_msg"].upper()
    at.button(key="tl_compare").click().run()
    assert not at.exception
    c = compare(at.session_state["tl_r0"], at.session_state["tl_r1"])
    assert abs(c["delta"]) < 1e-12 and not c["label_changed"]       # model lower-cases text
