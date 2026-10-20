"""Runs the real Streamlit app against the real trained model (skipped if not trained)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
needs_model = pytest.mark.skipif(not (ROOT / "models" / "matrix.joblib").exists(),
                                 reason="no trained model: run python -m src.train")


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)


@needs_model
def test_every_page_renders():
    at = _app().run()
    assert not at.exception
    for page in ["Analyzer", "Dashboard", "About", "Overview"]:
        at.session_state["page"] = page
        at.run()
        assert not at.exception, page


@needs_model
def test_analyzer_explains_a_message():
    at = _app().run()
    at.session_state["page"] = "Analyzer"
    at.run()
    at.session_state["msg"] = "URGENT! You have WON a £1,000 cash prize. Call 09061701461 now"
    at.button(key="analyze").click().run()
    assert not at.exception
    r = at.session_state["result"]
    assert r["suspicious"] and 0.5 <= r["prob"] <= 1
    page = " ".join(str(e.value) for kind in ("markdown", "html") for e in at.get(kind))
    assert "mx-verdict" in page and "What would change the decision" in page


@needs_model
def test_empty_message_is_not_analysed():
    at = _app().run()
    at.session_state["page"] = "Analyzer"
    at.run()
    at.button(key="analyze").click().run()
    assert not at.exception
    assert at.session_state["result"] is None
    assert any("Paste or type" in w.value for w in at.warning)
