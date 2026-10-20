"""Interface helpers: user text must be shown exactly as typed, and never run as HTML."""
import html
import re

from ui.analysis import analyze
from ui.components import highlight, pct, pct_fine


def test_user_html_is_escaped(model):
    r = analyze(model, "<script>alert(1)</script><img src=x onerror=alert(2)> WIN cash",
                with_probe=False)
    h = highlight(r, "SHAP")
    assert "<script" not in h and "<img" not in h
    assert "&lt;script&gt;" in h


def test_markdown_characters_kept_as_typed(model):
    text = "I owe you $5 and **now** _thanks_\n\nbye"
    r = analyze(model, text, with_probe=False)
    h = highlight(r, "SHAP")
    message = h.split('<div class="mx-legend">')[0]
    visible = html.unescape(re.sub(r"<[^>]+>", "", message))
    assert visible == text          # every character, including $ * _ and blank lines


def test_honest_rounding():
    assert pct(0.9999) == ">99%" and pct(0.0001) == "<1%" and pct(0.5) == "50%"
    assert pct_fine(0.99995) == ">99.99%" and pct_fine(0.9972) == "99.72%"


def test_very_short_message_is_high_uncertainty(model):
    assert analyze(model, "free", with_probe=False)["uncertainty"]["level"] == "high"
