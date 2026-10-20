import random

from src.robustness import (ATTACKS, capitalization, evaluate_robustness, leet, probe,
                            url_defang, url_swap)
from tests.conftest import make_messages


def test_edit_functions():
    assert leet("free") == "fr3e"
    assert capitalization("Call me") == "CALL ME"
    assert capitalization("CALL ME") == "call me"
    swapped = url_swap("visit www.claimnow.com today", random.Random(0))
    assert "claimnow" not in swapped and "bit.ly" in swapped
    assert url_defang("go to www.x.com") == "go to www[.]x[.]com"


def test_url_swap_keeps_link_placeholder(model):
    a = "visit www.claimnow.com today"
    b = url_swap(a, random.Random(1))
    assert model.normalise(a) == model.normalise(b)


def test_probe_reports_real_scores(model, spam_text):
    rows = probe(model, spam_text)
    assert rows and all(r["attack"] in ATTACKS for r in rows)
    for r in rows:
        assert abs(r["prob_after"] - model.proba([r["text"]])[0]) < 1e-12


def test_evaluate_robustness_runs(model):
    texts, labels = make_messages(60, seed=3)
    rows = evaluate_robustness({"m": model}, texts, labels)
    assert {r["message_class"] for r in rows} == {"spam", "normal"}
    assert {r["attack"] for r in rows} == set(ATTACKS)
