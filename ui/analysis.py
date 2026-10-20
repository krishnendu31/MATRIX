"""Glue between the ML code in src/ and the interface.

`analyze` runs the real model and every explanation once, and returns plain Python data
(no model objects) that the pages render. Nothing here is hard-coded: every number comes
from the trained model.

Privacy: the app keeps the result of a user's message only in that browser session's
memory (st.session_state). It is not written to disk, logged, or shared between sessions.
"""
from __future__ import annotations

import math

from src.explain import explain_message
from src.preprocess import PLACEHOLDERS
from src.robustness import probe

_PUNCT = ".,!?;:'\"()[]"


def display_name(text: str, token: str) -> str:
    """How a word is shown to people: placeholders get a description, words are shown
    as typed (without trailing punctuation)."""
    return PLACEHOLDERS.get(token) or (text.strip(_PUNCT) or text)


def threshold_logit(model) -> float | None:
    """Raw score at which the calibrated probability equals the decision threshold."""
    cal = getattr(model, "_cal", None)
    a, b = getattr(cal, "a_", None), getattr(cal, "b_", None)
    if a is None or b is None or a == 0:
        return None
    t = model.threshold
    # Platt: p = 1 / (1 + exp(a*z + b))  ->  z = (log(1/p - 1) - b) / a
    return (math.log(1 / t - 1) - b) / a


def analyze(model, text: str, lime_samples: int = 500, with_probe: bool = True) -> dict:
    e = explain_message(model, text, lime_samples=lime_samples)
    units = e["units"]
    cf = e["counterfactual"]
    return {
        "text": text,
        "prob": e["prob"],
        "suspicious": bool(e["suspicious"]),
        "threshold": model.threshold,
        "base_prob": e["base_prob"],
        "logit": e["shap"].logit,
        "base_logit": e["shap"].base_value,
        "rest": e["shap"].rest,
        "thr_logit": threshold_logit(model),
        "units": [(u.text, u.token, list(u.spans), u.kind) for u in units],
        "names": [display_name(u.text, u.token) for u in units],
        "shap": e["shap"].values.tolist(),
        "lime": e["lime"].values.tolist(),
        "lime_r2": e["lime"].score,
        "lime_samples": lime_samples,
        "reason": e["reason"],
        "agreement": {k: v for k, v in e["agreement"].items()},
        "uncertainty": e["uncertainty"],
        "cf": {
            "flipped": cf.flipped, "direction": cf.direction, "probs": cf.probs,
            "candidates_tested": cf.candidates_tested,
            "steps": [{"token": s.token,
                       "name": display_name(units[s.unit_ids[0]].text, s.token),
                       "count": len(s.unit_ids), "unit_ids": list(s.unit_ids),
                       "before": s.prob_before, "after": s.prob_after} for s in cf.steps],
        },
        "probe": probe(model, text) if (with_probe and units) else [],
    }
