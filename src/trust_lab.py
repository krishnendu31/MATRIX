"""MATRIX Trust Lab: challenge a prediction by editing the message and comparing.

Everything shown is computed by the existing model and explanation code:
  * transformations reuse src.robustness.attack (same edits as the robustness tests);
  * both messages are analysed by ui.analysis.analyze (real probability, SHAP, LIME);
  * conclusions are derived only from those numbers (see `compare`).
Pasted messages live only in this browser session's memory (st.session_state).
"""
from __future__ import annotations

import difflib

import streamlit as st

from src.explain import BORDERLINE
from src.robustness import ATTACK_LABELS, attack

# Edits that can be applied and checked reliably on any message. Word edits disguise the
# two words that drive this message's verdict; message edits change the whole text.
TRANSFORMS = ["typo", "leetspeak", "spacing", "capitalization", "synonym", "greeting"]
TOP_K = 5
EXPLANATION_SHIFT = 0.6   # top-5 SHAP word overlap below this = "explanation changed significantly"

# Instability heuristic (documented, not a trust score). Words are compared ignoring case and
# punctuation. An edit is MINOR when at most MINOR_EDIT_WORDS words were changed, added or
# removed, or at most 30% of the words for longer messages. A minor edit is flagged as a
# prediction instability when it flips the predicted label OR moves the spam probability by at
# least 20 percentage points.
MINOR_EDIT_WORDS = 4
MINOR_EDIT_SHARE = 0.30
SHIFT_PP = 20.0


def transform(model, text: str, kind: str, seed: int = 0) -> str:
    """Apply one existing robustness edit. Returns the text unchanged if it does not apply."""
    if kind not in TRANSFORMS:
        raise ValueError(f"unknown transformation: {kind}")
    return attack(text, model, kind, n_words=2, seed=seed)


def edit_summary(original: str, modified: str) -> dict:
    """Word-level difference between two messages (exactly what changed)."""
    a, b = original.split(), modified.split()
    na = [w.strip(".,!?;:'\"()").lower() for w in a]
    nb = [w.strip(".,!?;:'\"()").lower() for w in b]
    changes, changed = [], 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=na, b=nb, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
        changes.append((" ".join(a[i1:i2]) or "∅", " ".join(b[j1:j2]) or "∅"))
    share = changed / max(len(a), 1)
    return {"changes": changes, "words_changed": changed, "words_total": len(a),
            "share": share,
            "minor": 0 < changed and (changed <= MINOR_EDIT_WORDS or share <= MINOR_EDIT_SHARE)}


def instability(c: dict, edit: dict) -> bool:
    """Minor edit + (label flip or a shift of at least SHIFT_PP percentage points)."""
    return edit["minor"] and (c["label_changed"] or abs(c["delta"]) * 100 >= SHIFT_PP)


def _top_words(r: dict) -> list[str]:
    return list(r["agreement"].get("shap_top") or [])[:TOP_K]


def _uncertain(p: float) -> bool:
    return BORDERLINE[0] <= p <= BORDERLINE[1]


def compare(r0: dict, r1: dict) -> dict:
    """Compare two `analyze` results (original r0, modified r1). Pure function: every value
    comes from the two results, nothing is estimated or invented."""
    w0, w1 = _top_words(r0), _top_words(r1)
    overlap = (len(set(w0) & set(w1)) / max(len(w0), len(w1))) if (w0 and w1) else None
    out = {
        "prob_before": r0["prob"], "prob_after": r1["prob"],
        "delta": r1["prob"] - r0["prob"],
        "label_before": r0["suspicious"], "label_after": r1["suspicious"],
        "label_changed": r0["suspicious"] != r1["suspicious"],
        "shap_top_before": w0, "shap_top_after": w1,
        "words_lost": [w for w in w0 if w not in w1], "words_new": [w for w in w1 if w not in w0],
        "explanation_overlap": overlap,
        "agreement_before": r0["agreement"].get("topk_overlap"),
        "agreement_after": r1["agreement"].get("topk_overlap"),
        "uncertain_before": _uncertain(r0["prob"]), "uncertain_after": _uncertain(r1["prob"]),
        "uncertainty_level_before": r0["uncertainty"]["level"],
        "uncertainty_level_after": r1["uncertainty"]["level"],
    }
    out["insights"] = insights(out)
    return out


def insights(c: dict) -> list[tuple[str, str]]:
    """(tone, sentence) conclusions, each backed by a value in `c`."""
    pts = c["delta"] * 100
    res = []
    if c["label_changed"]:
        to = "suspicious" if c["label_after"] else "not flagged"
        res.append(("red" if c["label_after"] else "green",
                    f"Prediction changed: the edited message is now {to} "
                    f"({c['prob_before']:.1%} → {c['prob_after']:.1%})."))
    else:
        res.append(("green", f"Prediction remained stable ({pts:+.1f} percentage points)."))
    if c["explanation_overlap"] is not None and c["explanation_overlap"] < EXPLANATION_SHIFT:
        res.append(("amber", f"Explanation changed significantly: only "
                             f"{c['explanation_overlap']:.0%} of the top SHAP words stayed the same."))
    elif c["explanation_overlap"] is not None:
        res.append(("gray", f"Explanation stayed consistent: {c['explanation_overlap']:.0%} of the "
                            "top SHAP words stayed the same."))
    if c["uncertain_after"] and not c["uncertain_before"]:
        res.append(("amber", "Model became more uncertain: the edited message lands in the "
                             "borderline range (20–80%)."))
    elif c["uncertain_before"] and not c["uncertain_after"]:
        res.append(("gray", "Model became more certain: the edited message left the "
                            "borderline range (20–80%)."))
    return res


# =============================================================================
# Page
# =============================================================================
def _ui():
    # imported lazily so the calculations above can be tested without the interface
    from ui import components as C
    from ui.analysis import analyze
    from ui.pages import EXAMPLES
    from ui.theme import esc, icon
    return C, analyze, EXAMPLES, esc, icon


def _pick_example():
    from ui.pages import EXAMPLES
    choice = st.session_state.get("tl_example")
    if choice:
        st.session_state["tl_msg"] = EXAMPLES[choice]
    st.session_state["tl_example"] = None


def _analyze_original(model):
    from ui.analysis import analyze
    text = (st.session_state.get("tl_msg") or "").strip()
    st.session_state["tl_r1"] = None
    if not text:
        st.session_state["tl_r0"] = None
        st.session_state["tl_error"] = "Paste or type a message first."
        return
    st.session_state["tl_error"] = None
    st.session_state["tl_r0"] = analyze(model, text, with_probe=False)
    st.session_state["tl_mod"] = text            # editable copy to challenge


def _apply(model, kind: str):
    current = st.session_state.get("tl_mod") or ""
    new = transform(model, current, kind)
    st.session_state["tl_note"] = (None if new != current else
                                   f"“{ATTACK_LABELS[kind]}” does not apply to this message.")
    st.session_state["tl_mod"] = new
    st.session_state["tl_r1"] = None


def _compare(model):
    from ui.analysis import analyze
    text = (st.session_state.get("tl_mod") or "").strip()
    if not text:
        st.session_state["tl_note"] = "The modified message is empty."
        return
    st.session_state["tl_r1"] = analyze(model, text, with_probe=False)


def page(model):
    C, analyze, EXAMPLES, esc, icon = _ui()
    C.page_header("scale", "MATRIX Trust Lab", "Don't just trust the prediction. Challenge it.",
                  "Analyse a message, change it the way a spammer might, and see whether the "
                  "verdict and its explanation hold up. Every number comes from the real model.")

    # ---- A. message input
    with st.container(border=True, key="card-tl-input"):
        C.card_title("message", "1 · Your message")
        st.pills("Examples", list(EXAMPLES), key="tl_example", on_change=_pick_example)
        st.text_area("Message", key="tl_msg", height=120, max_chars=5000,
                     label_visibility="collapsed", placeholder="Paste an SMS or e-mail")
        st.button("Analyze", type="primary", icon=":material/search:", key="tl_analyze",
                  on_click=_analyze_original, args=(model,))
    if st.session_state.get("tl_error"):
        st.warning(st.session_state["tl_error"])
    r0 = st.session_state.get("tl_r0")
    if not r0:
        return

    # ---- B. challenge the AI
    with st.container(border=True, key="card-tl-challenge"):
        C.card_title("undo", "2 · Challenge the AI",
                     "Edit the text yourself or apply a quick change. Word changes target the "
                     "two words that drive this verdict.")
        with st.container(horizontal=True, gap="small", key="tl-transforms"):
            for kind in TRANSFORMS:
                st.button(ATTACK_LABELS[kind], key=f"tl_t_{kind}", on_click=_apply,
                          args=(model, kind))
        st.text_area("Modified message (exactly what will be analysed)", key="tl_mod", height=110,
                     max_chars=5000)
        if st.session_state.get("tl_note"):
            st.info(st.session_state["tl_note"])
        st.button("Compare", type="primary", icon=":material/compare_arrows:", key="tl_compare",
                  on_click=_compare, args=(model,))

    r1 = st.session_state.get("tl_r1")
    if not r1:
        return
    if (st.session_state.get("tl_mod") or "").strip() != r1["text"]:
        st.info("The modified message was edited after the last comparison; press Compare again.")
    c = compare(r0, r1)
    edit = edit_summary(r0["text"], r1["text"])
    unstable = instability(c, edit)

    # ---- what exactly changed (quick changes stack on top of manual edits)
    changes = "; ".join(f"{esc(x)} → {esc(y)}" for x, y in edit["changes"][:12]) or "no words changed"
    C.html(f'<p class="mx-caption" style="max-width:none;margin-top:1.2rem"><b>What changed '
           f'({edit["words_changed"]} of {edit["words_total"]} words):</b> {changes}</p>')
    if unstable:
        why = ("and changed the predicted label" if c["label_changed"] else
               f"by {abs(c['delta']) * 100:.1f} percentage points")
        C.html(f'<div class="mx-note high" style="margin-top:.8rem">{icon("alert", 18)}<span>'
               f'<b>Prediction instability detected.</b> A small edit ({edit["words_changed"]} of '
               f'{edit["words_total"]} words) moved the spam probability from '
               f'{esc(C.pct_fine(c["prob_before"]))} to {esc(C.pct_fine(c["prob_after"]))} {why}. '
               'A lower spam probability does not prove a message is safe: this model scores '
               'individual words, and small rewording or typos in its strongest words can hide a '
               'scam from it. Rule used: at most 4 words (or 30% of the words) changed, ignoring case, '
               'and the label flips or the '
               'probability moves by 20 points or more.</span></div>')

    # ---- C. side-by-side
    C.section("eye", "Side by side", "Same model, same settings, two messages.")
    cols = st.columns(2, gap="large")
    for col, r, title, key in [(cols[0], r0, "Original", "o"), (cols[1], r1, "Modified", "m")]:
        tone, ico, head, _ = C.verdict_state(r)
        with col, st.container(border=True, key=f"card-tl-{key}"):
            C.html(f'<div class="mx-card-title">{esc(title)}</div>'
                   f'<div style="display:flex;flex-wrap:wrap;gap:.5rem;margin:.2rem 0 .8rem">'
                   f'{C.chip(head, tone, ico)}{C.chip(C.pct_fine(r["prob"]) + " spam probability", "gray")}</div>'
                   f'{C.highlight(r, "SHAP")}')
            C.html('<div class="mx-card-hint" style="margin-top:.9rem">Potential phishing '
                   'indicators (rule-based, not the ML model)</div>' + C.phishing_block(r))
    tone = "red" if c["delta"] > 0 else "green"
    C.html('<div style="display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1rem">'
           + C.chip(f"Probability change: {c['delta'] * 100:+.2f} percentage points", tone)
           + C.chip("Prediction changed" if c["label_changed"] else "Prediction unchanged",
                    "amber" if c["label_changed"] else "green") + "</div>")

    # ---- D. explanation reliability
    C.section("scale", "Explanation reliability",
              "Do SHAP and LIME agree, and did the edit change which words matter?")
    with st.container(border=True, key="card-tl-reliability"):
        def fmt(v):
            return "not measurable" if v is None or v != v else f"{v:.0%}"
        rows = [("SHAP–LIME top-5 agreement", fmt(c["agreement_before"]), fmt(c["agreement_after"])),
                ("Top SHAP words", ", ".join(c["shap_top_before"]) or "–",
                 ", ".join(c["shap_top_after"]) or "–")]
        C.html('<div class="mx-table-wrap"><table class="mx-steps"><thead><tr><th></th><th>Original</th>'
               '<th>Modified</th></tr></thead><tbody>' + "".join(
                   f"<tr><td><b>{esc(a)}</b></td><td>{esc(b)}</td><td>{esc(d)}</td></tr>"
                   for a, b, d in rows) + "</tbody></table></div>")
        changes = []
        if c["words_lost"]:
            changes.append("no longer in the top words: " + ", ".join(c["words_lost"]))
        if c["words_new"]:
            changes.append("new in the top words: " + ", ".join(c["words_new"]))
        C.html(f'<p style="margin:.9rem 0 .3rem">'
               f'{esc("; ".join(changes).capitalize() + "." if changes else "The top SHAP words did not change.")}</p>'
               '<p class="mx-small" style="margin:0">Words are shown as the model reads them '
               '(e.g. “fr3e” as free, numbers as placeholders). SHAP is exact for this model; LIME '
               'is a sampled approximation, so when the score is near 0% or 100% it can rank minor '
               'words differently. Low agreement means “read the highlighting with care”, not that '
               'either method is wrong.</p>')

    # ---- E. final insight
    C.section("target", "What this experiment shows")
    with st.container(border=True, key="card-tl-insight"):
        C.html("".join(f'<div class="mx-note {"high" if t in ("red", "amber") else "low"}">'
                       f'{icon("info", 16)}<span>{esc(s)}</span></div>' for t, s in c["insights"]))
