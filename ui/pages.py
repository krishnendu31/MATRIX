"""The four MATRIX pages. Numbers shown here come from the trained model or from the files
written by `python -m src.train` (models/report.json and the CSVs next to it)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .analysis import analyze
from .components import (card_title, chip, explanation, highlight, html, page_header, pct,
                         section, show_fig, stat_tiles, verdict_state)
from .theme import (ACCENT, ALT_C, BORDER, GREEN, GREEN_BG, HERO_TRACE, LIME_C, MUTED,
                    NEUTRAL_C, PRIMARY, RED, RED_BG, SAFE_BAR, SHAP_C, SURFACE, SUS_BAR, TEXT,
                    esc, icon, style_fig)

# Example inputs written for the demo. The results shown for them are real model outputs.
EXAMPLES = {
    "Prize scam": "URGENT! Your mobile number has WON a £1,000 cash prize. To claim, call 09061701461 now.",
    "Disguised spam": "C0ngratulati0ns!! U have been selected 4 a fr3e iPh0ne. Reply Y to 85023 to cl@im",
    "Delivery link": "Your parcel could not be delivered. Confirm your address within 24 hours at www.parcel-redelivery.com",
    "Normal message": "Hey, are we still on for dinner tonight? I can pick you up at 7.",
    "Tricky normal": "Call me when you're free, I won the bet so dinner is on you tomorrow!",
}


def go_to(page: str):
    st.session_state["page"] = page


# =============================================================================
# A. Overview
# =============================================================================
@st.cache_data(show_spinner=False, max_entries=4)
def _preview_result(_model, model_id: int, text: str) -> dict:
    # model_id is part of the cache key, so a retrained model never reuses old results.
    # A fixed example message from EXAMPLES, never user input. Real model output.
    return analyze(_model, text, lime_samples=200, with_probe=False)


def _hero_preview(model):
    """Right side of the hero: the real analyzer output for one example message."""
    if model is None:
        steps = [("search", "Paste a message"), ("eye", "See the words behind the verdict"),
                 ("undo", "Find what would change it")]
        html('<div class="mx-preview"><div class="bar"><span class="label">How it works</span></div>'
             + "".join(f'<div class="mx-note low">{icon(i, 16)}<span>{esc(t)}</span></div>'
                       for i, t in steps) + "</div>")
        return
    text = EXAMPLES["Prize scam"]
    r = _preview_result(model, id(model), text)
    tone, ico, head, _ = verdict_state(r)
    reason = r["reason"].split(", backed up")[0].rstrip(".") + "."
    html(f'<div class="mx-preview"><div class="bar"><div class="dots"><i></i><i></i><i></i></div>'
         f'<span class="label">Live preview · example message</span></div>'
         f'{highlight(r, "SHAP", legend=False)}'
         f'<div class="foot">{chip(head, tone, ico)}{chip(pct(r["prob"]) + " spam probability", "gray")}</div>'
         f'<p class="why">{esc(reason)}</p></div>')


def overview(art: dict, model=None):
    with st.container(key="hero"):
        left, right = st.columns([3, 2], gap="large", vertical_alignment="center")
        with left:
            html(f"""
<div class="mx-eyebrow">{icon("shield-check", 15)} Explainable message security</div>
<div class="mx-wordmark">MATRIX</div>
<div class="mx-motto">Every threat leaves a trace.</div>
<p class="mx-lead">Paste any SMS or e-mail and see whether it looks suspicious, which words made
the model think so, and what would change its mind. Transparent, explainable AI.</p>""")
            with st.container(horizontal=True, gap="small", key="hero-cta"):
                st.button("Analyze a message", type="primary", icon=":material/search:",
                          on_click=go_to, args=("Analyzer",), key="cta-analyze")
                st.button("Analyst dashboard", icon=":material/monitoring:", on_click=go_to,
                          args=("Dashboard",), key="cta-dashboard")
        with right:
            _hero_preview(model)

    section("layers", "What MATRIX does for every message")
    features = [
        ("shield-check", "Detect", "Know if a message looks suspicious."),
        ("eye", "Understand", "See which words influenced the prediction."),
        ("compass", "Explore", "Discover what could change the result."),
    ]
    html('<div class="mx-grid mx-grid-3">' + "".join(
        f'<div class="mx-tile mx-feature"><div class="mx-tile-ico">{icon(i, 24, stroke=1.75)}</div>'
        f'<h4>{esc(t)}</h4><p>{esc(d)}</p></div>' for i, t, d in features) + "</div>")

    rep = art["report"]
    if rep:
        t, d = rep["test"], rep["dataset"]
        section("target", "Model at a glance",
                f"Measured on {d['test']:,} held-out messages the model never saw.")
        stat_tiles([
            (f"{t['f1']:.3f}", "F1 score", "balance of precision and recall"),
            (f"{t['precision']:.1%}", "Precision", "flagged messages that really were spam"),
            (f"{t['recall']:.1%}", "Recall", "spam that was caught"),
            (f"{t['false_positive_rate']:.2%}", "False-positive rate",
             f"{t['fp']} of {t['fp'] + t['tn']} normal messages flagged"),
        ], cols=4)

    # Closing call to action
    with st.container(key="cta"):
        html('<div class="mx-cta-title">Every message has a story. Understand yours.</div>'
             '<div class="mx-cta-sub">See what the model notices, why it matters, and what could '
             'change its mind.</div>')
        with st.container(horizontal=True, horizontal_alignment="center", key="cta-row"):
            st.button("Analyze a message", type="primary", icon=":material/search:",
                      on_click=go_to, args=("Analyzer",), key="cta-analyze-bottom")


# =============================================================================
# B. Message Analyzer
# =============================================================================
def _pick_example():
    choice = st.session_state.get("example")
    if choice:
        st.session_state["msg"] = EXAMPLES[choice]
    st.session_state["example"] = None


def _run(model):
    text = (st.session_state.get("msg") or "").strip()
    if not text:
        st.session_state["result"] = None
        st.session_state["analyze_error"] = "Paste or type a message first."
        return
    st.session_state["analyze_error"] = None
    st.session_state["result"] = analyze(model, text)


def _clear():
    st.session_state["msg"] = ""
    st.session_state["result"] = None
    st.session_state["analyze_error"] = None


def analyzer(model):
    page_header("search", "Message analyzer", "Is this message suspicious?",
                "Paste an SMS or e-mail. MATRIX tells you how likely it is to be spam, which "
                "words made it think so, and what would change its mind.")
    with st.container(border=True, key="card-input"):
        card_title("message", "Your message", "Paste your own text, or start from an example.")
        st.pills("Examples", list(EXAMPLES), key="example", on_change=_pick_example,
                 help="Example messages written for this demo; the results are real model output.")
        st.text_area("Message", key="msg", height=190, max_chars=5000,
                     placeholder="e.g. You have won a free holiday! Text WIN to 80086 now",
                     label_visibility="collapsed")
        b1, b2, note = st.columns([1.2, 0.8, 3.2], vertical_alignment="center")
        b1.button("Analyze", type="primary", icon=":material/search:", on_click=_run,
                  args=(model,), width="stretch", key="analyze")
        b2.button("Clear", on_click=_clear, width="stretch", key="clear")
        note.markdown(f'<span class="mx-small">{icon("lock", 14)} Processed in memory for this '
                      'session only. Not stored or logged.</span>', unsafe_allow_html=True)

    if st.session_state.get("analyze_error"):
        st.warning(st.session_state["analyze_error"])
    r = st.session_state.get("result")
    if not r:
        return
    current = (st.session_state.get("msg") or "").strip()
    if current and current != r["text"]:
        st.info("The message was edited. The result below is for the previous text; "
                "press Analyze to update it.")
    st.write("")
    explanation(r, key="user")
    html('<p class="mx-small" style="margin-top:1rem">MATRIX is a decision aid. “Suspicious” is a '
         'statistical signal with a calibrated score, never proof that a message is malicious.</p>')


# =============================================================================
# C. Analyst dashboard
# =============================================================================
@st.cache_data(show_spinner="Explaining…", max_entries=64)
def _analyze_dataset_message(_model, model_id: int, text: str) -> dict:
    # model_id (identity of the loaded model) keeps the cache tied to the current model.
    # Only messages from the public test set reach this cache, never user input.
    return analyze(_model, text, with_probe=True)


def dashboard(model, art: dict):
    rep = art["report"]
    page_header("chart", "Analyst dashboard", "How well does MATRIX work?",
                "Every number below is computed by src/train.py on held-out messages the model "
                "never saw during training, tuning or calibration.")
    if not rep:
        st.info("No evaluation results yet. Run `python -m src.train` to create them.")
        return
    t, d = rep["test"], rep["dataset"]
    cv = rep.get("cv", {})
    stat_tiles([
        (f"{(t['tp'] + t['tn']) / (t['tp'] + t['tn'] + t['fp'] + t['fn']):.2%}", "Accuracy",
         f"{t['tp'] + t['tn']:,} of {t['tp'] + t['tn'] + t['fp'] + t['fn']:,} correct"),
        (f"{t['precision']:.3f}", "Precision", "test set"),
        (f"{t['recall']:.3f}", "Recall", "test set"),
        (f"{t['f1']:.3f}", "F1", f"5-fold CV {cv['cv_f1_mean']:.3f} ± {cv['cv_f1_std']:.3f}"
         if cv else "test set"),
        (f"{t['false_positive_rate']:.2%}", "False-positive rate", f"{t['fp']} of {t['fp'] + t['tn']}"),
        (f"{t['roc_auc']:.4f}", "ROC-AUC", "ranking quality"),
    ])
    html(f'<p class="mx-small" style="margin-top:.5rem">UCI SMS Spam Collection · {d["messages"]:,} '
         f'unique messages ({d["spam_share"]:.1%} spam, {d["duplicates_removed"]} duplicates removed '
         f'before splitting) · stratified 80/20 split, seed {rep.get("seed", 42)} · train '
         f'{d["train"]:,} / test {d["test"]:,}.</p>')

    # Lazy tabs: only the open tab's code runs. Without this, every dashboard load also ran
    # the hidden Error explorer's full explanation (LIME + counterfactual + robustness probe),
    # which made the run take ~5 s; meanwhile Streamlit kept showing the previous page's
    # leftover elements, faded them, and finally removed them ("appears, then disappears").
    tabs = st.tabs(["Performance", "Error explorer", "Signals", "Explanation quality",
                    "Robustness", "Model comparison"], key="dash_tab", on_change="rerun")
    sections = [lambda: _performance(rep, art), lambda: _errors(model, art),
                lambda: _signals(art), lambda: _explanations(rep, art),
                lambda: _robustness(art), lambda: _comparison(art)]
    for tab, render in zip(tabs, sections):
        if tab.open:
            with tab:
                render()


def _performance(rep, art):
    t = rep["test"]
    c1, c2 = st.columns(2, gap="large")
    with c1, st.container(border=True, key="card-dash-cm"):
        card_title("target", "Confusion matrix", "Held-out test set")
        z = [[t["tn"], t["fp"]], [t["fn"], t["tp"]]]
        txt = [[f"{t['tn']}<br>correctly passed", f"{t['fp']}<br>wrongly flagged"],
               [f"{t['fn']}<br>missed spam", f"{t['tp']}<br>spam caught"]]
        fig = go.Figure(go.Heatmap(
            z=[[0.15, 1], [1, 0.15]], x=["Predicted legitimate", "Predicted spam"],
            y=["Actually legitimate", "Actually spam"], text=txt, texttemplate="%{text}",
            colorscale=[[0, GREEN_BG], [0.5, SURFACE], [1, RED_BG]], showscale=False,
            hoverinfo="skip", xgap=4, ygap=4, textfont=dict(size=14, color=TEXT)))
        fig.update_yaxes(autorange="reversed")
        show_fig(style_fig(fig, height=430), "cm")
    with c2, st.container(border=True, key="card-dash-cal"):
        card_title("scale", "Is the confidence honest?", "Calibration on the test set")
        cal_df = art.get("calibration")
        cal = rep.get("calibration")
        if cal_df is not None and cal:
            fig = px.line(cal_df, x="mean_predicted", y="fraction_spam", color="model",
                          markers=True, color_discrete_map={"raw": ALT_C, "calibrated": PRIMARY})
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                                     line=dict(dash="dot", color=MUTED)))
            fig.update_xaxes(title_text="predicted probability", tickformat=".0%")
            fig.update_yaxes(title_text="actual share of spam", tickformat=".0%")
            show_fig(style_fig(fig, height=330), "cal")
            html('<div style="display:flex;flex-wrap:wrap;gap:.4rem">'
                 + chip(f"ECE raw {cal['ece_raw']:.4f}", "gray")
                 + chip(f"ECE calibrated {cal['ece_calibrated']:.4f}", "green", "check")
                 + chip(f"Brier {cal['brier_raw']:.4f} → {cal['brier_calibrated']:.4f}", "gray")
                 + "</div>")
            tp_df = art.get("test_predictions")
            mid = int(((tp_df.prob > 0.1) & (tp_df.prob < 0.9)).sum()) if tp_df is not None else None
            n = len(tp_df) if tp_df is not None else None
            html('<p class="mx-caption">Lower ECE means the stated confidence matches reality '
                 'more closely.'
                 + (f' Only <b>{mid}</b> of {n:,} messages score between 10% and 90%, so the '
                    'middle range rests on few examples.' if mid is not None else '') + '</p>'
                 '<details class="mx-exact"><summary>How this is measured</summary>'
                 '<p class="mx-caption">All values use the held-out test set only; the calibrator '
                 'was fitted on training data. ECE: messages are put into 10 equal-width '
                 'probability bins; for each bin the gap between average predicted probability and '
                 'actual share of spam is weighted by the share of messages in it (empty bins are '
                 'skipped). Brier: mean squared error between probability and the 0/1 label. The '
                 'curve uses 8 bins with equal numbers of messages, so its points differ from the '
                 'ECE bins. A small ECE mostly reflects the confident predictions.</p></details>')


def _errors(model, art):
    tp = art.get("test_predictions")
    if tp is None:
        st.info("Run `python -m src.train` to create the test predictions.")
        return
    kind = st.segmented_control(
        "Show", ["false positive", "false negative"], default="false positive", key="err-kind",
        format_func={"false positive": "Normal messages wrongly flagged",
                     "false negative": "Spam that slipped through"}.get)
    kind = kind or "false positive"
    errs = tp[tp.outcome == kind].sort_values("prob", ascending=(kind == "false negative"))
    if errs.empty:
        st.success("None in the test set.")
        return
    st.dataframe(errs[["text", "prob"]], hide_index=True, height=min(240, 40 + 35 * len(errs)),
                 column_config={
        "text": st.column_config.TextColumn("Message", width="large"),
        "prob": st.column_config.ProgressColumn("Spam probability", min_value=0, max_value=1,
                                                format="percent")})
    pick = st.selectbox("Explain one of them", errs.text.tolist(), key=f"pick-{kind}",
                        format_func=lambda s: s[:110] + ("…" if len(s) > 110 else ""))
    if pick:
        explanation(_analyze_dataset_message(model, id(model), pick), key=f"err-{kind}", analyst=True)


def _signals(art):
    gi = art.get("global_importance")
    if gi is None:
        st.info("Run `python -m src.train` first.")
        return
    n_test = len(art["test_predictions"]) if art.get("test_predictions") is not None else None
    html('<p class="mx-muted">Mean |SHAP| per word or word pair: its absolute contribution to the '
         'raw score (log-odds), summed over the test messages that contain it and divided by all '
         + (f'{n_test:,} ' if n_test else '') + 'test messages (messages without it count as 0). '
         'Direction is the sign of the model weight.</p>')
    c1, c2 = st.columns(2, gap="medium")
    for col, direction, colour, title in [(c1, "suspicious", SUS_BAR, "Strongest spam signals"),
                                          (c2, "safe", SAFE_BAR, "Strongest legitimate signals")]:
        top = gi[gi.direction == direction].nlargest(15, "mean_abs_shap")[::-1]
        fig = px.bar(top, x="mean_abs_shap", y="label", orientation="h",
                     color_discrete_sequence=[colour], title=title)
        fig.update_xaxes(title_text="mean |SHAP| (log-odds)")
        fig.update_yaxes(title_text=None)
        with col, st.container(border=True, key=f"card-dash-sig-{direction}"):
            show_fig(style_fig(fig, height=470), f"sig-{direction}")


def _explanations(rep, art):
    a = rep.get("lime_shap_agreement")
    if not a:
        st.info("Run the full `python -m src.train` (not --fast) to measure explanation quality.")
        return
    html('<p class="mx-muted">We test the explanations, not only the classifier.</p>')
    stat_tiles([
        (f"{a['top5_overlap_mean']:.0%}", "LIME–SHAP top-5 overlap", f"{a['messages']} test messages"),
        (f"{a['spearman_mean']:.2f}", "Rank correlation", "Spearman, all words"),
        (f"{a.get('sign_agreement_mean', float('nan')):.0%}", "Same direction",
         "on shared top words"),
        (f"{a.get('lime_stability_mean', float('nan')):.0%}", "LIME repeatability",
         "top-5 overlap across 3 seeds"),
        (f"{a['lime_fidelity_r2_mean']:.2f}", "LIME local fit (R²)", "mean; low near 0%/100%"),
    ])
    c1, c2 = st.columns(2, gap="large")
    ag = art.get("agreement")
    with c1, st.container(border=True, key="card-dash-agree"):
        card_title("scale", "Agreement per message",
                   f"How many of the top-5 words LIME and SHAP share ({a['messages']} messages)")
        if ag is not None:
            lv = (ag.topk_overlap.dropna() * 5).round().div(5).value_counts().sort_index()
            fig = px.bar(x=lv.index, y=lv.values, color_discrete_sequence=["#A3708A"], text_auto=True)
            fig.update_traces(textposition="outside", textfont=dict(color=TEXT), cliponaxis=False)
            fig.update_xaxes(title_text="top-5 words shared by LIME and SHAP", tickformat=".0%",
                             tickvals=[0, .2, .4, .6, .8, 1])
            fig.update_yaxes(title_text="messages")
            show_fig(style_fig(fig, height=300), "agree-hist")
            html(f'<p class="mx-caption"><b>{a["share_low_agreement"]:.0%}</b> of messages fall '
                 'below 60% overlap; the analyzer shows a warning for these.</p>')
    f = rep.get("faithfulness")
    with c2, st.container(border=True, key="card-dash-faith"):
        card_title("target", "Faithfulness: deletion test",
                   f"Remove the top {f['k'] if f else 3} words and re-score")
        if f:
            df = pd.DataFrame({"words removed": ["Top SHAP words", "Top LIME words", "Random words"],
                               "drop": [f["drop_logodds_shap"], f["drop_logodds_lime"],
                                        f["drop_logodds_random"]]})
            fig = px.bar(df, x="words removed", y="drop", color="words removed",
                         color_discrete_sequence=[SHAP_C, LIME_C, ALT_C], text_auto=".2f")
            fig.update_layout(showlegend=False)
            fig.update_traces(textposition="outside", textfont=dict(color=TEXT), cliponaxis=False)
            fig.update_yaxes(title_text="score drop (log-odds)")
            fig.update_xaxes(title_text=None)
            show_fig(style_fig(fig, height=300), "faith")
            html(f'<p class="mx-caption">Across <b>{f["messages"]}</b> messages, removing the words '
                 'an explanation names lowers the score far more than removing random words, so '
                 'the explanations point at words the model really uses.</p>')
    cf = rep.get("counterfactuals")
    if cf:
        section("undo", "Counterfactuals on the test set")
        stat_tiles([
            (f"{cf['flip_found']:.0%}", "Flip found", f"of {cf['flagged_messages']} flagged messages"),
            (f"{cf['median_words_removed']:.0f}", "Median words removed", "when a flip is found"),
            (f"{cf['share_flipped_with_1_word']:.0%}", "Flipped by one word", "of flagged messages"),
        ])
    if ag is not None:
        with st.expander("Messages where LIME and SHAP disagree most"):
            st.dataframe(ag.nsmallest(10, "topk_overlap")[
                ["text", "label", "prob", "topk_overlap", "spearman", "lime_fidelity_r2"]],
                hide_index=True, column_config={
                    "text": st.column_config.TextColumn("Message", width="large"),
                    "prob": st.column_config.NumberColumn("Spam prob.", format="percent"),
                    "topk_overlap": st.column_config.NumberColumn("Top-5 overlap", format="percent")})


def _robustness(art):
    rb = art.get("robustness")
    if rb is None or "message_class" not in rb:
        st.info("Run the full `python -m src.train` to create the robustness results.")
        return
    cls = st.segmented_control("Messages", ["spam", "normal"], default="spam", key="rob-cls",
                               format_func={"spam": "Spam: still caught?",
                                            "normal": "Normal: still passed?"}.get) or "spam"
    sub = rb[(rb.message_class == cls) & (rb.n_tested > 0)].copy()
    sub["model"] = sub.model.str.replace(r" \(.*\)", "", regex=True).replace(
        {"MATRIX": "MATRIX (de-obfuscation on)", "Ablation": "Ablation (off)"})
    # Short axis labels so they fit beside the bars on laptop screens (full names in hover)
    short = {"synonym": "Synonym", "url_defang": "Defanged link", "stretching": "Stretched letters",
             "spacing": "Spaced letters", "url_swap": "Different link", "typo": "Typo"}
    sub["edit"] = [short.get(a, lbl) for a, lbl in zip(sub.attack, sub.attack_label)]
    c1, c2 = st.columns(2, gap="medium")
    for col, ycol, title in [(c1, "label_kept", "Verdict unchanged after the edit"),
                             (c2, "explanation_top3_kept", "Top-3 explanation words kept")]:
        fig = px.bar(sub, x=ycol, y="edit", color="model", barmode="group",
                     orientation="h", color_discrete_sequence=[PRIMARY, ALT_C], title=title,
                     hover_data={"n_tested": True, "attack_label": True, "edit": False},
                     text_auto=".0%")
        fig.update_traces(textposition="outside", textfont=dict(size=11, color=TEXT),
                          cliponaxis=False)
        fig.update_xaxes(tickformat=".0%", range=[0, 1.16], tickvals=[0, 0.5, 1], title_text=None)
        fig.update_yaxes(title_text=None, autorange="reversed")
        with col, st.container(border=True, key=f"card-dash-rob-{ycol}"):
            show_fig(style_fig(fig, height=470, legend_bottom=True), f"rob-{ycol}-{cls}")
    n_url = int(sub[sub.attack == "url_swap"].n_tested.max()) if (sub.attack == "url_swap").any() else 0
    html(f'<p class="mx-small">Each edit is applied to the words that drive each message\'s '
         f'verdict (word edits) or to the whole message. Link edits only apply to messages with '
         f'a link ({n_url} here); synonym edits only apply to spam-style words. Capital letters '
         'cannot change anything because MATRIX lower-cases text first.</p>')
    with st.expander("What the edits look like"):
        ex = sub.drop_duplicates("attack")[["attack_label", "example_before", "example_after", "n_tested"]]
        st.dataframe(ex, hide_index=True, column_config={
            "attack_label": "Edit", "example_before": st.column_config.TextColumn("Before", width="medium"),
            "example_after": st.column_config.TextColumn("After", width="medium"),
            "n_tested": "Messages tested"})


def _comparison(art):
    mc = art.get("model_comparison")
    if mc is None:
        st.info("Run the full `python -m src.train` to compare models.")
        return
    # Dots, not bars: the F1 axis is zoomed in, and bars would exaggerate the gaps.
    fig = go.Figure()
    fig.add_trace(go.Scatter(name="5-fold CV F1 (training data, ± std)", x=mc.model,
                             y=mc.cv_f1_mean, mode="markers",
                             error_y=dict(type="data", array=mc.cv_f1_std, color=MUTED),
                             marker=dict(size=13, color=ACCENT, line=dict(width=1.5, color=PRIMARY))))
    fig.add_trace(go.Scatter(name="Test F1 (held-out)", x=mc.model, y=mc.test_f1,
                             mode="markers", marker=dict(size=13, color=PRIMARY, symbol="diamond")))
    fig.update_yaxes(title_text="F1 (axis zoomed)", tickformat=".2f")
    with st.container(border=True, key="card-dash-cmp"):
        show_fig(style_fig(fig, height=400), "cmp-models")
    show = mc[["model", "cv_f1_mean", "cv_f1_std", "test_precision", "test_recall", "test_f1",
               "test_fpr", "test_roc_auc", "test_fp", "test_fn", "explainable_exactly"]]
    st.dataframe(show, hide_index=True, column_config={
        "model": st.column_config.TextColumn("Model", width="medium"),
        "cv_f1_mean": st.column_config.NumberColumn("CV F1", format="%.3f"),
        "cv_f1_std": st.column_config.NumberColumn("± std", format="%.3f"),
        "test_precision": st.column_config.NumberColumn("Precision", format="%.3f"),
        "test_recall": st.column_config.NumberColumn("Recall", format="%.3f"),
        "test_f1": st.column_config.NumberColumn("Test F1", format="%.3f"),
        "test_fpr": st.column_config.NumberColumn("FPR", format="percent"),
        "test_roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
        "test_fp": "Wrongly flagged", "test_fn": "Missed spam",
        "explainable_exactly": st.column_config.CheckboxColumn("Exact word-level SHAP")})
    html('<p class="mx-small">All models read text the same way and use the same folds. MATRIX keeps '
         'Logistic Regression: it is within noise of the best model, its word explanations are '
         'exact rather than approximated, and calibration makes its probabilities honest.</p>')


# =============================================================================
# D. About & methodology
# =============================================================================
def about(art: dict):
    rep = art["report"]
    d = rep.get("dataset", {}) if rep else {}
    t = rep.get("test", {}) if rep else {}
    page_header("book", "About & methodology", "How MATRIX works",
                "What MATRIX is for, how the model and its explanations are built, and where "
                "its limits are.")

    def block(ico, title, body, key):
        with st.container(border=True, key=f"card-about-{key}"):
            card_title(ico, title)
            st.markdown(body)

    with st.container(border=True, key="card-about-facts"):
        card_title("trace", "Project at a glance")
        facts = [("Project", "MATRIX: Every Threat Leaves a Trace"),
                 ("Team", "CharmedAI"),
                 ("Hackathon", "AI Innovation Hackathon 2026, Women in AI, Research, "
                               "Innovation & Entrepreneurship Club"),
                 ("Track", "Explainable AI (XAI)"),
                 ("Challenge", "05: Explain Why a Message Looks Suspicious")]
        if t:
            facts.append(("Held-out results", f"F1 {t['f1']:.3f} · precision {t['precision']:.1%} · "
                                              f"recall {t['recall']:.1%} · false-positive rate "
                                              f"{t['false_positive_rate']:.2%}"))
        html('<dl class="mx-kv">' + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in facts)
             + "</dl>")

    section("activity", "How it works", "Four steps from a raw message to a tested explanation.")
    steps = [
        ("1", "Prepare", "Links, numbers and money become placeholders; disguises like fr3e "
                         "are undone."),
        ("2", "Predict", "TF-IDF features and Logistic Regression with calibrated confidence."),
        ("3", "Explain", "Exact SHAP word attributions, checked against LIME."),
        ("4", "Test", "Agreement, faithfulness and robustness are measured, not assumed."),
    ]
    html('<div class="mx-grid mx-grid-4">' + "".join(
        f'<div class="mx-tile"><div class="mx-tile-ico">{n}</div>'
        f'<h4>{esc(t)}</h4><p>{esc(d)}</p></div>' for n, t, d in steps) + "</div>")
    section("book", "In detail")

    n = (f"{d['messages']:,} unique messages ({d['spam_share']:.1%} spam) after removing "
         f"{d['duplicates_removed']} exact duplicates") if d else "about 5,000 unique messages"
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        block("alert", "The problem", """
Spam and suspicious-message filters give a verdict, never a reason. Real messages get
blocked and users never learn why; harmful ones slip through and nobody can see what the
model missed; analysts cannot debug a bare label. MATRIX gives every verdict a calibrated
probability, the words behind it, a plain reason and the edit that would change it.
""", "problem")
        block("layers", "Dataset", f"""
**UCI SMS Spam Collection** (Almeida & Hidalgo, 2011, CC BY 4.0): 5,574 English SMS messages
labelled *ham* or *spam*; {n}. Duplicates are removed **before** splitting so the same message
never sits in both training and test data.

UCI Spambase and DistilBERT, mentioned in our proposal, are future work and not part of this
version.
""", "data")
        block("activity", "Model methodology", """
- **Features:** TF-IDF over words and word pairs. Links, phone numbers, short codes and money
  become placeholder tokens; disguises (`fr3e`, `f r e e`, `freeeee`, hidden zero-width
  characters) are undone.
- **Model:** Logistic Regression (C = 10) with **Platt calibration** fitted on 5-fold
  out-of-fold scores; the vectoriser is refit inside each fold (no leakage).
- **Evaluation:** stratified 80/20 split (seed 42), 5-fold cross-validation on the training
  part only; precision, recall, F1, false-positive rate, ROC-AUC, confusion matrix, ECE.
- **Comparisons:** uncalibrated Logistic Regression, Multinomial Naive Bayes and Linear SVM on
  identical features and folds. The test set is never used to train, tune or calibrate.
""", "ml")
        block("lock", "Ethics & privacy", """
- Built only on a public dataset; no private messages or credentials.
- Messages you paste stay in your browser session's memory: not written to disk, logged,
  cached across users, or sent anywhere. Pasted HTML or Markdown is shown, never run.
- Confidence is calibrated and shown with honest rounding, together with an uncertainty level.
- “Suspicious” is a statistical signal, never proof. MATRIX is a decision aid; people decide.
- The interface fonts load from Google Fonts; nothing else leaves your machine.
""", "ethics")
    with c2:
        block("eye", "Explainability", """
- **SHAP (exact):** for a linear model, the interventional SHAP value of a feature is
  w·(x − mean), computed on the raw model score (log-odds) before calibration. A word pair's
  value is shared equally by its two words; words *missing* from the message are shown as one
  extra term. A unit test checks the formula against brute-force Shapley values.
- **LIME (sampled):** hides random words, re-scores, and fits a weighted linear surrogate
  (cosine kernel, width 25, as in the original paper). Its fit (R²) is shown with each result.
- **Agreement:** top-5 overlap, rank correlation and direction agreement between the two.
- **Faithfulness:** removing the top words an explanation names must hurt the score more
  than removing random words.
- **Counterfactuals:** greedy word removal, every candidate re-scored by the model, then
  pruning. A small edit within this search, not guaranteed to be the smallest.
- **Robustness:** leetspeak, spacing, stretching, typos, synonyms, capitals, changed and
  defanged links, added greeting; on spam and normal messages, with and without de-obfuscation.
""", "xai")
        block("info", "Limitations", """
- Trained on English SMS from the UK (mostly 2000s): modern scams, other languages and
  e-mail formatting are less reliable. MATRIX warns when most words are unknown to it.
- Money amounts weigh heavily, so a normal message about money can score high.
- A bag-of-words model sees words, not meaning: an unseen synonym can weaken detection.
- Defanged links such as `www[.]site[.]com` are not recognised as links yet.
- LIME is random, so its ranking varies slightly between runs; SHAP is exact.
""", "limits")
        block("chart", "Technology", """
Python · scikit-learn · NumPy · pandas · SciPy · Streamlit · Plotly · pytest.
SHAP and LIME are implemented in `src/explain.py` so each attribution maps onto the exact
characters typed; `notebooks/verify_xai.py` compares them with the official `shap` and
`lime` packages. Fonts: DM Sans and Plus Jakarta Sans.
""", "tech")
