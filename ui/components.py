"""Reusable interface pieces. Every function renders values it is given; none invents data."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.explain import BORDERLINE, STOPWORDS, strength_levels
from src.indicators import phishing_caution, phishing_indicators

from .theme import (AMBER, ACCENT, GREEN, LIME_C, MUTED, PLOT_CONFIG, PRIMARY, RED, SAFE_BAR,
                    SHAP_C, SUS_BAR, TEXT, esc, icon, style_fig)


# ----------------------------------------------------------------------------- basics
def html(s: str):
    """Render MATRIX's own HTML. Any user text inside has gone through theme.esc, which
    neutralises HTML, Markdown and LaTeX, so "I owe you $5 and **now**" shows as typed."""
    st.markdown(s, unsafe_allow_html=True)


def pct(p: float) -> str:
    """Honest rounding: never show a calibrated score as a flat 0% or 100%."""
    return ">99%" if p > 0.99 else ("<1%" if p < 0.01 else f"{p:.0%}")


def pct_fine(p: float) -> str:
    """Like pct, but keeps detail near 0% and 100% so small changes stay visible."""
    if p >= 0.9999:
        return ">99.99%"
    if p <= 0.0001:
        return "<0.01%"
    if p > 0.99 or p < 0.01:
        return f"{p:.2%}"
    return f"{p:.0%}"


def page_header(ico: str, eyebrow: str, title: str, subtitle: str = ""):
    """The heading at the top of every page."""
    sub = f'<p class="mx-page-sub">{esc(subtitle)}</p>' if subtitle else ""
    html(f'<div class="mx-page"><div class="mx-eyebrow">{icon(ico, 15)}{esc(eyebrow)}</div>'
         f'<div class="mx-page-title">{esc(title)}</div>{sub}</div>')


def section(ico: str, title: str, subtitle: str = ""):
    sub = f'<p class="mx-muted">{esc(subtitle)}</p>' if subtitle else ""
    html(f'<div class="mx-section"><div class="ico">{icon(ico, 18)}</div>'
         f'<div><div class="h">{esc(title)}</div>{sub}</div></div>')


def card_title(ico: str, title: str, hint: str = ""):
    h = f'<div class="mx-card-hint">{esc(hint)}</div>' if hint else ""
    html(f'<div class="mx-card-title">{icon(ico, 18)}{esc(title)}</div>{h}')


def stat_tiles(items, cols: int | None = None):
    """items: [(value, label, sublabel)]. cols=4 gives a 4 / 2 / 1 column responsive grid;
    otherwise the tiles flow (at least 180px wide)."""
    cells = "".join(f'<div class="mx-stat"><div class="v">{esc(v)}</div>'
                    f'<div class="l">{esc(l)}</div>'
                    + (f'<div class="s">{esc(s)}</div>' if s else "") + "</div>"
                    for v, l, s in items)
    if cols == 4:
        html(f'<div class="mx-grid mx-grid-4">{cells}</div>')
    else:
        html(f'<div class="mx-grid" style="grid-template-columns:repeat(auto-fit,minmax(180px,1fr))">'
             f'{cells}</div>')


def chip(text: str, tone: str = "gray", ico: str | None = None) -> str:
    i = icon(ico, 14) if ico else ""
    return f'<span class="mx-chip {tone}">{i}{esc(text)}</span>'


# ----------------------------------------------------------------------------- verdict
def verdict_state(r: dict) -> tuple[str, str, str, str]:
    """(tone, icon, headline, sub-line) from the real model output."""
    p, sus = r["prob"], r["suspicious"]
    borderline = BORDERLINE[0] <= p <= BORDERLINE[1]
    if sus and not borderline:
        return "red", "alert", "Suspicious", "The model would flag this message."
    if sus:
        return "amber", "alert", "Leans suspicious", "Flagged, but the score is borderline."
    if borderline:
        return "amber", "info", "Leans legitimate", "Not flagged, but the score is borderline."
    return "green", "shield-check", "Looks legitimate", "The model would not flag this message."


def verdict_card(r: dict):
    tone, ico, head, sub = verdict_state(r)
    p, thr = r["prob"], r["threshold"]
    u = r["uncertainty"]
    u_tone = {"low": "green", "medium": "amber", "high": "amber"}[u["level"]]
    html(f"""
<div class="mx-verdict {tone}">
  <div class="head">{icon(ico, 24)}{esc(head)}</div>
  <div class="sub">{esc(sub)}</div>
  <div class="mx-prob">{esc(pct(p))}</div>
  <div class="mx-small">calibrated probability that this is spam</div>
  <div class="mx-meter" role="img" aria-label="score {esc(pct(p))}, threshold {thr:.0%}">
    <div class="thr" style="left:{thr * 100:.1f}%"></div>
    <div class="dot" style="left:{min(max(p, 0.01), 0.99) * 100:.1f}%"></div>
  </div>
  <div class="mx-scale"><span>legitimate</span><span>threshold {thr:.0%}</span><span>spam</span></div>
  <div style="margin-top:.75rem">{chip(f"Uncertainty: {u['level']}", u_tone, "info")}</div>
</div>""")
    for n in u["notes"]:
        html(f'<div class="mx-note {n["level"]}">{icon("info", 16)}<span>{esc(n["text"])}</span></div>')


# ----------------------------------------------------------------------------- highlighting
def highlight(r: dict, method: str, legend: bool = True) -> str:
    """Colour the original characters of each unit by its attribution."""
    text = r["text"]
    shap_view = method == "SHAP"
    values = np.asarray(r["shap"] if shap_view else r["lime"])
    # SHAP is in log-odds, LIME in probability points: different noise floors
    levels = strength_levels(values, 0.05 if shap_view else 0.002)
    style = [None] * len(text)
    for (_, _, spans, _), v, lvl in zip(r["units"], values, levels):
        if lvl == "none":
            continue
        cls = ("mx-strong" if lvl == "strong" else "mx-weak") if v > 0 else "mx-safe"
        if v < 0 and r["suspicious"] and lvl != "strong":
            continue  # keep the picture focused on why it was flagged
        for a, b in spans:
            for c in range(a, b):
                style[c] = cls
    out, i = [], 0
    while i < len(text):
        j = i
        while j < len(text) and style[j] == style[i]:
            j += 1
        chunk = esc(text[i:j])
        out.append(f'<span class="{style[i]}">{chunk}</span>' if style[i] else chunk)
        i = j
    legend_html = ('<div class="mx-legend">'
              '<span><span class="mx-strong">word</span> strong spam signal</span>'
              '<span><span class="mx-weak">word</span> weak spam signal</span>'
              '<span><span class="mx-safe">word</span> sign of a normal message</span></div>')
    return f'<div class="mx-msg">{"".join(out)}</div>' + (legend_html if legend else "")


# ----------------------------------------------------------------------------- counterfactual
def before_after(r: dict) -> str:
    """Original message with the counterfactual's removals struck through, next to the edited
    message. Both scores are the model's own outputs (first and last counterfactual step)."""
    cf = r["cf"]
    removed = {i for s in cf["steps"] for i in s.get("unit_ids", [])}
    if not removed:
        return ""
    text = r["text"]
    cut = [False] * len(text)
    for i in removed:
        for a, b in r["units"][i][2]:
            for c in range(a, b):
                cut[c] = True
    before, after, i = [], [], 0
    while i < len(text):
        j = i
        while j < len(text) and cut[j] == cut[i]:
            j += 1
        chunk = esc(text[i:j])
        before.append(f"<del>{chunk}</del>" if cut[i] else chunk)
        after.append('<span class="gone">…</span>' if cut[i] else chunk)
        i = j
    p0, p1 = cf["probs"][0], cf["probs"][-1]
    tone0 = "red" if p0 >= r["threshold"] else "green"
    tone1 = "red" if p1 >= r["threshold"] else "green"
    return (f'<div class="mx-ba">'
            f'<div class="side"><div class="lbl"><span>Before</span>{chip(pct(p0), tone0)}</div>'
            f'<div class="txt">{"".join(before)}</div></div>'
            f'<div class="side"><div class="lbl"><span>After the edit</span>{chip(pct(p1), tone1)}</div>'
            f'<div class="txt">{"".join(after)}</div></div></div>')


def counterfactual_block(r: dict, analyst: bool = False):
    """analyst=True also shows, for unflagged messages, which removals would get them
    flagged (useful for missed spam); users only see edits for flagged messages."""
    cf, p0 = r["cf"], r["prob"]
    steps = cf["steps"]
    if cf["direction"] == "to_suspicious" and (not analyst or not steps):
        html(f'<p class="mx-muted">{icon("check", 16, GREEN)} Already treated as a normal '
             'message, so there is nothing to change.</p>')
        return
    if not steps:
        html('<p class="mx-muted">No single word removal lowers the score: the decision '
             'rests on the message as a whole.</p>')
        return

    rows = [f'<tr><td><span class="mx-small">original</span></td><td></td>'
            f'<td><b>{esc(pct_fine(p0))}</b></td><td></td></tr>']
    for k, s in enumerate(steps, 1):
        d = (s["after"] - s["before"]) * 100
        dtxt = f"{d:+.2f}" if abs(d) < 1 else f"{d:+.1f}"
        last = k == len(steps) and cf["flipped"]
        times = f' <span class="mx-small">×{s["count"]}</span>' if s["count"] > 1 else ""
        rows.append(
            f'<tr class="{"flip" if last else ""}"><td>{k}</td>'
            f'<td>remove <span class="mx-word">{esc(s["name"])}</span>{times}</td>'
            f'<td>{esc(pct_fine(s["before"]))} → <b>{esc(pct_fine(s["after"]))}</b></td>'
            f'<td><span class="mx-delta {"up" if d > 0 else ""}">{dtxt} pts</span></td></tr>')
    table = ('<table class="mx-steps"><thead><tr><th>Step</th><th>Edit</th>'
             '<th>Spam probability</th><th>Change</th></tr></thead><tbody>'
             + "".join(rows) + "</tbody></table>")

    names = ", ".join(f"“{s['name']}”" for s in steps)
    final = cf["probs"][-1]
    if cf["direction"] == "to_safe":
        if cf["flipped"]:
            head = (f'{icon("undo", 16, GREEN)} Remove {esc(names)} and the score drops from '
                    f'<b>{esc(pct(p0))}</b> to <b>{esc(pct(final))}</b>: the message would '
                    f'<b>no longer be flagged</b>.')
        else:
            head = (f'{icon("info", 16, AMBER)} Even without {esc(names)} the score stays at '
                    f'<b>{esc(pct(final))}</b>: the suspicion is spread over many signals.')
    else:
        head = (f'{icon("info", 16, AMBER)} Removing {esc(names)} would move the score from '
                f'<b>{esc(pct(p0))}</b> to <b>{esc(pct(final))}</b>'
                + (": the model <b>would then flag it</b>." if cf["flipped"] else "."))
    html(f'<p style="margin:.1rem 0 .8rem">{head}</p>{before_after(r)}'
         f'<div class="mx-table-wrap">{table}</div>')
    if cf["direction"] == "to_safe":
        lbl = ("<b>Predicted label changed:</b> suspicious → not flagged." if cf["flipped"] else
               "<b>Predicted label unchanged:</b> still flagged after these removals.")
        note = ("This shows how the model reacts to the edit; it does not mean the edited message "
                "is genuinely safe.")
    else:
        lbl = ("<b>Predicted label changed:</b> not flagged → suspicious." if cf["flipped"] else
               "<b>Predicted label unchanged.</b>")
        note = "This shows how the model reacts to the edit, not what the message really is."
    rows_exact = "".join(
        f"<tr><td>{k}</td><td>{esc(s['name'])}</td><td>{s['before']:.6f}</td>"
        f"<td>{s['after']:.6f}</td></tr>" for k, s in enumerate(steps, 1))
    html(f'<p style="margin:.8rem 0 .3rem;color:#3A1628">{lbl} {esc(note)}</p>'
         f'<p class="mx-small" style="margin:0 0 .4rem">Every score above is a fresh prediction by '
         f'the model on the edited message ({cf["candidates_tested"]} edited versions tested). '
         'The search removes one word at a time and keeps only removals that are needed, so '
         'this is a small edit, not necessarily the smallest possible one.</p>'
         f'<details class="mx-exact"><summary>Exact probabilities (unrounded)</summary>'
         f'<div class="mx-table-wrap"><table class="mx-steps"><thead><tr><th>Step</th><th>Removed</th>'
         f'<th>Before</th><th>After</th></tr></thead><tbody>'
         f'<tr><td>original</td><td></td><td></td><td>{p0:.6f}</td></tr>{rows_exact}</tbody></table></div>'
         f'</details>')


# ----------------------------------------------------------------------------- charts
def _aggregate(r: dict, k: int = 10):
    agg = OrderedDict()
    for name, s, l in zip(r["names"], r["shap"], r["lime"]):
        a = agg.setdefault(name, [0.0, 0.0])
        a[0] += s
        a[1] += l
    top = sorted(agg.items(), key=lambda kv: -abs(kv[1][0]))[:k]
    return top


def shap_lime_chart(r: dict):
    top = _aggregate(r)
    if not top:
        return None
    words = [w for w, _ in top][::-1]
    shap = [v[0] for _, v in top][::-1]
    lime = [v[1] * 100 for _, v in top][::-1]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=("SHAP · exact · log-odds",
                                        f"LIME · {r['lime_samples']} samples · probability points"))
    for col, vals in [(1, shap), (2, lime)]:
        fig.add_trace(go.Bar(
            x=vals, y=words, orientation="h", showlegend=False,
            marker_color=[SUS_BAR if v > 0 else SAFE_BAR for v in vals],
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>"), row=1, col=col)
        fig.add_vline(x=0, line_width=1, line_color=MUTED, row=1, col=col)
    style_fig(fig, height=110 + 30 * len(words))
    fig.update_annotations(font=dict(size=12, color=MUTED))
    fig.update_xaxes(title_text="← legitimate · spam →", title_font=dict(size=11, color=MUTED))
    fig.update_layout(margin=dict(l=8, r=8, t=40, b=8))
    return fig


def waterfall_chart(r: dict, k: int = 6):
    """The SHAP decomposition: average message + each word + absent words = this message."""
    top = _aggregate(r, k=len(r["names"]))
    shown = top[:k]
    other = sum(v[0] for _, v in top[k:])
    labels = ["average message"] + [w for w, _ in shown]
    vals = [r["base_logit"]] + [v[0] for _, v in shown]
    measure = ["absolute"] + ["relative"] * len(shown)
    if top[k:]:
        labels.append(f"{len(top) - k} other words")
        vals.append(other)
        measure.append("relative")
    labels += ["words not in message", "this message"]
    vals += [r["rest"], r["logit"]]
    measure += ["relative", "total"]
    fig = go.Figure(go.Waterfall(
        x=labels, y=vals, measure=measure,
        increasing=dict(marker=dict(color=SUS_BAR)), decreasing=dict(marker=dict(color=SAFE_BAR)),
        totals=dict(marker=dict(color=TEXT)), connector=dict(line=dict(color=ACCENT, width=1)),
        hovertemplate="%{x}: %{y:+.2f}<extra></extra>"))
    if r.get("thr_logit") is not None:
        fig.add_hline(y=r["thr_logit"], line_dash="dot", line_color=MUTED,
                      annotation_text="flag threshold", annotation_font_color=MUTED)
    style_fig(fig, height=400)
    fig.update_yaxes(title_text="raw model score (log-odds)", title_font=dict(size=11, color=MUTED))
    fig.update_xaxes(tickangle=-35)
    return fig


def show_fig(fig, key: str):
    if fig is not None:
        st.plotly_chart(fig, key=key, config=PLOT_CONFIG, theme=None)


def agreement_line(r: dict):
    ag = r["agreement"]
    ov = ag.get("topk_overlap")
    if ov is None or ov != ov:
        html('<p class="mx-muted">Too few words to compare the two methods.</p>')
        return
    n, k = round(ov * ag["k"]), ag["k"]
    tone = "green" if ov >= 0.6 else "amber"
    sign = ag.get("sign_agreement")
    rho = ag.get("spearman")
    bits = [chip(f"{n} of top {k} words shared", tone, "check" if ov >= 0.6 else "alert")]
    if sign == sign and sign is not None:
        bits.append(chip(f"same direction on {sign:.0%} of them", "gray"))
    if rho == rho and rho is not None:
        bits.append(chip(f"rank correlation {rho:.2f}", "gray"))
    bits.append(chip(f"LIME fit R² {r['lime_r2']:.2f}", "gray"))
    html('<div style="display:flex;flex-wrap:wrap;gap:.4rem;margin:.2rem 0 .6rem">'
         + "".join(bits) + "</div>")


def probe_table(r: dict):
    rows = r.get("probe") or []
    if not rows:
        html('<p class="mx-muted">No edits apply to this message.</p>')
        return
    df = pd.DataFrame([{
        "Edit": x["attack_label"],
        "Edited message": x["text"],
        "Spam probability": f"{pct_fine(x['prob_before'])} → {pct_fine(x['prob_after'])}",
        "Verdict": "changed" if x["label_changed"] else "same",
        "Top words kept": "–" if x["top3_kept"] != x["top3_kept"] else f"{x['top3_kept']:.0%}",
    } for x in rows])
    st.dataframe(df, hide_index=True, column_config={
        "Edit": st.column_config.TextColumn(width="small"),
        "Edited message": st.column_config.TextColumn(width="medium"),
        "Spam probability": st.column_config.TextColumn(width="small"),
        "Verdict": st.column_config.TextColumn(width="small"),
        "Top words kept": st.column_config.TextColumn(width="small"),
    })
    changed = sum(x["label_changed"] for x in rows)
    html(f'<p class="mx-small">{len(rows)} edits tried; the verdict changed for {changed}. '
         '“Top words kept” is the share of the top-3 explanation words still on top after the edit.</p>')


# ----------------------------------------------------------------------------- full result
def risk_indicators(r: dict, k: int = 4) -> str:
    """The words that push hardest towards each label, with their real SHAP values."""
    agg = OrderedDict()
    for name, (_, token, _, _), v in zip(r["names"], r["units"], r["shap"]):
        if token in STOPWORDS:      # same rule as the plain-language reason
            continue
        agg[name] = agg.get(name, 0.0) + v
    spam = sorted([(n, v) for n, v in agg.items() if v > 0.05], key=lambda x: -x[1])
    safe = sorted([(n, v) for n, v in agg.items() if v < -0.05], key=lambda x: x[1])
    groups = ([("Pushing towards spam", spam[:k], "red"), ("Signs of a normal message", safe[:2], "green")]
              if r["suspicious"] else
              [("Signs of a normal message", safe[:k], "green"), ("Pushing towards spam", spam[:2], "red")])
    out = []
    for title, items, tone in groups:
        if not items:
            continue
        out.append(f'<div class="k">{esc(title)}</div>')
        out += [f'<div class="row"><span class="w">{esc(n)}</span>'
                f'<span class="v {tone}">{v:+.2f}</span></div>' for n, v in items]
    if not out:
        return '<p class="mx-muted">No single word stands out; the score comes from the overall mix.</p>'
    return ('<div class="mx-ind">' + "".join(out) + '</div>'
            '<p class="mx-small" style="margin:.7rem 0 0">SHAP values in log-odds: how far each word '
            'moves the raw score. Positive means more suspicious. Filler words such as “to” are '
            'left out.</p>')


def instability_note(r: dict) -> str:
    """Same rule as the Trust Lab: a small edit (the robustness check edits at most two words or
    the casing/greeting) that flips the verdict or moves the probability by >= 20 points."""
    worst = None
    for e in r.get("probe") or []:
        shift = abs(e["prob_after"] - e["prob_before"]) * 100
        if e["label_changed"] or shift >= 20:
            if worst is None or shift > abs(worst["prob_after"] - worst["prob_before"]) * 100:
                worst = e
    if not worst:
        return ""
    return ('<div class="mx-note high">' + icon("alert", 16) + '<span><b>Prediction instability '
            f'detected.</b> {esc(worst["attack_label"])} moves the score from '
            f'{esc(pct_fine(worst["prob_before"]))} to {esc(pct_fine(worst["prob_after"]))}'
            + (" and changes the verdict" if worst["label_changed"] else "")
            + '. A lower spam probability does not prove a message is safe. See “Robustness '
            'check” below.</span></div>')


def phishing_block(r: dict) -> str:
    found = phishing_indicators(r["text"])
    if not found:
        return '<p class="mx-muted">No common phishing patterns found.</p>'
    rows = "".join(f'<div class="row"><span class="w">{esc(i["label"])}</span>'
                   f'<span class="v" style="color:#685460;font-weight:600">'
                   f'{esc(", ".join(i["evidence"]))}</span></div>' for i in found)
    out = f'<div class="mx-ind">{rows}</div>'
    if phishing_caution(found, r["suspicious"]):
        out += ('<div class="mx-note high">' + icon("alert", 16) + '<span><b>Treat with caution.</b> '
                'The model scored this below the 50% threshold, but it shows several common '
                'phishing patterns. A lower spam probability does not prove a message is safe: '
                'MATRIX was trained on 2000s UK SMS spam (no delivery or account phishing) and '
                'can miss reworded or modern scams.</span></div>')
    return out


def explanation(r: dict, key: str, analyst: bool = False):
    """Everything about one analysed message. `key` keeps widget ids unique per place.
    Most important first: message + verdict side by side; deeper analysis below."""
    def why_card():
        with st.container(border=True, key=f"card-why-{key}"):
            card_title("message", "Why")
            html(f'<p style="margin:0;line-height:1.65">{esc(r["reason"])}</p>')

    def risk_card():
        with st.container(border=True, key=f"card-ind-{key}"):
            card_title("target", "Risk indicators")
            html(risk_indicators(r))

    def phishing_card():
        with st.container(border=True, key=f"card-phish-{key}"):
            card_title("alert", "Phishing patterns",
                       "Rule-based check of visible patterns, not the ML model. It never changes "
                       "the probability or verdict.")
            html(phishing_block(r))

    left, right = st.columns([7, 5], gap="large")
    with left:
        with st.container(border=True, key=f"card-msg-{key}"):
            card_title("eye", "Your message, explained",
                       "Highlighted words drove the decision.")
            method = st.segmented_control(
                "Highlight using", ["SHAP", "LIME"], default="SHAP", key=f"method-{key}",
                help="SHAP is exact for this model. LIME is an independent, sampled check.")
            html(highlight(r, method or "SHAP"))
        if analyst:
            # Error explorer: split the cards by measured height so neither column is left
            # mostly empty (message + risk indicators vs. verdict + why + phishing).
            risk_card()
    with right:
        with st.container(border=True, key=f"card-verdict-{key}"):
            verdict_card(r)
            html(instability_note(r))
        why_card()
        if not analyst:
            risk_card()
        phishing_card()

    section("undo", "What would change the decision",
            "Counterfactual: words are removed one at a time and every edit is re-scored by the model.")
    with st.container(border=True, key=f"card-cf-{key}"):
        counterfactual_block(r, analyst)

    section("layers", "Under the hood", "How the two explanation methods compare, how the score "
                                        "is built, and how robust this verdict is.")
    t1, t2, t3 = st.tabs(["SHAP vs LIME", "Score breakdown", "Robustness check"])
    with t1:
        agreement_line(r)
        show_fig(shap_lime_chart(r), f"cmp-{key}")
        html('<p class="mx-small" style="max-width:90ch">SHAP comes straight from the model\'s weights and is exact. '
             'LIME hides random words, re-scores the message and fits a small local model, so it '
             'is approximate and works in probability points. When the score is near 0% or 100%, '
             'hiding a few words barely moves the probability, so LIME can rank minor words '
             'differently. That is a limit of sampling, not proof either method is wrong.</p>')
    with t2:
        show_fig(waterfall_chart(r), f"wf-{key}")
        html(f'<p class="mx-small" style="max-width:90ch">SHAP splits the raw model score exactly: the average '
             f'training message (score {r["base_logit"]:+.2f}, about {esc(pct(r["base_prob"]))} '
             f'spam) plus each word, plus words this message does not contain '
             f'({r["rest"]:+.2f}), equals this message ({r["logit"]:+.2f}, {esc(pct(r["prob"]))}). '
             'Calibration then turns the raw score into the probability shown.</p>')
    with t3:
        probe_table(r)
