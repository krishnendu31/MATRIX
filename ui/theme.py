"""MATRIX design system: "Soft Plum & Ivory".

One place for every colour, the page CSS, icons and chart styling, so all pages look the same.
Streamlit's own widgets get the same colours from .streamlit/config.toml.

Contrast (WCAG): every text colour below is >= 4.5:1 on the surfaces it is used on
(dark plum 14.6:1 on the background, plum 6.4:1 on white, muted 6.4:1 on the background).
The lilac #D3C4CF is used for borders, fills and chips with dark text only, never with muted text.
"""
from __future__ import annotations

import html

# ----------------------------------------------------------------------------- palette
BG = "#F8F5F7"            # background, pale ivory
LILAC = "#D3C4CF"         # secondary / muted surfaces (fills, chips with dark text)
TINT = "#F1EAF0"          # soft card tint
PRIMARY = "#8C4A6B"       # primary accent, plum (6.4:1 on white: fine for text and buttons)
PRIMARY_DEEP = "#703653"  # darker plum: hover, active states
TEXT = "#3A1628"          # dark burgundy: headings, body text, strongest contrast
CARD = "#FFFFFF"          # cards
MUTED = "#685460"         # muted text and labels
BORDER = "#E7DAE3"        # light borders
HOVER = "#F1EAF0"         # hover background
SHADOW = "rgba(58, 22, 40, 0.06)"

# Older names kept so every module keeps working
SURFACE = TINT
ACCENT = LILAC
MUTED_ON_BLUSH = MUTED

# Result states only: green / red / amber never appear as decoration, and plum never means risk.
# *_C: icons, borders and chart marks.  *_TEXT: readable text shade.
RED_C, RED_TEXT, RED_BG, RED_LINE = "#C2453C", "#B23B33", "#FBEAE8", "#EFC2BD"        # suspicious
GREEN_C, GREEN_TEXT, GREEN_BG, GREEN_LINE = "#3E8159", "#2F7048", "#E7F2EA", "#B9D8C3"  # legitimate
AMBER_C, AMBER_TEXT, AMBER_BG, AMBER_LINE = "#C08A2E", "#8A5A16", "#FBF1DF", "#EBD3A5"  # uncertain
RED, GREEN, AMBER = RED_TEXT, GREEN_TEXT, AMBER_TEXT

# Chart colours: semantic where a mark means spam / legitimate, neutral greys otherwise,
# plum only as the accent series (SHAP, "MATRIX", "calibrated").
SUS_BAR = RED_C
SAFE_BAR = GREEN_C
SHAP_C = PRIMARY
LIME_C = "#C9A9BD"        # light plum, paired with SHAP
NEUTRAL_C = "#9A8692"     # neutral series (counts, distributions)
ALT_C = "#D3C4CF"         # comparison / ablation series (lilac)
GRID = "#EFE6EC"


# ----------------------------------------------------------------------------- icons
# Lucide-style line icons (24px grid, 2px rounded stroke), drawn for MATRIX.
_ICON_PATHS = {
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
    "shield-check": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "message": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "eye": '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
    "undo": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "lock": '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "book": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    "scale": '<path d="M12 3v18"/><path d="M5 7h14"/><path d="m5 7-3 7a4 4 0 0 0 6 0z"/><path d="m19 7-3 7a4 4 0 0 0 6 0z"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "compass": '<circle cx="12" cy="12" r="10"/><path d="m16.24 7.76-2.12 6.36-6.36 2.12 2.12-6.36z"/>',
    "trace": '<circle cx="5" cy="18" r="2"/><circle cx="19" cy="6" r="2"/><path d="M7 18h4a4 4 0 0 0 4-4v-4a4 4 0 0 1 4-4"/>',
}


def icon(name: str, size: int = 18, color: str = "currentColor", stroke: float = 2) -> str:
    return (f'<svg class="mx-ico" xmlns="http://www.w3.org/2000/svg" width="{size}" '
            f'height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{_ICON_PATHS[name]}</svg>')


# Characters that Markdown or Streamlit's LaTeX would interpret, plus line breaks (a blank
# line would end our HTML block). Encoding them as character references makes the browser
# show them exactly as typed, while the Markdown parser never sees them.
# Markdown/LaTeX-sensitive characters, as code points: * _ $ ` ~ [ ] # backslash | { }
_MD_CHARS = (42, 95, 36, 96, 126, 91, 93, 35, 92, 124, 123, 125)
_MD_SAFE = {chr(c): f"&#{c};" for c in _MD_CHARS}
_MD_SAFE[chr(10)] = "&#10;"          # line break


def esc(s) -> str:
    """Escape text (often user input) for MATRIX's HTML: no HTML, Markdown or LaTeX runs."""
    text = str(s).replace(chr(13) + chr(10), chr(10))
    return "".join(_MD_SAFE.get(c) or html.escape(c) for c in text)


# ----------------------------------------------------------------------------- CSS
# Selectors use only stable hooks: Streamlit's data-testid attributes, the documented
# st-key-<key> classes of keyed containers/widgets, ARIA attributes, and MATRIX's own mx-*
# classes. Specificity (e.g. div[class*=...]) is used instead of !important.
HEAD_FONT = "'Plus Jakarta Sans', 'DM Sans', system-ui, sans-serif"
BODY_FONT = "'DM Sans', system-ui, sans-serif"
NAV_BREAK = 1260   # below this width the one-line navigation becomes a dropdown

CSS = f"""
<style>
/* ================= layout: wide, centred, breathable ================= */
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: {BG}; }}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stMainBlockContainer"] {{
  max-width: 1440px; padding: 1.75rem clamp(1rem, 3.6vw, 3.5rem) 4rem;
}}
[data-testid="stSidebar"] {{ background: {TINT}; border-right: 1px solid {BORDER}; }}
p, li {{ line-height: 1.65; }}
h1, h2, h3, h4 {{ color: {TEXT}; letter-spacing: -0.015em; }}
.mx-ico {{ display: inline-block; vertical-align: -3px; flex-shrink: 0; }}
.mx-muted {{ color: {MUTED}; font-size: 0.97rem; margin: 0; max-width: 72ch; }}
.mx-small {{ color: {MUTED}; font-size: 0.82rem; }}
.mx-gap {{ height: 0.5rem; }}

/* ================= cards: st.container(border=True, key="card-...") ================= */
div[class*="st-key-card"] {{
  background: {CARD}; border: 1px solid {BORDER}; border-radius: 20px;
  padding: 1.6rem 1.75rem; box-shadow: 0 1px 2px {SHADOW}, 0 8px 24px {SHADOW}; gap: 0.9rem;
}}
.mx-card-title {{ font-family: {HEAD_FONT}; font-size: 1.12rem; font-weight: 700; color: {TEXT};
  display: flex; align-items: center; gap: 0.6rem; margin: 0 0 0.2rem; }}
.mx-card-title .mx-ico {{ color: {PRIMARY}; }}
.mx-card-hint {{ color: {MUTED}; font-size: 0.84rem; margin: 0 0 0.4rem; }}

/* ================= top bar: brand left, navigation right ================= */
.st-key-topbar {{ margin-bottom: 1.25rem; }}
.mx-brand {{ display: flex; align-items: center; gap: 0.8rem; white-space: nowrap; }}
.mx-logo {{ width: 44px; height: 44px; border-radius: 14px; background: {PRIMARY}; color: #fff;
  display: flex; align-items: center; justify-content: center; box-shadow: 0 6px 16px rgba(140, 74, 107, 0.25); }}
.mx-name {{ font-family: {HEAD_FONT}; font-weight: 800; font-size: 1.35rem; letter-spacing: 0.14em;
  color: {TEXT}; line-height: 1.1; }}
.mx-tag {{ font-size: 0.84rem; color: {MUTED}; font-style: italic; white-space: nowrap; }}
.st-key-nav_d {{ margin-left: auto; width: fit-content; max-width: 100%; }}
.st-key-nav_d [data-testid="stButtonGroup"] {{ display: flex; justify-content: flex-end; }}
.st-key-nav_d [data-testid="stButtonGroup"] > div {{
  background: {CARD}; border: 1px solid {BORDER}; border-radius: 999px; padding: 5px; gap: 4px;
  box-shadow: 0 1px 2px {SHADOW}; flex-wrap: nowrap; width: fit-content;
}}
.st-key-nav_d [data-testid="stButtonGroup"] button {{
  border: none; border-radius: 999px; background: transparent; color: {TEXT}; white-space: nowrap;
  font-weight: 550; padding: 0.5rem 1.05rem; min-height: 0; transition: background 0.15s ease;
}}
.st-key-nav_d [data-testid="stButtonGroup"] button:hover {{ background: {HOVER}; color: {TEXT}; }}
.st-key-nav_d [data-testid="stButtonGroup"] button[aria-checked="true"] {{
  background: {PRIMARY}; color: #fff; font-weight: 650;
}}
.st-key-nav_d [data-testid="stIconMaterial"] {{ color: {PRIMARY}; }}
.st-key-nav_d button[aria-checked="true"] [data-testid="stIconMaterial"] {{ color: #fff; }}
.st-key-nav_m {{ display: none; }}
@media (min-width: 641px) {{
  /* brand column sized to its content; the navigation takes the rest of the row */
  .st-key-topbar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {{
    flex: 0 0 auto; width: auto; min-width: max-content; }}
  .st-key-topbar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {{
    flex: 1 1 0; width: auto; min-width: 0; }}
  .st-key-nav_m {{ max-width: 320px; margin-left: auto; }}
}}
@media (max-width: {NAV_BREAK}px) {{
  .st-key-nav_d {{ display: none; }}
  .st-key-nav_m {{ display: block; }}
}}

/* ================= page header + sections ================= */
.mx-page {{ margin: 0.75rem 0 2rem; }}
.mx-eyebrow {{ font-size: 0.78rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
  color: {PRIMARY}; display: flex; align-items: center; gap: 0.45rem; margin-bottom: 0.6rem; }}
.mx-page-title {{ font-family: {HEAD_FONT}; font-size: clamp(1.9rem, 3vw, 2.6rem); font-weight: 800;
  color: {TEXT}; line-height: 1.12; letter-spacing: -0.025em; margin: 0 0 0.7rem; }}
.mx-page-sub {{ color: {MUTED}; font-size: 1.06rem; max-width: 68ch; margin: 0; line-height: 1.65; }}
.mx-section {{ display: flex; align-items: center; gap: 0.85rem; margin: 3.5rem 0 1.4rem; }}
.mx-section .ico {{ width: 42px; height: 42px; border-radius: 13px; background: {TINT}; color: {PRIMARY};
  display: flex; align-items: center; justify-content: center; flex-shrink: 0; border: 1px solid {BORDER}; }}
.mx-section .h {{ font-family: {HEAD_FONT}; font-size: clamp(1.3rem, 1.9vw, 1.55rem); font-weight: 750;
  color: {TEXT}; line-height: 1.25; margin: 0; }}
.mx-section .mx-muted {{ font-size: 0.93rem; margin-top: 0.15rem; }}

/* ================= hero (two columns, keyed container "hero") ================= */
div.st-key-hero {{
  background: {CARD}; border: 1px solid {BORDER}; border-radius: 28px;
  padding: clamp(1.6rem, 4vw, 3.5rem); box-shadow: 0 1px 2px {SHADOW}, 0 18px 48px {SHADOW};
}}
.mx-wordmark {{ font-family: {HEAD_FONT}; font-weight: 800; font-size: clamp(2.7rem, 4.6vw, 4rem);
  letter-spacing: 0.12em; color: {TEXT}; line-height: 1; margin: 0.35rem 0 0.75rem; }}
.mx-motto {{ font-family: {HEAD_FONT}; font-size: clamp(1.15rem, 1.6vw, 1.5rem); font-weight: 600;
  color: {PRIMARY}; font-style: italic; margin: 0 0 1.1rem; }}
.mx-lead {{ font-size: clamp(1rem, 1.15vw, 1.125rem); color: {MUTED}; max-width: 54ch; margin: 0 0 1.75rem;
  line-height: 1.7; }}
.mx-preview {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 20px; padding: 1.25rem 1.3rem; }}
.mx-preview .bar {{ display: flex; align-items: center; justify-content: space-between; gap: 0.6rem;
  margin-bottom: 0.85rem; }}
.mx-preview .dots {{ display: flex; gap: 6px; }}
.mx-preview .dots i {{ width: 9px; height: 9px; border-radius: 50%; background: {LILAC}; display: block; }}
.mx-preview .label {{ font-size: 0.74rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: {MUTED}; }}
.mx-preview .mx-msg {{ background: {CARD}; font-size: 0.98rem; line-height: 1.95; }}
.mx-preview .foot {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin-top: 0.9rem; }}
.mx-preview .why {{ font-size: 0.86rem; color: {MUTED}; margin: 0.7rem 0 0; line-height: 1.55; }}

/* ================= grids, tiles, stats ================= */
.mx-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.5rem; }}
.mx-grid-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.mx-grid-4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
@media (max-width: 1100px) {{ .mx-grid-3, .mx-grid-4 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 640px) {{ .mx-grid-3, .mx-grid-4 {{ grid-template-columns: 1fr; }} }}
.mx-tile {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 20px; padding: 1.6rem 1.7rem;
  box-shadow: 0 1px 2px {SHADOW}; transition: border-color 0.15s ease, box-shadow 0.15s ease; height: 100%; box-sizing: border-box; }}
.mx-tile:hover {{ border-color: {LILAC}; box-shadow: 0 10px 28px {SHADOW}; }}
.mx-tile-ico {{ width: 44px; height: 44px; border-radius: 13px; background: {TINT}; color: {PRIMARY};
  display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;
  font-family: {HEAD_FONT}; font-weight: 800; border: 1px solid {BORDER}; }}
.mx-tile h4 {{ font-family: {HEAD_FONT}; font-size: 1.08rem; margin: 0 0 0.4rem; font-weight: 700; padding: 0; color: {TEXT}; }}
.mx-tile p {{ font-size: 0.93rem; color: {MUTED}; margin: 0; line-height: 1.6; }}
/* homepage feature cards: three, roomy, centred */
.mx-feature {{ text-align: center; padding: 2.4rem 2rem 2.2rem; }}
.mx-feature .mx-tile-ico {{ width: 60px; height: 60px; border-radius: 18px; margin: 0 auto 1.2rem; }}
.mx-feature h4 {{ font-size: 1.3rem; margin-bottom: 0.45rem; }}
.mx-feature p {{ font-size: 1rem; max-width: 30ch; margin: 0 auto; }}
/* closing call to action (keyed container "cta") */
div.st-key-cta {{ margin-top: 4rem; background: {TINT}; border: 1px solid {BORDER}; border-radius: 28px;
  padding: clamp(2rem, 5vw, 3.5rem) clamp(1.25rem, 4vw, 3rem); text-align: center; gap: 0.5rem; }}
.mx-cta-title {{ font-family: {HEAD_FONT}; font-size: clamp(1.5rem, 2.6vw, 2.1rem); font-weight: 800;
  color: {TEXT}; letter-spacing: -0.02em; line-height: 1.2; margin: 0 auto 0.6rem; max-width: 28ch;
  text-align: center; }}
.mx-cta-sub {{ color: {MUTED}; font-size: 1.06rem; max-width: 52ch; margin: 0 auto 1.4rem; line-height: 1.65;
  text-align: center; }}
.mx-stat {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 18px; padding: 1.3rem 1.4rem;
  box-shadow: 0 1px 2px {SHADOW}; }}
.mx-stat .v {{ font-family: {HEAD_FONT}; font-size: clamp(1.6rem, 2.2vw, 2rem); font-weight: 800; color: {TEXT}; line-height: 1.15; }}
.mx-stat .l {{ font-size: 0.9rem; color: {TEXT}; font-weight: 600; margin-top: 0.3rem; }}
.mx-stat .s {{ font-size: 0.8rem; color: {MUTED}; margin-top: 0.1rem; }}

/* ================= verdict ================= */
.mx-verdict {{ border-radius: 18px; padding: 1.35rem 1.45rem; border: 1px solid; }}
.mx-verdict.red {{ background: {RED_BG}; border-color: {RED_LINE}; }}
.mx-verdict.amber {{ background: {AMBER_BG}; border-color: {AMBER_LINE}; }}
.mx-verdict.green {{ background: {GREEN_BG}; border-color: {GREEN_LINE}; }}
.mx-verdict .head {{ display: flex; align-items: center; gap: 0.6rem; font-family: {HEAD_FONT};
  font-weight: 800; font-size: 1.5rem; }}
.mx-verdict.red .head {{ color: {RED_TEXT}; }}
.mx-verdict.amber .head {{ color: {AMBER_TEXT}; }}
.mx-verdict.green .head {{ color: {GREEN_TEXT}; }}
.mx-verdict .sub {{ color: {TEXT}; font-size: 0.95rem; margin-top: 0.25rem; }}
.mx-prob {{ font-family: {HEAD_FONT}; font-size: clamp(2.3rem, 3vw, 2.8rem); font-weight: 800; color: {TEXT};
  line-height: 1; margin: 1rem 0 0.3rem; }}
.mx-verdict .mx-small {{ color: {TEXT}; }}
.mx-meter {{ position: relative; height: 10px; border-radius: 99px; margin: 0.8rem 0 0.4rem;
  background: linear-gradient(90deg, {GREEN_LINE} 0%, {AMBER_LINE} 50%, {RED_LINE} 100%); }}
.mx-meter .dot {{ position: absolute; top: 50%; width: 18px; height: 18px; border-radius: 99px;
  transform: translate(-50%, -50%); background: #fff; border: 3px solid {TEXT}; }}
.mx-meter .thr {{ position: absolute; top: -5px; bottom: -5px; width: 2px; background: {TEXT}; opacity: 0.35; }}
.mx-scale {{ display: flex; justify-content: space-between; font-size: 0.78rem; color: {TEXT}; }}

/* ================= chips, notes, indicators ================= */
.mx-chip {{ display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.22rem 0.75rem;
  border-radius: 99px; font-size: 0.82rem; font-weight: 600; border: 1px solid; line-height: 1.5; }}
.mx-chip.red {{ color: {RED_TEXT}; background: {RED_BG}; border-color: {RED_LINE}; }}
.mx-chip.amber {{ color: {AMBER_TEXT}; background: {AMBER_BG}; border-color: {AMBER_LINE}; }}
.mx-chip.green {{ color: {GREEN_TEXT}; background: {GREEN_BG}; border-color: {GREEN_LINE}; }}
.mx-chip.gray {{ color: {MUTED}; background: {CARD}; border-color: {BORDER}; }}
.mx-chip.rose, .mx-chip.plum {{ color: {TEXT}; background: {TINT}; border-color: {BORDER}; }}
.mx-note {{ display: flex; gap: 0.6rem; align-items: flex-start; padding: 0.7rem 0.85rem; border-radius: 12px;
  font-size: 0.9rem; margin-top: 0.6rem; line-height: 1.55; }}
.mx-note.high {{ background: {AMBER_BG}; color: {TEXT}; }}
.mx-note.high .mx-ico {{ color: {AMBER_C}; }}
.mx-note.medium {{ background: {TINT}; color: {TEXT}; }}
.mx-note.low {{ background: {BG}; color: {MUTED}; border: 1px solid {BORDER}; }}
.mx-ind {{ display: flex; flex-direction: column; gap: 0.5rem; }}
.mx-ind .row {{ display: flex; align-items: center; justify-content: space-between; gap: 0.8rem;
  padding: 0.55rem 0.75rem; border-radius: 12px; background: {BG}; border: 1px solid {BORDER}; }}
.mx-ind .w {{ font-weight: 600; color: {TEXT}; overflow-wrap: anywhere; }}
.mx-ind .v {{ font-variant-numeric: tabular-nums; font-weight: 700; font-size: 0.88rem; white-space: nowrap; }}
.mx-ind .v.red {{ color: {RED_TEXT}; }} .mx-ind .v.green {{ color: {GREEN_TEXT}; }}
.mx-ind .k {{ font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: {MUTED}; margin-top: 0.4rem; }}

/* ================= highlighted message ================= */
.mx-msg {{ font-size: 1.1rem; line-height: 2.1; padding: 1.2rem 1.35rem; border-radius: 16px; background: {BG};
  border: 1px solid {BORDER}; white-space: pre-wrap; overflow-wrap: anywhere; color: {TEXT}; }}
.mx-strong {{ color: {RED_TEXT}; font-weight: 700; background: {RED_BG}; border-bottom: 2px solid {RED_C};
  border-radius: 4px; padding: 1px 3px; }}
.mx-weak {{ color: {RED_TEXT}; background: #FDF5F4; border-bottom: 1px dashed {RED_LINE};
  border-radius: 4px; padding: 1px 3px; }}
.mx-safe {{ color: {GREEN_TEXT}; font-weight: 600; background: {GREEN_BG}; border-radius: 4px; padding: 1px 3px; }}
.mx-legend {{ display: flex; flex-wrap: wrap; gap: 1rem; font-size: 0.8rem; color: {MUTED}; margin-top: 0.8rem; }}
.mx-legend span {{ display: inline-flex; align-items: center; gap: 0.35rem; }}

/* ================= counterfactual ================= */
.mx-ba {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin: 0.6rem 0 1.25rem; }}
.mx-ba .side {{ border: 1px solid {BORDER}; border-radius: 16px; padding: 1rem 1.15rem; background: {BG}; }}
.mx-ba .lbl {{ display: flex; justify-content: space-between; align-items: center; font-size: 0.76rem;
  font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: {MUTED}; margin-bottom: 0.45rem; }}
.mx-ba .txt {{ font-size: 0.98rem; line-height: 1.8; color: {TEXT}; white-space: pre-wrap; overflow-wrap: anywhere; }}
.mx-ba del {{ color: {RED_TEXT}; background: {RED_BG}; text-decoration-thickness: 2px; border-radius: 3px; padding: 0 2px; }}
.mx-ba .gone {{ color: {MUTED}; font-weight: 600; }}
.mx-steps {{ width: 100%; border-collapse: collapse; font-size: 0.93rem; }}
.mx-steps th {{ text-align: left; font-weight: 700; color: {MUTED}; font-size: 0.75rem; text-transform: uppercase;
  letter-spacing: 0.06em; padding: 0.5rem 0.6rem; border-bottom: 1px solid {BORDER}; }}
.mx-steps td {{ padding: 0.65rem 0.6rem; border-bottom: 1px solid #F3ECF1; vertical-align: middle; }}
.mx-steps tr.flip td {{ background: {GREEN_BG}; }}
.mx-word {{ font-weight: 600; background: {CARD}; border: 1px solid {BORDER}; color: {TEXT}; border-radius: 6px; padding: 1px 7px; }}
.mx-delta {{ color: {GREEN_TEXT}; font-weight: 700; white-space: nowrap; }}
.mx-delta.up {{ color: {RED_TEXT}; }}
.mx-table-wrap {{ overflow-x: auto; }}
.mx-exact {{ margin-top: 0.4rem; }}
.mx-exact summary {{ cursor: pointer; color: {PRIMARY}; font-weight: 600; font-size: 0.88rem; }}
.mx-exact td {{ font-variant-numeric: tabular-nums; }}

/* ================= about ================= */
.mx-kv {{ display: grid; grid-template-columns: max-content 1fr; gap: 0.55rem 1.5rem; font-size: 0.97rem; }}
.mx-kv dt {{ color: {MUTED}; }} .mx-kv dd {{ margin: 0; color: {TEXT}; font-weight: 600; }}
@media (max-width: 640px) {{ .mx-kv {{ grid-template-columns: 1fr; gap: 0.1rem; }} .mx-kv dd {{ margin-bottom: 0.6rem; }} }}

/* ================= footer ================= */
.mx-footer {{ margin-top: 4rem; padding-top: 1.4rem; border-top: 1px solid {BORDER}; font-size: 0.84rem;
  color: {MUTED}; display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; justify-content: space-between; }}

/* ================= Streamlit widgets ================= */
[data-testid="stBaseButton-primary"] {{ background: {PRIMARY}; border-color: {PRIMARY}; color: #fff;
  font-weight: 650; padding: 0.65rem 1.6rem; box-shadow: 0 6px 16px rgba(140, 74, 107, 0.22); }}
[data-testid="stBaseButton-primary"]:hover {{ background: {PRIMARY_DEEP}; border-color: {PRIMARY_DEEP}; color: #fff; }}
[data-testid="stBaseButton-secondary"] {{ background: {CARD}; border: 1px solid {LILAC}; color: {TEXT};
  font-weight: 550; padding: 0.65rem 1.6rem; }}
[data-testid="stBaseButton-secondary"]:hover {{ background: {HOVER}; border-color: {PRIMARY}; color: {TEXT}; }}
.st-key-example [data-testid="stButtonGroup"] button {{ background: {CARD}; border: 1px solid {BORDER};
  color: {TEXT}; border-radius: 999px; padding: 0.35rem 0.95rem; }}
.st-key-example [data-testid="stButtonGroup"] button:hover {{ background: {HOVER}; border-color: {LILAC}; }}
[data-testid="stTextAreaRootElement"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px;
  transition: border-color 0.15s ease, box-shadow 0.15s ease; }}
[data-testid="stTextAreaRootElement"]:focus-within {{ border-color: {PRIMARY}; box-shadow: 0 0 0 4px rgba(140, 74, 107, 0.14); }}
[data-testid="stTextArea"] textarea {{ background: {CARD}; font-size: 1.04rem; line-height: 1.65; color: {TEXT}; padding: 0.9rem 1rem; }}
[data-testid="stWidgetLabel"] p {{ font-size: 0.86rem; color: {MUTED}; font-weight: 600; }}
[data-testid="stTabs"] {{ margin-top: 1.5rem; }}
[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 1.25rem; flex-wrap: wrap; border-bottom: 1px solid {BORDER}; }}
[data-testid="stTabs"] [role="tab"] {{ color: {MUTED}; font-weight: 600; padding: 0.6rem 0.1rem; }}
[data-testid="stTabs"] [role="tab"]:hover {{ color: {TEXT}; }}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{ color: {PRIMARY}; }}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {{ padding-top: 1.4rem; }}
[data-testid="stExpander"] details {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; }}
[data-testid="stExpander"] summary:hover {{ background: {HOVER}; }}
[data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 18px; padding: 1rem 1.2rem; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 14px; overflow: hidden; }}
[data-testid="stAlertContainer"] {{ border-radius: 14px; }}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {{ background: {CARD}; border-color: {BORDER}; border-radius: 14px; }}

/* Dashboard: cards that sit side by side get equal heights */
[data-testid="stHorizontalBlock"]:has(div[class*="st-key-card-dash"]) {{ align-items: stretch; }}
[data-testid="stColumn"]:has(div[class*="st-key-card-dash"]) > [data-testid="stVerticalBlock"] {{ height: 100%; }}
[data-testid="stLayoutWrapper"]:has(> div[class*="st-key-card-dash"]) {{ height: 100%; }}
div[class*="st-key-card-dash"] {{ height: 100%; padding: 1.35rem 1.5rem 1.25rem; gap: 0.6rem; }}
[role="tabpanel"] .mx-grid {{ margin-bottom: 0.9rem; }}
[role="tabpanel"] .mx-stat {{ height: 100%; box-sizing: border-box; display: flex; flex-direction: column; }}
[role="tabpanel"] .mx-stat .s {{ margin-top: auto; padding-top: 0.25rem; }}
.mx-caption {{ color: {MUTED}; font-size: 0.88rem; line-height: 1.6; margin: 0.25rem 0 0; max-width: 62ch; }}
.mx-caption b {{ color: {TEXT}; font-weight: 600; }}
/* Analyzer: the highlighted message stays in view beside the (taller) verdict column */
@media (min-width: 901px) {{
  [data-testid="stHorizontalBlock"]:has(div[class*="st-key-card-verdict"]) {{ align-items: stretch; }}
  [data-testid="stColumn"]:has(div[class*="st-key-card-msg"]) > [data-testid="stVerticalBlock"] {{ height: 100%; }}
  [data-testid="stLayoutWrapper"]:has(> div[class*="st-key-card-msg"]) {{ position: sticky; top: 1.25rem; z-index: 1; }}
}}
/* Stack two-column areas earlier than Streamlit's own 640px break, so nothing gets squeezed */
@media (max-width: 1000px) {{
  .st-key-hero [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; gap: 1.75rem; }}
  .st-key-hero [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    flex: 1 1 100%; width: 100%; min-width: 100%; }}
}}
@media (max-width: 900px) {{
  [data-testid="stHorizontalBlock"]:has(div[class*="st-key-card-verdict"]) {{ flex-wrap: wrap; }}
  [data-testid="stHorizontalBlock"]:has(div[class*="st-key-card-verdict"]) > [data-testid="stColumn"] {{
    flex: 1 1 100%; width: 100%; min-width: 100%; }}
  /* show the verdict before the message */
  [data-testid="stColumn"]:has(div[class*="st-key-card-verdict"]) {{ order: -1; }}
}}
@media (max-width: 640px) {{
  div[class*="st-key-card"] {{ padding: 1.2rem 1.15rem; }}
  .mx-section {{ margin-top: 2.5rem; }}
}}
</style>
"""

# Kept for compatibility; the hero no longer uses a decorative trace.
HERO_TRACE = ""


# ----------------------------------------------------------------------------- charts
def style_fig(fig, height: int | None = None, legend_bottom: bool = False):
    """Apply the MATRIX chart style to a Plotly figure."""
    fig.update_layout(
        font=dict(family="DM Sans, system-ui, sans-serif", color=TEXT, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=40, b=8),
        title_font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif", size=15, color=TEXT),
        legend=dict(title=None, orientation="h", yanchor="bottom",
                    y=-0.32 if legend_bottom else 1.02, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT)),
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER, font=dict(color=TEXT, family="DM Sans")),
    )
    # automargin: make room for long labels instead of clipping them
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="#DCCBD6", linecolor=BORDER, automargin=True,
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#DCCBD6", linecolor=BORDER, automargin=True,
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    if not (fig.layout.title and fig.layout.title.text):
        fig.update_layout(margin=dict(t=14))          # no empty band above untitled charts
    if any(tr.type == "bar" for tr in fig.data):
        fig.update_layout(barcornerradius=6, bargap=0.32)
    if height:
        fig.update_layout(height=height)
    return fig


PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
