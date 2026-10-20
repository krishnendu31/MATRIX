"""MATRIX: Every threat leaves a trace.

Streamlit interface with four pages:
  * Overview          - what MATRIX does, the model at a glance
  * Message Analyzer  - verdict, calibrated probability, uncertainty, highlighted words,
                        plain reason, SHAP vs LIME, score breakdown, counterfactual, robustness
  * Analyst Dashboard - metrics, confusion matrix, calibration, error explorer, global
                        signals, explanation quality, robustness, model comparison
  * About             - dataset, methodology, explainability, limitations, privacy, stack

The ML lives in src/, the interface in ui/. This file only loads the model and routes pages.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src import trust_lab
from ui import pages
from ui.components import html
from ui.theme import AMBER_C, CSS, icon

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

st.set_page_config(page_title="MATRIX · Explainable message checker", page_icon="🌸",
                   layout="wide", initial_sidebar_state="collapsed")
html(CSS)   # page-wide stylesheet


# =============================================================================
# Loading (cached; never trains: training is an explicit, separate step)
# =============================================================================
TRAIN_HELP = """
**How to fix it** (in a terminal, inside the project folder, with the virtual environment active):

1. Make sure the dataset is in `data/` (`SMSSpamCollection`, see `data/README.md`).
2. Train the model: `python -m src.train`
3. Restart the app: `streamlit run app.py`
"""


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner="Loading the MATRIX model…", max_entries=2)
def load_model(path_str: str, stamp: float):
    """Returns (model, problem) where problem is None or a message for the user.
    The file's path and modification time are part of the cache key, so a model retrained
    while the app is running is picked up on the next page load."""
    path = Path(path_str)
    if not path.exists():
        has_data = any((ROOT / "data" / f).exists() for f in ("SMSSpamCollection", "spam.csv"))
        why = ("No trained model was found in `models/`." if has_data else
               "No trained model was found in `models/`, and the dataset is missing from `data/`.")
        return None, why
    try:
        m = joblib.load(path)
    except Exception as ex:  # usually a model saved with a different scikit-learn version
        return None, (f"The saved model could not be loaded ({type(ex).__name__}). It was "
                      "probably trained with different package versions; retrain it.")
    import sklearn
    saved = getattr(m, "sklearn_version", None)
    if saved != sklearn.__version__:
        return None, (f"The saved model was trained with scikit-learn {saved or 'unknown'}, "
                      f"but {sklearn.__version__} is installed. Retrain it so predictions "
                      "are reliable.")
    return m, None


@st.cache_resource(show_spinner=False)
def warm_up_charts() -> bool:
    """Plotly builds its first figure slowly (~1 s: templates and validators load once per
    process). Doing it here, once per server and on the first page load, keeps that cost out
    of the first Analyst Dashboard visit, where it used to keep the previous page's leftover
    elements on screen (faded) until the run finished."""
    import plotly.express as px
    px.bar(x=[0, 1], y=[0, 1]).to_json()
    return True


@st.cache_resource(show_spinner=False)
def chart_library_warmup() -> str:
    """Browser side: the first Plotly chart after a fresh page load makes the browser load and
    run Streamlit's 4.6 MB chart library, which takes ~3 s. Until that finishes the browser
    keeps the previous page's leftover elements on screen. This script asks the browser to load
    the same module while it is idle on the first page, so the first chart no longer waits.
    It imports the exact file Streamlit itself uses (found in the installed package, so it
    follows upgrades); if that file cannot be found, nothing is emitted."""
    js_dir = Path(st.__file__).parent / "static" / "static" / "js"
    found = sorted(js_dir.glob("PlotlyChart.*.js"))
    if not found:
        return ""
    url = f"./static/js/{found[0].name}"
    # requestIdleCallback with a 2 s timeout: runs when the browser is idle, and at the latest
    # after 2 s (idle callbacks can be postponed indefinitely in background tabs).
    return ("<script>(function () { var go = function () { "
            f"import('{url}').catch(function () {{}}); }};"
            " if (window.requestIdleCallback) { window.requestIdleCallback(go, {timeout: 2000}); }"
            " else { setTimeout(go, 1000); } })();</script>")


ARTIFACTS = ["test_predictions", "global_importance", "model_comparison", "calibration",
             "agreement", "robustness", "faithfulness", "counterfactuals"]


@st.cache_data(show_spinner=False, max_entries=2)
def load_artifacts(models_dir: str, stamp: float):
    """Evaluation outputs from src/train.py. `stamp` (report modification time) refreshes
    the cache after retraining."""
    out = {}
    MODELS = Path(models_dir)
    for name in ARTIFACTS:
        f = MODELS / f"{name}.csv"
        try:
            # Only empty cells are missing values. pandas' defaults would also turn real words
            # and messages such as "nan", "NA" or "null" into NaN (e.g. the vocabulary word "nan").
            out[name] = pd.read_csv(f, keep_default_na=False, na_values=[""]) if f.exists() else None
        except Exception:
            out[name] = None
    f = MODELS / "report.json"
    try:
        out["report"] = json.loads(f.read_text()) if f.exists() else {}
    except Exception:
        out["report"] = {}
    return out


# =============================================================================
# Header + navigation
# =============================================================================
PAGES = {"Overview": "Overview", "Analyzer": "Message Analyzer", "Trust": "Trust Lab",
         "Dashboard": "Analyst Dashboard", "About": "About & Methodology"}
NAV_ICONS = {"Overview": "home", "Analyzer": "search", "Trust": "science", "Dashboard": "monitoring",
             "About": "menu_book"}

if st.session_state.get("page") not in PAGES:
    st.session_state["page"] = "Overview"
# "page" is the single source of truth. Two navigation widgets mirror it: a one-line pill bar
# on wide screens and a dropdown on narrow ones (CSS shows exactly one), so the navigation
# never wraps onto two lines.
st.session_state["nav_d"] = st.session_state["page"]
st.session_state["nav_m"] = st.session_state["page"]


def _navigate(widget_key: str):
    st.session_state["page"] = st.session_state[widget_key]


with st.container(key="topbar"):
    brand, nav = st.columns([1, 3], vertical_alignment="center", gap="medium")
    with brand:
        html(f'<div class="mx-brand"><div class="mx-logo">{icon("trace", 22, "#fff")}</div>'
             f'<div><div class="mx-name">MATRIX</div>'
             f'<div class="mx-tag">Every threat leaves a trace.</div></div></div>')
    with nav:
        st.segmented_control("Navigation", list(PAGES), key="nav_d", required=True,
                             format_func=lambda p: f":material/{NAV_ICONS[p]}: {PAGES[p]}",
                             on_change=_navigate, args=("nav_d",),
                             label_visibility="collapsed")
        st.selectbox("Go to page", list(PAGES), key="nav_m", format_func=PAGES.get,
                     on_change=_navigate, args=("nav_m",), label_visibility="collapsed")

model_file = MODELS / "matrix.joblib"
model, problem = load_model(str(model_file), _mtime(model_file))
warm_up_charts()
art = load_artifacts(str(MODELS), _mtime(MODELS / "report.json"))
page = st.session_state["page"]

# Every page renders into one placeholder. When the user switches page, the placeholder is
# emptied first, so the previous page disappears at once. Otherwise Streamlit keeps showing the
# previous page's remaining elements (faded, as "stale") until the new run has fully finished,
# which on a first dashboard visit takes a few seconds while the browser starts its chart
# library. Re-runs on the same page (e.g. pressing Analyze) are not cleared, so nothing flickers.
page_slot = st.empty()
if st.session_state.get("_rendered_page") != page:
    page_slot.empty()
st.session_state["_rendered_page"] = page

with page_slot.container():
    if problem and page in ("Analyzer", "Trust", "Dashboard"):
        with st.container(border=True, key="card-problem"):
            html(f'<div class="mx-card-title">{icon("alert", 20, AMBER_C)}The model is not ready</div>')
            st.markdown(problem)   # MATRIX's own message, not user input
            st.markdown(TRAIN_HELP)
    elif page == "Overview":
        pages.overview(art, model)
    elif page == "Analyzer":
        pages.analyzer(model)
    elif page == "Trust":
        trust_lab.page(model)
    elif page == "Dashboard":
        pages.dashboard(model, art)
    else:
        pages.about(art)

html('<div class="mx-footer"><span><b>MATRIX</b> · Team CharmedAI · Women in AI Innovation '
     'Hackathon 2026 · Challenge 05</span><span>A decision aid: “suspicious” is a calibrated '
     'statistical signal, never proof.</span></div>')

if chart_library_warmup():
    st.html(chart_library_warmup(), unsafe_allow_javascript=True)
