"""Cross-check MATRIX's own SHAP and LIME against the reference libraries.

MATRIX computes Linear SHAP and LIME itself (so the app has no heavy dependencies and can
map every attribution back onto the exact characters typed). This script shows that the
numbers match the official `shap` and `lime` packages.

    pip install -r requirements-dev.txt
    python -m src.train
    python notebooks/verify_xai.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

model = joblib.load(ROOT / "models" / "matrix.joblib")
test = pd.read_csv(ROOT / "models" / "test_predictions.csv").sample(50, random_state=0)
docs = [model.normalise(t) for t in test.text]
X = model.vectorize_docs(docs)

# ------------------------------------------------------------------ SHAP
ours = (X.toarray() - model.mu) * model.coef          # phi_j = w_j (x_j - mu_j)
try:
    import shap
    from scipy import sparse
    background = sparse.csr_matrix(model.mu.reshape(1, -1))
    explainer = shap.LinearExplainer(model.lr, background)
    ref = explainer.shap_values(X)
    ref = ref.toarray() if hasattr(ref, "toarray") else np.asarray(ref)
    print(f"SHAP: max |ours - shap.LinearExplainer| = {np.abs(ours - ref).max():.2e}")
    print(f"      base value ours {model.intercept + model.mu @ model.coef:.4f} "
          f"vs shap {float(np.ravel(explainer.expected_value)[0]):.4f}")
except ImportError:
    print("shap not installed: pip install -r requirements-dev.txt")

# ------------------------------------------------------------------ LIME
try:
    from lime.lime_text import LimeTextExplainer
    from src.explain import lime_units

    lime_exp = LimeTextExplainer(class_names=["ham", "spam"], split_expression=r"\s+",
                                 bow=False, random_state=0)
    overlaps = []
    for t, d in zip(test.text[:20], docs[:20]):
        units = model.tokenize(t)
        if len(units) < 3:
            continue
        mine = lime_units(model, units, n_samples=1000, seed=0)
        top_mine = {units[i].token for i in np.argsort(-np.abs(mine.values))[:5]}
        e = lime_exp.explain_instance(
            d, lambda ds: np.c_[1 - model.proba_docs(ds), model.proba_docs(ds)],
            num_features=5, num_samples=1000)
        top_ref = {w for w, _ in e.as_list()}
        overlaps.append(len(top_mine & top_ref) / max(len(top_ref), 1))
    print(f"LIME: top-5 words shared with the lime package: {np.mean(overlaps):.0%} "
          f"(both are random samplers, so 100% is not expected)")
except ImportError:
    print("lime not installed: pip install -r requirements-dev.txt")
