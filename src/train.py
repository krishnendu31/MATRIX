"""Train MATRIX and produce every number in the README.

    python -m src.train            # full run (about 1-3 minutes)
    python -m src.train --fast     # model + test metrics only (seconds)

Outputs (in models/):
    matrix.joblib            the trained, calibrated model used by the app
    report.json              all headline metrics
    model_comparison.csv     MATRIX vs uncalibrated LR vs Naive Bayes vs Linear SVM (5-fold CV + test)
    calibration.csv          reliability curve, raw vs calibrated
    global_importance.csv    strongest spam / safe signals overall
    test_predictions.csv     held-out messages with scores (false-positive explorer)
    agreement.csv            LIME-SHAP agreement (+ LIME stability) per test message
    faithfulness.csv         deletion test: score drop when top SHAP / LIME / random words are removed
    counterfactuals.csv      counterfactual search result per flagged or missed message
    robustness.csv           edit/attack results on spam and normal messages, with and without de-obfuscation
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from .data import load_sms
from .explain import (agreement, counterfactual, lime_stability, lime_units,
                      shap_units)
from .model import MatrixModel, build_word_vocab, make_vectorizer
from .preprocess import normalise, pretty_token
from .robustness import evaluate_robustness

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models"
SEED = 42


def scores(y, p, threshold=0.5) -> dict:
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "roc_auc": roc_auc_score(y, p),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def ece(y, p, bins=10) -> float:
    """Expected calibration error: average gap between confidence and reality."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


class TextPipeline:
    """A comparison model that reads text exactly like MATRIX does (same placeholders,
    same de-obfuscation vocabulary, same TF-IDF settings), so only the classifier differs."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, texts, labels):
        self.vocab = build_word_vocab(texts)
        self.pipe = make_pipeline(make_vectorizer(), self.estimator)
        self.pipe.fit([normalise(t, True, self.vocab) for t in texts], labels)
        return self

    def proba(self, texts):
        return self.pipe.predict_proba([normalise(t, True, self.vocab) for t in texts])[:, 1]


def comparison_models() -> dict:
    """Factories for the models in the proposal. Each call returns a fresh, unfitted model."""
    return {
        "MATRIX (Logistic Regression + Platt)": lambda: MatrixModel(robust=True),
        "Logistic Regression, uncalibrated": lambda: TextPipeline(
            LogisticRegression(C=10.0, max_iter=5000, solver="liblinear", random_state=SEED)),
        "Naive Bayes": lambda: TextPipeline(MultinomialNB(alpha=0.1)),
        # vectoriser is fit inside each calibration fold, as in MATRIX
        "Linear SVM (Platt)": lambda: _SvmPipeline(),
    }


class _SvmPipeline(TextPipeline):
    def __init__(self):
        super().__init__(None)

    def fit(self, texts, labels):
        self.vocab = build_word_vocab(texts)
        self.pipe = CalibratedClassifierCV(
            make_pipeline(make_vectorizer(), LinearSVC(C=1.0, random_state=SEED)),
            method="sigmoid", cv=5)
        self.pipe.fit([normalise(t, True, self.vocab) for t in texts], labels)
        return self


def run(fast: bool = False, lime_messages: int = 300, robust_messages: int = 150) -> dict:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    df = load_sms()
    tr, te = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    Xtr, ytr = tr["text"].tolist(), tr["label"].to_numpy()
    Xte, yte = te["text"].tolist(), te["label"].to_numpy()
    print(f"Data: {len(df)} messages ({df['label'].mean():.1%} spam), "
          f"{df.attrs.get('duplicates_removed', 0)} duplicates removed. "
          f"Train {len(tr)} / test {len(te)}")

    model = MatrixModel(robust=True).fit(Xtr, ytr)
    p = model.proba(Xte)
    report = {
        "dataset": {"messages": len(df), "spam_share": float(df["label"].mean()),
                    "duplicates_removed": int(df.attrs.get("duplicates_removed", 0)),
                    "train": len(tr), "test": len(te)},
        "test": scores(yte, p, model.threshold),
        "threshold": model.threshold,
        "seed": SEED,
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__},
    }
    joblib.dump(model, OUT / "matrix.joblib")

    pred = pd.DataFrame({"text": Xte, "label": yte, "prob": p,
                         "pred": (p >= model.threshold).astype(int)})
    pred["outcome"] = np.select(
        [(pred.label == 1) & (pred.pred == 1), (pred.label == 0) & (pred.pred == 1),
         (pred.label == 1) & (pred.pred == 0)],
        ["true positive", "false positive", "false negative"], "true negative")
    pred.to_csv(OUT / "test_predictions.csv", index=False)

    # ---------------------------------------------------- global importance
    coef = pd.DataFrame({"feature": model.feature_names, "weight": model.coef})
    Xv = model.vectorize_docs([model.normalise(t) for t in Xte])
    # mean |SHAP| over the test set, counting each feature only where it appears
    present = Xv.copy()
    present.data = np.abs(model.coef[present.indices] * (present.data - model.mu[present.indices]))
    mean_abs = np.asarray(present.sum(axis=0)).ravel() / max(len(Xte), 1)
    coef["mean_abs_shap"] = mean_abs
    coef["direction"] = np.where(coef.weight > 0, "suspicious", "safe")
    coef["label"] = coef.feature.map(pretty_token)
    coef.sort_values("mean_abs_shap", ascending=False).to_csv(
        OUT / "global_importance.csv", index=False)

    if fast:
        _save(report, t0)
        return report

    # ---------------------------------------------------- model comparison
    # 5-fold cross-validation on the TRAINING data only (same folds for every model),
    # then each model is refit on all training data and scored once on the test set.
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = list(cv.split(Xtr, ytr))
    Xtr_arr = np.asarray(Xtr, dtype=object)
    rows = []
    for name, make in comparison_models().items():
        fold_scores = []
        for a, b in folds:
            m = make().fit(list(Xtr_arr[a]), ytr[a])
            fold_scores.append(scores(ytr[b], m.proba(list(Xtr_arr[b]))))
        fs = pd.DataFrame(fold_scores)
        # the MATRIX model is the one already trained above: reuse it, do not retrain
        final = model if name.startswith("MATRIX") else make().fit(Xtr, ytr)
        s = scores(yte, final.proba(Xte))
        row = {"model": name,
               "cv_f1_mean": fs.f1.mean(), "cv_f1_std": fs.f1.std(ddof=0),
               "cv_precision_mean": fs.precision.mean(), "cv_recall_mean": fs.recall.mean(),
               "cv_fpr_mean": fs.false_positive_rate.mean(),
               "test_precision": s["precision"], "test_recall": s["recall"],
               "test_f1": s["f1"], "test_fpr": s["false_positive_rate"],
               "test_roc_auc": s["roc_auc"],
               "test_tp": s["tp"], "test_fp": s["fp"], "test_tn": s["tn"], "test_fn": s["fn"],
               "explainable_exactly": "Logistic Regression" in name}
        rows.append(row)
        if name.startswith("MATRIX"):
            report["cv"] = {k: row[k] for k in ("cv_f1_mean", "cv_f1_std", "cv_precision_mean",
                                                "cv_recall_mean", "cv_fpr_mean")}
        print(f"  {name:38s} CV F1 {row['cv_f1_mean']:.3f} +/- {row['cv_f1_std']:.3f}"
              f" | test F1 {s['f1']:.3f} FPR {s['false_positive_rate']:.4f}")
    pd.DataFrame(rows).to_csv(OUT / "model_comparison.csv", index=False)
    report["comparison"] = rows

    # ---------------------------------------------------- calibration
    raw = model.raw_proba(Xte)
    report["calibration"] = {
        "brier_raw": brier_score_loss(yte, raw), "brier_calibrated": brier_score_loss(yte, p),
        "ece_raw": ece(yte, raw), "ece_calibrated": ece(yte, p)}
    cal = []
    for name, probs in [("raw", raw), ("calibrated", p)]:
        fr, mp = calibration_curve(yte, probs, n_bins=8, strategy="quantile")
        cal += [{"model": name, "mean_predicted": a, "fraction_spam": b} for a, b in zip(mp, fr)]
    pd.DataFrame(cal).to_csv(OUT / "calibration.csv", index=False)

    # ---------------------------------------------------- LIME vs SHAP agreement
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xte), size=min(lime_messages, len(Xte)), replace=False)
    ag, faith = [], []
    for n, i in enumerate(idx):
        units = model.tokenize(Xte[i])
        if len(units) < 3:
            continue
        s = shap_units(model, units)
        lm = lime_units(model, units, n_samples=500, seed=int(i))
        a = agreement(units, s.values, lm.values, k=5)
        row = {"text": Xte[i], "label": int(yte[i]), "prob": float(p[i]),
               "n_words": len(units), "topk_overlap": a["topk_overlap"],
               "spearman": a["spearman"], "sign_agreement": a["sign_agreement"],
               "lime_fidelity_r2": lm.score, "lime_stability": float("nan")}
        if n < 100:   # LIME re-run with 3 seeds: how repeatable is it? (subset, it is slow)
            row["lime_stability"] = lime_stability(model, units, seeds=(1, 2, 3))
        ag.append(row)
        f = deletion_test(model, units, s.values, lm.values, rng)
        if f:
            faith.append(f)
    ag = pd.DataFrame(ag)
    ag.to_csv(OUT / "agreement.csv", index=False)
    report["lime_shap_agreement"] = {
        "messages": len(ag), "top5_overlap_mean": ag.topk_overlap.mean(),
        "spearman_mean": ag.spearman.mean(), "sign_agreement_mean": ag.sign_agreement.mean(),
        "lime_fidelity_r2_mean": ag.lime_fidelity_r2.mean(),
        "lime_stability_mean": ag.lime_stability.mean(),
        "share_low_agreement": float((ag.topk_overlap < 0.6).mean()),
        "top5_overlap_spam": ag[ag.label == 1].topk_overlap.mean(),
        "top5_overlap_ham": ag[ag.label == 0].topk_overlap.mean()}
    faith = pd.DataFrame(faith)
    faith.to_csv(OUT / "faithfulness.csv", index=False)
    report["faithfulness"] = {"messages": len(faith), "k": 3,
                              **{c: float(faith[c].mean()) for c in faith.columns}}

    # ---------------------------------------------------- counterfactual stats
    cf_rows = []
    for t, y, pp in zip(Xte, yte, p):
        if pp < model.threshold and y == 0:
            continue    # normal messages correctly passed: nothing to change
        cf = counterfactual(model, model.tokenize(t))
        cf_rows.append({"text": t, "label": int(y), "prob": float(pp),
                        "flagged": bool(pp >= model.threshold), "flipped": cf.flipped,
                        "words_removed": len(cf.steps),
                        "prob_after": cf.probs[-1], "candidates_tested": cf.candidates_tested})
    cfd = pd.DataFrame(cf_rows)
    cfd.to_csv(OUT / "counterfactuals.csv", index=False)
    fl, missed = cfd[cfd.flagged], cfd[~cfd.flagged]
    report["counterfactuals"] = {
        "search": "greedy removal tested on the model at every step, then pruning; max 10 words",
        "flagged_messages": len(fl), "flip_found": float(fl.flipped.mean()),
        "median_words_removed": float(fl[fl.flipped].words_removed.median()),
        "share_flipped_with_1_word": float((fl.flipped & (fl.words_removed == 1)).mean()),
        "missed_spam": len(missed), "missed_spam_flip_found": float(missed.flipped.mean())
        if len(missed) else None}

    # ---------------------------------------------------- robustness ablation
    plain = MatrixModel(robust=False).fit(Xtr, ytr)
    spam_i = [i for i, y in enumerate(yte) if y == 1][:robust_messages]
    ham_all = [i for i, y in enumerate(yte) if y == 0]
    ham_i = list(np.random.default_rng(SEED).choice(ham_all, size=min(robust_messages * 2,
                                                                         len(ham_all)),
                                                    replace=False))
    r_idx = spam_i + ham_i
    rob = evaluate_robustness({"MATRIX (de-obfuscation on)": model,
                               "Ablation (de-obfuscation off)": plain},
                              [Xte[i] for i in r_idx], yte[r_idx])
    pd.DataFrame(rob).to_csv(OUT / "robustness.csv", index=False)
    report["robustness"] = [{k: r[k] for k in ("attack", "model", "message_class", "n_tested",
                                                "label_kept", "explanation_top3_kept")}
                            for r in rob]
    report["ablation_clean_test_f1_without_deobfuscation"] = scores(
        yte, plain.proba(Xte))["f1"]

    _save(report, t0)
    return report


def deletion_test(model, units, shap_vals, lime_vals, rng, k: int = 3) -> dict | None:
    """Faithfulness: remove the k words each method says push hardest towards the verdict
    (and k random words, averaged over 5 draws), re-run the model, and measure how much
    its confidence in its own verdict falls. A faithful explanation should beat random."""
    from .preprocess import units_to_doc
    toks = list(dict.fromkeys(u.token for u in units))
    if len(toks) < 2 * k:
        return None
    p0 = float(model.proba_docs([units_to_doc(units)])[0])
    sign = 1 if p0 >= model.threshold else -1

    def score(removed):
        keep = [u.token not in removed for u in units]
        doc = units_to_doc(units, keep)
        return float(model.proba_docs([doc])[0]), float(model.logit_docs([doc])[0])

    def top(vals):
        agg = defaultdict(float)
        for u, v in zip(units, vals):
            agg[u.token] += v
        return {t for t, _ in sorted(agg.items(), key=lambda kv: -sign * kv[1])[:k]}

    z0 = float(model.logit_docs([units_to_doc(units)])[0])
    out = {}
    for name, removed in [("shap", top(shap_vals)), ("lime", top(lime_vals))]:
        pr, z = score(removed)
        out[f"drop_prob_{name}"] = sign * (p0 - pr)
        out[f"drop_logodds_{name}"] = sign * (z0 - z)
    rand = [score(set(rng.choice(toks, size=k, replace=False))) for _ in range(5)]
    out["drop_prob_random"] = float(np.mean([sign * (p0 - pr) for pr, _ in rand]))
    out["drop_logodds_random"] = float(np.mean([sign * (z0 - z) for _, z in rand]))
    return out

def _save(report, t0):
    report["runtime_seconds"] = round(time.time() - t0, 1)
    with open(OUT / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    t = report["test"]
    print(f"\nMATRIX on held-out test set: precision {t['precision']:.3f}  recall {t['recall']:.3f}"
          f"  F1 {t['f1']:.3f}  FPR {t['false_positive_rate']:.4f}  AUC {t['roc_auc']:.4f}")
    print(f"Saved to {OUT}  ({report['runtime_seconds']}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="model + test metrics only")
    a = ap.parse_args()
    run(fast=a.fast)
