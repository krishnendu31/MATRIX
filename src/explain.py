"""Explanations for MATRIX predictions.

Four ways of explaining one message:

1. Linear SHAP. What is explained: the raw score (log-odds) of the Logistic Regression,
   BEFORE Platt calibration. For a linear model f(x) = b + w.x, the interventional SHAP
   value of feature j, with the training data as background and features treated as
   independent, is exactly  phi_j = w_j * (x_j - mu_j)  (Lundberg & Lee, 2017), where mu is
   the average training document. This is what shap.LinearExplainer computes; it is exact,
   not sampled, and base + sum(phi) = f(x) holds to rounding error.
   Two things are OUR choices on top of SHAP, and are reported as such:
     * a word-pair feature's value is shared equally between its two words, so each
       word gets one number that can be highlighted;
     * features absent from the message also have SHAP values (w_j * (0 - mu_j)); they
       are summed into `rest` ("words this message does not contain") rather than hidden.
   Platt calibration is a monotonic function of the score, so it never changes the
   ranking of words, but SHAP values are in log-odds, not probability points.
2. LIME (Ribeiro et al., 2016). Randomly hide words, ask the model again, and fit a small
   weighted linear model on "which words were present" -> calibrated probability.
   Implemented here following the original algorithm (cosine-distance kernel, width 25,
   ridge surrogate) so it runs with no extra dependency. LIME is sampled, so it varies a
   little between runs; its fit quality (weighted R^2) is reported with every result.
3. Counterfactual. Repeatedly try removing each remaining word, re-run the real model on
   every candidate, and keep the removal that moves the score furthest towards the other
   label, until the label flips. Then drop any removal that turns out not to be needed.
   This finds a small edit within this search; it is not guaranteed to be the smallest.
4. A plain-language reason built from the SHAP words, grouped into signal families
   (prize offer, urgency, call-to-action ...).
5. Uncertainty notes: when the score, the input or the explanation should be trusted less.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from .preprocess import PLACEHOLDERS, Unit, tokenize, units_to_doc


# =============================================================================
# 1. Linear SHAP, mapped back onto words
# =============================================================================
@dataclass
class ShapResult:
    values: np.ndarray        # one attribution per unit, in log-odds (raw model score)
    base_value: float         # raw score of the average training message
    logit: float              # raw score of this message
    rest: float               # contribution of words NOT in the message (absent features)
    # Exact decomposition: base_value + values.sum() + rest == logit


def shap_units(model, units: list[Unit], keep: list[bool] | None = None) -> ShapResult:
    if keep is None:
        keep = [True] * len(units)
    idx_units = [i for i, k in enumerate(keep) if k]
    toks = [units[i].token for i in idx_units]
    doc = " ".join(toks)
    x = model.vectorize_docs([doc])
    logit = float(model.intercept + x.dot(model.coef)[0])
    base = float(model.intercept + model.mu @ model.coef)

    vocab = model.vectorizer.vocabulary_
    xd = dict(zip(x.indices, x.data))
    # every occurrence of every feature, with the units it covers
    occ = defaultdict(list)
    for p, t in enumerate(toks):
        occ[t].append((idx_units[p],))
        if p + 1 < len(toks):
            occ[t + " " + toks[p + 1]].append((idx_units[p], idx_units[p + 1]))

    vals = np.zeros(len(units))
    covered = 0.0
    for feat, places in occ.items():
        j = vocab.get(feat)
        if j is None:
            continue
        phi = model.coef[j] * (xd.get(j, 0.0) - model.mu[j])
        covered += phi
        share = phi / len(places)
        for unit_ids in places:
            for u in unit_ids:
                vals[u] += share / len(unit_ids)
    rest = (logit - base) - covered
    return ShapResult(values=vals, base_value=base, logit=logit, rest=rest)


# =============================================================================
# 2. LIME
# =============================================================================
@dataclass
class LimeResult:
    values: np.ndarray   # one weight per unit (effect on calibrated probability)
    intercept: float
    score: float         # weighted R^2 of the local surrogate (local fidelity)


def lime_units(model, units: list[Unit], n_samples: int = 500, seed: int = 0,
               kernel_width: float = 25.0) -> LimeResult:
    m = len(units)
    if m == 0:
        return LimeResult(np.zeros(0), float(model.proba_docs([""])[0]), 1.0)
    rng = np.random.default_rng(seed)
    # Same sampling scheme as lime.LimeTextExplainer: choose how many words to drop,
    # then which ones.
    Z = np.ones((n_samples, m), dtype=bool)
    sizes = rng.integers(1, m + 1, size=n_samples - 1)
    for r, s in enumerate(sizes, start=1):
        Z[r, rng.choice(m, size=s, replace=False)] = False
    docs = [units_to_doc(units, row) for row in Z]
    y = model.proba_docs(docs)
    k = Z.sum(axis=1)
    cos_sim = np.sqrt(k / m)
    d = (1.0 - cos_sim) * 100
    w = np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
    reg = Ridge(alpha=1.0, random_state=seed)
    reg.fit(Z.astype(float), y, sample_weight=w)
    return LimeResult(values=reg.coef_, intercept=float(reg.intercept_),
                      score=float(reg.score(Z.astype(float), y, sample_weight=w)))


# =============================================================================
# 3. Counterfactual: a small set of removals, each tested on the real model
# =============================================================================
@dataclass
class CfStep:
    token: str                  # normalised token removed (every occurrence of it)
    unit_ids: list[int]         # which units of the message that is
    prob_before: float          # calibrated P(suspicious) before this removal
    prob_after: float           # ... and after it (re-computed by the real model)


@dataclass
class Counterfactual:
    removed: list[int]          # unit indices removed, in order
    probs: list[float]          # probability after each removal (probs[0] = original)
    flipped: bool               # did the label change?
    steps: list[CfStep] = field(default_factory=list)
    direction: str = "to_safe"  # "to_safe" for flagged messages, else "to_suspicious"
    candidates_tested: int = 0  # how many edited messages the model scored in the search


def counterfactual(model, units: list[Unit], max_steps: int = 10) -> Counterfactual:
    """Remove words until the model's label flips, testing every option on the real model.

    Step 1 (greedy search): at each step, try removing each remaining distinct word (all of
    its occurrences), score every candidate with the model, keep the one that moves the
    probability furthest towards the other label.
    Step 2 (pruning): if the label flipped, try putting each removed word back; keep it back
    if the label stays flipped. The result is a small edit within this search, not a
    guaranteed global minimum.
    Filler words ("to", "a" ...) are never suggested.
    """
    thr = model.threshold
    p0 = float(model.proba_docs([units_to_doc(units)])[0])
    flagged = p0 >= thr
    direction = "to_safe" if flagged else "to_suspicious"
    better = (lambda a, b: a < b) if flagged else (lambda a, b: a > b)
    crossed = (lambda p: p < thr) if flagged else (lambda p: p >= thr)

    by_token = defaultdict(list)
    for i, u in enumerate(units):
        if u.token not in STOPWORDS:
            by_token[u.token].append(i)

    removed_tokens: list[str] = []
    keep = [True] * len(units)
    p, tested = p0, 0
    z = float(model.logit_docs([units_to_doc(units)])[0])
    for _ in range(max_steps):
        options = [t for t in by_token if t not in removed_tokens]
        if not options:
            break
        docs = []
        for t in options:
            k = keep.copy()
            for i in by_token[t]:
                k[i] = False
            docs.append(units_to_doc(units, k))
        # Rank candidates by the raw score: near 0% or 100% many calibrated probabilities
        # round to the same value, while the raw score still tells them apart. Calibration
        # is monotonic, so the order is the same.
        zs = model.logit_docs(docs)
        tested += len(docs)
        j = int(np.argmin(zs) if flagged else np.argmax(zs))
        if not better(zs[j], z):
            break               # no single removal moves the score the right way
        removed_tokens.append(options[j])
        for i in by_token[options[j]]:
            keep[i] = False
        z = float(zs[j])
        p = float(model.calibrate_logit(z)[0])
        if crossed(p):
            break

    flipped = crossed(p)
    if flipped and len(removed_tokens) > 1:
        for t in list(removed_tokens):              # try to put each word back
            trial = [x for x in removed_tokens if x != t]
            k = [True] * len(units)
            for x in trial:
                for i in by_token[x]:
                    k[i] = False
            tested += 1
            if crossed(float(model.proba_docs([units_to_doc(units, k)])[0])):
                removed_tokens = trial

    # Replay the final edit step by step so every number shown is a real model output
    keep, steps, probs, removed = [True] * len(units), [], [p0], []
    for t in removed_tokens:
        before = probs[-1]
        for i in by_token[t]:
            keep[i] = False
        after = float(model.proba_docs([units_to_doc(units, keep)])[0])
        steps.append(CfStep(t, list(by_token[t]), before, after))
        probs.append(after)
        removed += by_token[t]
    return Counterfactual(removed, probs, flipped=crossed(probs[-1]), steps=steps,
                          direction=direction, candidates_tested=tested)


# =============================================================================
# 4. Plain-language reason
# =============================================================================
SIGNAL_FAMILIES = {
    "a prize or reward offer": {
        "won", "win", "winner", "wins", "winning", "prize", "prizes", "cash", "award",
        "awarded", "reward", "free", "bonus", "gift", "voucher", "vouchers", "claim",
        "selected", "guaranteed", "congratulations", "congrats", "lucky", "jackpot",
        "draw", "holiday", "entry", "camcorder", "nokia", "ipod", "tickets",
    },
    "urgency or pressure": {
        "urgent", "now", "immediately", "today", "tonight", "expires", "expiry",
        "final", "last", "limited", "hurry", "asap", "quick", "act", "attempt",
        "2nd", "valid", "only", "important",
    },
    "a request to call, text or click": {
        "call", "text", "txt", "reply", "send", "click", "visit", "dial", "ring",
        "contact", "sms", "__phone__", "__shortcode__", "__url__", "__email__",
        "landline", "line", "msg",
    },
    "money or premium-rate pricing": {
        "__money__", "cost", "costs", "price", "per", "pmin", "ppm", "rate", "charge",
        "charged", "min", "mins", "minute", "week", "wk", "pw", "stop", "subscription",
    },
    "marketing or service language": {
        "customer", "service", "services", "offer", "offers", "unsubscribe", "optout",
        "opt", "tone", "ringtone", "network", "account", "chat", "dating", "mobile",
        "mob", "latest", "new", "club", "uk",
    },
}


# Short function words carry weight in the model but make a poor "reason".
STOPWORDS = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "is", "are", "be", "and", "or",
    "u", "ur", "you", "your", "i", "me", "my", "we", "it", "its", "this", "that", "has",
    "have", "had", "with", "from", "by", "as", "so", "do", "if", "no", "not", "just", "get",
    "got", "go", "will", "can", "all", "our", "us", "he", "she", "they", "but", "was",
}


def family_of(token: str) -> str | None:
    for fam, words in SIGNAL_FAMILIES.items():
        if token in words:
            return fam
    return None


def strength_levels(values: np.ndarray, floor: float = 0.05) -> list[str]:
    """'strong' / 'weak' / 'none' per unit, relative to the strongest push.
    floor: ignore attributions smaller than this in absolute terms (0.05 log-odds for SHAP)."""
    out = []
    peak = np.max(np.abs(values)) if len(values) else 0.0
    for v in values:
        a = abs(v)
        if peak == 0 or a < 0.12 * peak or a < floor:
            out.append("none")
        elif a >= 0.45 * peak:
            out.append("strong")
        else:
            out.append("weak")
    return out


_PUNCT = ".,!?;:'\"()[]"


def _quote(units, ids, limit=3):
    seen, out = set(), []
    for i in ids:
        u = units[i]
        label = PLACEHOLDERS.get(u.token)
        s = label if label else f"“{u.text.strip(_PUNCT) or u.text}”"
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) == limit:
            break
    return ", ".join(out)


def plain_reason(units: list[Unit], shap_vals: np.ndarray, prob: float,
                 threshold: float) -> str:
    suspicious = prob >= threshold
    order = np.argsort(-shap_vals if suspicious else shap_vals)
    sign = 1 if suspicious else -1
    pushing = [i for i in order if sign * shap_vals[i] > 0.05
               and units[i].token not in STOPWORDS]

    if suspicious:
        fam_score, fam_ids, other = Counter(), defaultdict(list), []
        for i in pushing:
            fam = family_of(units[i].token)
            if fam:
                fam_score[fam] += shap_vals[i]
                fam_ids[fam].append(i)
            else:
                other.append(i)
        fams = [f for f, _ in fam_score.most_common()]
        if not fams and not other:
            return "Flagged, but no single word stands out; the score comes from the overall mix."
        parts = []
        if fams:
            parts.append(f"Flagged mainly for {fams[0]} ({_quote(units, fam_ids[fams[0]])})")
            if len(fams) > 1:
                extra = " and ".join(f"{f} ({_quote(units, fam_ids[f], 2)})" for f in fams[1:3])
                parts.append(f"backed up by {extra}")
        else:
            parts.append(f"Flagged mainly for the wording {_quote(units, other)}")
        if other and fams:
            parts.append(f"other pushes come from {_quote(units, other, 2)}")
        return ", ".join(parts) + "."
    else:
        safe = _quote(units, pushing) if pushing else None
        weak = [i for i in np.argsort(-shap_vals) if shap_vals[i] > 0.05
                and units[i].token not in STOPWORDS]
        if prob >= BORDERLINE[0]:
            s = "Not flagged, but close to the line. "
            s += (f"The strongest safe signals are {safe}." if safe else "")
        else:
            s = "Not flagged. "
            s += (f"It reads like a personal message: the strongest safe signals are {safe}."
                  if safe else "Nothing in it looks like typical spam wording.")
        if weak:
            s += f" Weak suspicious signals ({_quote(units, weak, 2)}) were not enough to flag it."
        return s


# =============================================================================
# One-call explanation used by the app
# =============================================================================
def explain_message(model, text: str, lime_samples: int = 500, top_k: int = 5) -> dict:
    units = model.tokenize(text)
    prob = float(model.proba_docs([units_to_doc(units)])[0])
    s = shap_units(model, units)
    lime = lime_units(model, units, n_samples=lime_samples)
    cf = counterfactual(model, units)
    ag = agreement(units, s.values, lime.values, top_k)
    return {
        "text": text,
        "units": units,
        "prob": prob,
        "suspicious": prob >= model.threshold,
        "shap": s,
        # the same base value, as a probability: "an average message scores this"
        "base_prob": float(model.calibrate_logit(s.base_value)[0]),
        "lime": lime,
        "levels": strength_levels(s.values),
        "counterfactual": cf,
        "reason": plain_reason(units, s.values, prob, model.threshold),
        "agreement": ag,
        "uncertainty": uncertainty(model, units, prob, ag, lime.score),
    }


# =============================================================================
# 5. When to trust the result less
# =============================================================================
BORDERLINE = (0.2, 0.8)


def uncertainty(model, units: list[Unit], prob: float, ag: dict, lime_r2: float) -> dict:
    """Plain warnings about this particular result, each computed from the message, the
    model or the explanations. level: 'low' | 'medium' | 'high' uncertainty."""
    notes = []
    words = [u for u in units if u.kind == "word"]
    vocab = model.vectorizer.vocabulary_
    n_known = sum(u.token in vocab for u in words)
    known_share = n_known / len(words) if words else float("nan")

    if not units:
        notes.append(("high", "There are no words, numbers or links to analyse, so the score "
                              "only reflects the model's starting point."))
    if BORDERLINE[0] <= prob <= BORDERLINE[1]:
        notes.append(("high", f"Borderline score ({prob:.0%}). Calibrated means: messages "
                              "scored like this turn out to be spam about this often, so "
                              "either answer is plausible."))
    if words and known_share < 0.5:
        notes.append(("high", f"Only {n_known} of {len(words)} words were seen in training "
                              "(another language, new slang or unusual spelling). The model "
                              "has little to go on."))
    elif 0 < len(units) < 3:
        notes.append(("high", "Only one or two words: far too little evidence for a reliable "
                              "score, however confident the number looks."))
    elif len(units) == 3:
        notes.append(("medium", "Very short message: there is little evidence either way."))
    ov = ag.get("topk_overlap", float("nan"))
    if ov == ov and ov < 0.6:
        notes.append(("medium", "SHAP and LIME point at different words, so treat the word "
                                "highlighting with care."))
    if units and lime_r2 == lime_r2 and lime_r2 < 0.2:
        notes.append(("low", f"LIME's local fit is weak (R² {lime_r2:.2f}); the SHAP view is "
                             "the more reliable one for this message."))
    level = ("high" if any(l == "high" for l, _ in notes) else
             "medium" if any(l == "medium" for l, _ in notes) else "low")
    return {"level": level, "notes": [{"level": l, "text": t} for l, t in notes],
            "known_word_share": known_share, "n_words": len(words)}


# =============================================================================
# Do LIME and SHAP agree?
# =============================================================================
def _top_tokens(units, values, k):
    """Top-k distinct tokens by |attribution| (duplicates merged)."""
    agg = defaultdict(float)
    for u, v in zip(units, values):
        agg[u.token] += v
    ranked = sorted(agg.items(), key=lambda kv: -abs(kv[1]))
    return [t for t, v in ranked if abs(v) > 1e-9][:k], agg


def agreement(units, shap_vals, lime_vals, k: int = 5) -> dict:
    ts, agg_s = _top_tokens(units, shap_vals, k)
    tl, agg_l = _top_tokens(units, lime_vals, k)
    if not ts or not tl:
        return {"topk_overlap": float("nan"), "spearman": float("nan"), "k": k}
    kk = min(len(ts), len(tl))
    shared = set(ts[:kk]) & set(tl[:kk])
    overlap = len(shared) / kk
    # Of the words both methods rank highly, do they agree on the direction?
    sign = (float(np.mean([np.sign(agg_s[t]) == np.sign(agg_l[t]) for t in shared]))
            if shared else float("nan"))
    toks = list(agg_s)
    sa, la = [agg_s[t] for t in toks], [agg_l[t] for t in toks]
    rho = (spearmanr(sa, la).statistic
           if len(toks) > 2 and np.ptp(sa) > 0 and np.ptp(la) > 0 else float("nan"))
    return {"topk_overlap": overlap, "spearman": float(rho), "sign_agreement": sign,
            "k": kk, "shap_top": ts, "lime_top": tl}


def lime_stability(model, units, seeds=(0, 1, 2), n_samples: int = 500, k: int = 5) -> float:
    """How repeatable is LIME? Mean pairwise top-k overlap between runs with different
    random seeds (1.0 = always the same top words)."""
    tops = [set(_top_tokens(units, lime_units(model, units, n_samples, seed=s).values, k)[0])
            for s in seeds]
    pairs = [(a, b) for i, a in enumerate(tops) for b in tops[i + 1:]]
    vals = [len(a & b) / max(len(a | b) and min(len(a), len(b)), 1) for a, b in pairs]
    return float(np.mean(vals)) if vals else float("nan")
