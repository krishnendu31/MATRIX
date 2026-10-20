"""Robustness test: do the label and the explanation survive small edits?

Spammers disguise trigger words ("fr3e", "f r e e", "freeeee", typos, synonyms), and real
messages vary in harmless ways (capital letters, a different link, a friendly greeting).
We edit messages and check:

  * label stability       - is the message still classified the same way?
  * explanation stability - are the top words of the explanation still the same words?

Two kinds of edits:
  * word attacks change the 2 words that drive THIS message's verdict most (top SHAP words
    in the direction of the verdict): leetspeak, spacing, stretching, typo, synonym;
  * message edits change the whole message: capitalization, link swap, defanged link
    ("www[.]site[.]com"), an added greeting.

Spam AND normal messages are tested: a robust filter keeps catching disguised spam and
also does not start flagging normal messages after harmless changes.

The same edits are run against a model with de-obfuscation switched off (robust=False),
so we can show what that step actually buys us.
"""
from __future__ import annotations

import random
import re

import numpy as np

from .explain import STOPWORDS, shap_units
from .preprocess import _URL

SYNONYMS = {
    "free": "complimentary", "win": "get", "won": "got", "winner": "lucky one",
    "prize": "reward", "cash": "money", "call": "ring", "urgent": "important",
    "claim": "collect", "now": "today", "text": "message", "txt": "message",
    "reply": "respond", "mobile": "phone", "award": "gift", "guaranteed": "assured",
    "congratulations": "well done", "offer": "deal", "stop": "end",
}
LEET = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$"}
URL_RE = re.compile(_URL, re.IGNORECASE)


def leet(w: str) -> str:
    """free -> fr3e, prize -> pr1ze, cash -> c@sh (every other swappable letter)."""
    out, n = [], 0
    for c in w:
        if c.lower() in LEET:
            out.append(LEET[c.lower()] if n % 2 == 0 else c)
            n += 1
        else:
            out.append(c)
    return "".join(out)


def spaced(w: str) -> str:
    return " ".join(w)


def stretch(w: str) -> str:
    vowels = [k for k, c in enumerate(w) if c.lower() in "aeiou"]
    k = vowels[-1] if vowels else len(w) - 1
    return w[:k + 1] + w[k] * 3 + w[k + 1:]


def typo(w: str, rng: random.Random) -> str:
    if len(w) < 3:
        return w + w[-1]
    k = rng.randrange(len(w) - 1)
    return w[:k] + w[k + 1] + w[k] + w[k + 2:]


def synonym(w: str) -> str:
    return SYNONYMS.get(w.lower(), w)


# ---------------------------------------------------------------- message edits
def capitalization(text: str) -> str:
    """Whole message in capitals (or in lower case if it already is all capitals)."""
    return text.lower() if text == text.upper() else text.upper()


def url_swap(text: str, rng: random.Random) -> str:
    """Replace every link with a different, random-looking short link."""
    def new(_):
        return "http://bit.ly/" + "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789")
                                          for _ in range(6))
    return URL_RE.sub(new, text)


def url_defang(text: str) -> str:
    """'www.site.com' -> 'www[.]site[.]com', 'http' -> 'hxxp' (a common way to hide links)."""
    return URL_RE.sub(lambda m: m.group(0).replace(".", "[.]").replace("http", "hxxp"), text)


def greeting(text: str) -> str:
    return "Hi, just so you know: " + text


WORD_ATTACKS = ["leetspeak", "spacing", "stretching", "typo", "synonym"]
MESSAGE_EDITS = ["capitalization", "url_swap", "url_defang", "greeting"]
ATTACKS = WORD_ATTACKS + MESSAGE_EDITS
ATTACK_LABELS = {
    "leetspeak": "Leetspeak (fr3e)", "spacing": "Spaced letters (f r e e)",
    "stretching": "Stretched letters (freeee)", "typo": "Typo (swapped letters)",
    "synonym": "Synonym (free → complimentary)", "capitalization": "Capital letters",
    "url_swap": "Different link", "url_defang": "Defanged link (www[.]x[.]com)",
    "greeting": "Added greeting",
}


def _verdict_sign(model, units) -> int:
    """+1 if the model flags the message, -1 if not: the direction its explanation points."""
    from .preprocess import units_to_doc
    return 1 if model.proba_docs([units_to_doc(units)])[0] >= model.threshold else -1


def attack(text: str, model, kind: str, n_words: int = 2, seed: int = 0) -> str:
    """Apply one edit. Word attacks disguise the n words that push hardest towards this
    message's verdict (words only: not numbers or links)."""
    rng = random.Random(seed)
    if kind == "capitalization":
        return capitalization(text)
    if kind == "url_swap":
        return url_swap(text, rng)
    if kind == "url_defang":
        return url_defang(text)
    if kind == "greeting":
        return greeting(text)
    units = model.tokenize(text)
    s = shap_units(model, units)
    sign = _verdict_sign(model, units)
    order = [i for i in np.argsort(-sign * s.values)
             if units[i].kind == "word" and len(units[i].spans) == 1
             and units[i].text.isalpha() and len(units[i].text) >= 3
             and units[i].token not in STOPWORDS and sign * s.values[i] > 0][:n_words]
    edits = []
    for i in order:
        w = units[i].text
        new = {"leetspeak": leet, "spacing": spaced, "stretching": stretch,
               "synonym": synonym}.get(kind, lambda x: typo(x, rng))(w)
        edits.append((units[i].spans[0], new))
    out = text
    for (a, b), new in sorted(edits, reverse=True):
        out = out[:a] + new + out[b:]
    return out


def top_tokens(model, text: str, k: int = 3, sign: int = 1) -> set[str]:
    """Top-k tokens pushing in direction `sign` (+1 suspicious, -1 safe), as the model reads
    them. With de-obfuscation, 'fr3e' is read as 'free', so the explanation can still point
    at the same word after an attack; without it, 'fr3e' is an unknown token and the
    explanation loses that word."""
    units = model.tokenize(text)
    s = shap_units(model, units)
    agg = {}
    for i, u in enumerate(units):
        agg[u.token] = agg.get(u.token, 0.0) + s.values[i]
    return {t for t, v in sorted(agg.items(), key=lambda kv: -sign * kv[1])[:k] if sign * v > 0}


def evaluate_robustness(models: dict, texts: list[str], labels, n_words: int = 2) -> list[dict]:
    """models: {"name": model}. Edits are computed against the first (reference) model so
    every model faces exactly the same edited messages. Only messages the edit actually
    changed, and that the model got right before the edit, are scored."""
    labels = np.asarray(labels)
    ref = next(iter(models.values()))
    rows = []
    for kind in ATTACKS:
        attacked = [attack(t, ref, kind, n_words, seed=j) for j, t in enumerate(texts)]
        changed = np.array([a != t for a, t in zip(attacked, texts)])
        for name, m in models.items():
            p0, p1 = m.proba(texts), m.proba(attacked)
            y0, y1 = p0 >= m.threshold, p1 >= m.threshold
            for cls, cls_name in [(1, "spam"), (0, "normal")]:
                in_cls = labels == cls
                correct = in_cls & (y0 == bool(cls))
                idx = np.where(correct & changed)[0]
                sign = 1 if cls == 1 else -1
                overlap = []
                for i in idx:
                    a = top_tokens(m, texts[i], sign=sign)
                    if a:
                        overlap.append(len(a & top_tokens(m, attacked[i], sign=sign)) / len(a))
                kept = float((y1[idx] == y0[idx]).mean()) if len(idx) else float("nan")
                ex = idx[0] if len(idx) else None
                rows.append({
                    "attack": kind, "attack_label": ATTACK_LABELS[kind], "model": name,
                    "message_class": cls_name,
                    "n_messages": int(in_cls.sum()), "n_tested": int(len(idx)),
                    "correct_before": float(correct.sum() / max(in_cls.sum(), 1)),
                    "label_kept": kept,
                    # kept for the existing dashboard: share of caught spam still caught
                    "still_detected_after": kept if cls == 1 else float("nan"),
                    "mean_prob_change": float((p1 - p0)[idx].mean()) if len(idx) else float("nan"),
                    "explanation_top3_kept": float(np.mean(overlap)) if overlap else float("nan"),
                    "example_before": texts[ex] if ex is not None else "",
                    "example_after": attacked[ex] if ex is not None else "",
                })
    return rows


def probe(model, text: str, n_words: int = 2) -> list[dict]:
    """Run every edit on ONE message and report what changed (for the analyzer page)."""
    p0 = float(model.proba([text])[0])
    y0 = p0 >= model.threshold
    sign = 1 if y0 else -1
    base_top = top_tokens(model, text, sign=sign)
    out = []
    for j, kind in enumerate(ATTACKS):
        new = attack(text, model, kind, n_words, seed=j)
        if new == text:
            continue                     # e.g. no link to swap
        p1 = float(model.proba([new])[0])
        top = top_tokens(model, new, sign=sign)
        out.append({"attack": kind, "attack_label": ATTACK_LABELS[kind], "text": new,
                    "prob_before": p0, "prob_after": p1,
                    "label_changed": (p1 >= model.threshold) != y0,
                    "top3_kept": (len(base_top & top) / len(base_top)) if base_top else float("nan")})
    return out
