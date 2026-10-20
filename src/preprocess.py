"""Text preparation for MATRIX.

The text is split into "units". A unit is one piece of the original message (a word,
a URL, a money amount, a phone number...) together with:
  * its character span(s) in the original text, so we can highlight it later, and
  * a normalised token that the model actually sees.

Keeping both lets every explanation point back at the exact characters the user typed.

Normalisation does three things:
  1. Replaces URLs, e-mails, money amounts, phone numbers, short codes and plain numbers
     with placeholder tokens (__url__, __money__ ...). The model learns "contains a phone
     number" rather than memorising one particular number.
  2. Undoes common obfuscation tricks (when robust=True): leetspeak such as "fr3e" or
     "c@sh", stretched letters such as "freeeee", spaced-out words such as "f r e e", and
     invisible zero-width characters hidden inside words.
  3. Lower-cases everything.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Token patterns (order matters: the first matching alternative wins)
# ---------------------------------------------------------------------------
_URL = r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|co\.uk|uk|biz|info|ly|io)(?:/\S*)?"
_EMAIL = r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"
_MONEY = (
    r"[£$€]\s?\d[\d,]*(?:\.\d+)?(?:k|m)?"          # £1,000  $50  €2.5k
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:pounds|gbp|usd|dollars|euros)\b"
    r"|\b\d+(?:\.\d+)?p\b"                          # 150p (UK pence pricing)
)
_PHONE = r"\+?\d[\d\-\s]{5,}\d"                     # long digit runs, may include spaces
_NUMBER = r"\b\d+(?:[.,]\d+)?\b"
# A word needs at least one letter; letters may be accented ("café"). [^\W\d_] = any letter.
# Invisible zero-width characters inside a word ("fr<zero-width space>ee") are kept in the word here and
# removed during normalisation, so they cannot be used to split a trigger word in two.
_ZW = "\u200b\u200c\u200d\u2060\ufeff"   # zero-width space/joiners, word joiner, BOM
_WORD = (rf"(?:[^\W_]|[@$!{_ZW}])*[^\W\d_](?:[^\W_]|[@$!'{_ZW}])*"
         r"|(?:[^\W_]|[@$])+")
_ZW_DROP = str.maketrans("", "", _ZW)

TOKEN_RE = re.compile(
    rf"(?P<url>{_URL})|(?P<email>{_EMAIL})|(?P<money>{_MONEY})|(?P<phone>{_PHONE})"
    rf"|(?P<number>{_NUMBER})|(?P<word>{_WORD})",
    re.IGNORECASE,
)

LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                          "7": "t", "@": "a", "$": "s", "!": "i"})

PLACEHOLDERS = {
    "__url__": "a web link",
    "__email__": "an e-mail address",
    "__money__": "a money amount",
    "__phone__": "a phone number",
    "__shortcode__": "a short code to text",
    "__num__": "a number",
}


@dataclass
class Unit:
    """One explainable piece of a message."""
    text: str                      # original surface text (may join several spans)
    token: str                     # normalised token the model sees
    spans: list = field(default_factory=list)   # [(start, end), ...] in the original
    kind: str = "word"             # url / email / money / phone / number / word


def _destretch(w: str, vocab) -> str:
    """freeeee -> free, wooon -> won, prizeeee -> prize.

    Runs of 3+ identical letters are collapsed. If the model's vocabulary is known we pick
    whichever reading it recognises (keep a double letter, or collapse to one); otherwise
    we keep two letters, which is right for the most common spam words (free, good, too).
    """
    if not re.search(r"(.)\1{2,}", w):
        return w
    two = re.sub(r"(.)\1{2,}", r"\1\1", w)
    one = re.sub(r"(.)\1{2,}", r"\1", w)
    if vocab is not None:
        for cand in (two, one):
            if cand in vocab:
                return cand
    return two


def _normalise_word(word: str, robust: bool, vocab=None) -> str:
    if robust:
        word = word.translate(_ZW_DROP)
    w = word.lower().strip("'!")
    if not w:
        w = word.lower()
    if robust:
        letters = sum(c.isalpha() for c in w)
        tricks = sum(c in "0134578@$!" for c in w)
        if letters >= 2 and tricks >= 1:
            w = w.translate(LEET_MAP)
        w = _destretch(w, vocab)
    return w


def _digits(s: str) -> int:
    return sum(c.isdigit() for c in s)


def tokenize(text: str, robust: bool = True, vocab=None) -> list[Unit]:
    """Split a raw message into explainable units."""
    units: list[Unit] = []
    for m in TOKEN_RE.finditer(text):
        kind = m.lastgroup
        surface = m.group(0)
        span = (m.start(), m.end())
        if kind == "url":
            tok = "__url__"
        elif kind == "email":
            tok = "__email__"
        elif kind == "money":
            tok = "__money__"
        elif kind == "phone":
            n = _digits(surface)
            tok = "__phone__" if n >= 7 else ("__shortcode__" if n >= 4 else "__num__")
        elif kind == "number":
            n = _digits(surface)
            tok = "__phone__" if n >= 7 else ("__shortcode__" if n >= 4 else "__num__")
            kind = "phone" if n >= 4 else "number"
        else:
            tok = _normalise_word(surface, robust, vocab)
        units.append(Unit(text=surface, token=tok, spans=[span], kind=kind))

    if robust:
        units = _merge_spaced_letters(units, text, vocab)
    return units


def _merge_spaced_letters(units: list[Unit], text: str, vocab=None) -> list[Unit]:
    """Join runs like 'f r e e' or 'F.R.E.E' into one unit.

    Single letters are also normal SMS shorthand ("u r a star"), so a run is only joined
    when the result is a word the model knows (3+ letters). Without a vocabulary we only
    join runs of 4+ letters, which ordinary shorthand rarely produces.
    """
    out: list[Unit] = []
    i = 0
    while i < len(units):
        j = i
        while (j < len(units) and units[j].kind == "word" and len(units[j].text) == 1
               and units[j].text.isalpha()):
            # letters must be separated only by spaces / dots / dashes
            if j > i:
                gap = text[units[j - 1].spans[-1][1]:units[j].spans[0][0]]
                if not re.fullmatch(r"[\s.\-_*]{1,2}", gap):
                    break
            j += 1
        joined = "".join(u.token for u in units[i:j])
        ok = (joined in vocab) if vocab is not None else (j - i >= 4)
        if j - i >= 3 and ok:
            run = units[i:j]
            out.append(Unit(
                text=" ".join(u.text for u in run),
                token="".join(u.token for u in run),
                spans=[s for u in run for s in u.spans],
                kind="word",
            ))
            i = j
        else:
            out.append(units[i])
            i += 1
    return out


def normalise(text: str, robust: bool = True, vocab=None) -> str:
    """Raw text -> space-separated normalised tokens (what the vectoriser sees)."""
    return " ".join(u.token for u in tokenize(text, robust, vocab))


def units_to_doc(units: list[Unit], keep: list[bool] | None = None) -> str:
    """Rebuild a model document from units, optionally dropping some of them."""
    if keep is None:
        return " ".join(u.token for u in units)
    return " ".join(u.token for u, k in zip(units, keep) if k)


def pretty_token(tok: str) -> str:
    """Human-readable name for a token or bigram, used in charts and reasons."""
    return " ".join(PLACEHOLDERS.get(t, t) for t in tok.split(" "))
