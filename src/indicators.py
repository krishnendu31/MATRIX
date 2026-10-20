"""Rule-based phishing indicators: observable patterns, NOT a machine-learning prediction.

Why this exists: the model was trained on the UCI SMS Spam Collection (UK, mostly 2000s),
which contains prize / premium-rate spam but no delivery or account phishing (for example,
the word "parcel" never appears). Such messages can therefore score below the threshold even
when they show classic phishing patterns. These rules surface those patterns separately and
never change the model's probability or verdict.
"""
from __future__ import annotations

import re

# Each indicator: (key, label, compiled pattern). Patterns avoid backslash escapes on purpose.
_W = "(?<![A-Za-z])"          # start of a word
_E = "(?![A-Za-z])"           # end of a word
_RULES = [
    ("link", "Contains a web link",
     r"(https?://|www[.]|hxxp|[A-Za-z0-9-]+[.](com|net|org|info|ly|io|co[.]uk|uk|biz|xyz|top)(/|" + _E + ")|"
     r"[A-Za-z0-9-]+" + re.escape("[.]") + r"[A-Za-z]+)"),      # defanged: site[.]com
    ("urgency", "Urgency or a deadline",
     _W + "(urgent|urgently|immediately|asap|now|today|tonight|expires?|expired|final (notice|reminder)|"
          "last chance|act now|within [0-9]+ ?(hours?|hrs?|days?|minutes?|mins?))" + _E),
    ("verify", "Asks you to confirm, verify or update details",
     _W + "(confirm|verify|validate|update|re-?enter|reactivate|unlock)" + _E),
    ("personal", "Mentions personal, account or payment details",
     _W + "(address|details|password|pin|login|log in|identity|id|account|acc0?unt|billing|"
          "payment|card|bank|ssn|passcode)" + _E),
    ("delivery", "Delivery or parcel lure",
     _W + "(parcel|package|delivery|delivered|deliver|courier|postage|redelivery|reschedule|"
          "tracking|shipment)" + _E),
    ("prize", "Prize, reward or free-gift claim",
     _W + "(won|win|winner|prize|reward|claim|cash prize|free gift|gift card|voucher|selected|"
          "congratulations|lottery|jackpot)" + _E),
    ("threat", "Account problem or threat",
     _W + "(suspended|locked|l0cked|blocked|disabled|unusual activity|failed|on hold|"
          "unpaid|overdue|refund)" + _E),
]
_COMPILED = [(k, label, re.compile(p, re.IGNORECASE)) for k, label, p in _RULES]


def phishing_indicators(text: str) -> list[dict]:
    """Every indicator found in the text, with the exact words that triggered it."""
    found = []
    for key, label, rx in _COMPILED:
        hits = []
        for m in rx.finditer(text or ""):
            word = m.group(0)
            if word.lower() not in (h.lower() for h in hits):
                hits.append(word)
        if hits:
            found.append({"key": key, "label": label, "evidence": hits[:4]})
    return found


STRONG = {"prize", "urgency", "verify", "personal", "threat"}


def phishing_caution(indicators: list[dict], suspicious: bool) -> bool:
    """True when the model did NOT flag the message but it shows a link plus at least one other
    pattern, or at least two of: prize claim, urgency, request to confirm/verify, personal or
    payment details, account threat. A caution only: the model's verdict is shown unchanged."""
    keys = {i["key"] for i in indicators}
    if suspicious:
        return False
    return ("link" in keys and len(keys - {"link"}) >= 1) or len(keys & STRONG) >= 2
