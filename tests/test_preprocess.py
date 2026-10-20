from src.preprocess import normalise, tokenize


def tokens(text, vocab=None):
    return [u.token for u in tokenize(text, robust=True, vocab=vocab)]


def test_placeholders():
    t = tokens("Call 09061701461 or text 85023, win £1,000 at www.prize.com or 150p/msg")
    for ph in ["__phone__", "__shortcode__", "__money__", "__url__"]:
        assert ph in t


def test_deobfuscation():
    assert tokens("fr3e c@sh") == ["free", "cash"]
    assert tokens("freeeee", vocab={"free"}) == ["free"]
    assert tokens("you won a f r e e prize", vocab={"free"}) == ["you", "won", "a", "free", "prize"]


def test_sms_shorthand_is_not_merged():
    # "u r a" is normal texting, not a disguised word
    assert tokens("u r a star", vocab={"free", "star"}) == ["u", "r", "a", "star"]
    assert tokens("u r a star") == ["u", "r", "a", "star"]       # no vocabulary known


def test_accents_kept():
    assert tokens("très bien, café") == ["très", "bien", "café"]


def test_spans_point_at_original_text():
    text = "Claim your f r e e prize now"
    for u in tokenize(text, robust=True, vocab={"free"}):
        pieces = [text[a:b] for a, b in u.spans]
        assert " ".join(pieces) == u.text


def test_without_robust_no_deobfuscation():
    assert normalise("fr3e", robust=False) == "fr3e"


def test_zero_width_characters_cannot_split_words():
    zw = chr(0x200B)   # zero-width space
    assert tokens(f"fr{zw}ee pr{zw}ize") == ["free", "prize"]
    # spans still point at the original characters, invisible ones included
    u = tokenize(f"fr{zw}ee", robust=True)[0]
    assert u.spans == [(0, 5)]
