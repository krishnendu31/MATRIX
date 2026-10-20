"""Shared test fixtures.

The unit tests train a tiny model on a few made-up messages so they run in seconds and do
not need the dataset. That model only exercises the code; its numbers mean nothing.
"""
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import MatrixModel  # noqa: E402

SPAM = ["URGENT! You have WON a {m} cash prize. Call {p} now to claim",
        "FREE entry to win a {m} voucher. Text WIN to {s}",
        "Congratulations you are awarded a holiday! Reply YES to {s} cost 150p per msg",
        "Your mobile account is selected for a bonus {m}. Visit www.claimnow.com",
        "Final attempt to contact you! Claim your prize, call {p} landline only"]
HAM = ["Hey are we still on for dinner tonight? I can pick you up at 7",
       "Ok I will call you when I reach home",
       "Can you send me the notes from class tomorrow",
       "Mum says thanks for the cake, see you on sunday",
       "I will not be late, traffic is fine. Meet at the station at 8",
       "Did you watch the match? That last goal was crazy"]


def make_messages(n=300, seed=0):
    rng = random.Random(seed)
    texts, labels = [], []
    for _ in range(n):
        spam = rng.random() < 0.3
        t = rng.choice(SPAM if spam else HAM).format(
            m=rng.choice(["£1000", "£500"]), p="0906" + str(rng.randint(1000000, 9999999)),
            s=str(rng.randint(80000, 89999)))
        texts.append(t + " " + rng.choice(["", "now", "thanks", "x", "free"]))
        labels.append(int(spam))
    return texts, labels


@pytest.fixture(scope="session")
def model():
    texts, labels = make_messages()
    return MatrixModel(robust=True).fit(texts, labels)


@pytest.fixture(scope="session")
def spam_text():
    return "URGENT! You have WON a £1000 cash prize. Call 09061234567 now to claim"


@pytest.fixture(scope="session")
def ham_text():
    return "Hey are we still on for dinner tonight? I can pick you up at 7"
