"""Loading the UCI SMS Spam Collection.

Accepts either format:
  * data/SMSSpamCollection  (original UCI file: "label<TAB>text" per line)
  * data/spam.csv           (Kaggle copy: columns v1=label, v2=text, latin-1)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_sms(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    uci = data_dir / "SMSSpamCollection"
    kaggle = data_dir / "spam.csv"
    if uci.exists():
        rows = []
        with open(uci, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if "\t" in line:
                    label, text = line.split("\t", 1)
                    rows.append((label.strip(), text))
        df = pd.DataFrame(rows, columns=["label", "text"])
    elif kaggle.exists():
        df = pd.read_csv(kaggle, encoding="latin-1")
        df = df.rename(columns={"v1": "label", "v2": "text"})[["label", "text"]]
    else:
        raise FileNotFoundError(
            f"No dataset found in {data_dir}. Put the UCI file 'SMSSpamCollection' "
            "or Kaggle 'spam.csv' there (see README, 'Dataset')."
        )
    df = df.dropna()
    df["label"] = df["label"].str.lower().map({"ham": 0, "spam": 1})
    df = df.dropna().astype({"label": int})
    n_before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df.attrs["duplicates_removed"] = n_before - len(df)
    return df
