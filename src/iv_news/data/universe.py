from __future__ import annotations

import pandas as pd

from iv_news.config import DATA_DIR


def load_universe() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "universe.csv")
    df["ticker"] = df["ticker"].str.upper()
    return df


def tickers() -> list[str]:
    return load_universe()["ticker"].tolist()
