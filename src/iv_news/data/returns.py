from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from iv_news.config import DATA_DIR

DEFAULT_START = "2022-01-01"
DEFAULT_END = "2025-04-30"
RETURNS_PATH = DATA_DIR / "returns.parquet"


def fetch_prices(
    tickers: list[str],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.levels[0]})
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.sort_index().dropna(how="all")
    return prices


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def build_returns(
    tickers: list[str],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache: Path = RETURNS_PATH,
    refresh: bool = False,
) -> pd.DataFrame:
    if cache.exists() and not refresh:
        cached = pd.read_parquet(cache)
        if set(tickers).issubset(cached.columns):
            return cached[tickers].dropna(how="all")
    prices = fetch_prices(tickers, start, end)
    rets = to_log_returns(prices)
    rets = rets.dropna(axis=1, thresh=int(0.95 * len(rets)))
    rets = rets.dropna()
    cache.parent.mkdir(parents=True, exist_ok=True)
    rets.to_parquet(cache)
    return rets
