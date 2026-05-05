"""Sanity-check the universe file."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iv_news.data.universe import load_universe


def main() -> None:
    df = load_universe()
    print(f"Universe size: {len(df)}")
    print("\nBy sector:")
    print(df.groupby("sector").size().sort_values(ascending=False).to_string())
    dups = df["ticker"][df["ticker"].duplicated()].tolist()
    if dups:
        raise SystemExit(f"Duplicate tickers: {dups}")


if __name__ == "__main__":
    main()
