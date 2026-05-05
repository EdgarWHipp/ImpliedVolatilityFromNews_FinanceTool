"""Stage 1 end-to-end: fetch returns, compute sample cov + RMT-cleaned cov,
print eigenvalue summary."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from iv_news.cov.rmt import cov_rmt, marchenko_pastur_edges, sample_cov
from iv_news.data.returns import build_returns
from iv_news.data.universe import tickers as universe_tickers


def main() -> None:
    tks = universe_tickers()
    print(f"Fetching returns for {len(tks)} tickers...")
    rets = build_returns(tks)
    print(f"Returns shape: {rets.shape}  (T x N)")
    print(f"  Date range: {rets.index.min().date()}  ->  {rets.index.max().date()}")
    print(f"  Tickers kept: {len(rets.columns)}  (dropped if >5% missing)")

    T, N = rets.shape
    q = N / T
    lo, hi = marchenko_pastur_edges(q)
    print(f"\nMarchenko-Pastur:  q = N/T = {q:.4f}   bulk in [{lo:.3f}, {hi:.3f}]")

    cov_s = sample_cov(rets)
    cond_s = np.linalg.cond(cov_s.values)
    print(f"\nSample covariance:    cond = {cond_s:,.1f}")

    res = cov_rmt(rets)
    cond_c = np.linalg.cond(res.cov_clean.values)
    print(f"RMT-cleaned cov:      cond = {cond_c:,.1f}")
    print(f"Signal eigenvalues kept: {res.n_signal}")
    print(f"Top 5 raw eigenvalues:   {np.round(res.eigenvalues_raw[:5], 3)}")
    print(f"Top 5 clean eigenvalues: {np.round(res.eigenvalues_clean[:5], 3)}")


if __name__ == "__main__":
    main()
