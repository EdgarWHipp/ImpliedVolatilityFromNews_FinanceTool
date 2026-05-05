from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iv_news.cov.rmt import cov_rmt, marchenko_pastur_edges, sample_cov


def make_factor_returns(
    n_assets: int = 50,
    n_obs: int = 750,
    n_factors: int = 3,
    factor_vol: float = 0.02,
    idio_vol: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n_obs, n_factors)) * factor_vol
    B = rng.standard_normal((n_assets, n_factors))
    eps = rng.standard_normal((n_obs, n_assets)) * idio_vol
    R = F @ B.T + eps
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(R, columns=cols)


def test_mp_edges():
    lo, hi = marchenko_pastur_edges(q=0.1)
    assert hi == pytest.approx((1 + np.sqrt(0.1)) ** 2)
    assert lo == pytest.approx((1 - np.sqrt(0.1)) ** 2)


def test_rmt_recovers_factor_structure():
    rets = make_factor_returns(n_assets=50, n_obs=750, n_factors=3, seed=42)
    res = cov_rmt(rets)

    # Top 3 eigenvalues should sit well above MP bulk.
    top3 = res.eigenvalues_raw[:3]
    assert (top3 > res.lambda_plus * 1.5).all(), top3
    # Cleaned matrix must remain PSD.
    eig = np.linalg.eigvalsh(res.cov_clean.values)
    assert eig.min() > -1e-10
    # Bulk must be flattened (replaced by their mean).
    bulk_clean = res.eigenvalues_clean[res.n_signal:]
    assert np.allclose(bulk_clean, bulk_clean.mean(), atol=1e-10)
    # We identified a non-trivial number of signal eigenvalues.
    assert 1 <= res.n_signal <= 10


def test_rmt_pure_noise_collapses_to_identity_like():
    rng = np.random.default_rng(1)
    # Pure noise: T much larger than N so MP bulk is tight.
    rets = pd.DataFrame(rng.standard_normal((2000, 30)) * 0.01)
    res = cov_rmt(rets)
    # With pure noise we expect at most 1-2 spurious "signal" eigenvalues.
    assert res.n_signal <= 3
    # Off-diagonal correlation magnitude should be small after cleaning.
    corr_clean = res.cov_clean.values / np.sqrt(
        np.outer(np.diag(res.cov_clean.values), np.diag(res.cov_clean.values))
    )
    off = corr_clean[~np.eye(30, dtype=bool)]
    assert np.abs(off).mean() < 0.05


def test_sample_cov_matches_numpy():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.standard_normal((200, 10)))
    expected = np.cov(df.values, rowvar=False, ddof=1)
    got = sample_cov(df).values
    np.testing.assert_allclose(got, expected, atol=1e-12)
