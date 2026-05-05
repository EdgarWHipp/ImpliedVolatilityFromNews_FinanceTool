"""Random Matrix Theory cleaning for sample covariance matrices.

Following Laloux, Cizeau, Bouchaud, Potters (1999). The sample correlation
matrix of T observations on N assets has eigenvalues that, under the null
of i.i.d. noise, follow the Marchenko-Pastur distribution with edges

    lambda_pm = (1 +/- sqrt(q))**2,   q = N / T.

Eigenvalues inside the bulk [lambda_-, lambda_+] are noise; we replace them
by their mean to preserve trace. Eigenvalues above lambda_+ are signal
(market + sector factors) and are kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    X = returns.values - returns.values.mean(axis=0, keepdims=True)
    T = X.shape[0]
    cov = (X.T @ X) / (T - 1)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def cov_to_corr(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sigma = np.sqrt(np.diag(cov))
    inv = 1.0 / sigma
    corr = cov * np.outer(inv, inv)
    np.fill_diagonal(corr, 1.0)
    return corr, sigma


def corr_to_cov(corr: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return corr * np.outer(sigma, sigma)


def marchenko_pastur_edges(q: float) -> tuple[float, float]:
    s = np.sqrt(q)
    return (1.0 - s) ** 2, (1.0 + s) ** 2


@dataclass
class RMTResult:
    cov_clean: pd.DataFrame
    eigenvalues_raw: np.ndarray
    eigenvalues_clean: np.ndarray
    lambda_plus: float
    n_signal: int


def cov_rmt(returns: pd.DataFrame, keep_market: bool = True) -> RMTResult:
    """Clean the sample covariance via MP bulk replacement.

    Steps: sample cov -> correlation -> eigendecompose -> replace bulk
    eigenvalues with their average (trace preserving) -> reconstruct corr
    -> rescale to covariance.
    """
    T, N = returns.shape
    if T <= N:
        raise ValueError(f"Need T > N for sample cov; got T={T}, N={N}.")
    q = N / T
    _, lam_plus = marchenko_pastur_edges(q)

    cov_s = sample_cov(returns).values
    corr, sigma = cov_to_corr(cov_s)

    eigvals, eigvecs = np.linalg.eigh(corr)
    bulk_mask = eigvals < lam_plus
    if keep_market and bulk_mask.all():
        bulk_mask[-1] = False  # always keep largest

    eigvals_clean = eigvals.copy()
    if bulk_mask.any():
        bulk_mean = eigvals[bulk_mask].mean()
        eigvals_clean[bulk_mask] = bulk_mean

    corr_clean = (eigvecs * eigvals_clean) @ eigvecs.T
    np.fill_diagonal(corr_clean, 1.0)
    corr_clean = 0.5 * (corr_clean + corr_clean.T)

    cov_clean = corr_to_cov(corr_clean, sigma)
    cov_df = pd.DataFrame(cov_clean, index=returns.columns, columns=returns.columns)
    return RMTResult(
        cov_clean=cov_df,
        eigenvalues_raw=eigvals[::-1],
        eigenvalues_clean=eigvals_clean[::-1],
        lambda_plus=lam_plus,
        n_signal=int((~bulk_mask).sum()),
    )
