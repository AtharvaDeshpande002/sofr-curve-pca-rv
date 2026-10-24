"""How fast does a residual mean-revert, and is it stationary at all?

AR(1):  e(t) = c + phi * e(t-1) + noise.   phi close to 1 = slow, phi close to 0 = fast.
Half-life = the number of days for a gap to halve:  phi ** h = 0.5   ->   h = -ln 2 / ln phi.
This is the discrete version of an Ornstein-Uhlenbeck process (mean-reversion speed kappa = -ln phi).

ADF (augmented Dickey-Fuller) tests "the series has a unit root" (= does NOT mean-revert). A small
p-value (say < 0.05) rejects that, i.e. evidence the residual does revert.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def ar1_phi(x: np.ndarray | pd.Series) -> float:
    """OLS slope of x(t) on x(t-1) with a constant."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) < 10:
        return np.nan
    y, lag = x[1:], x[:-1]
    lag_c = lag - lag.mean()
    denom = (lag_c ** 2).sum()
    return float((lag_c * (y - y.mean())).sum() / denom) if denom > 0 else np.nan


def half_life(phi: float) -> float:
    """Days for a deviation to halve. Infinite if the series does not revert (phi >= 1 or phi <= 0)."""
    if not np.isfinite(phi) or phi <= 0 or phi >= 1:
        return np.inf
    return float(-np.log(2) / np.log(phi))


def adf_pvalue(x: np.ndarray | pd.Series) -> float:
    """ADF p-value with automatic lag choice (AIC). NaN if the series is too short or constant."""
    x = pd.Series(x).dropna()
    if len(x) < 30 or x.std() == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(adfuller(x.to_numpy(), autolag="AIC")[1])
