"""Shared fixtures: a small synthetic SR3 market (so tests run without the licensed data) and the real config."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sofr_pca.config import load_config  # noqa: E402
from sofr_pca.data import build_curve  # noqa: E402

MONTHS = ["03", "06", "09", "12"]


def synthetic_market(n_days: int = 700, n_contracts: int = 20, seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quarterly contracts expiring every ~63 business days; prices follow level/slope/curvature random walks
    plus a small mean-reverting idiosyncratic piece per contract (so residuals exist)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    codes, months, last = [], [], []
    for i in range(n_contracts):
        y, q = 2019 + i // 4, i % 4
        codes.append(f"SR3{'HMUZ'[q]}{str(y)[2:]}")
        months.append(f"{y}-{MONTHS[q]}")
        last.append(dates[min(40 + 63 * i, n_days - 1)] if 40 + 63 * i < n_days else dates[-1] + pd.offsets.BDay(63 * i))
    contracts = pd.DataFrame({"contract": codes, "ref_month": months, "first_trade": dates[0], "last_trade": last,
                              "quarterly": True})
    lvl = np.cumsum(rng.normal(0, 4, n_days)); slope = np.cumsum(rng.normal(0, 1, n_days))
    curv = np.cumsum(rng.normal(0, 0.3, n_days))
    rows = []
    for i, c in enumerate(codes):
        idio = np.zeros(n_days)
        for t in range(1, n_days):
            idio[t] = 0.7 * idio[t - 1] + rng.normal(0, 0.4)
        x = i / n_contracts
        rate_bp = 300 + lvl + slope * (x - 0.5) * 4 + curv * (4 * (x - 0.5) ** 2 - 0.3) * 4 + idio
        for t, d in enumerate(dates):
            if d <= last[i]:
                rows.append((d, c, round(100 - rate_bp[t] / 100, 4), np.nan, np.nan))
    settles = pd.DataFrame(rows, columns=["date", "contract", "settle", "volume", "open_interest"])
    return settles, contracts


@pytest.fixture(scope="session")
def market():
    return synthetic_market()


@pytest.fixture(scope="session")
def curve(market):
    s, k = market
    return build_curve(s, k, n_positions=8)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw_available(cfg):
    return (cfg["paths"]["raw"] / cfg["data"]["settles_file"]).exists()
