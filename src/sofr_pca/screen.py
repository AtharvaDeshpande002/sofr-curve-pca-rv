"""Today's rich/cheap screen: the 'use it on the desk tomorrow' output.

For the latest date in the data:
    1. fit PCA + residual regressions on the last ``window`` days (everything known at today's close),
    2. z-score each structure's residual against its own trailing mean / std,
    3. rank, and attach what a trader needs to act: residual in bp, half-life, PCA-neutral weights,
       the out-of-sample hit rate of the rule on that structure (before and after costs), and a target / stop in bp.

Sign convention (price terms, what you see on a screen):
    z > 0  -> the structure's RATE is above fair value -> its PRICE is too low  -> CHEAP -> buy the package
    z < 0  -> the structure's RATE is below fair value -> its PRICE is too high -> RICH  -> sell the package
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CurveData
from .pca import pca_neutral_weights
from .residuals import fit_residual_model, rolling_z
from .structures import Structure


def screen_today(curve: CurveData, universe: list[Structure], window: int, z_window: int, exit_z: float,
                 stop_z: float, trades: pd.DataFrame | None = None, n_components: int = 3) -> pd.DataFrame:
    """Rank every structure by today's z-score. Returns one row per structure (all of them)."""
    lv = curve.levels.iloc[-window:]
    ch = curve.changes.iloc[-window:]
    model = fit_residual_model(lv, ch, universe, n_components)
    resid = model.residuals(lv, universe)
    z_all = rolling_z(resid, z_window)
    today = curve.dates[-1]
    mu = resid.iloc[-z_window:].mean()
    sd = resid.iloc[-z_window:].std()
    hit = {}
    if trades is not None and len(trades):
        hit = trades.groupby("structure").apply(lambda g: pd.Series({"hit": (g.net_bp > 0).mean(), "hit_g": (g.gross_bp > 0).mean(), "n": len(g)}),
                                                include_groups=False).to_dict("index")
    rows = []
    for u in universe:
        z = float(z_all[u.name].iloc[-1])
        dev = float(resid[u.name].iloc[-1] - mu[u.name])           # bp away from its own recent mean
        w_pca = pca_neutral_weights(model.pca, u.positions, u.leg_weights)
        h = hit.get(u.name, {})
        rows.append(dict(
            date=today.date(), structure=u.name, family=u.family,
            contracts="/".join(curve.contracts.loc[today, u.positions]),
            z=round(z, 2), residual_bp=round(dev, 2),
            view="cheap (buy package)" if z > 0 else "rich (sell package)",
            half_life_days=round(float(model.params.loc[u.name, "half_life"]), 1),
            pca_neutral_weights=" : ".join(f"{x:.2f}" for x in w_pca),
            oos_hit_gross=round(h["hit_g"], 2) if h else np.nan, oos_hit_net=round(h["hit"], 2) if h else np.nan,
            oos_trades=int(h["n"]) if h else 0,
            target_bp=round(max(abs(z) - exit_z, 0) * float(sd[u.name]), 2),
            stop_bp=round(max(stop_z - abs(z), 0) * float(sd[u.name]), 2),
        ))
    return pd.DataFrame(rows).sort_values("z").reset_index(drop=True)


def top_rich_cheap(table: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n richest (most negative z) and n cheapest (most positive z) structures."""
    rich = table.nsmallest(n, "z")
    cheap = table.nlargest(n, "z").iloc[::-1]
    return pd.concat([rich, cheap]).reset_index(drop=True)
