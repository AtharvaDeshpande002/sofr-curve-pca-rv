"""From raw SR3 settlement prices to a clean, roll-safe curve.

Three objects come out of this module, all indexed by date with columns = constant positions 1..N:

* ``levels``    - the rate (bp) of whichever contract sits in each position that day.
* ``contracts`` - which fixed contract sits in each position that day (e.g. position 1 = SR3Z26).
* ``changes``   - the daily change (bp) of THAT fixed contract since the previous session.

Why the care with ``changes``: on a roll day every position jumps to a later contract. Differencing the
generic ``levels`` series would book the gap between two different contracts as a "move" - a false
signal of several bp. Instead we difference each fixed contract and only then place the result on the
position the contract occupies today. A daily change is therefore always a real price change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

QUARTERLY_MONTHS = {"03", "06", "09", "12"}


@dataclass
class CurveData:
    """Everything downstream code needs about the curve."""

    levels: pd.DataFrame       # date x position, rate in bp
    changes: pd.DataFrame      # date x position, fixed-contract daily change in bp (first row NaN)
    contracts: pd.DataFrame    # date x position, contract code
    rates: pd.DataFrame        # date x contract, rate in bp (fixed contracts, used for trade P&L)
    last_trade: pd.Series      # contract -> last trading day (Timestamp)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.levels.index


def price_to_bp(price: pd.Series | pd.DataFrame | float):
    """SR3 quotes 100 - rate. Rate in bp = (100 - price) * 100.  Example: 96.125 -> 387.5 bp."""
    return (100.0 - price) * 100.0


def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two input files described in data/README.md and keep quarterly contracts only."""
    raw = cfg["paths"]["raw"]
    settles = pd.read_csv(raw / cfg["data"]["settles_file"], parse_dates=["date"])
    contracts = pd.read_csv(raw / cfg["data"]["contracts_file"], parse_dates=["first_trade", "last_trade"])
    # Quarterly (H/M/U/Z) contracts only: serial months are thin and would break the 3-month spacing.
    contracts = contracts[contracts["ref_month"].str[5:7].isin(QUARTERLY_MONTHS)].copy()
    settles = settles[settles["contract"].isin(contracts["contract"])].copy()
    return settles, contracts


def build_curve(settles: pd.DataFrame, contracts: pd.DataFrame, n_positions: int,
                start: str | None = None, end: str | None = None, min_oi: float = 0.0) -> CurveData:
    """Assign contracts to constant positions each day and build roll-safe levels and changes.

    Position 1 = the first quarterly contract whose last trading day is on or after today.
    A date is kept only if all ``n_positions`` contracts have a settle that day.
    """
    contracts = contracts.sort_values("ref_month").reset_index(drop=True)
    last_trade = contracts.set_index("contract")["last_trade"]
    # Wide table of rates on fixed contracts: one column per contract.
    rates = price_to_bp(settles.pivot_table(index="date", columns="contract", values="settle"))
    rates = rates.sort_index()
    oi = settles.pivot_table(index="date", columns="contract", values="open_interest").reindex(rates.index)

    # Daily change of each FIXED contract (previous session to this session). Done before any
    # position mapping, so a roll can never create a fake jump.
    fixed_changes = rates.diff()

    order = contracts["contract"].tolist()
    lt = contracts["last_trade"].to_numpy()
    rows_lvl, rows_chg, rows_ctr, keep = [], [], [], []
    for date in rates.index:
        # First quarterly contract still trading today, then the next n-1 in calendar order.
        first = int(np.searchsorted(lt, np.datetime64(date), side="left"))
        codes = order[first:first + n_positions]
        if len(codes) < n_positions:
            continue
        lvl = rates.loc[date, codes].to_numpy(dtype=float)
        if np.isnan(lvl).any():
            continue                      # a leg has no settle today: skip the date
        if min_oi > 0:
            leg_oi = oi.loc[date, codes].to_numpy(dtype=float)
            if np.nanmin(np.where(np.isnan(leg_oi), np.inf, leg_oi)) < min_oi:
                continue                  # liquidity filter: a leg is too thin
        rows_lvl.append(lvl)
        rows_chg.append(fixed_changes.loc[date, codes].to_numpy(dtype=float))
        rows_ctr.append(codes)
        keep.append(date)

    cols = list(range(1, n_positions + 1))
    idx = pd.DatetimeIndex(keep, name="date")
    levels = pd.DataFrame(rows_lvl, index=idx, columns=cols)
    changes = pd.DataFrame(rows_chg, index=idx, columns=cols)
    ctr = pd.DataFrame(rows_ctr, index=idx, columns=cols)

    if start:
        mask = idx >= pd.Timestamp(start)
        levels, changes, ctr = levels[mask], changes[mask], ctr[mask]
    if end:
        mask = levels.index <= pd.Timestamp(end)
        levels, changes, ctr = levels[mask], changes[mask], ctr[mask]
    # The first kept row has no previous session inside the sample: blank its change.
    changes.iloc[0] = np.nan
    # A change is only valid if the previous session in the wide table is also the previous kept date;
    # if dates were skipped (missing legs), the fixed-contract diff would span a gap - blank those rows.
    prev_kept = levels.index.to_series().shift(1)
    prev_any = pd.Series(rates.index, index=rates.index).shift(1).reindex(levels.index)
    gap = prev_kept.notna() & (prev_kept != prev_any)
    changes.loc[gap.to_numpy()] = np.nan
    return CurveData(levels=levels, changes=changes, contracts=ctr, rates=rates.loc[:, order], last_trade=last_trade)


def load_curve(cfg: dict) -> CurveData:
    """Convenience: raw files -> CurveData using the config."""
    settles, contracts = load_raw(cfg)
    d = cfg["data"]
    return build_curve(settles, contracts, d["n_positions"], d.get("start_date"), d.get("end_date"),
                       d.get("min_open_interest", 0) or 0)


def roll_days(curve: CurveData) -> pd.Series:
    """True on the first session after the front contract expired (every position shifts one contract)."""
    front = curve.contracts[1]
    return front.ne(front.shift(1)) & front.shift(1).notna()


def regime_labels(dates: pd.DatetimeIndex, regimes: list) -> pd.Series:
    """Label each date with its policy regime from the config (hiking / hold / cutting / 2026)."""
    out = pd.Series("other", index=dates)
    for label, a, b in regimes:
        out[(dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))] = label
    return out
