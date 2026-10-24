"""Walk-forward, net-of-cost backtest of PCA residual mean-reversion.

Timeline (no look-ahead):
    1. At the start of each ~3-month block, fit PCA + residual regressions on the previous ``window`` days only.
    2. Freeze those parameters. Every day in the block, compute each structure's z-score at the close (t).
    3. A signal at the close of t is traded at the close (settle) of t+1. Exits work the same way.
    4. At the end of the block, refit on the latest window and repeat.
Only dates after ``oos_start`` are ever traded, so every reported trade is out-of-sample.

Trades are held on FIXED contracts. If the curve rolls while a trade is open, the trade keeps its contracts;
its z-score is read from the structure those contracts form today (one position earlier). A trade whose
front leg reaches its last trading day is closed at that day's settle - you cannot hold an expiring leg.

Rules (from config.yaml): enter |z| > entry_z (bet on reversion), exit |z| < exit_z, stop if z extends
beyond stop_z, time stop = time_stop_mult x half-life (capped). Overlap control: at most ``max_per_region``
open trades per curve region and no two trades sharing a contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .data import CurveData
from .residuals import ResidualModel, fit_residual_model, rolling_z
from .structures import Structure, make_structure, region_of


@dataclass(frozen=True)
class BTParams:
    window: int = 252
    z_window: int = 63
    block_days: int = 63
    oos_start: str = "2020-07-01"
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    time_stop_mult: float = 2.0
    max_hold_days: int = 60
    fixed_hold_days: int = 0            # >0: ignore the half-life time stop and hold exactly this long (rule variant)
    max_per_region: int = 2
    allow_shared_legs: bool = False
    min_half_life: float = 1.0
    max_half_life: float = 40.0
    n_components: int = 3
    tick_bp_per_leg: float = 0.5
    commission_per_contract: float = 2.5
    dollars_per_bp: float = 25.0


def params_from_config(cfg: dict, **overrides) -> BTParams:
    """Collect every backtest parameter from the config (overrides win, e.g. for the robustness grid)."""
    b, m, c = cfg["backtest"], cfg["model"], cfg["costs"]
    p = BTParams(window=cfg["pca"]["window"], z_window=m["z_window"], block_days=b["block_days"], oos_start=b["oos_start"],
                 entry_z=b["entry_z"], exit_z=b["exit_z"], stop_z=b["stop_z"], time_stop_mult=b["time_stop_mult"],
                 max_hold_days=b["max_hold_days"], max_per_region=b["max_per_region"],
                 allow_shared_legs=b["allow_shared_legs"], min_half_life=m["min_half_life"],
                 max_half_life=m["max_half_life"], n_components=cfg["pca"]["n_components"],
                 tick_bp_per_leg=c["tick_bp_per_leg"], commission_per_contract=c["commission_per_contract"],
                 dollars_per_bp=cfg["contract"]["dollars_per_bp"])
    return replace(p, **overrides)


def round_trip_cost_bp(s: Structure, p: BTParams, tick_bp: float | None = None) -> float:
    """Cost of opening AND closing one unit, expressed in bp of the structure.

    Per side, every contract pays ``tick_bp`` (the price of crossing the bid-offer) plus commission.
    Commission in bp = $ / ($ per bp). Two sides per round trip.
    Example, a fly (4 contracts) at 0.5bp + $2.50:  2 x 4 x (0.5 + 2.5/25) = 4.8 bp.
    """
    t = p.tick_bp_per_leg if tick_bp is None else tick_bp
    return 2.0 * s.n_contracts * (t + p.commission_per_contract / p.dollars_per_bp)


@dataclass
class Block:
    """One walk-forward block: a frozen model and the dates it trades."""

    start: pd.Timestamp
    end: pd.Timestamp                    # last date on which this block may OPEN trades
    model: ResidualModel
    z: pd.DataFrame                      # z-scores from start until the block's trades can no longer be open
    fair: pd.DataFrame                   # fair value + trailing residual mean: the level z is measured from


@dataclass
class BacktestResult:
    params: BTParams
    trades: pd.DataFrame
    daily_pnl: pd.Series                 # $ per day, net of costs (costs booked half at entry, half at exit)
    daily_gross: pd.Series
    blocks: list[Block] = field(repr=False, default_factory=list)


def fit_blocks(curve: CurveData, universe: list[Structure], p: BTParams, insample: bool = False) -> list[Block]:
    """Fit one frozen model per block. ``insample=True`` fits a single model on ALL data (for comparison)."""
    dates = curve.dates
    i0 = int(dates.searchsorted(pd.Timestamp(p.oos_start)))
    if i0 < p.window and not insample:
        raise ValueError("oos_start leaves less history than the estimation window")
    full = fit_residual_model(curve.levels, curve.changes.iloc[1:], universe, p.n_components) if insample else None
    blocks = []
    for b in range(i0, len(dates), p.block_days):
        if insample:
            model = full
        else:
            lv = curve.levels.iloc[b - p.window:b]                     # strictly before the block
            ch = curve.changes.iloc[b - p.window:b]
            model = fit_residual_model(lv, ch, universe, p.n_components)
        hi = min(len(dates), b + p.block_days + p.max_hold_days + 5)
        # residuals from the frozen model, starting z_window days early so the trailing stats exist on day one
        lv_hist = curve.levels.iloc[max(0, b - p.z_window):hi]
        resid = model.residuals(lv_hist, universe)
        z = rolling_z(resid, p.z_window).loc[dates[b]:]
        # the level the trade aims back to: model fair value shifted by the residual's trailing mean
        anchor = model.fair_value(lv_hist, universe) + resid.rolling(p.z_window, min_periods=p.z_window // 2).mean()
        blocks.append(Block(start=dates[b], end=dates[min(b + p.block_days, len(dates)) - 1], model=model,
                            z=z, fair=anchor.loc[dates[b]:]))
    return blocks


def stitched(blocks: list[Block], what: str = "z") -> pd.DataFrame:
    """One panel where each date uses the model of the block it belongs to (what a trader saw that day)."""
    parts = [getattr(b, what).loc[b.start:b.end] for b in blocks]
    return pd.concat(parts)


@dataclass
class _Trade:
    structure: Structure
    contracts: list[str]
    weights: list[float]
    side: int                     # +1 = long the structure's rate, -1 = short the structure's rate
    block: int
    signal_date: pd.Timestamp
    entry_z: float
    half_life: float
    region: str
    entry_i: int = -1
    exit_i: int = -1
    exit_reason: str = ""
    exit_signal_z: float = np.nan


def run_backtest(curve: CurveData, universe: list[Structure], p: BTParams, regions: dict,
                 insample: bool = False, blocks: list[Block] | None = None) -> BacktestResult:
    """Simulate the rules day by day over the out-of-sample period."""
    blocks = blocks if blocks is not None else fit_blocks(curve, universe, p, insample)
    dates = curve.dates
    rates = curve.rates.reindex(dates).ffill()
    last_trade = curve.last_trade
    by_name = {u.name: u for u in universe}
    pos_of = [dict(zip(row, range(1, len(row) + 1))) for row in curve.contracts.to_numpy()]
    block_of_date = np.full(len(dates), -1)
    for k, b in enumerate(blocks):
        block_of_date[dates.searchsorted(b.start):dates.searchsorted(b.end) + 1] = k
    i0 = int(dates.searchsorted(blocks[0].start))

    def value(tr: _Trade, i: int) -> float:
        return float(np.dot(tr.weights, rates.iloc[i][tr.contracts].to_numpy()))

    def current_z(tr: _Trade, i: int) -> float:
        """z of the trade's contracts today, read from the structure they form today, under the entry model."""
        first_pos = pos_of[i].get(tr.contracts[0])
        if first_pos is None:
            return np.nan
        name = make_structure(tr.structure.family, first_pos, tr.structure.weights).name
        z = blocks[tr.block].z
        if name not in z.columns or dates[i] not in z.index:
            return np.nan
        return float(z.at[dates[i], name])

    open_tr: list[_Trade] = []
    pend_in: list[_Trade] = []
    pend_out: list[_Trade] = []
    done: list[_Trade] = []

    for i in range(i0, len(dates)):
        t = dates[i]
        # 1) execute yesterday's exit signals at today's settle
        for tr in pend_out:
            tr.exit_i = i
            open_tr.remove(tr)
            done.append(tr)
        pend_out = []
        # 2) execute yesterday's entry signals at today's settle
        for tr in pend_in:
            tr.entry_i = i
            open_tr.append(tr)
        pend_in = []
        # 3) exits decided at today's close
        for tr in open_tr:
            if any(last_trade[c] <= t for c in tr.contracts):       # expiring leg: close at today's settle
                tr.exit_i, tr.exit_reason = i, "expiry"
                continue
            z = current_z(tr, i)
            tr.exit_signal_z = z
            signed = -tr.side * z                                    # entry z expressed as a positive number
            held = i - tr.entry_i
            if not np.isfinite(z):
                tr.exit_reason = "model"
            elif signed < p.exit_z:
                tr.exit_reason = "target"
            elif signed > p.stop_z:
                tr.exit_reason = "stop"
            elif held >= (p.fixed_hold_days or min(p.max_hold_days, max(1, round(p.time_stop_mult * tr.half_life)))):
                tr.exit_reason = "time"
            else:
                continue
            if i + 1 < len(dates):
                pend_out.append(tr)
            else:
                tr.exit_i = i
        for tr in [x for x in open_tr if x.exit_i == i]:
            open_tr.remove(tr)
            done.append(tr)
        # 4) new entry signals at today's close (traded tomorrow)
        k = block_of_date[i]
        if k < 0 or i + 1 >= len(dates):
            continue
        zrow = blocks[k].z.loc[t]
        hl = blocks[k].model.params["half_life"]
        live = [tr for tr in open_tr if tr not in pend_out]
        busy = {c for tr in live for c in tr.contracts}
        region_count: dict[str, int] = {}
        for tr in live:
            region_count[tr.region] = region_count.get(tr.region, 0) + 1
        cands = zrow[(zrow.abs() > p.entry_z) & (zrow.abs() < p.stop_z)].abs().sort_values(ascending=False).index
        for name in cands:
            s = by_name[name]
            if not (p.min_half_life <= hl[name] <= p.max_half_life):
                continue
            codes = [curve.contracts.iat[i, q - 1] for q in s.positions]
            if any(last_trade[c] <= dates[min(i + 3, len(dates) - 1)] for c in codes):
                continue                                             # a leg expires within 3 sessions
            if not p.allow_shared_legs and busy.intersection(codes):
                continue
            reg = region_of(s, regions)
            if region_count.get(reg, 0) >= p.max_per_region:
                continue
            z = float(zrow[name])
            pend_in.append(_Trade(structure=s, contracts=codes, weights=s.leg_weights, side=-int(np.sign(z)),
                                  block=k, signal_date=t, entry_z=z, half_life=float(hl[name]), region=reg))
            busy.update(codes)
            region_count[reg] = region_count.get(reg, 0) + 1
    for tr in open_tr:                                               # still open at the end of the data
        tr.exit_i, tr.exit_reason = len(dates) - 1, "open at end"
        done.append(tr)

    return _assemble(done, dates, rates, p, blocks)


def _assemble(done: list[_Trade], dates: pd.DatetimeIndex, rates: pd.DataFrame, p: BTParams,
              blocks: list[Block]) -> BacktestResult:
    """Turn simulated trades into a trade table and daily $ P&L series (gross and net)."""
    oos = dates[dates >= blocks[0].start]
    gross = pd.Series(0.0, index=oos)
    net = pd.Series(0.0, index=oos)
    rows = []
    for tr in done:
        if tr.entry_i < 0 or tr.exit_i <= tr.entry_i:
            continue
        path = rates.iloc[tr.entry_i:tr.exit_i + 1][tr.contracts].to_numpy() @ np.array(tr.weights)
        daily = tr.side * np.diff(path) * p.dollars_per_bp          # $ per day while the trade is open
        cost_bp = round_trip_cost_bp(tr.structure, p)
        cost_usd = cost_bp * p.dollars_per_bp
        d_idx = dates[tr.entry_i + 1:tr.exit_i + 1]
        gross.loc[d_idx] += daily
        net.loc[d_idx] += daily
        net.loc[dates[tr.entry_i]] -= cost_usd / 2
        net.loc[dates[tr.exit_i]] -= cost_usd / 2
        g_bp = float(tr.side * (path[-1] - path[0]))
        rows.append(dict(structure=tr.structure.name, family=tr.structure.family, region=tr.region,
                         contracts="/".join(tr.contracts), side="long rate" if tr.side > 0 else "short rate",
                         signal_date=tr.signal_date.date(), entry_date=dates[tr.entry_i].date(),
                         exit_date=dates[tr.exit_i].date(), hold_days=tr.exit_i - tr.entry_i,
                         entry_z=round(tr.entry_z, 2), exit_signal_z=round(tr.exit_signal_z, 2) if np.isfinite(tr.exit_signal_z) else np.nan,
                         half_life=round(tr.half_life, 1), exit_reason=tr.exit_reason,
                         gross_bp=round(g_bp, 3), cost_bp=round(cost_bp, 3), net_bp=round(g_bp - cost_bp, 3),
                         net_usd=round((g_bp - cost_bp) * p.dollars_per_bp, 2), n_contracts=tr.structure.n_contracts))
    trades = pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True) if rows else pd.DataFrame()
    return BacktestResult(params=p, trades=trades, daily_pnl=net, daily_gross=gross, blocks=blocks)


# ---------------------------------------------------------------- metrics

def max_drawdown(pnl: pd.Series) -> float:
    """Largest peak-to-trough fall of cumulative P&L ($, reported as a negative number)."""
    cum = pnl.cumsum()
    return float((cum - cum.cummax()).min())


def sharpe(pnl: pd.Series) -> float:
    """Annualised Sharpe of daily P&L (252 days). Zero-P&L days count: being flat is part of the strategy."""
    sd = pnl.std(ddof=1)
    return float(pnl.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan


def trade_stats(tr: pd.DataFrame) -> dict:
    """Per-trade statistics in bp (net of costs)."""
    if tr is None or len(tr) == 0:
        return dict(trades=0, hit_rate=np.nan, hit_rate_gross=np.nan, avg_gross_bp=np.nan, avg_win_bp=np.nan, avg_loss_bp=np.nan, avg_hold_days=np.nan,
                    gross_bp=0.0, net_bp=0.0, net_usd=0.0)
    w, l = tr[tr.net_bp > 0], tr[tr.net_bp <= 0]
    return dict(trades=len(tr), hit_rate=len(w) / len(tr), hit_rate_gross=float((tr.gross_bp > 0).mean()),
                avg_gross_bp=tr.gross_bp.mean(), avg_win_bp=w.net_bp.mean() if len(w) else np.nan,
                avg_loss_bp=l.net_bp.mean() if len(l) else np.nan, avg_hold_days=tr.hold_days.mean(),
                gross_bp=tr.gross_bp.sum(), net_bp=tr.net_bp.sum(), net_usd=tr.net_usd.sum())


def summary(res: BacktestResult) -> dict:
    """Portfolio headline numbers."""
    d = trade_stats(res.trades)
    d.update(sharpe_net=sharpe(res.daily_pnl), sharpe_gross=sharpe(res.daily_gross),
             max_dd_usd=max_drawdown(res.daily_pnl), days=len(res.daily_pnl),
             start=str(res.daily_pnl.index[0].date()), end=str(res.daily_pnl.index[-1].date()))
    return d


def by_group(res: BacktestResult, col: str) -> pd.DataFrame:
    """Trade statistics per family / structure / region."""
    if len(res.trades) == 0:
        return pd.DataFrame()
    return pd.DataFrame({k: trade_stats(g) for k, g in res.trades.groupby(col)}).T


def pnl_by_regime(res: BacktestResult, labels: pd.Series) -> pd.DataFrame:
    """Net $ P&L, Sharpe and active days per policy regime (daily P&L grouped by the regime of each day)."""
    lab = labels.reindex(res.daily_pnl.index)
    out = {}
    for k, g in res.daily_pnl.groupby(lab):
        out[k] = dict(net_usd=g.sum(), sharpe_net=sharpe(g), days=len(g))
    return pd.DataFrame(out).T


def net_with_cost(res: BacktestResult, tick_bp: float) -> pd.Series:
    """Re-price the same trades under another cost assumption (decisions don't depend on costs)."""
    net = res.daily_gross.copy()
    if len(res.trades) == 0:
        return net
    p = res.params
    for _, r in res.trades.iterrows():
        c = 2.0 * r.n_contracts * (tick_bp + p.commission_per_contract / p.dollars_per_bp) * p.dollars_per_bp
        net.loc[pd.Timestamp(r.entry_date)] -= c / 2
        net.loc[pd.Timestamp(r.exit_date)] -= c / 2
    return net
