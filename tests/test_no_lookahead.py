"""No look-ahead: a model used on day t must be identical whatever happens after t."""
import numpy as np
import pandas as pd

from sofr_pca import backtest as bt
from sofr_pca.data import CurveData
from sofr_pca.structures import build_universe

FAMS = {"fly": [1, -2, 1], "dfly": [1, -3, 3, -1], "spread3m": [1, -1]}
REGIONS = {"front": [1, 3], "belly": [3.5, 5.5], "back": [6, 8]}


def _shock_after(curve: CurveData, cut: pd.Timestamp, bp: float) -> CurveData:
    """Copy of the curve with every level/rate after ``cut`` moved by ``bp`` (and a twist) - i.e. a different future."""
    lv = curve.levels.copy(); ch = curve.changes.copy(); rt = curve.rates.copy()
    twist = np.linspace(0, bp, lv.shape[1])
    lv.loc[lv.index > cut] += bp + twist
    first = lv.index[lv.index > cut][0]
    ch.loc[first] += bp + twist
    rt.loc[rt.index > cut] += bp
    return CurveData(levels=lv, changes=ch, contracts=curve.contracts, rates=rt, last_trade=curve.last_trade)


def test_block_models_ignore_the_future(curve):
    uni = build_universe(FAMS, 8)
    p = bt.BTParams(window=150, z_window=40, block_days=40, oos_start=str(curve.dates[200].date()))
    base = bt.fit_blocks(curve, uni, p)
    cut = base[3].start                                   # change everything from block 3 onwards
    shocked = bt.fit_blocks(_shock_after(curve, cut, 25.0), uni, p)
    for b0, b1 in zip(base[:4], shocked[:4]):             # blocks fitted on data up to `cut` are unchanged
        pd.testing.assert_frame_equal(b0.model.params, b1.model.params)
    # z-scores on days up to the cut are unchanged
    z0, z1 = bt.stitched(base, "z"), bt.stitched(shocked, "z")
    pd.testing.assert_frame_equal(z0.loc[:cut - pd.Timedelta(days=1)], z1.loc[:cut - pd.Timedelta(days=1)])


def test_training_window_ends_before_block(curve):
    uni = build_universe(FAMS, 8)
    p = bt.BTParams(window=150, z_window=40, block_days=40, oos_start=str(curve.dates[200].date()))
    for b in bt.fit_blocks(curve, uni, p):
        assert b.model.fit_end < b.start


def test_trades_execute_the_day_after_the_signal(curve):
    uni = build_universe(FAMS, 8)
    p = bt.BTParams(window=150, z_window=40, block_days=40, oos_start=str(curve.dates[200].date()), entry_z=1.5)
    t = bt.run_backtest(curve, uni, p, REGIONS).trades
    assert len(t) > 0
    assert (pd.to_datetime(t.entry_date) > pd.to_datetime(t.signal_date)).all()
    assert (pd.to_datetime(t.exit_date) > pd.to_datetime(t.entry_date)).all()


def test_no_trade_shares_a_contract_with_another_open_trade(curve):
    uni = build_universe(FAMS, 8)
    p = bt.BTParams(window=150, z_window=40, block_days=40, oos_start=str(curve.dates[200].date()), entry_z=1.5)
    res = bt.run_backtest(curve, uni, p, REGIONS)
    t = res.trades.assign(e=lambda d: pd.to_datetime(d.entry_date), x=lambda d: pd.to_datetime(d.exit_date))
    for i, a in t.iterrows():
        for j, b in t.iterrows():
            if j <= i:
                continue
            overlap = a.e < b.x and b.e < a.x
            if overlap:
                assert not set(a.contracts.split("/")) & set(b.contracts.split("/"))
