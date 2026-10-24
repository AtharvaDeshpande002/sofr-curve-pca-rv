"""Spread / fly / double-fly maths and cost maths."""
import numpy as np
import pandas as pd
import pytest

from sofr_pca import backtest as bt
from sofr_pca.structures import build_universe, make_structure, structure_values

FAMS = {"spread3m": [1, -1], "fly": [1, -2, 1], "dfly": [1, -3, 3, -1], "spread1y": [1, 0, 0, 0, -1]}


def test_names_and_legs():
    s = make_structure("fly", 5, [1, -2, 1])
    assert s.name == "FLY 5-6-7" and s.positions == [5, 6, 7] and s.n_contracts == 4
    y = make_structure("spread1y", 3, [1, 0, 0, 0, -1])
    assert y.name == "1Y 3-7" and y.positions == [3, 7] and y.leg_weights == [1.0, -1.0] and y.n_contracts == 2
    d = make_structure("dfly", 2, [1, -3, 3, -1])
    assert d.n_contracts == 8 and d.centre == 3.5


def test_universe_size():
    u = build_universe(FAMS, 12)
    fam = pd.Series([x.family for x in u]).value_counts()
    assert fam["spread3m"] == 11 and fam["fly"] == 10 and fam["dfly"] == 9 and fam["spread1y"] == 8


def test_structure_values_match_hand_formula():
    lv = pd.DataFrame([[400.0, 390.0, 385.0, 383.0, 382.0]], columns=range(1, 6))
    u = build_universe(FAMS, 5)
    v = structure_values(lv, u).iloc[0]
    assert v["SPR 1-2"] == 10.0                       # front minus back
    assert v["FLY 1-2-3"] == 400 - 2 * 390 + 385      # = 5
    assert v["DFLY 1-2-3-4"] == 400 - 3 * 390 + 3 * 385 - 383
    assert v["1Y 1-5"] == 400 - 382


def test_round_trip_cost():
    p = bt.BTParams(tick_bp_per_leg=0.5, commission_per_contract=2.5, dollars_per_bp=25.0)
    fly = make_structure("fly", 1, [1, -2, 1])
    spr = make_structure("spread3m", 1, [1, -1])
    # 2 sides x 4 contracts x (0.5 bp + $2.50/$25) = 4.8 bp ;  spread: 2 x 2 x 0.6 = 2.4 bp
    assert bt.round_trip_cost_bp(fly, p) == pytest.approx(4.8)
    assert bt.round_trip_cost_bp(spr, p) == pytest.approx(2.4)
    assert bt.round_trip_cost_bp(fly, p, tick_bp=0.0) == pytest.approx(0.8)   # commission only


def test_net_equals_gross_minus_costs(curve):
    fams = {"fly": [1, -2, 1], "dfly": [1, -3, 3, -1]}
    uni = build_universe(fams, 8)
    p = bt.BTParams(window=150, z_window=40, block_days=40, oos_start=str(curve.dates[200].date()), entry_z=1.5)
    res = bt.run_backtest(curve, uni, p, {"front": [1, 3], "belly": [3.5, 5.5], "back": [6, 8]})
    assert len(res.trades) > 5
    assert res.daily_gross.sum() - res.daily_pnl.sum() == pytest.approx(res.trades.cost_bp.sum() * 25, abs=1e-6)
    assert res.trades.gross_bp.sum() * 25 == pytest.approx(res.daily_gross.sum(), abs=1e-6)
    assert np.allclose(res.trades.net_bp, res.trades.gross_bp - res.trades.cost_bp)
    # re-pricing with the same cost reproduces the net series
    assert bt.net_with_cost(res, p.tick_bp_per_leg).sum() == pytest.approx(res.daily_pnl.sum())
