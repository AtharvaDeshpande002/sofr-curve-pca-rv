"""Roll handling: a quarterly roll must never show up as a price move."""
import numpy as np
import pandas as pd

from sofr_pca.data import price_to_bp, roll_days


def test_price_to_bp():
    assert price_to_bp(96.125) == 387.5
    assert price_to_bp(100.0) == 0.0


def test_position_one_is_first_live_contract(curve, market):
    _, k = market
    for d in curve.dates[::50]:
        expected = k[k.last_trade >= d].sort_values("ref_month").contract.iloc[0]
        assert curve.contracts.at[d, 1] == expected


def test_changes_on_roll_days_use_the_same_contract(curve):
    rolls = roll_days(curve)
    assert rolls.sum() >= 5
    for d in curve.dates[rolls.to_numpy()]:
        i = curve.dates.get_loc(d)
        prev = curve.dates[i - 1]
        new_front = curve.contracts.at[d, 1]
        # today's position-1 change = that contract's own move since yesterday...
        own_move = curve.rates.at[d, new_front] - curve.rates.at[prev, new_front]
        assert np.isclose(curve.changes.at[d, 1], own_move)
        # ...and NOT the jump of the generic position-1 series (old front -> new front)
        generic_jump = curve.levels.at[d, 1] - curve.levels.at[prev, 1]
        assert not np.isclose(curve.changes.at[d, 1], generic_jump) or np.isclose(own_move, generic_jump)


def test_contracts_shift_by_exactly_one_position_on_a_roll(curve):
    rolls = roll_days(curve)
    for d in curve.dates[rolls.to_numpy()][:3]:
        i = curve.dates.get_loc(d)
        before = curve.contracts.iloc[i - 1].tolist()
        after = curve.contracts.iloc[i].tolist()
        assert after[:-1] == before[1:]


def test_sum_of_daily_changes_equals_fixed_contract_move(curve):
    """Chaining changes of one fixed contract reproduces its total move - no leakage across rolls."""
    code = curve.contracts.iloc[300, 4]
    pos = curve.contracts.eq(code)
    seen = pos.any(axis=1)
    dates = curve.dates[seen.to_numpy()]
    chg = sum(curve.changes.at[d, int(pos.loc[d].idxmax())] for d in dates[1:])
    assert np.isclose(chg, curve.rates.at[dates[-1], code] - curve.rates.at[dates[0], code])
