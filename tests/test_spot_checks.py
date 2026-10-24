"""Hand spot-checks against the real settlement file (skipped when data/raw is not present).

The expected numbers below were worked out by hand from the raw settles, independently of the package:
    rate (bp) = (100 - settle) x 100 ;  fly = r5 - 2 r6 + r7 ;  spread change = change(r1) - change(r2).

    2021-06-15  pos 5/6/7 = SR3H22 / SR3M22 / SR3U22, settles 99.940 / 99.910 / 99.855
                rates 6.0 / 9.0 / 14.5 bp  ->  FLY 5-6-7 = 6 - 18 + 14.5 = 2.5 bp
    2023-03-13  (the day after the SVB failure) pos 5/6/7 = SR3Z23 / SR3H24 / SR3M24, settles 96.090 / 96.325 / 96.515
                rates 391.0 / 367.5 / 348.5  ->  FLY 5-6-7 = 391 - 735 + 348.5 = 4.5 bp ;  SPR 1-2 change = +30.75 bp
    2025-09-17  pos 5/6/7 = SR3U26 / SR3Z26 / SR3H27, settles 96.980 / 97.040 / 97.050
                rates 302 / 296 / 295  ->  FLY 5-6-7 = 302 - 592 + 295 = 5.0 bp
    Trade check: the first out-of-sample trade, DFLY 6-7-8-9 long the rate on SR3U21/Z21/H22/M22,
                entered 2020-07-13 at settles 100.025 / 100.035 / 100.045 / 100.030 and exited 2020-07-16 at
                100.020 / 100.030 / 100.035 / 100.025: value moves from -2.5 to -1.0 -> +1.5 bp gross, minus 9.6 bp of
                costs (8 contracts x 2 sides x 0.6 bp) = -8.1 bp net.
"""
import numpy as np
import pandas as pd
import pytest

from sofr_pca.data import load_curve
from sofr_pca.structures import build_universe, structure_values

CASES = [
    ("2021-06-15", ["SR3H22", "SR3M22", "SR3U22"], 2.5, None),
    ("2023-03-13", ["SR3Z23", "SR3H24", "SR3M24"], 4.5, 30.75),
    ("2025-09-17", ["SR3U26", "SR3Z26", "SR3H27"], 5.0, None),
]


@pytest.fixture(scope="module")
def real_curve(cfg, raw_available):
    if not raw_available:
        pytest.skip("licensed settlement data not present in data/raw")
    return load_curve(cfg)


@pytest.mark.parametrize("day,legs,fly,spr_chg", CASES)
def test_spot_dates(real_curve, cfg, day, legs, fly, spr_chg):
    d = pd.Timestamp(day)
    assert real_curve.contracts.loc[d, [5, 6, 7]].tolist() == legs
    uni = build_universe(cfg["structures"]["families"], 12)
    v = structure_values(real_curve.levels.loc[[d]], uni).iloc[0]
    assert v["FLY 5-6-7"] == pytest.approx(fly, abs=1e-9)
    if spr_chg is not None:
        c = structure_values(real_curve.changes.loc[[d]], uni).iloc[0]
        assert c["SPR 1-2"] == pytest.approx(spr_chg, abs=1e-9)


def test_first_trade_by_hand(cfg, raw_available):
    path = cfg["paths"]["results"] / "trades_oos.csv"
    if not raw_available or not path.exists():
        pytest.skip("needs data/raw and reports/results (run scripts/run_all.py)")
    t = pd.read_csv(path).iloc[0]
    assert (t.structure, t.entry_date, t.exit_date, t.side) == ("DFLY 6-7-8-9", "2020-07-13", "2020-07-16", "long rate")
    entry = np.array([100.025, 100.035, 100.045, 100.030]); exit_ = np.array([100.020, 100.030, 100.035, 100.025])
    w = np.array([1, -3, 3, -1])
    gross = w @ ((100 - exit_) * 100) - w @ ((100 - entry) * 100)
    assert gross == pytest.approx(1.5) and t.gross_bp == pytest.approx(gross)
    assert t.net_bp == pytest.approx(gross - 2 * 8 * (0.5 + 2.5 / 25))
