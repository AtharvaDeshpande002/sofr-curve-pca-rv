"""The whole study in one function: data -> PCA -> residuals -> mean reversion -> backtests -> screen.

``run_analysis`` returns a dict of results (DataFrames / objects) that ``scripts/run_all.py`` saves and charts,
and that the notebook walks through step by step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest as bt
from .data import CurveData, load_curve, regime_labels, roll_days
from .meanrev import adf_pvalue, ar1_phi, half_life
from .pca import PCAModel, fit_pca, pca_neutral_weights, rolling_explained
from .residuals import fit_residual_model, add_adf
from .screen import screen_today
from .structures import Structure, build_universe, structure_values


def pca_section(curve: CurveData, cfg: dict) -> dict:
    """Step 2: full-sample PCA, rolling explained variance, loadings per regime."""
    k = cfg["pca"]["n_components"]
    full = fit_pca(curve.changes.iloc[1:], curve.levels, k)
    explained = pd.DataFrame({"share": full.explained[:5], "cumulative": np.cumsum(full.explained[:5])},
                             index=[f"PC{i + 1}" for i in range(5)])
    roll = rolling_explained(curve.changes.iloc[1:], curve.levels.iloc[1:], cfg["pca"]["window"],
                             cfg["pca"]["rolling_step"], k)
    by_regime = {}
    for label, (a, b) in cfg["pca"]["regime_periods"].items():
        m = (curve.dates >= pd.Timestamp(a)) & (curve.dates <= pd.Timestamp(b))
        ch = curve.changes[m].dropna()
        if len(ch) > 60:
            r = fit_pca(ch, curve.levels[m], k)
            by_regime[label] = r
    return dict(full=full, explained=explained, rolling=roll, by_regime=by_regime)


def structure_stats(curve: CurveData, universe: list[Structure], blocks: list[bt.Block], k: int) -> pd.DataFrame:
    """Step 4: half-life and ADF of every structure's residual.

    Two views: one fit on the full sample (in-sample, descriptive) and the median half-life across the
    walk-forward training windows (what the backtest actually used).
    """
    full = add_adf(fit_residual_model(curve.levels, curve.changes.iloc[1:], universe, k), curve.levels, universe)
    hl_blocks = pd.DataFrame({b.start: b.model.params["half_life"] for b in blocks})
    out = pd.DataFrame({
        "family": [u.family for u in universe], "centre": [u.centre for u in universe],
        "resid_sigma_bp": full.params["sigma"].round(2), "half_life_full": full.params["half_life"].round(1),
        "adf_p_full": full.params["adf_p"].round(3),
        "half_life_wf_median": hl_blocks.replace(np.inf, np.nan).median(axis=1).round(1),
        "share_blocks_tradable": ((hl_blocks >= 1) & (hl_blocks <= 40)).mean(axis=1).round(2),
    }, index=[u.name for u in universe])
    out.index.name = "structure"
    return out


def pca_vs_traditional(curve: CurveData, full: PCAModel, universe: list[Structure]) -> pd.DataFrame:
    """Step 3.3: does the PCA-weighted fly behave better than 1:-2:1?

    For each fly compare the two weightings on: daily-change exposure to PC1 and PC2 (regression beta of the
    fly's daily change on the PC scores' daily changes), half-life of the fly level, and ADF p-value.
    """
    flies = [u for u in universe if u.family == "fly"]
    sc_chg = full.scores(curve.levels).diff().iloc[1:]
    rows = []
    for u in flies:
        for label, w in [("1:-2:1", np.array(u.leg_weights)), ("PCA-neutral", pca_neutral_weights(full, u.positions, u.leg_weights))]:
            lvl = curve.levels[u.positions].to_numpy() @ w
            chg = pd.Series(curve.changes[u.positions].to_numpy() @ w, index=curve.dates).iloc[1:]
            X = np.column_stack([np.ones(len(sc_chg)), sc_chg[["PC1", "PC2"]].to_numpy()])
            m = ~np.isnan(chg.to_numpy())
            beta = np.linalg.lstsq(X[m], chg.to_numpy()[m], rcond=None)[0]
            rows.append(dict(structure=u.name, weighting=label, weights=" : ".join(f"{x:.2f}" for x in w),
                             beta_pc1=round(beta[1], 3), beta_pc2=round(beta[2], 3),
                             level_half_life=round(half_life(ar1_phi(lvl)), 1), level_adf_p=round(adf_pvalue(lvl), 3),
                             daily_vol_bp=round(float(np.nanstd(chg)), 2)))
    return pd.DataFrame(rows)


def signal_quality(curve: CurveData, universe: list[Structure], z: pd.DataFrame, horizons: list[int],
                   buckets: list[float], p: bt.BTParams) -> pd.DataFrame:
    """Average reversion (bp) h days after the close on which |z| sat in each bucket, out-of-sample.

    Reversion = -sign(z) x (change of the structure over the next h days), so positive = it moved back toward
    fair value. Compared with the round-trip cost of one unit. This isolates the SIGNAL from the trading rules.
    The h-day change is the sum of daily fixed-contract changes of the structure at the same positions.
    """
    fam = {u.name: u for u in universe}
    chg = structure_values(curve.changes, universe)
    rows = []
    for h in horizons:
        fut = chg.shift(-1).rolling(h).sum().shift(-(h - 1)).reindex(z.index)  # change from close t to close t+h
        for f in sorted({u.family for u in universe}):
            cols = [c for c in z.columns if fam[c].family == f]
            zz, ff = z[cols].stack(), fut[cols].stack()
            ok = zz.notna() & ff.notna()
            zz, ff = zz[ok], ff[ok]
            rev = -np.sign(zz) * ff
            cost = bt.round_trip_cost_bp(fam[cols[0]], p)
            for lo, hi in zip(buckets[:-1], buckets[1:]):
                k = (zz.abs() >= lo) & (zz.abs() < hi)
                rows.append(dict(family=f, horizon_days=h, z_bucket=f"{lo}-{hi}", signals=int(k.sum()),
                                 avg_reversion_bp=round(float(rev[k].mean()), 3),
                                 hit_rate=round(float((rev[k] > 0).mean()), 3),
                                 t_stat=round(float(rev[k].mean() / rev[k].std() * np.sqrt(k.sum() / h)), 2),
                                 round_trip_cost_bp=round(cost, 2)))
    return pd.DataFrame(rows)


def backtest_section(curve: CurveData, cfg: dict) -> dict:
    """Step 5: headline walk-forward, in-sample comparison, per-family, robustness, costs, rule variants."""
    fams, n, regions = cfg["structures"]["families"], cfg["data"]["n_positions"], cfg["structures"]["regions"]
    b = cfg["backtest"]
    p = bt.params_from_config(cfg)
    uni = build_universe(fams, n, b["families"])
    main = bt.run_backtest(curve, uni, p, regions)
    ins = bt.run_backtest(curve, uni, p, regions, insample=True)
    labels = regime_labels(curve.dates, cfg["regimes"])

    headline = pd.DataFrame({"out-of-sample (walk-forward)": bt.summary(main), "in-sample (one fit, all data)": bt.summary(ins)}).T

    fam_rows, fam_trades = {}, []
    for f in b["compare_families"]:
        r = bt.run_backtest(curve, build_universe(fams, n, [f]), p, regions)
        fam_rows[f] = bt.summary(r)
        fam_trades.append(r.trades)
    by_family = pd.DataFrame(fam_rows).T

    grid = []
    for w in cfg["pca"]["windows_tested"]:
        blocks = bt.fit_blocks(curve, uni, bt.replace(p, window=w))
        for ez in b["entry_z_tested"]:
            pp = bt.replace(p, window=w, entry_z=ez)
            s = bt.summary(bt.run_backtest(curve, uni, pp, regions, blocks=blocks))
            grid.append(dict(window=w, entry_z=ez, trades=s["trades"], sharpe_net=s["sharpe_net"],
                             sharpe_gross=s["sharpe_gross"], net_usd=s["net_usd"], gross_bp=s["gross_bp"]))
    grid = pd.DataFrame(grid)

    costs = []
    for name, tick in cfg["costs"]["scenarios"].items():
        net = bt.net_with_cost(main, tick)
        costs.append(dict(scenario=name, tick_bp_per_leg=tick, net_usd=net.sum(), sharpe_net=bt.sharpe(net),
                          max_dd_usd=bt.max_drawdown(net)))
    costs.append(dict(scenario="gross (no costs at all)", tick_bp_per_leg=np.nan, net_usd=main.daily_gross.sum(),
                      sharpe_net=bt.sharpe(main.daily_gross), max_dd_usd=bt.max_drawdown(main.daily_gross)))
    costs = pd.DataFrame(costs)

    variants = []
    for name, over in b["rule_variants"].items():
        s = bt.summary(bt.run_backtest(curve, uni, bt.replace(p, **over), regions, blocks=main.blocks))
        variants.append(dict(variant=name, **{k: s[k] for k in ["trades", "hit_rate", "avg_hold_days", "gross_bp",
                                                                  "net_bp", "net_usd", "sharpe_gross", "sharpe_net"]}))
    variants = pd.DataFrame(variants)

    # break-even: the per-contract, per-side cost at which the average trade nets zero
    t = main.trades
    breakeven = float((t.gross_bp / (2 * t.n_contracts)).mean()) if len(t) else np.nan

    return dict(params=p, universe=uni, main=main, family_trades=pd.concat(fam_trades, ignore_index=True), insample=ins, headline=headline, by_family=by_family,
                by_structure=bt.by_group(main, "structure"), by_region=bt.by_group(main, "region"),
                exit_reasons=t.exit_reason.value_counts() if len(t) else pd.Series(dtype=int),
                regime=bt.pnl_by_regime(main, labels), grid=grid, costs=costs, variants=variants,
                breakeven_bp_per_contract_side=breakeven, labels=labels)


def run_analysis(cfg: dict) -> dict:
    """Run every step. Takes ~1 minute on a laptop."""
    curve = load_curve(cfg)
    n, k = cfg["data"]["n_positions"], cfg["pca"]["n_components"]
    universe = build_universe(cfg["structures"]["families"], n)
    res: dict = dict(curve=curve, universe=universe, rolls=int(roll_days(curve).sum()))
    res["pca"] = pca_section(curve, cfg)
    res["bt"] = backtest_section(curve, cfg)
    p = res["bt"]["params"]
    blocks_all = bt.fit_blocks(curve, universe, p)                 # every family, for signal study + heatmap
    res["blocks_all"] = blocks_all
    res["z_oos"] = bt.stitched(blocks_all, "z")
    res["fair_oos"] = bt.stitched(blocks_all, "fair")
    res["stats"] = structure_stats(curve, universe, blocks_all, k)
    res["pca_vs_trad"] = pca_vs_traditional(curve, res["pca"]["full"], universe)
    res["signal"] = signal_quality(curve, universe, res["z_oos"], cfg["backtest"]["forward_horizons"],
                                   cfg["backtest"]["z_buckets"], p)
    scr_uni = [u for u in universe if u.family in cfg["screen"]["families"]]
    res["screen"] = screen_today(curve, scr_uni, p.window, p.z_window, p.exit_z, p.stop_z, res["bt"]["family_trades"], k)
    return res
