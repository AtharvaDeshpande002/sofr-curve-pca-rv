"""Reproduce every result, chart and the interactive report from the raw data in one command.

    python scripts/run_all.py                 # everything, including re-executing notebooks/walkthrough.ipynb
    python scripts/run_all.py --no-notebook   # results, charts and report only (~25 s)

Outputs (all small, derived - safe to commit):
    reports/results/*.csv     result tables quoted in the README
    reports/results/summary.json   headline numbers
    reports/figures/*.png     static charts
    reports/report.html       interactive report (Plotly)
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from sofr_pca import plots  # noqa: E402
from sofr_pca.analysis import run_analysis  # noqa: E402
from sofr_pca.backtest import net_with_cost, sharpe  # noqa: E402
from sofr_pca.config import load_config  # noqa: E402
from sofr_pca.screen import top_rich_cheap  # noqa: E402
from sofr_pca.structures import structure_values  # noqa: E402


def save_tables(r: dict, out: pathlib.Path) -> dict:
    """Write the result tables and return the headline numbers for summary.json."""
    out.mkdir(parents=True, exist_ok=True)
    b = r["bt"]
    rd = lambda df, n=3: df.round(n) if hasattr(df, "round") else df  # noqa: E731
    pc = r["pca"]
    rd(pc["explained"], 4).to_csv(out / "pca_explained_variance.csv", index_label="component")
    rd(pc["full"].loadings, 4).to_csv(out / "pca_loadings_full_sample.csv", index_label="position")
    pd.concat({k: m.loadings for k, m in pc["by_regime"].items()}, names=["period", "position"]).round(4) \
        .to_csv(out / "pca_loadings_by_period.csv")
    pd.DataFrame({k: m.explained[:3] for k, m in pc["by_regime"].items()}, index=["PC1", "PC2", "PC3"]).round(4) \
        .to_csv(out / "pca_explained_by_period.csv", index_label="component")
    rd(pc["rolling"], 4).to_csv(out / "pca_rolling_explained.csv", index_label="date")
    r["stats"].to_csv(out / "structure_mean_reversion.csv")
    r["pca_vs_trad"].to_csv(out / "pca_neutral_vs_traditional_flies.csv", index=False)
    r["signal"].to_csv(out / "signal_quality_by_z_bucket.csv", index=False)
    b["headline"].drop(columns=["start", "end"]).astype(float).round(3).T \
        .to_csv(out / "backtest_headline_oos_vs_insample.csv", index_label="metric")
    rd(b["by_family"]).to_csv(out / "backtest_by_family.csv", index_label="family")
    rd(b["by_structure"]).to_csv(out / "backtest_by_structure.csv", index_label="structure")
    rd(b["by_region"]).to_csv(out / "backtest_by_region.csv", index_label="region")
    rd(b["regime"]).to_csv(out / "backtest_pnl_by_regime.csv", index_label="regime")
    rd(b["grid"]).to_csv(out / "robustness_grid.csv", index=False)
    rd(b["costs"]).to_csv(out / "cost_sensitivity.csv", index=False)
    rd(b["variants"]).to_csv(out / "rule_variants.csv", index=False)
    b["main"].trades.drop(columns=["contracts"]).to_csv(out / "trades_oos.csv", index=False)
    b["family_trades"].drop(columns=["contracts"]).to_csv(out / "trades_oos_all_families.csv", index=False)
    pd.DataFrame({"cum_net_usd": b["main"].daily_pnl.cumsum(), "cum_gross_usd": b["main"].daily_gross.cumsum()}) \
        .round(2).to_csv(out / "equity_curve_oos.csv", index_label="date")
    h = b["headline"]
    s10 = r["signal"][r["signal"].horizon_days == 10]
    curv = s10[s10.family.isin(["fly", "dfly"])]
    return {
        "data_first_date": str(r["curve"].dates[0].date()), "data_last_date": str(r["curve"].dates[-1].date()),
        "sessions": int(len(r["curve"].dates)), "quarterly_rolls": r["rolls"],
        "pc1_share": round(float(pc["full"].explained[0]), 4), "pc2_share": round(float(pc["full"].explained[1]), 4),
        "pc3_share": round(float(pc["full"].explained[2]), 4), "pc123_share": round(float(pc["full"].explained[:3].sum()), 4),
        "oos": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.iloc[0].to_dict().items()},
        "insample": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.iloc[1].to_dict().items()},
        "breakeven_cost_bp_per_contract_side": round(b["breakeven_bp_per_contract_side"], 4),
        "curvature_10d_reversion_bp_range": [round(float(curv.avg_reversion_bp.min()), 3), round(float(curv.avg_reversion_bp.max()), 3)],
        "curvature_10d_hit_rate_range": [round(float(curv.hit_rate.min()), 3), round(float(curv.hit_rate.max()), 3)],
        "robustness_positive_net_sharpe": int((b["grid"].sharpe_net > 0).sum()), "robustness_cells": int(len(b["grid"])),
        "robustness_best_gross_sharpe": round(float(b["grid"].sharpe_gross.max()), 3),
        "screen_date": str(r["screen"].date.iloc[0]),
        "screen_beyond_entry": int((r["screen"].z.abs() > b["params"].entry_z).sum()),
    }


def make_charts(r: dict, cfg: dict, figs: pathlib.Path) -> list:
    """Static PNGs + the matching interactive figures for the report."""
    figs.mkdir(parents=True, exist_ok=True)
    plots.apply_style()
    b, p, pc = r["bt"], r["bt"]["params"], r["pca"]
    curvature = [u.name for u in r["universe"] if u.family in ("fly", "dfly")]
    name = cfg["example_structure"]
    ex_u = [u for u in r["universe"] if u.name == name]
    n_ex = cfg.get("example_window_days", 504)
    level = structure_values(r["curve"].levels, ex_u)[name].iloc[-n_ex:]
    z_ex, anchor = r["z_oos"][name].reindex(level.index), r["fair_oos"][name].reindex(level.index)
    tr = b["main"].trades
    tr_ex = tr[(tr.structure == name) & (pd.to_datetime(tr.entry_date) >= level.index[0])] if len(tr) else tr
    top = top_rich_cheap(r["screen"], cfg["screen"]["top_n"])
    ev, rg = cfg["events"], cfg["regimes"]

    plots.fig_signal_vs_cost(r["signal"], 10, figs / "01_signal_vs_cost.png")
    plots.fig_pca(pc["full"], figs / "02_pca_explained_loadings.png")
    plots.fig_regime_loadings(pc["by_regime"], figs / "03_loadings_by_regime.png")
    plots.fig_rolling_explained(pc["rolling"], ev, figs / "04_rolling_explained.png")
    plots.fig_heatmap(r["z_oos"], curvature, figs / "05_zscore_heatmap.png")
    plots.fig_example(level, anchor, z_ex, tr_ex, name, p, figs / "06_example_fly.png")
    plots.fig_half_life(r["stats"], figs / "07_half_life_by_position.png")
    pkg_name, pkg_tick = list(cfg["costs"]["scenarios"].items())[1]
    plots.fig_equity(b["main"], rg, ev, figs / "08_equity_curve.png", alt=net_with_cost(b["main"], pkg_tick),
                     alt_label=f"net, {pkg_name} + commission")
    plots.fig_robustness(b["grid"], figs / "09_robustness_heatmap.png")
    plots.fig_today(top, p.entry_z, figs / "10_today_rich_cheap.png")
    plots.plt.close("all")

    head = b["headline"][["trades", "hit_rate_gross", "avg_gross_bp", "gross_bp", "net_bp", "net_usd", "sharpe_gross",
                          "sharpe_net", "max_dd_usd", "avg_hold_days"]]
    T = plots.table_html
    return [
        ("1. The answer in one picture", "Average move back toward fair value after a signal, by size of the signal, versus "
         "the cost of trading one unit. Out-of-sample z-scores only.", plots.plotly_signal(r["signal"])),
        ("2. PCA of daily curve changes", "Positions 1-12 = the next twelve quarterly SR3 contracts. Daily changes are "
         "computed on fixed contracts, so quarterly rolls never create false moves.", plots.plotly_pca(pc["full"])),
        ("3. Are the factors stable?", "Loadings estimated separately in each policy period, and the explained variance "
         "on a rolling 252-day window.", plots.plotly_regime_loadings(pc["by_regime"], pc["rolling"])),
        ("4. Rich/cheap through time", "z-score of each fly's residual from the curve's level/slope/curvature, from the "
         "walk-forward model in force on that day.", plots.plotly_heatmap(r["z_oos"], curvature)),
        ("5. Worked example", f"{name} over the last {n_ex} sessions, with the out-of-sample trades the rule took.",
         plots.plotly_example(level, anchor, z_ex, tr_ex, name, p)),
        ("6. Mean-reversion speed", "Median AR(1) half-life of each residual across the walk-forward training windows.",
         plots.plotly_half_life(r["stats"])),
        ("7. Walk-forward backtest", "Fly and double-fly book, one unit per trade, signal at the close of t, trade at the "
         "settle of t+1, at most 2 open trades per curve region.", plots.plotly_equity(b["main"], rg)),
        ("Headline numbers", "Out-of-sample (walk-forward) vs in-sample (one model fitted on all data).", T(head)),
        ("Results by family", "Same rule run separately on each family.", T(b["by_family"][head.columns])),
        ("Cost sensitivity", "The same trades re-priced under different execution assumptions.", T(b["costs"].set_index("scenario"))),
        ("Rule variants", "Alternative exits, all reported.", T(b["variants"].set_index("variant"))),
        ("P&L by policy regime", "Net daily P&L grouped by the regime of each day.", T(b["regime"])),
        ("8. Robustness", "Every combination of entry threshold and estimation window.", plots.plotly_robustness(b["grid"])),
        ("PCA-neutral vs 1:-2:1 flies", "Exposure of each weighting's daily change to PC1/PC2, and mean reversion of its level.",
         T(r["pca_vs_trad"].set_index("structure"), "{:,.3f}")),
        ("9. Today's screen", f"Latest session {r['screen'].date.iloc[0]}. Hit rates = share of out-of-sample trades on that "
         "structure that made money before (gross) and after (net) costs.", plots.plotly_today(top)),
        ("Today's table", "", T(top.drop(columns=["date"]).set_index("structure"))),
    ]


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    r = run_analysis(cfg)
    summary = save_tables(r, cfg["paths"]["results"])
    sections = make_charts(r, cfg, cfg["paths"]["figures"])
    o = summary["oos"]
    intro = (f"PCA rich/cheap model of the SOFR futures curve, {summary['data_first_date']} to {summary['data_last_date']}. "
             f"Walk-forward backtest of fly and double-fly mean reversion: {o['trades']} trades, gross Sharpe {o['sharpe_gross']:.2f}, "
             f"net Sharpe {o['sharpe_net']:.2f} after costs. The signal is real but smaller than the cost of crossing every leg.")
    plots.write_report(cfg["paths"]["report"], sections, "SOFR curve PCA & butterfly mean reversion", intro)
    (cfg["paths"]["results"] / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    top_rich_cheap(r["screen"], cfg["screen"]["top_n"]).to_csv(cfg["paths"]["results"] / "screen_today.csv", index=False)
    print(json.dumps(summary, indent=2))
    if "--no-notebook" not in sys.argv:                       # re-execute the walkthrough so its outputs match
        from build_notebook import main as build_nb
        build_nb()
    print(f"done in {time.time() - t0:.0f}s; net Sharpe check {sharpe(r['bt']['main'].daily_pnl):.2f}")


if __name__ == "__main__":
    main()
