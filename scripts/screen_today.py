"""Desk-ready output for the latest date in the data: top rich and cheap structures.

    python scripts/screen_today.py            # uses the latest settles in data/raw
    python scripts/screen_today.py --top 5

Writes reports/screen/screen_<date>.csv, a PNG chart and an HTML table, and prints the table.
Hit rates come from reports/results/trades_oos_all_families.csv (produced by run_all.py); if that file is
missing the column is left blank rather than recomputing the whole backtest.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from sofr_pca import plots  # noqa: E402
from sofr_pca.backtest import params_from_config  # noqa: E402
from sofr_pca.config import load_config  # noqa: E402
from sofr_pca.data import load_curve  # noqa: E402
from sofr_pca.screen import screen_today, top_rich_cheap  # noqa: E402
from sofr_pca.structures import build_universe  # noqa: E402


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top", type=int, default=cfg["screen"]["top_n"], help="rows per side (rich / cheap)")
    args = ap.parse_args()

    p = params_from_config(cfg)
    curve = load_curve(cfg)
    uni = build_universe(cfg["structures"]["families"], cfg["data"]["n_positions"], cfg["screen"]["families"])
    tr_path = cfg["paths"]["results"] / "trades_oos_all_families.csv"
    trades = pd.read_csv(tr_path) if tr_path.exists() else None
    table = screen_today(curve, uni, p.window, p.z_window, p.exit_z, p.stop_z, trades, p.n_components)
    top = top_rich_cheap(table, args.top)

    out = ROOT / "reports" / "screen"
    out.mkdir(parents=True, exist_ok=True)
    day = str(table.date.iloc[0])
    top.to_csv(out / f"screen_{day}.csv", index=False)
    plots.apply_style()
    plots.fig_today(top, p.entry_z, out / f"screen_{day}.png")
    (out / f"screen_{day}.html").write_text(
        f"<h2>SOFR rich/cheap screen - {day}</h2>" + plots.table_html(top.drop(columns=["date"]).set_index("structure")),
        encoding="utf-8")
    cols = ["structure", "contracts", "z", "residual_bp", "view", "half_life_days", "pca_neutral_weights",
            "oos_hit_gross", "oos_hit_net", "target_bp", "stop_bp"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"Screen for {day}  (|z| > {p.entry_z:g} = entry signal; target = back to |z| {p.exit_z:g}, stop = |z| {p.stop_z:g})")
        print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
