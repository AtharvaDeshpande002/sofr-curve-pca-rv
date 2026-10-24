"""Build notebooks/walkthrough.ipynb from the cells below and execute it so the outputs are saved.

    python scripts/build_notebook.py

Kept as a script so the notebook is reproducible and diff-friendly: edit the story here, not in Jupyter.
"""
from __future__ import annotations

import pathlib

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "walkthrough.ipynb"

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

CELLS = [
    md("""# SOFR curve PCA and butterfly mean reversion - walkthrough

**Question.** Which SOFR spreads and butterflies are rich or cheap relative to the curve's normal shape, and does
trading them back to fair value make money **after costs, out of sample**?

**Answer in one line.** Curvature (flies and double flies) does mean-revert, measurably and out of sample, but by
only ~0.2-1.2 bp over ten days - far less than the 4.8-9.6 bp it costs to cross every leg. As a stand-alone
taker strategy it loses money; as a screen for *timing* trades you already want to do, it is useful.

This notebook walks through every step with the real numbers. All of it runs from `src/sofr_pca`; the same code
produces the README charts via `scripts/run_all.py`."""),
    code("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20)

from sofr_pca import plots, backtest as bt
from sofr_pca.config import load_config
from sofr_pca.analysis import run_analysis
from sofr_pca.data import roll_days
from sofr_pca.structures import structure_values
from sofr_pca.screen import top_rich_cheap
plots.apply_style()
cfg = load_config()
r = run_analysis(cfg)          # ~20 seconds: data -> PCA -> residuals -> backtests -> screen
curve = r["curve"]
print(f"{len(curve.dates)} sessions, {curve.dates[0].date()} to {curve.dates[-1].date()}, "
      f"{curve.levels.shape[1]} constant positions, {r['rolls']} quarterly rolls")"""),
    md("""## Step 1 - A roll-safe curve

* Rate = 100 - price, in basis points. Position 1 = the first quarterly contract (H/M/U/Z) still trading; positions
  1-12 cover the next three years.
* On a roll day every position moves to a later contract. If we differenced the generic position series we would book
  the gap between two *different* contracts as a move. So daily changes are computed on each **fixed** contract and only
  then placed on the position the contract occupies today.

Below: on each roll day, the change we use for position 1 versus the fake "generic" jump."""),
    code("""rolls = curve.dates[roll_days(curve).to_numpy()]
rows = []
for d in rolls[-6:]:
    i = curve.dates.get_loc(d); prev = curve.dates[i - 1]
    rows.append(dict(roll_day=d.date(), old_front=curve.contracts.iat[i - 1, 0], new_front=curve.contracts.iat[i, 0],
                     fixed_contract_change_bp=round(curve.changes.iat[i, 0], 2),
                     generic_jump_bp=round(curve.levels.iat[i, 0] - curve.levels.iat[i - 1, 0], 2)))
pd.DataFrame(rows)"""),
    md("""The generic jump is often several bp (the front contract is replaced by one three months later); the real move
of the new front contract is usually a fraction of that. Every number downstream uses the real move.

## Step 2 - PCA of daily changes

PCA finds the few "shapes" that explain most of the curve's daily moves. We use **changes** (not levels) because
levels trend for years and are not stationary."""),
    code("""pc = r["pca"]
display(pc["explained"].round(3))
plots.fig_pca(pc["full"]);"""),
    md("""* **PC1 (level)**: the whole Fed path re-prices together. It explains ~90% of daily variance. Note the loading is
  small at position 1: the front contract is already partly fixed by realised SOFR.
* **PC2 (slope)**: the front 1-2 years move against the back - steepening / flattening.
* **PC3 (curvature)**: the belly (positions 4-8) moves against both wings. Flies are bets on this - and on what is left
  after it.

### Are the shapes stable?"""),
    code("""plots.fig_regime_loadings(pc["by_regime"]);
plots.fig_rolling_explained(pc["rolling"], cfg["events"]);"""),
    md("""Since 2022 the three shapes are almost identical across hiking, cutting and 2026. The outlier is the zero-rate hold
(2020-21): with the Fed pinned at zero, the front stopped moving and the level factor lived in the back. A model fitted
in 2021 would therefore misread 2022 - one reason to re-estimate on a rolling window.

## Step 3 - Rich / cheap residuals

Each structure's level is regressed on the PC1-PC3 scores over the fitting window:
`y(t) = a + b1 PC1(t) + b2 PC2(t) + b3 PC3(t) + e(t)`. The residual `e` is what the curve's normal shapes cannot
explain; its z-score against its own trailing 63-day mean and standard deviation is the signal."""),
    code("""blk = r["blocks_all"][-1]
print("latest walk-forward model, fitted on data up to", blk.model.fit_end.date())
blk.model.params.loc[["FLY 3-4-5", "FLY 5-6-7", "FLY 8-9-10", "DFLY 4-5-6-7", "SPR 4-5"]].round(3)"""),
    md("""### PCA-neutral weights vs the textbook 1:-2:1

The PCA-neutral fly keeps the middle at -2 and re-sizes the wings so the fly has zero exposure to PC1 and PC2.
It does what it promises - the betas go to zero - but its *level* is **less** mean-reverting than 1:-2:1, not more.
The textbook fly's apparent mean reversion partly comes from a small, mean-reverting level/slope exposure."""),
    code("""r["pca_vs_trad"].query("structure in ['FLY 2-3-4','FLY 4-5-6','FLY 5-6-7','FLY 8-9-10']")"""),
    md("""## Step 4 - How fast do residuals revert?

AR(1): `e(t) = c + phi e(t-1) + noise`; half-life = `-ln 2 / ln phi`. ADF tests whether the residual has a unit root."""),
    code("""st = r["stats"]
display(st[st.family.isin(["fly", "dfly"])][["resid_sigma_bp", "half_life_full", "adf_p_full", "half_life_wf_median"]])
plots.fig_half_life(st);"""),
    md("""Every residual is stationary by ADF on the full sample, and half-lives are short: 3-7 days for flies in the
walk-forward windows. Short half-life is good news for a mean-reversion trade - and bad news for costs, because the
move you can capture before it disappears is small.

## Step 5 - Is the signal real? (before any trading rules)

For every out-of-sample day and structure: if |z| sat in a bucket, how far did the structure move back toward fair
value over the next 5 / 10 / 20 days?"""),
    code("""sig = r["signal"]
display(sig[sig.horizon_days == 10].pivot(index="z_bucket", columns="family", values="avg_reversion_bp").round(2))
display(sig[sig.horizon_days == 10].pivot(index="z_bucket", columns="family", values="hit_rate").round(2))
plots.fig_signal_vs_cost(sig, 10);"""),
    md("""Flies and double flies revert 0.2-1.2 bp in ten days with 53-60% hit rates - a real, positive edge that grows
with |z|. Spreads do not revert at all (hit rates below 50%): 3M and 1Y spreads are dominated by the Fed path, which
trends. But the edge is an order of magnitude smaller than the cost of trading a fly leg by leg.

## Step 6 - Walk-forward backtest, net of costs

* Fit on the previous 252 days, trade the next 63 with frozen parameters, refit, repeat. Out-of-sample from Jul-2020.
* Enter when 2 < |z| < 3.5 (signal at close t, trade at the settle of t+1). Exit at |z| < 0.5, stop at |z| > 3.5,
  time stop 2 x half-life. At most two open trades per curve region; no two trades share a contract.
* Costs: 1 tick (0.5 bp) per leg per side + $2.50 per contract per side."""),
    code("""b = r["bt"]
display(b["headline"][["trades", "hit_rate_gross", "avg_gross_bp", "gross_bp", "net_usd", "sharpe_gross", "sharpe_net", "max_dd_usd"]].round(2))
display(b["by_family"][["trades", "hit_rate_gross", "avg_gross_bp", "net_usd", "sharpe_gross", "sharpe_net"]].round(2))
plots.fig_equity(b["main"], cfg["regimes"], cfg["events"], alt=bt.net_with_cost(b["main"], 0.125),
                 alt_label="net, package execution (1/4 tick per leg) + commission");"""),
    code("""display(b["costs"].round(2)); display(b["variants"].round(2)); display(b["regime"].round(2))
print("break-even cost per contract per side: %.3f bp ($%.2f)" % (b["breakeven_bp_per_contract_side"], 25 * b["breakeven_bp_per_contract_side"]))
plots.fig_robustness(b["grid"]);"""),
    md("""**Verdict.** Out of sample the fly/double-fly book is slightly positive before costs (gross Sharpe ~0.5) and
clearly negative after them, in every regime, under every exit variant and in all 12 window x threshold combinations.
The average trade earns ~0.25 bp gross; to break even, costs would have to fall below ~0.04 bp per contract per side
(about $0.90 all-in), which is below even commission alone. The in-sample fit looks better (gross Sharpe ~0.9) -
the usual decay from in-sample to out-of-sample.

## Step 7 - Today's screen"""),
    code("""top = top_rich_cheap(r["screen"], 10)
display(top[["structure", "contracts", "z", "residual_bp", "view", "half_life_days", "pca_neutral_weights", "oos_hit_gross", "target_bp", "stop_bp"]])
plots.fig_today(top, b["params"].entry_z);"""),
    md("""## What a trader takes away

1. The SOFR curve is ~99% three shapes; anything a fly does beyond them is small (1-3 bp sigma in the belly and back)
   and short-lived (half-life 3-7 days).
2. Those dislocations do revert, but the edge is about a quarter of a basis point per trade - smaller than crossing one
   tick on one leg. Do not run this as a stand-alone strategy that pays the offer.
3. Where it helps: as a timing overlay for trades you want anyway (enter a fly you like on the cheap side of its
   residual), for sizing PCA-neutral hedges, and for spotting the rare |z| > 3.5 dislocation, where the reversion is
   larger (~1 bp in 10 days).
4. To turn it into a strategy you would need passive execution (earning, not paying, the bid-offer) or a sharper signal
   - e.g. add flow / open-interest or event information.""")
]


def main() -> None:
    nb = nbformat.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NB.parent.mkdir(parents=True, exist_ok=True)
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}}).execute()
    nbformat.write(nb, NB)
    print("wrote", NB)


if __name__ == "__main__":
    main()
