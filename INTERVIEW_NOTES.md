# Interview notes - SOFR Curve PCA & Butterfly Mean-Reversion

Plain-English guide to this project: what it does, why each choice was made, where every number comes from, and how
to answer the questions you are likely to get. Read it top to bottom once; then use the Q&A before interviews.

---

## 1. The project in 10 sentences

1. I took eight years of daily SR3 (3-month SOFR futures) settlement prices, Jun-2018 to Oct-2026, and built a curve of
   the next 12 quarterly contracts (positions 1-12, about three years).
2. I computed daily changes on fixed contracts so the quarterly roll never shows up as a fake price move.
3. A PCA of those daily changes shows three shapes - level, slope, curvature - explain 99.3% of how the curve moves.
4. For every spread, fly and double fly I regressed its level on those three shapes; what is left over (the residual)
   is how rich or cheap it is relative to the rest of the curve.
5. I turned the residual into a z-score against its own recent history and measured how fast it mean-reverts:
   half-lives of 3-7 days for flies.
6. Before writing any trading rule I checked the raw signal: after a fly is 2 sigma cheap, it moves back 0.2-1.2 bp
   over ten days, with a 53-60% hit rate. Spreads do not revert at all.
7. I then ran a proper walk-forward backtest: fit on the past year, trade the next quarter with frozen parameters,
   refit, repeat - signal at today's close, trade at tomorrow's settle, with costs.
8. Out of sample the fly book was slightly profitable before costs (+0.25 bp a trade, Sharpe 0.49) but lost $55k after
   paying one tick per leg plus commission, and no setting of the window or threshold changed that.
9. So the honest conclusion is that curvature dislocations are real but too small to pay four to eight bid-offers for;
   the model is useful as a timing and hedging tool, not as a stand-alone taker strategy.
10. Everything runs from one command, is tested for roll handling and look-ahead, and every README number is
    reproduced in `reports/results/`.

---

## 2. Each method, intuitively

### 2.1 Rates in bp, positions and rolls
*Definition.* An SR3 future quotes 100 minus the expected 3-month SOFR rate. **Rate (bp) = (100 - price) x 100.**
Example: price 96.125 -> rate 3.875% -> 387.5 bp.

*Positions.* Position 1 is the first quarterly contract (H, M, U, Z) still trading; position 2 the next; ... position 12
is about three years out. Every quarter the front contract expires and every contract moves up one position - a
**roll**.

*Why fixed-contract changes.* Suppose position 1 is SR3H26 at 360 bp and SR3M26 (position 2) is at 345 bp. H26 expires;
tomorrow position 1 is M26 at 346 bp. The generic position-1 series "fell" 14 bp - but nothing moved 14 bp; M26 moved
+1 bp. I always compute the change of the **same contract** (M26: 345 -> 346 = +1 bp) and then file it under the
position it holds today. Test: `tests/test_rolls.py`.

### 2.2 Spreads and flies (all in rate terms, front minus back)
* 3M spread = r(i) - r(i+1). 1Y spread = r(i) - r(i+4).
* Fly = r(i) - 2 r(i+1) + r(i+2). Double fly = r(i) - 3 r(i+1) + 3 r(i+2) - r(i+3).

Worked example (2021-06-15, hand-checked in `tests/test_spot_checks.py`): positions 5/6/7 settle 99.940 / 99.910 /
99.855 -> rates 6.0 / 9.0 / 14.5 bp -> 5-6-7 fly = 6 - 18 + 14.5 = **2.5 bp**.

Money: one unit of a fly = 1 / 2 / 1 contracts; each contract is $25 per bp, so a 1 bp move of the fly = $25 per unit.

In **price** terms the sign flips: a fly whose rate is "too high" has a price that is "too low" - it is cheap, you buy
the package. The screen prints it that way.

### 2.3 PCA (principal component analysis)
*Definition.* PCA finds the few directions (shapes across the curve) along which daily moves are largest, ordered by
how much of the total variance each one explains.

*Intuition.* On most days the whole curve moves up or down together (PC1, **level**: the market re-pricing the Fed
path). Next most common: the front and back moving in opposite directions (PC2, **slope**). Then the belly against
both wings (PC3, **curvature**). Here: 90.5% / 7.3% / 1.5%.

*Tiny example.* If every position moved +5 bp, that is pure PC1. If positions 1-4 moved +5 and 9-12 moved -5, that is
mostly PC2. A fly r5 - 2 r6 + r7 barely changes in either case - which is why flies are "curvature" trades.

*How it is computed.* Covariance matrix of daily changes (12 x 12) -> eigenvectors (the shapes, "loadings") and
eigenvalues (their variances). Implemented with `numpy.linalg.eigh` and cross-checked against scikit-learn in
`tests/test_models.py`.

### 2.4 Rich / cheap residual and z-score
*Scores.* Today's curve minus its average, projected on each loading = how much level, slope and curvature the curve
holds today.

*Regression.* For each structure, over the fitting window: fly(t) = a + b1 PC1(t) + b2 PC2(t) + b3 PC3(t) + e(t).
The fitted part is "what the fly should be given today's curve shape"; e(t) is the dislocation, in bp.

*z-score.* z = (e(t) - average of e over the last 63 days) / standard deviation of e over the last 63 days.
z = +2: the fly's rate is 2 sigma above where the curve shape puts it.

*Why the trailing mean.* Out of sample the curve visits levels the fitting window never saw, so the frozen regression
drifts slowly. Measuring against the residual's own recent mean keeps the signal about **short-term** dislocations.
(Without it, an earlier version took 2,100 trades that mostly hit the stop - the z-score was measuring drift.)

### 2.5 PCA-neutral weights
Keep the middle leg at -2; choose the two wing weights so the fly's exposure to PC1 and PC2 is exactly zero (two
equations, two unknowns). Example (full sample): 5-6-7 fly -> 1.00 : -2.00 : 1.05; front 2-3-4 fly -> 1.97 : -2.00 :
0.76, because the front contract moves much less than the belly. The textbook 1:-2:1 is neutral only to a
**parallel** move, and real level moves are not parallel.

### 2.6 Half-life and ADF
*AR(1).* e(t) = c + phi x e(t-1) + noise. If phi = 0.8, 80% of today's gap is still there tomorrow.
*Half-life* = -ln 2 / ln phi. phi = 0.8 -> 3.1 days. phi >= 1 -> never reverts (infinite).
*ADF (augmented Dickey-Fuller) test.* Null hypothesis: the series has a unit root (does not revert). A p-value
below 0.05 rejects that - evidence of mean reversion.

### 2.7 Walk-forward backtest
*Definition.* Fit on data up to a date, trade the next period with the parameters frozen, then move the window
forward. Every trade is decided with information available at the time.

*Here.* Fit on 252 days, trade the next 63, refit; first out-of-sample day 1-Jul-2020. Signal at the close of day t,
trade at the settle of t+1. Enter when 2 < |z| < 3.5; exit when |z| < 0.5 (target), |z| > 3.5 (stop) or after
2 x half-life (time stop, max 60 days). A trade holds its **fixed** contracts; if the curve rolls, its z is read from
the structure those contracts form after the roll; a trade is closed on the last trading day of any leg.

*Overlap control.* At most two open trades per curve region (front / belly / back) and no two trades sharing a
contract - one dislocation should not become five trades.

*Costs.* One tick (0.5 bp) per leg per side + $2.50 per contract per side. A fly round trip:
2 sides x 4 contracts x (0.5 + 2.5/25) = **4.8 bp**; a double fly 9.6 bp; a spread 2.4 bp.

### 2.8 Signal quality vs trading rules
I measured the raw signal separately from the rules: for each out-of-sample day and structure, if |z| was in a bucket,
how far did the structure move back over the next 5/10/20 days? This tells you whether the problem is the signal or
the rules. Here the signal is positive but small; the rules don't destroy it, the costs do.

---

## 3. Design choices and alternatives

| Choice | Why | Alternatives considered |
|---|---|---|
| PCA on **changes**, not levels | Levels trend for years (non-stationary); their PCA is dominated by the trend | Levels PCA (common in sell-side screens; unstable), Nelson-Siegel factors |
| Positions 1-12 | Three years: where SR3 RV trading lives and data is liquid since 2018 | 16 positions (thinner back end early in the sample) |
| Constant position, fixed-contract changes | Correct P&L; no fake roll moves | Constant-maturity interpolation (smoother, but not tradable contracts) |
| 252-day window | One year of regimes; tested 126 and 504 | Expanding window (slow to adapt after 2020-21) |
| z vs trailing 63-day residual mean | Removes out-of-sample drift of the frozen fit | Raw residual / in-window sigma (tested: whipsaw, 2,100 trades, mostly stops) |
| Next-day settle execution | Cannot trade at the close you used to compute the signal | Same-close execution (look-ahead) |
| Region cap + no shared legs | Avoid counting one dislocation many times | Correlation clustering |
| Full tick per leg default | Conservative; spec standard | Package tick (shown in cost table) |
| Headline universe = flies + double flies | The study is about curvature; spreads are reported separately | All families together |

---

## 4. Every number in the README and where it comes from

| Number | Value | File |
|---|---|---|
| Sample | 2018-06-01 to 2026-10-19, 2,112 sessions, 33 rolls | `reports/results/summary.json` |
| PC1 / PC2 / PC3 share | 90.5% / 7.3% / 1.5% (99.3% total) | `pca_explained_variance.csv` |
| Loadings per period | 2018-19, 2020-21, 2022-23, 2024-25, 2026 | `pca_loadings_by_period.csv`, `pca_explained_by_period.csv` |
| Fly half-lives | 3.1-7.0 days (walk-forward median) | `structure_mean_reversion.csv` |
| 10-day reversion, flies & double flies | 0.19-1.16 bp, hit rate 52.8-60.4% | `signal_quality_by_z_bucket.csv` |
| Spreads' 10-day hit rates | 46.5-50.0% | same |
| Out-of-sample headline | 361 trades, +0.25 bp gross/trade, gross Sharpe 0.49, net -$55,060, net Sharpe -6.27 | `backtest_headline_oos_vs_insample.csv` |
| In-sample headline | 245 trades, +0.70 bp, gross Sharpe 0.89, net Sharpe -5.16 | same |
| Break-even cost | 0.036 bp per contract per side (~$0.90) | `summary.json` |
| Robustness | 0 of 12 cells with positive net Sharpe; best gross Sharpe 0.77 (504 days, abs(z) > 2) | `robustness_grid.csv` |
| Cost table | +$2,300 (no costs) ... -$55,060 (full tick) | `cost_sensitivity.csv` |
| Rule variants | net Sharpe -6.27 / -6.13 / -5.87 / -5.21 | `rule_variants.csv` |
| Regimes | net Sharpe -3.97 hiking, -7.47 cutting, -7.73 hold, -7.12 2026 | `backtest_pnl_by_regime.csv` |
| PCA-neutral vs 1:-2:1 | 5-6-7 fly level half-life 39.5 -> 177.8 days; ADF p 0.000 -> 0.442 | `pca_neutral_vs_traditional_flies.csv` |
| Today | 19-Oct-2026, nothing beyond abs(z) 2; DFLY 8-9-10-11 at -1.90 | `screen_today.csv` |

Why is net Sharpe so negative (-6) when the gross is only slightly positive? Costs are paid on every trade with
certainty, so they form a steady downward drift with little day-to-day noise; Sharpe divides that drift by a small
volatility. In dollars it is $55k over six years of one-lot trades - a slow, certain bleed, not a blow-up.

---

## 5. What didn't work (and why)

* **The first version of the z-score** (residual / in-window sigma, no trailing mean): out of sample the residuals
  drifted, z spent long spells beyond 3.5, and the rule entered and got stopped out repeatedly (2,100 trades,
  net Sharpe -10). Also fixed: entries are now only taken when |z| is below the stop level.
* **Spreads**: no reversion. 3M and 1Y spreads are mostly the Fed path, which trends.
* **PCA-neutral flies**: they hedge level and slope perfectly but their levels revert more slowly.
* **Changing the knobs**: entry thresholds 1.5-3, windows 126-504, exit at 0, no stop, fixed holds - none turns net
  positive. That is the point of the robustness grid: no single lucky choice.
* **Cheaper execution**: even with commission only, the book loses money.

---

## 6. Fifteen likely interview questions - with model answers

**1. Why PCA on changes and not levels?**
Levels are non-stationary - they trend for years with the Fed cycle - so a level PCA mostly describes the trend in the
sample and its factors change meaning when the sample changes. Daily changes are close to stationary; their
covariance describes how the curve actually moves day to day, which is what a hedge or a relative-value trade cares
about. I then project *levels* on those change-based shapes to measure where the curve sits today.

**2. What do the three components mean for a trader?**
PC1 is the market re-pricing the whole path of Fed policy - every contract moves the same way, the belly most. PC2 is
the front against the back - steepeners and flatteners, e.g. "more cuts sooner but the same terminal". PC3 is the belly
against both wings - what a fly trades. Together 99.3%; a fly lives in the last 0.7% plus PC3.

**3. How did you avoid look-ahead bias?**
Three ways. The model for each quarter is fitted only on data before the quarter starts (tested: changing all data
after a date leaves earlier models and z-scores identical). Signals use today's close and trade at tomorrow's settle.
The z-score uses only trailing statistics. Contract expiries are known in advance, so closing a trade on a leg's last
trading day is not look-ahead.

**4. How do you handle the quarterly roll?**
Changes are computed on fixed contracts and then mapped to positions, so the roll never creates a move. Trades hold
fixed contracts; after a roll their z-score is read from the structure they now form (one position earlier), and any
trade is closed on the last trading day of a leg. Tests check that chained daily changes equal the contract's actual
move across rolls.

**5. Is the signal real or noise?**
Real but small. Out of sample, flies and double flies move back toward fair value in all four |z| buckets at all three
horizons (5/10/20 days), with hit rates above 50% and larger moves for larger |z|. Spreads, as a control, show nothing.
The t-stats are approximate because horizons overlap and neighbouring structures are correlated, so I lean on the
consistency across buckets, horizons and families rather than on one t-stat.

**6. Then why does it lose money?**
Size. The average reversion is ~0.25-1 bp, the half-life is a few days, and crossing four legs costs 4.8 bp round trip
(9.6 for a double fly). Break-even is about $0.90 per contract per side - less than commission. The signal is fine; the
execution assumption kills it.

**7. Why is the in-sample result better than out-of-sample?**
In-sample the model has seen the whole history, including the future, so its fair values fit better (gross +0.70 bp per
trade vs +0.25). The gap is the honest measure of how much the in-sample fit flatters you. That is why only
out-of-sample numbers are the headline.

**8. What would break this strategy even if costs were zero?**
A regime change in the curve's shapes - the 2020-21 zero bound changed the loadings sharply, so a model fitted in 2021
misreads 2022. Also crowding (others fading the same dislocations), event days (CPI/FOMC produce dislocations that
re-price rather than revert), and liquidity in the back end.

**9. How would you size it?**
By risk, not by contracts: target a fixed bp-volatility per trade using the residual's sigma (e.g. a 1.2 bp-sigma back
fly gets more units than a 4 bp-sigma front fly), with a cap per region and on total DV01 of the book's PC1/PC2
exposure. With PCA-neutral weights the level/slope risk is hedged, so the residual sigma is the right risk unit.

**10. Why do PCA-neutral flies revert more slowly?**
Because some of the textbook fly's apparent mean reversion is a small exposure to level and slope, which themselves
partly mean-revert in a range-bound market. Hedging those out leaves the pure curvature residual, which drifts more.
It is a better *hedge*, not a better *signal*.

**11. Why a trailing mean in the z-score - isn't that curve-fitting?**
It is a structural choice made for a stated reason (frozen fits drift out of sample), applied identically everywhere,
and the robustness grid varies the other knobs around it. I also report the earlier version without it, which was
worse. It is computed from past data only.

**12. Why did you use settlement prices?**
They are the official, synchronous daily prices for every contract - all legs at the same moment, no stale quotes.
The limitation is no intraday timing and no bid-offer, which is why costs are an explicit assumption with a
sensitivity table.

**13. How did you control for overlapping trades?**
Neighbouring flies share legs, so one dislocation in, say, the June-28 contract lights up three flies and two double
flies. I allow at most two open trades per curve region and never two trades sharing a contract; candidates are ranked
by |z| so the strongest one is taken.

**14. What would you do next to make it tradable?**
Trade passively: rest orders at the package bid/offer and earn the spread instead of paying it - the edge is the right
size for a market-maker, not a taker. Add conditioning information (open-interest changes, event calendar, flows) and
test on intraday data where the dislocation and the reversion actually happen.

**15. How did you validate the numbers?**
Unit tests for roll handling, look-ahead, structure maths and cost maths; a PCA cross-check against scikit-learn; three
dates checked by hand from raw settles (including 13-Mar-2023, the day after SVB, when the front spread moved 30.75 bp);
and the first trade's P&L recomputed by hand: +1.5 bp gross, -8.1 bp net.

**Bonus - "So what did you learn?"** That a statistically real edge and a tradable edge are different things, and
that the cost of crossing four or eight legs is the first thing to check on any curve RV idea.

---

## 7. Walk me through the code

`config.yaml` - every parameter: data dates, positions, families, PCA windows, model and backtest settings, costs,
regimes, chart events.

`src/sofr_pca/`
* `config.py` - `load_config()`: reads the YAML and makes paths absolute from the repo root.
* `data.py` - `load_raw()` reads the two CSVs, keeps quarterly contracts; `build_curve()` assigns contracts to positions
  each day and builds `levels` (bp), roll-safe `changes`, the `contracts` map and a fixed-contract `rates` table;
  `roll_days()` flags the first session after each expiry; `regime_labels()` tags dates hiking/hold/cutting/2026.
* `structures.py` - `Structure` (family, first position, weights; properties for legs, centre, contract count);
  `build_universe()` creates every spread/fly/double fly that fits in 12 positions; `structure_values()` applies the
  weights to a levels or changes table.
* `pca.py` - `fit_pca()` (covariance -> `eigh` -> sorted, sign-fixed loadings; `PCAModel.scores()` projects levels);
  `rolling_explained()`; `pca_neutral_weights()` solves the wing weights.
* `residuals.py` - `fit_residual_model()` regresses all structures on PC scores in one least-squares call and stores
  a, b, sigma, phi, half-life; `ResidualModel.residuals()/zscores()`; `rolling_z()` the trailing z-score.
* `meanrev.py` - `ar1_phi()`, `half_life()`, `adf_pvalue()`.
* `backtest.py` - `BTParams` and `params_from_config()`; `fit_blocks()` fits one frozen model per quarter and the
  z-scores it produces; `run_backtest()` is the day-by-day simulator (execute yesterday's signals, check exits, take new
  entries with the region cap and shared-leg rule); `_assemble()` builds the trade table and daily gross/net P&L;
  `summary()`, `by_group()`, `pnl_by_regime()`, `net_with_cost()` for reporting.
* `analysis.py` - `run_analysis()`: the whole study; returns every table the scripts and notebook use
  (`pca_section`, `structure_stats`, `pca_vs_traditional`, `signal_quality`, `backtest_section`).
* `screen.py` - `screen_today()` fits on the latest window and ranks structures with target/stop/hit rates;
  `top_rich_cheap()`.
* `plots.py` - one style for every chart; matplotlib PNG and Plotly versions of each figure; `write_report()`.

`scripts/`
* `run_all.py` - runs the analysis, writes `reports/results/*.csv` and `summary.json`, the PNGs, `report.html`, then
  re-executes the notebook.
* `screen_today.py` - the desk output for the latest date.
* `build_notebook.py` - the walkthrough notebook's cells; executes and saves it.

`tests/` - `conftest.py` builds a small synthetic SR3 market so tests run without licensed data;
`test_rolls.py`, `test_no_lookahead.py`, `test_structures_and_costs.py`, `test_models.py`; `test_spot_checks.py`
runs only when `data/raw` is present.
