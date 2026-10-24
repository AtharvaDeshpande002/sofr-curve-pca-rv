# Resume bullets (numbers from `reports/results/summary.json`)

* Built a PCA-based relative-value model of the SOFR futures curve (12 quarterly SR3 contracts, 2018-2026,
  roll-safe fixed-contract changes); three factors explain 99.3% of daily curve moves, and fly/double-fly residuals
  mean-revert out of sample with 3-7 day half-lives and 53-60% ten-day hit rates.

* Designed a walk-forward, net-of-cost backtest of butterfly mean reversion (quarterly refits, next-day execution,
  overlap and expiry controls, 12-setting robustness grid); showed the 0.25 bp/trade gross edge (gross Sharpe 0.49,
  361 trades, 2020-2026) is wiped out by execution costs (net Sharpe -6.3; break-even below $0.90 per contract),
  and packaged the model as a daily rich/cheap screen for timing discretionary RV entries.

* Delivered the research as a reproducible Python package (pytest checks for roll handling, look-ahead and P&L maths,
  hand-verified spot checks) with an interactive Plotly report and a plain-language interview guide.
