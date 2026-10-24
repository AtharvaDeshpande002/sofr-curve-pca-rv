"""SOFR curve PCA and relative-value (rich/cheap) toolkit.

Modules, in the order the pipeline uses them:
    config      - load config.yaml
    data        - raw settles -> constant-position curve of rates (bp) and roll-safe daily changes
    structures  - spreads, flies, double flies as weight vectors on positions
    pca         - principal components of daily curve changes
    residuals   - rich/cheap residuals and z-scores from regressing structures on PC scores
    meanrev     - AR(1) half-life and ADF stationarity test
    backtest    - walk-forward, net-of-cost mean-reversion backtest
    screen      - today's ranked rich/cheap table
    plots       - matplotlib (static) and plotly (interactive) charts
"""

__all__ = ["config", "data", "structures", "pca", "residuals", "meanrev", "backtest", "screen", "plots"]
