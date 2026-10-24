"""Rich / cheap: the part of a structure the curve's normal shapes cannot explain.

For each structure y (e.g. the 5-6-7 fly), over a fitting window:
    y(t) = a + b1 * PC1score(t) + b2 * PC2score(t) + b3 * PC3score(t) + e(t)
The fitted part a + b.s is the structure's "fair value" given today's level, slope and curvature.
The residual e(t) is how far it sits from fair value, in bp. Dividing by the residual's standard
deviation in the fitting window gives the z-score: z = +2 means the structure's rate is 2 sigma higher
than the curve shape implies (in price terms it is cheap to own; we would SELL the rate = buy the package).

Everything is estimated on the fitting window only. Applying the frozen parameters to later dates is
what makes the walk-forward test out-of-sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .meanrev import adf_pvalue, ar1_phi, half_life
from .pca import PCAModel, fit_pca
from .structures import Structure, structure_values


@dataclass
class ResidualModel:
    """PCA + per-structure regression, frozen at the end of a fitting window."""

    pca: PCAModel
    params: pd.DataFrame          # structure x [a, b_PC1.., sigma, phi, half_life, adf_p]
    fit_end: pd.Timestamp         # last date used in the fit (no data after this was seen)

    def fair_value(self, levels: pd.DataFrame, universe: list[Structure]) -> pd.DataFrame:
        s = self.pca.scores(levels).to_numpy()
        p = self.params.loc[[u.name for u in universe]]
        b = p[[c for c in p.columns if c.startswith("b_")]].to_numpy()
        fv = p["a"].to_numpy() + s @ b.T
        return pd.DataFrame(fv, index=levels.index, columns=p.index)

    def residuals(self, levels: pd.DataFrame, universe: list[Structure]) -> pd.DataFrame:
        """Structure level minus fair value (bp)."""
        return structure_values(levels, universe) - self.fair_value(levels, universe)

    def zscores(self, levels: pd.DataFrame, universe: list[Structure]) -> pd.DataFrame:
        """Residual / residual sigma of the fitting window."""
        return self.residuals(levels, universe) / self.params.loc[[u.name for u in universe], "sigma"].to_numpy()


def fit_residual_model(levels: pd.DataFrame, changes: pd.DataFrame, universe: list[Structure],
                       n_components: int = 3) -> ResidualModel:
    """Fit PCA on the window's daily changes, then regress every structure's level on the PC scores."""
    pca = fit_pca(changes, levels, n_components)
    s = pca.scores(levels)
    X = np.column_stack([np.ones(len(s)), s.to_numpy()])
    Y = structure_values(levels, universe)
    coef, *_ = np.linalg.lstsq(X, Y.to_numpy(), rcond=None)       # one OLS per structure, solved together
    resid = Y.to_numpy() - X @ coef
    rows = []
    for j, u in enumerate(universe):
        e = resid[:, j]
        phi = ar1_phi(e)
        row = {"a": coef[0, j], **{f"b_PC{k + 1}": coef[k + 1, j] for k in range(n_components)},
               "sigma": float(e.std(ddof=1)), "phi": phi, "half_life": half_life(phi)}
        rows.append(row)
    params = pd.DataFrame(rows, index=[u.name for u in universe])
    return ResidualModel(pca=pca, params=params, fit_end=levels.index[-1])


def rolling_z(resid: pd.DataFrame, window: int) -> pd.DataFrame:
    """z = (residual - its trailing mean) / its trailing std, using data up to and including each day.

    Why a trailing mean: out of sample the curve visits levels the fitting window never saw, so the
    frozen regression's residual drifts slowly. Measuring the residual against its own recent mean keeps
    the signal about short-term dislocations (what mean-reversion trades) rather than slow drift.
    """
    mu = resid.rolling(window, min_periods=window // 2).mean()
    sd = resid.rolling(window, min_periods=window // 2).std()
    return (resid - mu) / sd


def add_adf(model: ResidualModel, levels: pd.DataFrame, universe: list[Structure]) -> ResidualModel:
    """ADF p-value of each in-window residual (slower, so only computed where it is reported)."""
    r = model.residuals(levels, universe)
    model.params["adf_p"] = [adf_pvalue(r[u.name]) for u in universe]
    return model
