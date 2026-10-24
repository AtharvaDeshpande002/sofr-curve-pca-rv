"""Principal component analysis of daily curve changes.

Why changes and not levels: rate levels trend for years (they are not stationary), so a PCA of levels is
dominated by the trend and its "factors" change meaning with the sample. Daily changes are close to
stationary, so their covariance describes how the curve actually moves day to day:
    PC1 ~ level  (all rates move together: the market re-prices the whole Fed path)
    PC2 ~ slope  (front and back move in opposite directions: steepening / flattening)
    PC3 ~ curvature (the belly moves against both wings)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PCAModel:
    """A fitted PCA: loadings are unit-length columns, one per component."""

    loadings: pd.DataFrame        # position x PC (PC1, PC2, ...)
    eigvals: np.ndarray           # variance of each component (bp^2 per day), all components
    level_mean: pd.Series         # mean curve level over the window (bp) - the centre for level scores

    @property
    def explained(self) -> np.ndarray:
        """Share of total daily variance explained by each component."""
        return self.eigvals / self.eigvals.sum()

    def scores(self, levels: pd.DataFrame) -> pd.DataFrame:
        """Project (level - mean) onto the loadings: how much of each 'shape' the curve holds today."""
        x = levels.to_numpy() - self.level_mean.to_numpy()
        return pd.DataFrame(x @ self.loadings.to_numpy(), index=levels.index, columns=self.loadings.columns)


def _orient(vecs: np.ndarray) -> np.ndarray:
    """Fix the arbitrary sign of each eigenvector so charts are comparable across windows.

    PC1 positive on average (all rates up), PC2 positive at the back minus front (steepener),
    PC3 positive in the belly relative to the wings.
    """
    n = vecs.shape[0]
    out = vecs.copy()
    if out[:, 0].sum() < 0:
        out[:, 0] *= -1
    if out.shape[1] > 1 and out[-1, 1] - out[0, 1] < 0:
        out[:, 1] *= -1
    if out.shape[1] > 2:
        mid = out[n // 3: 2 * n // 3, 2].mean()
        wings = (out[: n // 3, 2].mean() + out[2 * n // 3:, 2].mean()) / 2
        if mid - wings < 0:
            out[:, 2] *= -1
    return out


def fit_pca(changes: pd.DataFrame, levels: pd.DataFrame, n_components: int = 3) -> PCAModel:
    """Eigen-decompose the covariance of daily changes (rows with any NaN dropped).

    ``levels`` (same window) is only used to store the mean level, the centre for level scores.
    """
    x = changes.dropna().to_numpy()
    cov = np.cov(x, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)                 # eigh: symmetric matrix, ascending order
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], _orient(vecs[:, order])
    cols = [f"PC{k + 1}" for k in range(n_components)]
    load = pd.DataFrame(vecs[:, :n_components], index=changes.columns, columns=cols)
    return PCAModel(loadings=load, eigvals=vals, level_mean=levels.mean())


def rolling_explained(changes: pd.DataFrame, levels: pd.DataFrame, window: int, step: int, k: int = 3) -> pd.DataFrame:
    """Explained-variance share of PC1..PCk re-estimated every ``step`` days on a trailing window."""
    rows = {}
    for end in range(window, len(changes) + 1, step):
        m = fit_pca(changes.iloc[end - window:end], levels.iloc[end - window:end], k)
        rows[changes.index[end - 1]] = m.explained[:k]
    return pd.DataFrame(rows, index=[f"PC{i + 1}" for i in range(k)]).T


def pca_neutral_weights(model: PCAModel, positions: list[int], weights: list[float]) -> np.ndarray:
    """Re-weight a structure so it has zero exposure to the curve's actual PC moves.

    * Flies and double flies (3+ legs): keep the inner legs fixed (e.g. -2 for a fly, -3/+3 for a double
      fly) and solve the two outer weights so that PC1 and PC2 exposure are both zero:
          w_first * L(first) + w_last * L(last) = -(inner legs' exposure)   for PC1 and PC2.
    * Spreads (2 legs): keep the front leg at 1 and size the back leg so PC1 (level) exposure is zero.
    Why: a 1:-2:1 fly is only immune to a perfectly parallel move. Real "level" moves are not parallel
    (the front moves less than the belly), so the textbook fly carries some hidden level/slope risk.
    """
    L = model.loadings
    w = np.array(weights, dtype=float)
    if len(positions) == 2:
        w[1] = -L.loc[positions[0], "PC1"] / L.loc[positions[1], "PC1"]
        return w
    pcs = ["PC1", "PC2"]
    inner = sum(w[j] * L.loc[positions[j], pcs].to_numpy() for j in range(1, len(positions) - 1))
    a = np.column_stack([L.loc[positions[0], pcs].to_numpy(), L.loc[positions[-1], pcs].to_numpy()])
    w[0], w[-1] = np.linalg.solve(a, -inner)
    return w


def pca_neutral_fly(model: PCAModel, first: int, middle_weight: float = -2.0) -> np.ndarray:
    """PCA-neutral weights for the fly on (first, first+1, first+2) with the middle fixed at -2."""
    return pca_neutral_weights(model, [first, first + 1, first + 2], [1.0, middle_weight, 1.0])
