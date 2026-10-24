"""PCA, PCA-neutral weights and mean-reversion statistics on data with known answers."""
import numpy as np
import pandas as pd
import pytest

from sofr_pca.meanrev import adf_pvalue, ar1_phi, half_life
from sofr_pca.pca import fit_pca, pca_neutral_weights


def _factor_changes(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.arange(1, 13)
    level = np.ones(12) / np.sqrt(12)
    slope = (pos - pos.mean()); slope = slope / np.linalg.norm(slope)
    x = rng.normal(0, 6, (n, 1)) * level + rng.normal(0, 2, (n, 1)) * slope + rng.normal(0, 0.3, (n, 12))
    return pd.DataFrame(x, columns=pos)


def test_pca_recovers_level_and_slope():
    ch = _factor_changes()
    m = fit_pca(ch, ch.cumsum(), 3)
    assert m.explained[0] > 0.85
    assert np.all(m.loadings["PC1"] > 0)                       # sign convention: PC1 = all rates up
    assert m.loadings["PC2"].iloc[-1] > m.loadings["PC2"].iloc[0]  # sign convention: PC2 = steepener
    assert np.allclose(np.linalg.norm(m.loadings, axis=0), 1.0)


def test_pca_matches_scikit_learn():
    """Our eigen-decomposition gives the same explained variance and (up to sign) the same vectors as sklearn."""
    from sklearn.decomposition import PCA
    ch = _factor_changes()
    m = fit_pca(ch, ch.cumsum(), 3)
    sk = PCA(n_components=3).fit(ch.to_numpy())
    assert np.allclose(m.explained[:3], sk.explained_variance_ratio_, atol=1e-10)
    for k in range(3):
        assert abs(abs(m.loadings.iloc[:, k].to_numpy() @ sk.components_[k]) - 1) < 1e-8


def test_pca_neutral_fly_has_no_pc1_pc2_exposure():
    ch = _factor_changes()
    m = fit_pca(ch, ch.cumsum(), 3)
    w = pca_neutral_weights(m, [4, 5, 6], [1, -2, 1])
    assert w[1] == -2
    L = m.loadings.loc[[4, 5, 6]]
    assert abs(w @ L["PC1"]) < 1e-10 and abs(w @ L["PC2"]) < 1e-10
    w2 = pca_neutral_weights(m, [3, 7], [1, -1])                 # spread: PC1-neutral hedge ratio
    assert abs(w2 @ m.loadings.loc[[3, 7], "PC1"]) < 1e-10


def test_half_life_of_known_ar1():
    rng = np.random.default_rng(3)
    x = np.zeros(20000)
    for t in range(1, len(x)):
        x[t] = 0.8 * x[t - 1] + rng.normal()
    phi = ar1_phi(x)
    assert phi == pytest.approx(0.8, abs=0.01)
    assert half_life(phi) == pytest.approx(np.log(0.5) / np.log(0.8), rel=0.05)   # ~3.1 days
    assert adf_pvalue(x[:2000]) < 0.01


def test_random_walk_has_infinite_half_life_or_high_adf():
    rng = np.random.default_rng(5)
    rw = np.cumsum(rng.normal(size=3000))
    assert half_life(1.0) == np.inf and half_life(-0.2) == np.inf
    assert adf_pvalue(rw) > 0.05
