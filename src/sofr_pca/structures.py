"""Spreads, butterflies and double flies as weight vectors on curve positions.

Convention (stated once, used everywhere): everything is in RATE terms, front minus back.
    3M spread  = r(i) - r(i+1)
    1Y spread  = r(i) - r(i+4)
    fly        = r(i) - 2 r(i+1) + r(i+2)
    double fly = r(i) - 3 r(i+1) + 3 r(i+2) - r(i+3)
A structure's level (or change) is simply the weighted sum of the leg rates (or leg changes).
Because the weights are integers, one "unit" is a real package: a fly is 1 / 2 / 1 contracts, a double
fly 1 / 3 / 3 / 1, and every 1 bp move of the structure is worth $25 per unit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LABEL = {"spread3m": "SPR", "spread1y": "1Y", "fly": "FLY", "dfly": "DFLY"}


@dataclass(frozen=True)
class Structure:
    """One tradable structure on consecutive positions starting at ``first``."""

    family: str
    first: int                                # position of the first (front) leg, 1-based
    weights: tuple[float, ...]                # weights on positions first, first+1, ...
    name: str = field(default="", compare=False)

    @property
    def positions(self) -> list[int]:
        return [self.first + j for j, w in enumerate(self.weights) if w != 0]

    @property
    def leg_weights(self) -> list[float]:
        return [w for w in self.weights if w != 0]

    @property
    def centre(self) -> float:
        """Centre of the structure on the curve (used to assign a curve region)."""
        return self.first + (len(self.weights) - 1) / 2

    @property
    def n_contracts(self) -> int:
        """Contracts traded per unit: sum of |weights| (fly = 4, double fly = 8, spread = 2)."""
        return int(sum(abs(w) for w in self.weights))

    def vector(self, n_positions: int) -> np.ndarray:
        """Full-length weight vector over positions 1..n (zeros elsewhere)."""
        v = np.zeros(n_positions)
        v[self.first - 1:self.first - 1 + len(self.weights)] = self.weights
        return v


def make_structure(family: str, first: int, weights) -> Structure:
    """Build a Structure with a trader-readable name such as 'FLY 5-6-7' or '1Y 3-7'."""
    w = tuple(float(x) for x in weights)
    legs = [first + j for j, x in enumerate(w) if x != 0]
    return Structure(family, first, w, f"{LABEL.get(family, family.upper())} {'-'.join(map(str, legs))}")


def build_universe(families: dict[str, list[float]], n_positions: int, include: list[str] | None = None) -> list[Structure]:
    """Every structure of each family that fits inside positions 1..n_positions."""
    out = []
    for fam, w in families.items():
        if include is not None and fam not in include:
            continue
        for first in range(1, n_positions - len(w) + 2):
            out.append(make_structure(fam, first, w))
    return out


def structure_matrix(universe: list[Structure], n_positions: int) -> pd.DataFrame:
    """Weight matrix (structures x positions) so that levels @ W.T gives every structure at once."""
    return pd.DataFrame([s.vector(n_positions) for s in universe], index=[s.name for s in universe],
                        columns=range(1, n_positions + 1))


def structure_values(panel: pd.DataFrame, universe: list[Structure]) -> pd.DataFrame:
    """Apply weights to a date x position panel (levels or changes) -> date x structure."""
    W = structure_matrix(universe, panel.shape[1])
    return pd.DataFrame(panel.to_numpy() @ W.to_numpy().T, index=panel.index, columns=W.index)


def region_of(s: Structure, regions: dict[str, list[float]]) -> str:
    """Curve region (front / belly / back) from the structure's centre position."""
    for name, (lo, hi) in regions.items():
        if lo <= s.centre <= hi:
            return name
    return "back" if s.centre > 8 else "front"
