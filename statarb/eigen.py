"""Correlation matrices, eigen decomposition, and eigenportfolios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .standardize import standardize


def correlation_matrix(market, when, window: int = 240, assets=None,
                       min_periods: int | None = None) -> pd.DataFrame:
    """Correlation of standardized returns for the universe at `when`.

    Uses the `window` hours ending at that timestamp; rows/cols keep universe
    rank order. Tokens with no data in the window are dropped.
    """
    stamp = market.asof(when)
    assets = list(assets) if assets is not None else market.members(stamp)
    raw = market.window(stamp, window=window, assets=assets)
    Y = standardize(raw).dropna(axis=1, how="all")

    corr = Y.corr(min_periods=min_periods or max(2, window // 2))
    corr.attrs.update(asof=stamp, window=window)
    return corr


@dataclass
class Eigen:
    """Eigen decomposition of a correlation matrix, sorted descending."""

    values: pd.Series        # indexed by component number
    vectors: pd.DataFrame    # rows = tokens, cols = component number
    asof: pd.Timestamp | None = None
    window: int | None = None

    @property
    def explained(self) -> pd.Series:
        """Variance share of each component."""
        return (self.values / self.values.sum()).rename("explained")

    @property
    def cumulative(self) -> pd.Series:
        return self.explained.cumsum().rename("cumulative")

    @property
    def top_share(self) -> float:
        """Variance explained by the largest eigenvalue alone."""
        return float(self.explained.iloc[0])

    def n_components(self, threshold: float = 0.90) -> int:
        """How many leading components it takes to reach `threshold` variance."""
        return int((self.cumulative >= threshold).argmax()) + 1

    def head(self, threshold: float = 0.90) -> "Eigen":
        """Truncate to the leading components covering `threshold` variance."""
        m = self.n_components(threshold)
        return Eigen(self.values.iloc[:m], self.vectors.iloc[:, :m],
                     self.asof, self.window)

    def vector(self, component: int = 0) -> pd.Series:
        return self.vectors.iloc[:, component]

    def summary(self, threshold: float = 0.90) -> pd.DataFrame:
        return pd.DataFrame({"eigenvalue": self.values,
                             "explained": self.explained,
                             "cumulative": self.cumulative})


def decompose(corr: pd.DataFrame) -> Eigen:
    """Eigenvalues/vectors of a correlation matrix, largest first."""
    corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all").fillna(0)
    values, vectors = np.linalg.eigh(corr.to_numpy())
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    idx = pd.RangeIndex(len(values), name="component")
    return Eigen(values=pd.Series(values, index=idx, name="eigenvalue"),
                 vectors=pd.DataFrame(vectors, index=corr.index, columns=idx),
                 asof=corr.attrs.get("asof"), window=corr.attrs.get("window"))


@dataclass
class EigenPortfolio:
    """One eigenportfolio and the window it was estimated on."""

    returns: pd.Series          # raw (unstandardized) portfolio returns
    weights: pd.Series          # indexed by token
    component: int
    asof: pd.Timestamp
    window: int

    @property
    def name(self) -> str:
        return f"eigen{self.component}"

    @property
    def standardized(self) -> pd.Series:
        return standardize(self.returns).rename(self.name)

    @property
    def cumulative(self) -> pd.Series:
        return (1 + self.returns.fillna(0)).cumprod().rename(self.name)


def eigen_portfolio(market, when, window: int = 240, component: int = 0,
                    assets=None, normalize: bool = True) -> EigenPortfolio:
    """Build an eigenportfolio from `component` of the correlation matrix.

    Avellaneda-Lee weights Q_i = v_i / sigma_i, applied to raw returns:
    F(t) = sum_i Q_i * R_i(t). The eigenvector sign is arbitrary, so it is
    flipped to leave the portfolio net long.
    """
    return eigen_portfolios(
        market, when, window=window, components=[component], assets=assets,
        normalize=normalize,
    )[0]


def eigen_portfolios(market, when, window: int = 240, components=None,
                     assets=None, normalize: bool = True) -> list[EigenPortfolio]:
    """Build several eigenportfolios from one eigendecomposition.

    This is equivalent to repeated :func:`eigen_portfolio` calls, but the
    correlation matrix and its eigendecomposition are calculated only once.
    """
    corr = correlation_matrix(market, when, window=window, assets=assets)
    eig = decompose(corr)
    return eigen_portfolios_from_eigen(
        market, eig, components=components, normalize=normalize,
    )


def eigen_portfolios_from_eigen(market, eig: Eigen, components=None,
                                normalize: bool = True) -> list[EigenPortfolio]:
    """Build portfolios from an existing decomposition without recomputing it."""
    components = list(range(len(eig.values))) if components is None else list(components)
    if any(component < 0 or component >= len(eig.values) for component in components):
        raise IndexError(f"component must be between 0 and {len(eig.values) - 1}")

    if eig.asof is None or eig.window is None:
        raise ValueError("eigen decomposition must carry asof and window metadata")
    stamp = eig.asof
    window = eig.window
    raw = market.window(stamp, window=window, assets=list(eig.vectors.index))
    volatility = raw.std(ddof=1).replace(0, np.nan)
    portfolios = []
    for component in components:
        vector = eig.vector(component)
        weights = (vector.reindex(raw.columns) / volatility).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if weights.empty:
            raise ValueError(f"eigenportfolio {component} has no finite weights")
        if weights.sum() < 0:
            weights = -weights
        if normalize:
            gross = weights.abs().sum()
            if not np.isfinite(gross) or gross == 0:
                raise ValueError(f"eigenportfolio {component} cannot be normalized")
            weights = weights / gross

        returns = (raw[weights.index] * weights).sum(
            axis=1, min_count=1
        ).rename(f"eigen{component}")
        portfolios.append(EigenPortfolio(
            returns=returns, weights=weights, component=component,
            asof=stamp, window=window,
        ))
    return portfolios
