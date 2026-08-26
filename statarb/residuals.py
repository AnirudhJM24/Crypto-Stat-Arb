"""Residuals of each token against the eigenportfolio factors.

Standardized returns are regressed on the leading eigenportfolios,

    Y_i(t) = alpha_i + sum_j beta_ij F_j(t) + eps_i(t),

and the residual eps_i is what a mean-reversion signal trades. Its running sum
X_i(t) = sum_{s<=t} eps_i(s) is the cumulative residual (the Avellaneda-Lee
auxiliary process).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .eigen import correlation_matrix, decompose, eigen_portfolios
from .standardize import standardize


@dataclass
class ResidualModel:
    """Factor fit for one estimation window."""

    residuals: pd.DataFrame     # rows = time, cols = token
    betas: pd.DataFrame         # rows = token, cols = factor
    alphas: pd.Series           # rows = token
    factors: pd.DataFrame       # standardized eigenportfolio returns
    standardized: pd.DataFrame  # standardized token returns that were fitted
    asof: pd.Timestamp
    window: int

    @property
    def cumulative(self) -> pd.DataFrame:
        """X_i(t) = running sum of the residuals."""
        return self.residuals.cumsum()

    @property
    def r_squared(self) -> pd.Series:
        """Share of each token's variance explained by the factors."""
        var = self.standardized.var(ddof=1)
        return (1 - self.residuals.var(ddof=1) / var).rename("r_squared")

    def zscore(self) -> pd.Series:
        """Latest cumulative residual in std devs of its own window."""
        cum = self.cumulative
        return ((cum.iloc[-1] - cum.mean()) / cum.std(ddof=1)).rename("z")

    def ranked(self, ascending: bool = True) -> pd.Series:
        """Tokens sorted by that z-score - most negative is the cheapest leg."""
        return self.zscore().sort_values(ascending=ascending)


def factor_returns(market, when, window: int = 240, n_factors: int = 1,
                   assets=None, variance: float | None = None) -> pd.DataFrame:
    """Standardized returns of the leading `n_factors` eigenportfolios."""
    corr = correlation_matrix(market, when, window=window, assets=assets)
    if variance is not None:
        n_factors = decompose(corr).n_components(variance)
    portfolios = eigen_portfolios(
        market, corr.attrs["asof"], window=window,
        components=range(n_factors), assets=list(corr.index),
    )
    return pd.DataFrame({pf.name: pf.standardized for pf in portfolios})


def n_factors_for(market, when, window: int = 240, variance: float = 0.90,
                  assets=None) -> int:
    """How many components it takes to explain `variance` of the correlation."""
    corr = correlation_matrix(market, when, window=window, assets=assets)
    return decompose(corr).n_components(variance)


def fit_residuals(market, when=None, window: int = 240, n_factors: int = 1,
                  assets=None, variance: float | None = None,
                  include=None) -> ResidualModel:
    """Regress every token in the universe on the leading eigenportfolios.

    `assets` defaults to the top-40 universe at `when`, and also defines the
    factors. `include` fits extra tokens that are outside that universe (ETH, for
    one, is absent from the universe file at many hours). Pass `variance` (e.g.
    0.90) instead of `n_factors` to size the factor set by explained variance.
    """
    when = market.returns.index[-1] if when is None else when
    stamp = market.asof(when)
    F = factor_returns(market, stamp, window=window, n_factors=n_factors,
                       assets=assets, variance=variance)
    universe = list(assets) if assets is not None else market.members(stamp)
    if include is not None:
        include = [include] if isinstance(include, str) else list(include)
        universe += [t for t in include
                     if t in market.tokens and t not in universe]
    Y = standardize(market.window(stamp, window=window, assets=universe))
    Y = Y.dropna(axis=1, how="all").reindex(F.index)

    X = np.column_stack([np.ones(len(F)), F.to_numpy()])
    resid, betas, alphas = {}, {}, {}
    for token in Y.columns:
        y = Y[token]
        ok = y.notna() & np.isfinite(X).all(axis=1)
        if ok.sum() <= X.shape[1]:
            continue
        coef, *_ = np.linalg.lstsq(X[ok.to_numpy()], y[ok].to_numpy(), rcond=None)
        alphas[token] = coef[0]
        betas[token] = coef[1:]
        resid[token] = y - (X @ coef)

    return ResidualModel(
        residuals=pd.DataFrame(resid, index=Y.index),
        betas=pd.DataFrame(betas, index=F.columns).T,
        alphas=pd.Series(alphas, name="alpha"),
        factors=F,
        standardized=Y[list(resid)],
        asof=stamp,
        window=window,
    )
