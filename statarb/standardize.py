"""Standardization of returns, for single assets and for portfolios."""

from __future__ import annotations

import numpy as np
import pandas as pd


def standardize(returns, ddof: int = 1):
    """Standardize returns over the estimation window.

        Rbar_i   = (1/T) * sum_t R_i(t)
        sigbar_i = sqrt( (1/(T-1)) * sum_t (R_i(t) - Rbar_i)^2 )
        Y_i(t)   = (R_i(t) - Rbar_i) / sigbar_i

    Accepts a Series (one asset) or a DataFrame (column-wise). Columns with zero
    or undefined volatility come back as NaN rather than +/-inf.
    """
    mu = returns.mean(axis=0)
    sigma = returns.std(axis=0, ddof=ddof)
    if isinstance(sigma, pd.Series):
        sigma = sigma.replace(0, np.nan)
    elif sigma == 0:
        sigma = np.nan
    return (returns - mu) / sigma


def portfolio_returns(returns: pd.DataFrame, weights) -> pd.Series:
    """Weighted return of a basket. `weights` may be a dict, Series, or array
    aligned to `returns.columns`; missing tokens are dropped."""
    if isinstance(weights, dict):
        weights = pd.Series(weights, dtype=float)
    elif not isinstance(weights, pd.Series):
        weights = pd.Series(np.asarray(weights, dtype=float), index=returns.columns)

    cols = [c for c in weights.index if c in returns.columns]
    if not cols:
        raise KeyError("none of the weighted tokens are in the returns frame")
    w = weights.loc[cols]
    return (returns[cols] * w).sum(axis=1, min_count=1)


def equal_weights(assets) -> pd.Series:
    assets = list(assets)
    return pd.Series(1.0 / len(assets), index=assets)


def standardized_panel(market, assets=None, when=None, window: int = 240,
                       weights=None, portfolio_name: str = "portfolio",
                       extra: pd.Series | None = None) -> pd.DataFrame:
    """Standardized returns over one estimation window, ready to plot.

    `assets` are plotted individually; `weights` (or `weights=True` for equal
    weight) adds a portfolio column built from them; `extra` appends any other
    return series, e.g. an eigenportfolio.
    """
    when = market.returns.index[-1] if when is None else when
    assets = list(assets) if assets is not None else market.members(when)
    raw = market.window(when, window=window, assets=assets)

    panel = raw.copy()
    if weights is not None:
        w = equal_weights(raw.columns) if weights is True else weights
        panel[portfolio_name] = portfolio_returns(raw, w)
    if extra is not None:
        panel[extra.name or "extra"] = extra

    return standardize(panel)
