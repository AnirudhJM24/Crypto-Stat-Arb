"""Divergence of a token (or portfolio) from its eigenportfolio."""

from __future__ import annotations

import pandas as pd

from .eigen import eigen_portfolio
from .standardize import standardize, portfolio_returns


def divergence(market, asset, when, window: int = 240, component: int = 0,
               weights=None) -> pd.Series:
    """Spread of standardized returns: Y_asset(t) - Y_eigen(t).

    `asset` is a ticker, or a name for the basket described by `weights`.
    """
    portfolio = eigen_portfolio(market, when, window=window, component=component)
    stamp = portfolio.asof

    if weights is not None:
        raw = market.window(stamp, window=window, assets=list(weights))
        series = portfolio_returns(raw, weights).rename(asset)
    else:
        series = market.window(stamp, window=window, assets=[asset])[asset]

    panel = pd.DataFrame({asset: series, portfolio.name: portfolio.returns})
    Y = standardize(panel)

    spread = (Y[asset] - Y[portfolio.name]).dropna()
    spread = spread.rename(f"{asset} - {portfolio.name}")
    spread.attrs.update(asof=stamp, window=window, asset=asset,
                        component=component)
    return spread


def divergence_stats(spread: pd.Series) -> pd.Series:
    """Shape of the divergence distribution, including tail frequencies."""
    sigma = spread.std(ddof=1)
    z = spread.abs() / sigma
    return pd.Series({
        "n": float(len(spread)),
        "mean": spread.mean(),
        "std": sigma,
        "skew": spread.skew(),
        "kurtosis": spread.kurtosis(),
        "min": spread.min(),
        "q05": spread.quantile(0.05),
        "q25": spread.quantile(0.25),
        "median": spread.median(),
        "q75": spread.quantile(0.75),
        "q95": spread.quantile(0.95),
        "max": spread.max(),
        "pct_beyond_1sd": (z >= 1).mean(),
        "pct_beyond_2sd": (z >= 2).mean(),
        "pct_beyond_3sd": (z >= 3).mean(),
    }, name=spread.name)


def separation_mask(spread: pd.Series, threshold: float = 1.5) -> pd.Series:
    """True where the spread exceeds `threshold` standard deviations. The mask is
    extended one bar forward so plotted segments stay connected."""
    mask = spread.abs() >= threshold * spread.std(ddof=1)
    return mask | mask.shift(-1, fill_value=False)


def separation_regions(spread: pd.Series, threshold: float = 1.5,
                       min_len: int = 1) -> pd.DataFrame:
    """Contiguous runs of separation, one row each."""
    flag = separation_mask(spread, threshold)
    groups = (flag != flag.shift()).cumsum()

    rows = []
    for _, run in spread[flag].groupby(groups[flag]):
        if len(run) < min_len:
            continue
        peak = run.loc[run.abs().idxmax()]
        rows.append({"start": run.index[0], "end": run.index[-1], "bars": len(run),
                     "peak": peak, "direction": "above" if peak > 0 else "below"})

    return pd.DataFrame(rows, columns=["start", "end", "bars", "peak", "direction"])
