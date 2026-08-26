"""Plot layer. Every function takes a Market, draws on an Axes, and returns it."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .divergence import divergence, divergence_stats, separation_mask
from .eigen import correlation_matrix, decompose, eigen_portfolio
from .residuals import fit_residuals
from .standardize import standardize, standardized_panel

WIDE = (12, 4)


def _axes(ax, figsize=WIDE):
    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=figsize)
    return ax


def _zero_line(ax):
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)


def _finish(ax, title, ylabel, legend=True):
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    return ax


def _cumulative(panel):
    """Compound standardized series so paths are comparable."""
    return (1 + panel.fillna(0)).cumprod()


def plot_standardized_returns(market, assets=None, when=None, window: int = 240,
                              weights=None, portfolio_name: str = "portfolio",
                              only_portfolio: bool = False, ax=None):
    """Standardized returns of one or more tokens, and/or of a portfolio of them.

    weights=True equal-weights `assets`; a dict/Series gives custom weights;
    only_portfolio=True hides the individual legs.
    """
    if isinstance(assets, str):
        assets = [assets]
    panel = standardized_panel(market, assets=assets, when=when, window=window,
                               weights=weights, portfolio_name=portfolio_name)
    if only_portfolio:
        panel = panel[[portfolio_name]]

    ax = _axes(ax)
    panel.plot(ax=ax, linewidth=0.9)
    _zero_line(ax)
    stamp = panel.index[-1]
    return _finish(ax, f"standardized returns as of {stamp:%Y-%m-%d %H:%M} "
                       f"({window}h window)", "standardized return")


def plot_eigen_spectrum(market, when=None, window: int = 240, threshold: float = 0.90,
                        top: int | None = 20, ax=None):
    """Scree plot of the correlation eigenvalues with cumulative variance."""
    when = market.returns.index[-1] if when is None else when
    corr = correlation_matrix(market, when, window=window)
    eig = decompose(corr)

    values = eig.values if top is None else eig.values.iloc[:top]
    explained = eig.explained.reindex(values.index)
    cumulative = eig.cumulative.reindex(values.index)

    ax = _axes(ax, figsize=(10, 4))
    ax.bar(values.index, explained, color="0.6", label="variance share")
    ax.set_ylabel("variance share")
    ax.set_xlabel("component")

    twin = ax.twinx()
    twin.plot(cumulative.index, cumulative, color="tab:blue", marker="o",
              markersize=3, linewidth=1, label="cumulative")
    twin.axhline(threshold, color="red", linestyle="--", linewidth=1,
                 label=f"{threshold:.0%}")
    twin.set_ylabel("cumulative variance")
    twin.set_ylim(0, 1.02)

    n = eig.n_components(threshold)
    ax.set_title(f"eigen spectrum as of {eig.asof:%Y-%m-%d %H:%M} ({window}h) - "
                 f"PC1 {eig.top_share:.1%}, {n} components reach {threshold:.0%}")
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="center right", fontsize=8)
    return ax


def plot_eigen_portfolio(market, when=None, window: int = 240, component: int = 0,
                         ax=None):
    """Standardized returns of the eigenportfolio itself (not cumulative)."""
    when = market.returns.index[-1] if when is None else when
    pf = eigen_portfolio(market, when, window=window, component=component)

    ax = _axes(ax)
    pf.standardized.plot(ax=ax, linewidth=1.0, color="tab:blue")
    _zero_line(ax)
    return _finish(ax, f"eigenportfolio {component} standardized returns "
                       f"as of {pf.asof:%Y-%m-%d %H:%M} ({window}h)",
                   "standardized return", legend=False)


def plot_cumulative_vs_eigen(market, assets=("BTC", "ETH"), when=None,
                             window: int = 240, component: int = 0, ax=None):
    """Cumulative standardized returns of token(s) against the eigenportfolio."""
    if isinstance(assets, str):
        assets = [assets]
    when = market.returns.index[-1] if when is None else when
    pf = eigen_portfolio(market, when, window=window, component=component)

    panel = standardized_panel(market, assets=list(assets), when=pf.asof,
                               window=window, extra=pf.returns)
    ax = _axes(ax)
    _cumulative(panel).plot(ax=ax, linewidth=1.0)
    return _finish(ax, f"cumulative standardized vs eigenportfolio {component} "
                       f"as of {pf.asof:%Y-%m-%d %H:%M} ({window}h)",
                   "cumulative growth of 1")


def plot_vs_eigen(market, assets=("BTC", "ETH"), when=None, window: int = 240,
                  component: int = 0, threshold: float | None = 1.5, ax=None):
    """Standardized returns of token(s) against the eigenportfolio (not cumulative).

    With a single asset and a `threshold`, the stretches where it pulls away from
    the eigenportfolio are drawn in red.
    """
    if isinstance(assets, str):
        assets = [assets]
    assets = list(assets)
    when = market.returns.index[-1] if when is None else when
    pf = eigen_portfolio(market, when, window=window, component=component)

    panel = standardized_panel(market, assets=assets, when=pf.asof, window=window,
                               extra=pf.returns)
    ax = _axes(ax)
    ax.plot(panel.index, panel[pf.name], color="tab:blue", linewidth=1.0,
            label=pf.name)
    for asset in assets:
        ax.plot(panel.index, panel[asset], linewidth=1.0, label=asset)

    if threshold is not None and len(assets) == 1:
        asset = assets[0]
        spread = divergence(market, asset, pf.asof, window=window,
                            component=component)
        mask = separation_mask(spread, threshold).reindex(panel.index, fill_value=False)
        ax.plot(panel.index, panel[asset].where(mask), color="red", linewidth=2.0,
                label=f"separation >{threshold} sd")

    _zero_line(ax)
    return _finish(ax, f"{', '.join(assets)} vs eigenportfolio {component} "
                       f"as of {pf.asof:%Y-%m-%d %H:%M} ({window}h)",
                   "standardized return")


def plot_divergence_distribution(market, asset="BTC", when=None, window: int = 240,
                                 component: int = 0, bins: int = 50, ax=None):
    """Histogram of a token's divergence from the eigenportfolio, with sd markers."""
    when = market.returns.index[-1] if when is None else when
    spread = divergence(market, asset, when, window=window, component=component)
    stats = divergence_stats(spread)
    sigma = stats["std"]

    ax = _axes(ax, figsize=(8, 4))
    ax.hist(spread, bins=bins, color="0.6", edgecolor="white")
    for k, style in ((1, ":"), (2, "--")):
        for sign in (-1, 1):
            ax.axvline(sign * k * sigma, color="red", linestyle=style, linewidth=1,
                       label=f"{k} sd" if sign > 0 else None)
    ax.axvline(stats["mean"], color="tab:blue", linewidth=1, label="mean")

    ax.set_xlabel("divergence (standardized return spread)")
    ax.set_ylabel("bars")
    ax.set_title(f"{asset} divergence from eigenportfolio {component} as of "
                 f"{spread.attrs['asof']:%Y-%m-%d %H:%M} ({window}h) - "
                 f"skew {stats['skew']:.2f}, kurtosis {stats['kurtosis']:.2f}")
    ax.legend(fontsize=8)
    return ax


def _residual_frame(model, cumulative, assets):
    frame = model.cumulative if cumulative else model.residuals
    if assets is not None:
        assets = [assets] if isinstance(assets, str) else list(assets)
        missing = [a for a in assets if a not in frame.columns]
        if missing:
            raise KeyError(f"no residuals fitted for: {missing}")
        frame = frame[assets]
    return frame


def plot_residuals(market, when=None, window: int = 240, n_factors: int = 1,
                   assets=None, universe=None, cumulative: bool = False,
                   variance: float | None = None, highlight=None, model=None, ax=None):
    """Residual returns of every token in the universe against the eigenportfolios.

    cumulative=True plots the running sum X_i(t) instead. `assets` narrows the
    lines drawn (the fit still uses the whole universe); `highlight` draws those
    tokens in color with the rest greyed out.
    """
    if model is None:
        model = fit_residuals(market, when=when, window=window, n_factors=n_factors,
                              assets=universe, variance=variance, include=assets)
    frame = _residual_frame(model, cumulative, assets)

    ax = _axes(ax)
    if highlight:
        highlight = [highlight] if isinstance(highlight, str) else list(highlight)
        rest = [c for c in frame.columns if c not in highlight]
        if rest:
            ax.plot(frame.index, frame[rest], color="0.8", linewidth=0.6)
        for token in highlight:
            ax.plot(frame.index, frame[token], linewidth=1.4, label=token)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    else:
        frame.plot(ax=ax, linewidth=0.7,
                   legend=frame.shape[1] <= 10)
        if frame.shape[1] <= 10:
            ax.legend(loc="upper left", fontsize=8, ncol=2)

    _zero_line(ax)
    kind = "cumulative residual" if cumulative else "residual"
    k = model.factors.shape[1]
    return _finish(ax, f"{kind} vs {k} eigenportfolio factor(s) as of "
                       f"{model.asof:%Y-%m-%d %H:%M} ({window}h, "
                       f"{frame.shape[1]} tokens)",
                   "cumulative residual" if cumulative else "residual return",
                   legend=False)


def plot_cumulative_residuals(market, **kwargs):
    """plot_residuals with the running sum X_i(t)."""
    kwargs["cumulative"] = True
    return plot_residuals(market, **kwargs)


def plot_token_residual(market, token, when=None, window: int = 240,
                        n_factors: int = 1, model=None, axes=None):
    """One token: residual returns on top, cumulative residual below."""
    if model is None:
        model = fit_residuals(market, when=when, window=window,
                              n_factors=n_factors, include=[token])
    if token not in model.residuals.columns:
        raise KeyError(f"no residuals fitted for {token}")

    eps = model.residuals[token]
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    top, bottom = axes

    top.plot(eps.index, eps, linewidth=0.9, color="tab:blue")
    _zero_line(top)
    top.set_ylabel("residual return")
    top.set_title(f"{token} residual vs {model.factors.shape[1]} eigenportfolio "
                  f"factor(s) as of {model.asof:%Y-%m-%d %H:%M} ({window}h)")

    cum = eps.cumsum()
    bottom.plot(cum.index, cum, linewidth=1.1, color="tab:purple")
    _zero_line(bottom)
    bottom.set_ylabel("cumulative residual")
    bottom.set_xlabel("")
    return axes
