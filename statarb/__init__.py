"""Crypto stat-arb toolkit: standardized returns, eigenportfolios, divergence."""

from .data import Market, to_utc
from .divergence import (divergence, divergence_stats, separation_mask,
                         separation_regions)
from .eigen import (Eigen, EigenPortfolio, correlation_matrix, decompose,
                    eigen_portfolio, eigen_portfolios)
from .ar1 import AR1Fit, fit_ar1, residual_ar1, rolling_ar1
from .backtest import (BacktestResult, ResidualPath, backtest_residual_path,
                       backtest_token, prepare_residual_path,
                       search_ar_windows, search_entry_exit)
from .pipeline import TokenAnalysis, analyze_token
from .residuals import (ResidualModel, factor_returns, fit_residuals,
                        n_factors_for)
from .standardize import (equal_weights, portfolio_returns, standardize,
                          standardized_panel)

_PLOT_EXPORTS = []
try:
    from .plots import (plot_cumulative_residuals, plot_residuals,
                        plot_token_residual,
                        plot_cumulative_vs_eigen, plot_divergence_distribution,
                        plot_eigen_portfolio, plot_eigen_spectrum,
                        plot_standardized_returns, plot_vs_eigen)
    _PLOT_EXPORTS = [
        "plot_standardized_returns", "plot_eigen_spectrum",
        "plot_eigen_portfolio", "plot_cumulative_vs_eigen", "plot_vs_eigen",
        "plot_divergence_distribution", "plot_residuals",
        "plot_cumulative_residuals", "plot_token_residual",
    ]
except ModuleNotFoundError as exc:
    if exc.name != "matplotlib":
        raise

__all__ = [
    "Market", "to_utc",
    "standardize", "standardized_panel", "portfolio_returns", "equal_weights",
    "correlation_matrix", "decompose", "eigen_portfolio", "eigen_portfolios",
    "Eigen", "EigenPortfolio",
    "divergence", "divergence_stats", "separation_mask", "separation_regions",
    "fit_residuals", "factor_returns", "n_factors_for", "ResidualModel",
    "AR1Fit", "fit_ar1", "residual_ar1", "rolling_ar1",
    "BacktestResult", "ResidualPath", "backtest_token",
    "prepare_residual_path", "backtest_residual_path", "search_ar_windows",
    "search_entry_exit",
    "TokenAnalysis", "analyze_token",
] + _PLOT_EXPORTS
