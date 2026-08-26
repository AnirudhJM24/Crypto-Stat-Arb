"""Ordered token analysis: eigen factors, residual, AR(1), then backtest."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .ar1 import AR1Fit, residual_ar1
from .backtest import BacktestResult, backtest_token
from .eigen import Eigen, correlation_matrix, decompose
from .residuals import ResidualModel, fit_residuals


@dataclass
class TokenAnalysis:
    token: str
    eigen: Eigen
    residual_model: ResidualModel
    residual: pd.Series
    cumulative_residual: pd.Series
    ar1: AR1Fit
    backtest: BacktestResult | None = None

    @property
    def factor_count(self) -> int:
        return self.residual_model.factors.shape[1]

    @property
    def regression(self) -> pd.Series:
        """Regression intercept, factor betas, and R-squared for the token."""
        values = {"alpha": self.residual_model.alphas[self.token]}
        values.update(self.residual_model.betas.loc[self.token].to_dict())
        values["r_squared"] = self.residual_model.r_squared[self.token]
        return pd.Series(values, name=self.token)


def analyze_token(market, token: str = "BTC", when=None,
                  estimation_window: int = 240, variance: float | None = None,
                  n_factors: int = 2, ar_window: int = 20, backtest_start=None,
                  backtest_end=None, entry: float = 2.0, exit: float = 0.5,
                  cost_bps: float = 5.0, hedged: bool = True) -> TokenAnalysis:
    """Execute the complete analysis in the statistically required order."""
    when = market.returns.index[-1] if when is None else when
    corr = correlation_matrix(market, when, window=estimation_window)
    eigen = decompose(corr)
    model = fit_residuals(
        market, when=when, window=estimation_window, n_factors=n_factors,
        variance=variance,
        include=[token],
    )
    if token not in model.residuals:
        raise KeyError(f"no residuals fitted for {token}")
    residual = model.residuals[token].rename(token)
    cumulative = residual.cumsum().rename(token)
    ar1 = residual_ar1(model, token, window=ar_window)

    result = None
    if backtest_start is not None or backtest_end is not None:
        result = backtest_token(
            market, token=token, start=backtest_start, end=backtest_end,
            estimation_window=estimation_window, variance=variance,
            n_factors=n_factors,
            ar_window=ar_window, entry=entry, exit=exit, cost_bps=cost_bps,
            hedged=hedged,
        )
    return TokenAnalysis(
        token=token, eigen=eigen, residual_model=model, residual=residual,
        cumulative_residual=cumulative, ar1=ar1, backtest=result,
    )
