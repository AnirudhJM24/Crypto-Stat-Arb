"""AR(1) / Ornstein-Uhlenbeck estimates for cumulative factor residuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AR1Fit:
    """One discrete AR(1) fit and its continuous-time OU interpretation."""

    intercept: float
    slope: float
    mean: float
    kappa: float
    tau_days: float
    innovation_std: float
    sigma_eq: float
    s_score: float
    nobs: int
    asof: pd.Timestamp

    @property
    def stationary(self) -> bool:
        """Whether the fit implies a conventional mean-reverting OU process."""
        return bool(0 < self.slope < 1 and np.isfinite(self.sigma_eq))

    def to_series(self) -> pd.Series:
        """Return the estimates in a compact, display-friendly form."""
        return pd.Series({
            "a": self.intercept,
            "b": self.slope,
            "m": self.mean,
            "kappa": self.kappa,
            "tau_days": self.tau_days,
            "innovation_std": self.innovation_std,
            "sigma_eq": self.sigma_eq,
            "s_score": self.s_score,
            "nobs": self.nobs,
            "stationary": self.stationary,
        }, name=self.asof)


def fit_ar1(cumulative: pd.Series, window: int | None = 20,
            bars_per_year: int = 24 * 365) -> AR1Fit:
    """Fit ``X[t+1] = a + b X[t] + innovation[t+1]``.

    ``cumulative`` is normally a token's cumulative eigen-factor residual.
    Only information through the series' final timestamp is used. ``window``
    selects the trailing observations; pass ``None`` to use the full series.
    """
    values = pd.Series(cumulative, dtype=float).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if window is not None:
        if window < 3:
            raise ValueError("AR(1) window must contain at least 3 observations")
        values = values.tail(window)
    if len(values) < 3:
        raise ValueError("at least 3 finite observations are required for AR(1)")

    x0 = values.iloc[:-1].to_numpy()
    x1 = values.iloc[1:].to_numpy()
    design = np.column_stack([np.ones(len(x0)), x0])
    intercept, slope = np.linalg.lstsq(design, x1, rcond=None)[0]
    innovations = x1 - (intercept + slope * x0)
    innovation_std = float(np.std(innovations, ddof=1))

    mean = (intercept / (1 - slope)
            if not np.isclose(slope, 1.0) else np.nan)
    if 0 < slope < 1:
        kappa = float(-np.log(slope) * bars_per_year)
        tau_days = float(365 / kappa)
    else:
        kappa = tau_days = np.nan
    sigma_eq = (float(innovation_std / np.sqrt(1 - slope ** 2))
                if abs(slope) < 1 else np.nan)
    s_score = (float((values.iloc[-1] - mean) / sigma_eq)
               if np.isfinite(mean) and np.isfinite(sigma_eq) and sigma_eq > 0
               else np.nan)

    return AR1Fit(
        intercept=float(intercept), slope=float(slope), mean=float(mean),
        kappa=kappa, tau_days=tau_days, innovation_std=innovation_std,
        sigma_eq=sigma_eq, s_score=s_score, nobs=len(values),
        asof=pd.Timestamp(values.index[-1]),
    )


def residual_ar1(model, token: str, window: int | None = 20,
                 bars_per_year: int = 24 * 365) -> AR1Fit:
    """Fit AR(1) to one token's cumulative residual in a ResidualModel."""
    if token not in model.residuals:
        raise KeyError(f"no residuals fitted for {token}")
    return fit_ar1(model.cumulative[token], window=window,
                   bars_per_year=bars_per_year)


def rolling_ar1(cumulative: pd.Series, window: int = 20,
                bars_per_year: int = 24 * 365) -> pd.DataFrame:
    """Refit AR(1) on every trailing window, indexed by each fit's as-of bar."""
    values = pd.Series(cumulative, dtype=float)
    if window < 3:
        raise ValueError("AR(1) window must contain at least 3 observations")
    rows = []
    for end in range(window, len(values) + 1):
        fit = fit_ar1(values.iloc[end - window:end], window=None,
                      bars_per_year=bars_per_year)
        rows.append(fit.to_series())
    if not rows:
        return pd.DataFrame(columns=[
            "a", "b", "m", "kappa", "tau_days", "innovation_std",
            "sigma_eq", "s_score", "nobs", "stationary",
        ]).rename_axis("startTime")
    return pd.DataFrame(rows).rename_axis("startTime")
