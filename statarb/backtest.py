"""Walk-forward trading backtest driven by cumulative-residual s-scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ar1 import fit_ar1
from .data import to_utc
from .eigen import (correlation_matrix, decompose,
                    eigen_portfolios_from_eigen)
from .standardize import standardize


@dataclass
class BacktestResult:
    """Hourly signals, positions, returns, trade ledger, and summary metrics."""

    frame: pd.DataFrame
    trades: pd.DataFrame
    metrics: pd.Series
    config: dict

    @property
    def equity(self) -> pd.Series:
        return self.frame["equity"]


@dataclass
class ResidualPath:
    """Reusable walk-forward residuals and unit-hedge returns."""

    frame: pd.DataFrame
    config: dict


def _target_position(current: int, score: float, entry: float,
                     exit: float, stationary: bool) -> int:
    if not stationary or not np.isfinite(score):
        return 0
    if current == 0:
        if score <= -entry:
            return 1
        if score >= entry:
            return -1
        return 0
    if current == 1:
        return 0 if score >= -exit else 1
    return 0 if score <= exit else -1


def _trade_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["side", "entry", "exit", "bars", "return", "closed"]
    rows = []
    active = None
    previous = 0
    for stamp, row in frame.iterrows():
        position = int(row["position"])
        if active is None and position != 0:
            active = {"side": "long" if position > 0 else "short",
                      "entry": stamp, "returns": []}
        if active is not None:
            active["returns"].append(float(row["strategy_return"]))
        if active is not None and previous != 0 and position == 0:
            returns = np.asarray(active.pop("returns"), dtype=float)
            rows.append({**active, "exit": stamp, "bars": len(returns),
                         "return": float(np.prod(1 + returns) - 1),
                         "closed": True})
            active = None
        previous = position
    if active is not None:
        returns = np.asarray(active.pop("returns"), dtype=float)
        rows.append({**active, "exit": frame.index[-1], "bars": len(returns),
                     "return": float(np.prod(1 + returns) - 1),
                     "closed": False})
    return pd.DataFrame(rows, columns=columns)


def _metrics(frame: pd.DataFrame, trades: pd.DataFrame,
             bars_per_year: int) -> pd.Series:
    returns = frame["strategy_return"].fillna(0)
    equity = frame["equity"]
    total = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    annual_return = (float(equity.iloc[-1] ** (bars_per_year / len(equity)) - 1)
                     if len(equity) and equity.iloc[-1] > 0 else np.nan)
    volatility = float(returns.std(ddof=1) * np.sqrt(bars_per_year))
    sharpe = (float(returns.mean() / returns.std(ddof=1) * np.sqrt(bars_per_year))
              if returns.std(ddof=1) > 0 else np.nan)
    drawdown = equity / equity.cummax() - 1
    closed = trades[trades["closed"]] if len(trades) else trades
    return pd.Series({
        "total_return": total,
        "annualized_return": annual_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "exposure": float(frame["position"].abs().mean()) if len(frame) else 0.0,
        "entries": int(len(trades)),
        "closed_trades": int(len(closed)),
        "trade_win_rate": (float((closed["return"] > 0).mean())
                           if len(closed) else np.nan),
        "ending_position": int(frame["position"].iloc[-1]) if len(frame) else 0,
    }, name="BTC backtest")


def _one_step_residual(market, token: str, observation_time,
                       estimation_window: int, variance: float | None,
                       n_factors: int):
    """Residual at one bar using a factor regression fitted through t-1."""
    index = market.returns.index
    location = index.get_loc(observation_time)
    if location == 0:
        raise ValueError("no prior bar is available for model estimation")
    fit_time = index[location - 1]

    corr = correlation_matrix(market, fit_time, window=estimation_window)
    eig = decompose(corr)
    n_factors = (eig.n_components(variance)
                 if variance is not None else n_factors)
    portfolios = eigen_portfolios_from_eigen(
        market, eig, components=range(n_factors),
    )
    factors = pd.DataFrame({pf.name: pf.standardized for pf in portfolios})
    token_raw = market.window(
        fit_time, window=estimation_window, assets=[token]
    )[token]
    token_standardized = standardize(token_raw).reindex(factors.index)
    design = np.column_stack([np.ones(len(factors)), factors.to_numpy()])
    ok = token_standardized.notna() & np.isfinite(design).all(axis=1)
    if ok.sum() <= design.shape[1]:
        raise ValueError(f"insufficient observations to fit {token} at {fit_time}")
    coefficients = np.linalg.lstsq(
        design[ok.to_numpy()], token_standardized[ok].to_numpy(), rcond=None
    )[0]

    token_sigma = token_raw.std(ddof=1)
    token_now = market.returns.at[observation_time, token]
    if not np.isfinite(token_now) or not np.isfinite(token_sigma) or token_sigma == 0:
        return np.nan, n_factors, pd.Series({token: 1.0})
    token_now = (token_now - token_raw.mean()) / token_sigma

    factor_now = []
    current_returns = market.returns.loc[observation_time]
    for portfolio in portfolios:
        available = portfolio.weights.index.intersection(current_returns.dropna().index)
        raw_now = float((current_returns[available] *
                         portfolio.weights[available]).sum())
        sigma = portfolio.returns.std(ddof=1)
        factor_now.append((raw_now - portfolio.returns.mean()) / sigma
                          if np.isfinite(sigma) and sigma != 0 else np.nan)
    vector = np.r_[1.0, factor_now]
    residual = (float(token_now - vector @ coefficients)
                if np.isfinite(vector).all() else np.nan)

    # Convert standardized regression betas to raw-return hedge ratios, then
    # combine overlapping eigenportfolio holdings into one unit-gross spread.
    spread_weights = pd.Series({token: 1.0}, dtype=float)
    for beta, portfolio in zip(coefficients[1:], portfolios):
        factor_sigma = portfolio.returns.std(ddof=1)
        hedge_ratio = beta * token_sigma / factor_sigma
        spread_weights = spread_weights.add(
            -hedge_ratio * portfolio.weights, fill_value=0.0
        )
    gross = spread_weights.abs().sum()
    if np.isfinite(gross) and gross > 0:
        spread_weights /= gross
    return residual, n_factors, spread_weights


def prepare_residual_path(market, token: str = "BTC", start=None, end=None,
                          estimation_window: int = 240,
                          variance: float | None = None,
                          n_factors: int = 2) -> ResidualPath:
    """Calculate the expensive walk-forward factor residual path once.

    A model fitted through ``t-1`` produces the residual at ``t`` and a
    unit-gross factor hedge whose return is measured at ``t+1``. The resulting
    path can be reused to compare many rolling AR windows without recomputing
    correlation matrices or factor regressions.
    """
    if token not in market.tokens:
        raise KeyError(f"unknown token {token}")
    if estimation_window < 3 or n_factors < 1:
        raise ValueError("estimation_window >= 3 and n_factors >= 1 are required")
    if variance is not None and not 0 < variance <= 1:
        raise ValueError("variance must be in (0, 1]")

    index = market.returns.index
    start_stamp = (index[min(estimation_window, len(index) - 2)]
                   if start is None else to_utc(start))
    end_stamp = index[-1] if end is None else to_utc(end)
    signal_times = index[(index >= start_stamp) & (index < end_stamp)]
    rows = []
    previous_weights = None
    for signal_time in signal_times:
        location = index.get_loc(signal_time)
        execution_time = index[location + 1]
        if execution_time > end_stamp:
            break
        residual, selected, weights = _one_step_residual(
            market, token, signal_time, estimation_window, variance, n_factors,
        )
        next_returns = market.returns.loc[execution_time].reindex(
            weights.index
        ).fillna(0.0)
        unit_return = float((weights * next_returns).sum())
        if previous_weights is None:
            rebalance = np.nan
        else:
            holdings = previous_weights.index.union(weights.index)
            rebalance = float((
                weights.reindex(holdings, fill_value=0.0) -
                previous_weights.reindex(holdings, fill_value=0.0)
            ).abs().sum())
        rows.append({
            "startTime": execution_time, "signal_time": signal_time,
            "residual": residual, "n_factors": selected,
            "unit_spread_return": unit_return,
            "rebalance_turnover": rebalance,
            "token_return": float(market.returns.at[execution_time, token]),
        })
        previous_weights = weights
    frame = pd.DataFrame(rows).set_index("startTime")
    return ResidualPath(frame=frame, config={
        "token": token, "start": start_stamp, "end": end_stamp,
        "estimation_window": estimation_window, "variance": variance,
        "n_factors": n_factors,
    })


def backtest_residual_path(path: ResidualPath, ar_window: int,
                           start=None, end=None, entry: float = 2.0,
                           exit: float = 0.5, cost_bps: float = 5.0,
                           bars_per_year: int = 24 * 365) -> BacktestResult:
    """Backtest one rolling AR window on a precomputed residual path."""
    from .ar1 import rolling_ar1

    if ar_window < 3:
        raise ValueError("ar_window must be at least 3")
    base = path.frame.copy()
    cumulative = base.set_index("signal_time")["residual"].fillna(0).cumsum()
    fits = rolling_ar1(cumulative, window=ar_window,
                       bars_per_year=bars_per_year).rename(columns={
        "a": "ar_a", "b": "ar_b", "m": "moving_m",
    })
    signal = base.reset_index().set_index("signal_time").join(fits, how="inner")
    signal["cumulative_residual"] = cumulative.reindex(signal.index)
    signal["moving_lower"] = signal["moving_m"] - signal["sigma_eq"]
    signal["moving_upper"] = signal["moving_m"] + signal["sigma_eq"]
    signal = signal.reset_index().set_index("startTime")

    if start is not None:
        signal = signal.loc[signal.index >= to_utc(start)]
    if end is not None:
        signal = signal.loc[signal.index <= to_utc(end)]
    if signal.empty:
        raise ValueError("no backtest bars remain after AR warmup and date filters")

    positions, turnovers, returns = [], [], []
    current = 0
    for _, row in signal.iterrows():
        target = _target_position(
            current, float(row["s_score"]), entry, exit,
            bool(row["stationary"]),
        )
        if current == 0 and target != 0:
            turnover = 1.0
        elif current != 0 and target == 0:
            turnover = 1.0
        elif current == target and target != 0:
            turnover = float(row["rebalance_turnover"])
            if not np.isfinite(turnover):
                turnover = 0.0
        elif current != target:
            turnover = 2.0
        else:
            turnover = 0.0
        strategy_return = (target * float(row["unit_spread_return"]) -
                           turnover * cost_bps / 10_000)
        positions.append(target)
        turnovers.append(turnover)
        returns.append(strategy_return)
        current = target

    signal["position"] = positions
    signal["turnover"] = turnovers
    signal["spread_return"] = signal["position"] * signal["unit_spread_return"]
    signal["strategy_return"] = returns
    signal["equity"] = (1 + signal["strategy_return"]).cumprod()
    signal["drawdown"] = signal["equity"] / signal["equity"].cummax() - 1
    trades = _trade_ledger(signal)
    config = {**path.config, "ar_window": ar_window, "entry": entry,
              "exit": exit, "cost_bps": cost_bps,
              "bars_per_year": bars_per_year}
    return BacktestResult(signal, trades, _metrics(signal, trades, bars_per_year),
                          config)


def search_ar_windows(path: ResidualPath, windows, development_end,
                      development_start=None, entry: float = 2.0,
                      exit: float = 0.5, cost_bps: float = 5.0,
                      min_trades: int = 5) -> pd.DataFrame:
    """Empirically rank AR windows on development-period net Sharpe."""
    rows = []
    for window in sorted(set(int(value) for value in windows)):
        result = backtest_residual_path(
            path, window, start=development_start, end=development_end,
            entry=entry, exit=exit, cost_bps=cost_bps,
        )
        metrics = result.metrics
        eligible = metrics["closed_trades"] >= min_trades
        rows.append({
            "ar_window": window, "objective": metrics["sharpe"] if eligible else np.nan,
            "sharpe": metrics["sharpe"], "total_return": metrics["total_return"],
            "max_drawdown": metrics["max_drawdown"],
            "closed_trades": int(metrics["closed_trades"]),
            "trade_win_rate": metrics["trade_win_rate"], "eligible": eligible,
        })
    table = pd.DataFrame(rows).set_index("ar_window").sort_index()
    if table["objective"].notna().sum() == 0:
        raise ValueError("no AR window met the minimum trade requirement")
    table.attrs["selected_window"] = int(table["objective"].idxmax())
    table.attrs["objective"] = "development-period net Sharpe"
    return table


def search_entry_exit(path: ResidualPath, ar_window: int, entries, exits,
                      development_end, development_start=None,
                      cost_bps: float = 5.0,
                      min_trades: int = 5) -> pd.DataFrame:
    """Rank valid entry/exit pairs on development-period net Sharpe.

    The AR window is held fixed. Only pairs satisfying ``entry > exit >= 0``
    are evaluated, and the returned selection metadata can be frozen for a
    single subsequent OOS run.
    """
    # Compute the rolling AR estimates once. Threshold pairs only change the
    # state machine, so rebuilding thousands of AR fits per pair is redundant.
    template = backtest_residual_path(
        path, ar_window, start=development_start, end=development_end,
        entry=max(float(value) for value in entries) + 1.0,
        exit=0.0, cost_bps=cost_bps,
    ).frame
    observations = list(template[[
        "s_score", "stationary", "rebalance_turnover", "unit_spread_return",
    ]].itertuples(index=False, name=None))

    rows = []
    for entry in sorted(set(float(value) for value in entries)):
        for exit in sorted(set(float(value) for value in exits)):
            if exit < 0 or entry <= exit:
                continue
            positions, strategy_returns = [], []
            current = 0
            for score, stationary, rebalance, unit_return in observations:
                target = _target_position(
                    current, float(score), entry, exit, bool(stationary),
                )
                if current == 0 and target != 0:
                    turnover = 1.0
                elif current != 0 and target == 0:
                    turnover = 1.0
                elif current == target and target != 0:
                    turnover = float(rebalance) if np.isfinite(rebalance) else 0.0
                elif current != target:
                    turnover = 2.0
                else:
                    turnover = 0.0
                strategy_returns.append(
                    target * float(unit_return) - turnover * cost_bps / 10_000
                )
                positions.append(target)
                current = target
            simulation = pd.DataFrame({
                "position": positions,
                "strategy_return": strategy_returns,
            }, index=template.index)
            simulation["equity"] = (1 + simulation["strategy_return"]).cumprod()
            trades = _trade_ledger(simulation)
            metrics = _metrics(simulation, trades, 24 * 365)
            eligible = metrics["closed_trades"] >= min_trades
            rows.append({
                "entry": entry, "exit": exit,
                "objective": metrics["sharpe"] if eligible else np.nan,
                "sharpe": metrics["sharpe"],
                "total_return": metrics["total_return"],
                "max_drawdown": metrics["max_drawdown"],
                "closed_trades": int(metrics["closed_trades"]),
                "trade_win_rate": metrics["trade_win_rate"],
                "eligible": eligible,
            })
    table = pd.DataFrame(rows).set_index(["entry", "exit"]).sort_index()
    if table["objective"].notna().sum() == 0:
        raise ValueError("no entry/exit pair met the minimum trade requirement")
    selected = table["objective"].idxmax()
    table.attrs["selected_entry"] = float(selected[0])
    table.attrs["selected_exit"] = float(selected[1])
    table.attrs["ar_window"] = int(ar_window)
    table.attrs["objective"] = "development-period net Sharpe"
    return table


def backtest_token(market, token: str = "BTC", start=None, end=None,
                   estimation_window: int = 240, variance: float | None = None,
                   n_factors: int = 2,
                   ar_window: int = 20, entry: float = 2.0,
                   exit: float = 0.5, cost_bps: float = 5.0,
                   bars_per_year: int = 24 * 365,
                   hedged: bool = True) -> BacktestResult:
    """Run a no-look-ahead, next-bar directional backtest for one token.

    At each observation hour the top-universe correlation matrix is decomposed, the
    first ``n_factors`` eigenportfolios are used in the token
    regression using data only through *t-1*. The one-step residual at *t* is
    appended to the cumulative process, then AR(1) is fitted to its last
    ``ar_window`` observations. A signal formed at *t* is applied to the token's
    return at *t+1*. By default the executed portfolio is the unit-gross token
    minus its fitted raw-return eigen-factor hedge. Pass ``hedged=False`` to
    test a naked directional token leg instead.

    Positions enter long below ``-entry`` and short above ``entry``. Longs exit
    at ``-exit`` and shorts at ``exit``. Non-stationary AR(1) fits force a flat
    position. ``cost_bps`` is charged on each unit of position turnover.
    """
    if token not in market.tokens:
        raise KeyError(f"unknown token {token}")
    if estimation_window < 3:
        raise ValueError("estimation_window must be at least 3")
    if ar_window < 3 or ar_window > estimation_window:
        raise ValueError("ar_window must be between 3 and estimation_window")
    if variance is not None and not 0 < variance <= 1:
        raise ValueError("variance must be in (0, 1]")
    if n_factors < 1:
        raise ValueError("n_factors must be at least 1")
    if entry <= exit or exit < 0:
        raise ValueError("thresholds must satisfy entry > exit >= 0")
    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")

    index = market.returns.index
    end_stamp = index[-1] if end is None else to_utc(end)
    start_stamp = (end_stamp - pd.Timedelta(days=30)
                   if start is None else to_utc(start))
    signal_times = index[(index >= start_stamp) & (index < end_stamp)]
    if len(signal_times) == 0:
        raise ValueError("the requested backtest range contains no signal bars")

    first_location = index.get_loc(signal_times[0])
    warmup_location = max(1, first_location - ar_window + 1)
    calculation_times = index[warmup_location:index.get_loc(signal_times[-1]) + 1]

    rows = []
    residuals = []
    residual_index = []
    current = 0
    current_weights = pd.Series(dtype=float)
    for signal_time in calculation_times:
        location = index.get_loc(signal_time)
        if location + 1 >= len(index):
            break
        execution_time = index[location + 1]
        if execution_time > end_stamp:
            break

        residual, n_factors, spread_weights = _one_step_residual(
            market, token, signal_time, estimation_window, variance, n_factors,
        )
        residuals.append(residual)
        residual_index.append(signal_time)
        if len(residuals) < ar_window or signal_time < start_stamp:
            continue
        cumulative = pd.Series(
            residuals, index=residual_index, dtype=float
        ).fillna(0).cumsum()
        # This is a genuinely rolling fit: m(t), sigma_eq(t), and the s-score
        # are re-estimated from the trailing AR window at every signal bar.
        fit = fit_ar1(cumulative, window=ar_window,
                      bars_per_year=bars_per_year)
        target = _target_position(current, fit.s_score, entry, exit,
                                  fit.stationary)
        unit_weights = (spread_weights if hedged
                        else pd.Series({token: 1.0}, dtype=float))
        target_weights = target * unit_weights
        holdings = current_weights.index.union(target_weights.index)
        turnover = float((target_weights.reindex(holdings, fill_value=0.0) -
                          current_weights.reindex(holdings, fill_value=0.0)).abs().sum())
        token_return = float(market.returns.at[execution_time, token])
        if not np.isfinite(token_return):
            token_return = 0.0
        next_returns = market.returns.loc[execution_time].reindex(
            target_weights.index
        ).fillna(0.0)
        spread_return = float((target_weights * next_returns).sum())
        strategy_return = spread_return - turnover * cost_bps / 10_000
        rows.append({
            "startTime": execution_time,
            "signal_time": signal_time,
            "residual": residual,
            "cumulative_residual": cumulative.iloc[-1],
            "s_score": fit.s_score,
            "ar_a": fit.intercept,
            "ar_b": fit.slope,
            "ar_mean": fit.mean,
            "moving_m": fit.mean,
            "sigma_eq": fit.sigma_eq,
            "moving_lower": fit.mean - fit.sigma_eq,
            "moving_upper": fit.mean + fit.sigma_eq,
            "tau_days": fit.tau_days,
            "stationary": fit.stationary,
            "n_factors": n_factors,
            "position": target,
            "turnover": turnover,
            "token_return": token_return,
            "spread_return": spread_return,
            "strategy_return": strategy_return,
        })
        current = target
        current_weights = target_weights

    frame = pd.DataFrame(rows).set_index("startTime")
    frame["equity"] = (1 + frame["strategy_return"]).cumprod()
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1
    trades = _trade_ledger(frame)
    config = {
        "token": token, "start": start_stamp, "end": end_stamp,
        "estimation_window": estimation_window, "variance": variance,
        "n_factors": n_factors,
        "ar_window": ar_window, "entry": entry, "exit": exit,
        "cost_bps": cost_bps, "bars_per_year": bars_per_year,
        "hedged": hedged,
    }
    return BacktestResult(frame=frame, trades=trades,
                          metrics=_metrics(frame, trades, bars_per_year),
                          config=config)
