"""Loading and shaping the raw market data."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

PRICES_FILE = "coin_all_prices_full.csv"
UNIVERSE_FILE = "coin_universe_150K_40.csv"


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["startTime"] = pd.to_datetime(frame["startTime"], utc=True)
    return frame.set_index("startTime").drop(columns="time").sort_index()


def to_utc(when) -> pd.Timestamp:
    """Coerce anything Timestamp-like to a tz-aware UTC Timestamp."""
    ts = pd.Timestamp(when)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


@dataclass
class Market:
    """Prices, returns and the hourly top-N universe, indexed by timestamp."""

    prices: pd.DataFrame
    universe: pd.DataFrame

    @classmethod
    def load(cls, data_dir: str | Path = "data") -> "Market":
        data_dir = Path(data_dir)
        prices = _read(data_dir / PRICES_FILE)
        universe = _read(data_dir / UNIVERSE_FILE)
        universe.columns = universe.columns.astype(int)
        return cls(prices=prices, universe=universe)

    @cached_property
    def returns(self) -> pd.DataFrame:
        """Hourly simple returns. Unlisted coins are 0.0 in the file, which would
        blow up pct_change, so zeros become NaN first."""
        return self.prices.replace(0, np.nan).pct_change()

    @property
    def tokens(self) -> list[str]:
        return list(self.prices.columns)

    def asof(self, when) -> pd.Timestamp:
        """The most recent universe hour on or before `when`."""
        stamp = self.universe.index.asof(to_utc(when))
        if pd.isna(stamp):
            raise KeyError(f"{to_utc(when)} precedes the data ({self.universe.index[0]})")
        return stamp

    def members(self, when, n: int | None = None) -> list[str]:
        """Tickers of the top-N universe at `when`, ordered by rank."""
        row = self.universe.loc[self.asof(when)].dropna()
        members = [t for t in row if t in self.prices.columns]
        return members[:n] if n else members

    def window(self, when, window: int = 240, assets=None) -> pd.DataFrame:
        """The `window` hours of returns ending at `when` (inclusive)."""
        stamp = self.asof(when)
        cols = list(assets) if assets is not None else self.tokens
        frame = self.returns.loc[:stamp, cols].tail(window)
        return frame.dropna(axis=1, how="all")
