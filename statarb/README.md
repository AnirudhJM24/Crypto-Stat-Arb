# statarb

Modular toolkit for the hourly crypto data in `data/`.
See the [project README](../README.md) for the model and the results.

```
statarb/
  data.py         Market: prices, returns, hourly top-40 universe, window slicing
  standardize.py  standardize(), portfolio_returns(), standardized_panel()
  eigen.py        correlation_matrix(), decompose() -> Eigen, eigen_portfolio()
  divergence.py   divergence(), divergence_stats(), separation_mask/regions()
  residuals.py    fit_residuals() -> ResidualModel (residuals, cumulative, betas, r2, z)
  ar1.py          fit_ar1(), residual_ar1(), rolling_ar1()
  backtest.py     no-look-ahead, next-bar token backtest
  pipeline.py     ordered eigen -> residual -> AR(1) -> backtest analysis
  plots.py        the six views, each returning a matplotlib Axes
  cli.py          python -m statarb <view>
```

Every function takes a `Market` explicitly. There are no globals, and nothing depends on a
notebook cell having run.

## Library

```python
from statarb import Market, eigen_portfolio, decompose, correlation_matrix, divergence

market = Market.load("data")

pf   = eigen_portfolio(market, "2021-06-01 12:00", window=240, component=0)
pf.standardized          # standardized returns of the eigenportfolio
pf.weights               # Q_i = v_i / sigma_i, normalized

eig  = decompose(correlation_matrix(market, "2021-06-01 12:00", window=240))
eig.top_share            # variance explained by PC1
eig.n_components(0.90)   # components needed for 90%
eig.head(0.90).vectors   # those eigenvectors

divergence(market, "SOL", "2021-06-01 12:00", window=240)

model = fit_residuals(market, "2021-06-01 12:00", window=240, n_factors=1)
model.residuals          # eps_i(t) for every token in the universe
model.cumulative         # X_i(t) = running sum
model.betas, model.r_squared, model.ranked()
```

## BTC: two eigen factors through trading

```python
from statarb import Market, analyze_token, backtest_token

market = Market.load("data")
analysis = analyze_token(
    market, "BTC", when="2021-06-01 12:00", estimation_window=240,
    n_factors=2, variance=None, ar_window=20,
)

analysis.factor_count          # exactly two leading eigenportfolios
analysis.regression            # alpha, every factor beta, and R-squared
analysis.residual
analysis.cumulative_residual
analysis.ar1.to_series()

test = backtest_token(
    market, "BTC", start="2021-05-22 12:00", end="2021-06-01 12:00",
    estimation_window=240, n_factors=2, variance=None, ar_window=20,
    entry=2.0, exit=0.5, cost_bps=5.0,
)
test.metrics
test.trades
test.frame
```

The backtest is walk-forward. A model estimated only through hour `t-1`
produces BTC's one-step residual at `t`; AR(1) then forms the s-score at `t`,
and the position is applied to BTC's return at `t+1`. This avoids the zero-sum
endpoint of an in-sample OLS residual path and charges costs on position
turnover. The default executed portfolio is the unit-gross BTC-minus-fitted-
eigen-factors spread; pass `hedged=False` for a naked BTC leg.

## CLI

```
python -m statarb standardized -a BTC ETH SOL --portfolio   # standardized returns (+ basket)
python -m statarb spectrum                                  # eigenvalue analysis
python -m statarb eigenpf -c 0                              # eigenportfolio standardized returns
python -m statarb cumulative -a BTC ETH                     # cumulative standardized vs eigenportfolio
python -m statarb compare -a SOL --threshold 1.5            # standardized vs eigenportfolio, red where separated
python -m statarb divergence -a SOL                         # divergence distribution + stats
```

Shared options: `-w/--when` window end (default last hour), `-n/--window` hours
(default 240), `-c/--component`, `--data-dir`, `--save out.png`.
