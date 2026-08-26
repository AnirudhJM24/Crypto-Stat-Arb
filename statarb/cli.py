"""Command line entry point: python -m statarb <view> [options]."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from .data import Market
from .divergence import divergence, divergence_stats
from .eigen import correlation_matrix, decompose
from .residuals import fit_residuals
from . import plots

VIEWS = ("standardized", "spectrum", "eigenpf", "cumulative", "compare",
         "divergence", "residuals", "cumresiduals")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="statarb", description=__doc__)
    p.add_argument("view", choices=VIEWS, help="which chart to draw")
    p.add_argument("-a", "--assets", nargs="+", default=["BTC", "ETH"],
                   help="tickers to plot (default: BTC ETH)")
    p.add_argument("-w", "--when", default=None,
                   help="estimation window end, e.g. '2021-06-01 12:00' (default: last hour)")
    p.add_argument("-n", "--window", type=int, default=240, help="window length in hours")
    p.add_argument("-c", "--component", type=int, default=0, help="eigen component")
    p.add_argument("--threshold", type=float, default=1.5,
                   help="separation threshold in std devs (compare view)")
    p.add_argument("--variance", type=float, default=0.90,
                   help="cumulative variance marker (spectrum view)")
    p.add_argument("--portfolio", action="store_true",
                   help="standardized view: also plot an equal-weight basket of --assets")
    p.add_argument("--only-portfolio", action="store_true",
                   help="standardized view: plot just the basket, not its legs")
    p.add_argument("--factors", type=int, default=1,
                   help="number of eigenportfolio factors (residual views)")
    p.add_argument("--highlight", nargs="+", default=None,
                   help="residual views: draw these tokens in color, grey the rest")
    p.add_argument("--all-assets", action="store_true",
                   help="residual views: plot the whole universe, not just --assets")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--save", default=None, help="write the figure here instead of showing it")
    return p


def draw(market: Market, args) -> None:
    common = dict(when=args.when, window=args.window)

    if args.view == "standardized":
        plots.plot_standardized_returns(
            market, assets=args.assets,
            weights=True if (args.portfolio or args.only_portfolio) else None,
            only_portfolio=args.only_portfolio, **common)

    elif args.view == "spectrum":
        plots.plot_eigen_spectrum(market, threshold=args.variance, **common)
        corr = correlation_matrix(market, args.when or market.returns.index[-1],
                                  window=args.window)
        eig = decompose(corr)
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(eig.summary().head(10))
        print(f"PC1 explains {eig.top_share:.2%}; "
              f"{eig.n_components(args.variance)} components reach {args.variance:.0%}")

    elif args.view == "eigenpf":
        plots.plot_eigen_portfolio(market, component=args.component, **common)

    elif args.view == "cumulative":
        plots.plot_cumulative_vs_eigen(market, assets=args.assets,
                                       component=args.component, **common)

    elif args.view == "compare":
        plots.plot_vs_eigen(market, assets=args.assets, component=args.component,
                            threshold=args.threshold, **common)

    elif args.view in ("residuals", "cumresiduals"):
        model = fit_residuals(market, when=args.when, window=args.window,
                              n_factors=args.factors,
                              variance=args.variance if args.factors == 0 else None,
                              include=None if args.all_assets else args.assets)
        plots.plot_residuals(market, model=model, assets=None if args.all_assets
                             else args.assets, highlight=args.highlight,
                             cumulative=args.view == "cumresiduals",
                             window=args.window)
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(pd.DataFrame({"beta": model.betas.iloc[:, 0],
                                "r_squared": model.r_squared,
                                "z": model.zscore()}).sort_values("z").head(10))

    elif args.view == "divergence":
        asset = args.assets[0]
        plots.plot_divergence_distribution(market, asset=asset,
                                           component=args.component, **common)
        spread = divergence(market, asset, args.when or market.returns.index[-1],
                            window=args.window, component=args.component)
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(divergence_stats(spread))


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    market = Market.load(args.data_dir)
    draw(market, args)

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
