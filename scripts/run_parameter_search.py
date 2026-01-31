"""Command-line entry point for Simple Moving Average parameter search."""

import argparse
import logging
import sys
from pathlib import Path

from stock_autotrade.strategy.optimization.grid_search import grid_search_simple_ma


DEFAULTS = {
    "ticker": "7203.T",
    "period": "1y",
    "start": None,
    "end": None,
    "initial_cash": 1_000_000.0,
    "fee_rate": 0.0,
    "trade_size": 100,
    "max_position": 1.0,
    "short_min": 3,
    "short_max": 15,
    "short_step": 1,
    "long_min": 10,
    "long_max": 60,
    "long_step": 5,
    "objective": "return",
    "n_jobs": 1,
    "output": None,
    "top_n": 10,
    "log_level": "INFO",
}


def _configure_logging(level: str) -> None:
    """Configure application logging.

    Args:
        level: Logging level name.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run parameter search for Simple Moving Average strategy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic grid search with defaults
  python run_parameter_search.py 7203.T

  # Custom parameter ranges
  python run_parameter_search.py 7203.T --short-min 5 --short-max 20 --long-min 20 --long-max 100

  # Optimize for Sharpe ratio and save results
  python run_parameter_search.py 7203.T --objective sharpe --output results.csv

  # Parallel execution
  python run_parameter_search.py 7203.T --n-jobs 4
        """,
    )

    # Data arguments
    parser.add_argument("ticker", nargs="?", default=DEFAULTS["ticker"], help="Ticker symbol, e.g. 7203.T")
    parser.add_argument("--period", default=DEFAULTS["period"], help="Data period (e.g., 1mo, 6mo, 1y, 2y)")
    parser.add_argument("--start", default=DEFAULTS["start"], help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULTS["end"], help="End date (YYYY-MM-DD)")

    # Backtest arguments
    parser.add_argument("--initial-cash", type=float, default=DEFAULTS["initial_cash"], help="Initial cash")
    parser.add_argument("--fee-rate", type=float, default=DEFAULTS["fee_rate"], help="Fee rate per trade")
    parser.add_argument("--trade-size", type=float, default=DEFAULTS["trade_size"], help="Trade size (shares)")
    parser.add_argument("--max-position", type=float, default=DEFAULTS["max_position"], help="Maximum position size")

    # Parameter search arguments
    parser.add_argument("--short-min", type=int, default=DEFAULTS["short_min"], help="Minimum short window")
    parser.add_argument("--short-max", type=int, default=DEFAULTS["short_max"], help="Maximum short window")
    parser.add_argument("--short-step", type=int, default=DEFAULTS["short_step"], help="Short window step")
    parser.add_argument("--long-min", type=int, default=DEFAULTS["long_min"], help="Minimum long window")
    parser.add_argument("--long-max", type=int, default=DEFAULTS["long_max"], help="Maximum long window")
    parser.add_argument("--long-step", type=int, default=DEFAULTS["long_step"], help="Long window step")

    # Optimization arguments
    parser.add_argument(
        "--objective",
        default=DEFAULTS["objective"],
        choices=["return", "sharpe", "drawdown", "calmar"],
        help="Objective to optimize",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=DEFAULTS["n_jobs"], help="Number of parallel workers (1=sequential, -1=all CPUs)"
    )

    # Output arguments
    parser.add_argument("--output", "-o", default=DEFAULTS["output"], help="Output CSV file path")
    parser.add_argument("--top-n", type=int, default=DEFAULTS["top_n"], help="Number of top results to display")

    parser.add_argument("--log-level", default=DEFAULTS["log_level"], help="Log level (DEBUG, INFO, WARNING)")

    return parser.parse_args()


def main() -> None:
    """Run parameter search from CLI."""
    args = _parse_args()
    _configure_logging(args.log_level)

    logger = logging.getLogger(__name__)

    # Build parameter ranges
    short_windows = range(args.short_min, args.short_max + 1, args.short_step)
    long_windows = range(args.long_min, args.long_max + 1, args.long_step)

    total_combinations = len(list(short_windows)) * len(list(long_windows))
    logger.info("Starting parameter search for %s", args.ticker)
    logger.info("Short windows: %d-%d (step %d)", args.short_min, args.short_max, args.short_step)
    logger.info("Long windows: %d-%d (step %d)", args.long_min, args.long_max, args.long_step)
    logger.info("Total combinations to evaluate: %d", total_combinations)
    logger.info("Objective: %s", args.objective)

    try:
        result = grid_search_simple_ma(
            ticker=args.ticker,
            short_windows=range(args.short_min, args.short_max + 1, args.short_step),
            long_windows=range(args.long_min, args.long_max + 1, args.long_step),
            period=args.period,
            start=args.start,
            end=args.end,
            initial_cash=args.initial_cash,
            fee_rate=args.fee_rate,
            trade_size=args.trade_size,
            max_position=args.max_position,
            objective=args.objective,
            n_jobs=args.n_jobs,
            verbose=1 if args.log_level.upper() == "DEBUG" else 0,
        )
    except Exception as e:
        logger.error("Parameter search failed: %s", e)
        sys.exit(1)

    # Display results
    logger.info("")
    logger.info("=" * 60)
    logger.info("PARAMETER SEARCH RESULTS")
    logger.info("=" * 60)
    logger.info("Best Parameters:")
    logger.info("  Short Window: %d", result.best_params["short_window"])
    logger.info("  Long Window:  %d", result.best_params["long_window"])
    logger.info("  Best Score:   %.4f", result.best_score)

    logger.info("Top %d Results:", args.top_n)
    top_results = result.top_n(args.top_n, ascending=False)
    # Filter out invalid combinations (score == -inf)
    top_results = top_results[top_results["score"] > float("-inf")]
    logger.info("\n%s", top_results.to_string(index=False))

    # Save to CSV if requested
    if args.output:
        output_path = Path(args.output)
        all_results = result.all_results[result.all_results["score"] > float("-inf")]
        all_results.to_csv(output_path, index=False)
        logger.info("Full results saved to: %s", output_path)


if __name__ == "__main__":
    main()
