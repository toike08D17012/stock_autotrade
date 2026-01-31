"""Command-line entry point for Simple Moving Average parameter search."""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from stock_autotrade.strategy.optimization.grid_search import grid_search_simple_ma


DEFAULTS = {
    "tickers": ["7203.T"],
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
    "aggregation": "mean",
    "n_jobs": None,
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
  # Basic grid search with single ticker
  python run_parameter_search.py 7203.T

  # Multiple tickers (scores aggregated across all tickers)
  python run_parameter_search.py 7203.T 6758.T 9984.T

  # Load tickers from CSV file (e.g., output of run_stock_screener.py)
  python run_parameter_search.py --tickers-file nikkei_225.csv

  # Load tickers from CSV with custom column name
  python run_parameter_search.py --tickers-file stocks.csv --tickers-column symbol

  # Multiple tickers with minimum aggregation (conservative approach)
  python run_parameter_search.py 7203.T 6758.T --aggregation min

  # Custom parameter ranges
  python run_parameter_search.py 7203.T --short-min 5 --short-max 20 --long-min 20 --long-max 100

  # Optimize for Sharpe ratio and save results
  python run_parameter_search.py 7203.T --objective sharpe --output results.csv

  # Parallel execution with multiple tickers
  python run_parameter_search.py 7203.T 6758.T 9984.T --n-jobs 4
        """,
    )

    # Data arguments
    parser.add_argument(
        "tickers",
        nargs="*",
        default=DEFAULTS["tickers"],
        help="Ticker symbol(s), e.g. 7203.T 6758.T 9984.T",
    )
    parser.add_argument(
        "--tickers-file",
        type=str,
        default=None,
        help="CSV file containing tickers (e.g., output from run_stock_screener.py)",
    )
    parser.add_argument(
        "--tickers-column",
        type=str,
        default="ticker",
        help="Column name containing tickers in the CSV file",
    )
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
        "--aggregation",
        default=DEFAULTS["aggregation"],
        choices=["mean", "min", "median"],
        help="Aggregation method for multiple tickers (mean, min, median)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULTS["n_jobs"],
        help="Number of parallel workers (None=auto based on CPU cores, 1=sequential, -1=all CPUs)",
    )
    parser.add_argument(
        "--download-delay",
        type=float,
        default=2.0,
        help="Delay in seconds between data downloads to avoid rate limiting (default: 2.0)",
    )

    # Cache arguments
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable data caching",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for cache files (default: ~/.cache/stock_autotrade)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache before running",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode: download all tickers first, then run grid search (default: streaming enabled)",
    )

    # Output arguments
    parser.add_argument("--output", "-o", default=DEFAULTS["output"], help="Output CSV file path")
    parser.add_argument("--top-n", type=int, default=DEFAULTS["top_n"], help="Number of top results to display")

    parser.add_argument("--log-level", default=DEFAULTS["log_level"], help="Log level (DEBUG, INFO, WARNING)")

    return parser.parse_args()


def _load_tickers_from_file(file_path: str, column: str) -> list[str]:
    """Load ticker symbols from a CSV file.

    Args:
        file_path: Path to the CSV file.
        column: Column name containing ticker symbols.

    Returns:
        List of ticker symbols.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the column is not found in the file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Tickers file not found: {file_path}")

    df = pd.read_csv(path)
    if column not in df.columns:
        available_columns = ", ".join(df.columns.tolist())
        raise ValueError(f"Column '{column}' not found in {file_path}. Available columns: {available_columns}")

    tickers = df[column].dropna().astype(str).tolist()
    return tickers


def main() -> None:
    """Run parameter search from CLI."""
    args = _parse_args()
    _configure_logging(args.log_level)

    logger = logging.getLogger(__name__)

    # Setup cache
    from stock_autotrade.data.cache import StockDataCache, set_global_cache

    if args.no_cache:
        logger.info("Caching disabled")
        set_global_cache(StockDataCache(enabled=False))
    else:
        cache = StockDataCache(cache_dir=args.cache_dir, enabled=True)
        set_global_cache(cache)
        logger.info("Cache directory: %s", cache.cache_dir)

        if args.clear_cache:
            cache.clear()
            logger.info("Cache cleared")

    # Determine tickers: from file or command line arguments
    if args.tickers_file:
        try:
            tickers = _load_tickers_from_file(args.tickers_file, args.tickers_column)
            logger.info("Loaded %d tickers from %s", len(tickers), args.tickers_file)
        except (FileNotFoundError, ValueError) as e:
            logger.error("Failed to load tickers: %s", e)
            sys.exit(1)
    else:
        tickers = args.tickers

    if not tickers:
        logger.error("No tickers specified. Provide tickers as arguments or use --tickers-file.")
        sys.exit(1)

    # Build parameter ranges
    short_windows = range(args.short_min, args.short_max + 1, args.short_step)
    long_windows = range(args.long_min, args.long_max + 1, args.long_step)

    # Determine n_jobs: None means auto-detect based on CPU cores
    n_jobs = args.n_jobs
    if n_jobs is None:
        cpu_count = os.cpu_count() or 1
        n_jobs = max(1, cpu_count - 1)  # Leave one core free for system
        logger.info("Auto-detected CPU cores: %d, using %d workers", cpu_count, n_jobs)

    total_combinations = len(list(short_windows)) * len(list(long_windows))
    tickers_str = ", ".join(tickers) if len(tickers) <= 10 else f"{len(tickers)} tickers"
    logger.info("Starting parameter search for %s", tickers_str)
    if len(tickers) > 1:
        logger.info("Aggregation method: %s", args.aggregation)
    logger.info("Short windows: %d-%d (step %d)", args.short_min, args.short_max, args.short_step)
    logger.info("Long windows: %d-%d (step %d)", args.long_min, args.long_max, args.long_step)
    logger.info("Total combinations to evaluate: %d", total_combinations)
    logger.info("Objective: %s", args.objective)

    try:
        result = grid_search_simple_ma(
            ticker=tickers,
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
            n_jobs=n_jobs,
            verbose=1 if args.log_level.upper() == "DEBUG" else 0,
            aggregation=args.aggregation,
            download_delay=args.download_delay,
            streaming=not args.no_streaming,
        )
    except Exception as e:
        logger.error("Parameter search failed: %s", e)
        sys.exit(1)

    # Display results
    logger.info("")
    logger.info("=" * 60)
    logger.info("PARAMETER SEARCH RESULTS")
    logger.info("=" * 60)
    logger.info("Tickers: %s", tickers_str)
    if len(tickers) > 1:
        logger.info("Aggregation: %s", args.aggregation)
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
