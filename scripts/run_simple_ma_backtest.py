"""Command-line entry point for Simple Moving Average backtests."""

import argparse
import logging

import pandas as pd

from stock_autotrade.backtest.runner import run_strategy_backtest, summarize_result
from stock_autotrade.strategy.simple_moving_average import generate_signals


DEFAULTS = {
    "ticker": "7203.T",
    "period": "6mo",
    "start": None,
    "end": None,
    "initial_cash": 1_000_000.0,
    "fee_rate": 0.0,
    "trade_size": 100,
    "max_position": 1.0,
    "short_window": 5,
    "long_window": 20,
    "log_level": "INFO",
}


def _configure_logging(level: str) -> None:
    """Configure application logging.

    Args:
        level (str): Logging level name.
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
    parser = argparse.ArgumentParser(description="Run Simple MA stock backtests.")
    parser.add_argument("ticker", nargs="?", default=DEFAULTS["ticker"], help="Ticker symbol, e.g. 7203.T")
    parser.add_argument(
        "--period",
        default=DEFAULTS["period"],
        help="Data period (e.g., 1mo, 6mo, 1y)",
    )
    parser.add_argument("--start", default=DEFAULTS["start"], help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULTS["end"], help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial-cash", type=float, default=DEFAULTS["initial_cash"], help="Initial cash")
    parser.add_argument("--fee-rate", type=float, default=DEFAULTS["fee_rate"], help="Fee rate per position change")
    parser.add_argument("--trade-size", type=float, default=DEFAULTS["trade_size"], help="Trade size per step (shares)")
    parser.add_argument(
        "--max-position",
        type=float,
        default=DEFAULTS["max_position"],
        help="Maximum position size (shares)",
    )
    parser.add_argument(
        "--short-window",
        type=int,
        default=DEFAULTS["short_window"],
        help="Short moving average window",
    )
    parser.add_argument(
        "--long-window",
        type=int,
        default=DEFAULTS["long_window"],
        help="Long moving average window",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULTS["log_level"],
        help="Log level (DEBUG, INFO, WARNING)",
    )
    return parser.parse_args()


def main() -> None:
    """Run a Simple MA backtest from CLI."""
    args = _parse_args()
    _configure_logging(args.log_level)

    def _signals(prices: pd.Series) -> pd.Series:
        return generate_signals(prices, short_window=args.short_window, long_window=args.long_window)

    result = run_strategy_backtest(
        ticker=args.ticker,
        signal_generator=_signals,
        period=args.period,
        start=args.start,
        end=args.end,
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        trade_size=args.trade_size,
        max_position=args.max_position,
    )

    logger = logging.getLogger(__name__)
    summary = summarize_result(result)
    logger.info("Backtest Summary:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
