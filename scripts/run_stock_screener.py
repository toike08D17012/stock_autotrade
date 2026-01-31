#!/usr/bin/env python3
"""Sample script to demonstrate stock screening functionality.

This script shows how to use the StockScreener to filter stocks
based on various criteria such as price range, average price, and volume.

Example usage:
    python scripts/run_stock_screener.py

    # Screen US stocks with custom criteria
    python scripts/run_stock_screener.py --market us --min-price 100 --max-price 500

    # Screen Japanese stocks
    python scripts/run_stock_screener.py --market jp --min-price 1000 --max-price 5000

    # Save results to different formats
    python scripts/run_stock_screener.py --output results.csv     # CSV format
    python scripts/run_stock_screener.py --output results.json   # JSON format
    python scripts/run_stock_screener.py --output results.xlsx   # Excel format
    python scripts/run_stock_screener.py --output results.pkl    # Pickle format
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stock_autotrade.data.screener import (
    ScreeningCriteria,
    ScreeningResult,
    StockScreener,
    get_nikkei225_tickers,
    get_sp500_sample_tickers,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Screen stocks based on various criteria.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Market selection
    parser.add_argument(
        "--market",
        type=str,
        choices=["us", "jp"],
        default="us",
        help="Market to screen (us: S&P 500 sample, jp: Nikkei 225 sample)",
    )

    # Price criteria
    parser.add_argument("--min-price", type=float, default=None, help="Minimum latest closing price")
    parser.add_argument("--max-price", type=float, default=None, help="Maximum latest closing price")

    # Average price criteria
    parser.add_argument("--min-avg-price", type=float, default=None, help="Minimum n-day average price")
    parser.add_argument("--max-avg-price", type=float, default=None, help="Maximum n-day average price")
    parser.add_argument("--avg-price-days", type=int, default=20, help="Number of days for average price calculation")

    # Price range criteria
    parser.add_argument("--min-low-price", type=float, default=None, help="Minimum n-day low price")
    parser.add_argument("--max-low-price", type=float, default=None, help="Maximum n-day low price")
    parser.add_argument("--min-high-price", type=float, default=None, help="Minimum n-day high price")
    parser.add_argument("--max-high-price", type=float, default=None, help="Maximum n-day high price")
    parser.add_argument(
        "--price-range-days", type=int, default=20, help="Number of days for high/low price calculation"
    )

    # Volume criteria
    parser.add_argument("--min-volume", type=int, default=None, help="Minimum average daily volume")

    # Market cap criteria
    parser.add_argument("--min-market-cap", type=float, default=None, help="Minimum market capitalization")
    parser.add_argument("--max-market-cap", type=float, default=None, help="Maximum market capitalization")

    # Volatility criteria
    parser.add_argument("--min-volatility", type=float, default=None, help="Minimum daily return volatility")
    parser.add_argument("--max-volatility", type=float, default=None, help="Maximum daily return volatility")
    parser.add_argument("--volatility-days", type=int, default=20, help="Number of days for volatility calculation")

    # Other options
    parser.add_argument("--days", type=int, default=60, help="Number of days of historical data to fetch")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path. Format determined by extension (.csv, .json, .xlsx, .pkl)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def save_results_to_file(screener: StockScreener, result: ScreeningResult, output_path: str) -> None:
    """Save screening results to file in various formats.

    Args:
        screener: The StockScreener instance.
        result: The screening result.
        output_path: Output file path with extension.
    """
    output_path_obj = Path(output_path)
    suffix = output_path_obj.suffix.lower()

    if len(result.passed) == 0:
        LOGGER.warning("No stocks passed screening. Creating empty file.")

    if suffix == ".csv":
        df_passed = screener.to_dataframe(result.passed)
        df_passed.to_csv(output_path_obj, index=False)
        LOGGER.info("Results saved to CSV: %s", output_path_obj)

    elif suffix == ".json":
        # Create JSON-serializable data
        data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "passed": len(result.passed),
                "failed": len(result.failed),
                "errors": len(result.errors),
            },
            "passed_stocks": [
                {
                    "ticker": m.ticker,
                    "latest_price": m.latest_price,
                    "avg_price": m.avg_price,
                    "low_price": m.low_price,
                    "high_price": m.high_price,
                    "avg_volume": m.avg_volume,
                    "market_cap": m.market_cap,
                    "volatility": m.volatility,
                    "data_start": m.data_start_date.isoformat() if m.data_start_date else None,
                    "data_end": m.data_end_date.isoformat() if m.data_end_date else None,
                }
                for m in result.passed
            ],
            "errors": [{"ticker": ticker, "error": error} for ticker, error in result.errors],
        }
        with open(output_path_obj, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        LOGGER.info("Results saved to JSON: %s", output_path_obj)

    elif suffix == ".xlsx":
        try:
            df_passed = screener.to_dataframe(result.passed)
            df_passed.to_excel(output_path_obj, index=False, engine="openpyxl")
            LOGGER.info("Results saved to Excel: %s", output_path_obj)
        except ImportError:
            LOGGER.error("openpyxl not installed. Cannot save to Excel format.")
            raise

    elif suffix == ".pkl":
        # Save the entire result object as pickle
        with open(output_path_obj, "wb") as f:
            pickle.dump(result, f)
        LOGGER.info("Results saved to Pickle: %s", output_path_obj)

    else:
        # Default to CSV for unknown extensions
        LOGGER.warning("Unknown file extension '%s'. Defaulting to CSV format.", suffix)
        df_passed = screener.to_dataframe(result.passed)
        df_passed.to_csv(output_path_obj, index=False)
        LOGGER.info("Results saved to CSV: %s", output_path_obj)


def main() -> None:
    """Run the stock screening process."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Select tickers based on market
    if args.market == "jp":
        tickers = get_nikkei225_tickers()
        LOGGER.info("Screening Japanese stocks (Nikkei 225 sample): %d tickers", len(tickers))
    else:
        tickers = get_sp500_sample_tickers()
        LOGGER.info("Screening US stocks (S&P 500 sample): %d tickers", len(tickers))

    # Build screening criteria
    criteria = ScreeningCriteria(
        min_price=args.min_price,
        max_price=args.max_price,
        min_avg_price=args.min_avg_price,
        max_avg_price=args.max_avg_price,
        avg_price_days=args.avg_price_days,
        min_low_price=args.min_low_price,
        max_low_price=args.max_low_price,
        min_high_price=args.min_high_price,
        max_high_price=args.max_high_price,
        price_range_days=args.price_range_days,
        min_volume=args.min_volume,
        min_market_cap=args.min_market_cap,
        max_market_cap=args.max_market_cap,
        min_volatility=args.min_volatility,
        max_volatility=args.max_volatility,
        volatility_days=args.volatility_days,
    )

    LOGGER.info("Screening criteria: %s", criteria)

    # Run screening
    screener = StockScreener()
    result = screener.screen(tickers, criteria, days=args.days)

    # Print results
    print("\n" + "=" * 60)
    print("SCREENING RESULTS")
    print("=" * 60)

    print(f"\nPassed: {len(result.passed)} stocks")
    print(f"Failed: {len(result.failed)} stocks")
    print(f"Errors: {len(result.errors)} stocks")

    if result.passed:
        print("\n--- Passed Stocks ---")
        df_passed = screener.to_dataframe(result.passed)
        print(df_passed.to_string(index=False))

        # Save to file if output specified
        if args.output:
            save_results_to_file(screener, result, args.output)

    if result.errors:
        print("\n--- Errors ---")
        for ticker, error in result.errors:
            print(f"  {ticker}: {error}")

    print("\n" + "=" * 60)

    # Print passed tickers for easy copy-paste
    if result.passed:
        print("\nPassed tickers (for backtest):")
        print(result.get_passed_tickers())


if __name__ == "__main__":
    main()
