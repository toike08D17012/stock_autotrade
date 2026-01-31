"""Data module for loading and screening stock data.

This module provides functionality for:
- Loading stock price data using yfinance
- Caching downloaded data to reduce API calls
- Screening stocks based on various criteria
"""

from stock_autotrade.data.cache import (
    StockDataCache,
    get_global_cache,
    set_global_cache,
)
from stock_autotrade.data.loader import load_stock_data
from stock_autotrade.data.screener import (
    ScreeningCriteria,
    ScreeningResult,
    StockMetrics,
    StockScreener,
    get_nikkei225_tickers,
    get_sp500_sample_tickers,
)


__all__ = [
    "load_stock_data",
    "StockDataCache",
    "get_global_cache",
    "set_global_cache",
    "ScreeningCriteria",
    "ScreeningResult",
    "StockMetrics",
    "StockScreener",
    "get_nikkei225_tickers",
    "get_sp500_sample_tickers",
]
