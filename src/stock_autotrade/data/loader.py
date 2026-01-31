import logging
from collections.abc import Callable
from typing import cast

import pandas as pd
import yfinance as yf


LOGGER = logging.getLogger(__name__)


def load_stock_data(
    ticker: str,
    period: str = "1mo",
    start: str | None = None,
    end: str | None = None,
    ticker_factory: Callable[[str], yf.Ticker] | None = None,
) -> pd.DataFrame:
    """Load stock data for a given ticker using yfinance.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL", "7203.T").
        period (str, optional): Data period to download (e.g., "1mo", "1y", "max"). Defaults to "1mo".
            Ignored if start/end are provided.
        start (str | None, optional): Start date string (YYYY-MM-DD) or datetime. Defaults to None.
        end (str | None, optional): End date string (YYYY-MM-DD) or datetime. Defaults to None.
        ticker_factory (Callable[[str], yf.Ticker] | None, optional): Factory for creating ticker clients.
            Defaults to None, which uses ``yfinance.Ticker``.

    Returns:
        pd.DataFrame: A DataFrame containing the stock price history.
    """
    if not ticker or not ticker.strip():
        raise ValueError("Ticker must be a non-empty string.")

    LOGGER.info("Loading data for %s (Period: %s, Start: %s, End: %s).", ticker, period, start, end)
    ticker_factory = ticker_factory or yf.Ticker
    ticker_obj = ticker_factory(ticker)
    if start is not None or end is not None:
        df = ticker_obj.history(start=start, end=end)
    else:
        df = ticker_obj.history(period=period)

    if df.empty:
        LOGGER.warning("No data found for %s.", ticker)

    return cast(pd.DataFrame, df)
