"""Stock data loading module with caching support."""

import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Literal, cast, overload

import pandas as pd

from .cache import StockDataCache, get_global_cache
from .ticker_protocols import TickerLike


LOGGER = logging.getLogger(__name__)


def _default_ticker_factory(ticker: str) -> TickerLike:
    """Create a yfinance ticker client lazily.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Ticker client compatible with :class:`TickerLike`.

    Raises:
        ModuleNotFoundError: If yfinance is not installed.
    """
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "yfinance is required to download stock data. Install it with `pip install yfinance`."
        ) from exc

    return cast(TickerLike, yf.Ticker(ticker))


def _parse_period_to_days(period: str) -> int:
    """Convert period string to approximate number of days.

    Args:
        period: Period string like '1mo', '1y', '5d', etc.

    Returns:
        Approximate number of days.
    """
    period = period.lower().strip()

    # Extract number and unit
    num = ""
    unit = ""
    for char in period:
        if char.isdigit():
            num += char
        else:
            unit += char

    num_val = int(num) if num else 1

    unit_days = {
        "d": 1,
        "w": 7,
        "wk": 7,
        "mo": 30,
        "m": 30,
        "y": 365,
        "yr": 365,
        "max": 365 * 50,  # ~50 years
    }

    return num_val * unit_days.get(unit, 30)


def _calculate_missing_ranges(
    requested_start: datetime,
    requested_end: datetime,
    cached_start: datetime | None,
    cached_end: datetime | None,
) -> list[tuple[datetime, datetime]]:
    """Calculate date ranges that need to be downloaded.

    Args:
        requested_start: Start date of requested range.
        requested_end: End date of requested range.
        cached_start: Start date of cached data (or None).
        cached_end: End date of cached data (or None).

    Returns:
        List of (start, end) tuples for ranges to download.
    """
    if cached_start is None or cached_end is None:
        # No cache, download everything
        return [(requested_start, requested_end)]

    missing_ranges: list[tuple[datetime, datetime]] = []

    # Need data before cache start?
    if requested_start < cached_start:
        missing_ranges.append((requested_start, cached_start - timedelta(days=1)))

    # Need data after cache end?
    if requested_end > cached_end:
        missing_ranges.append((cached_end + timedelta(days=1), requested_end))

    return missing_ranges


@overload
def load_stock_data(
    ticker: str,
    period: str = "1mo",
    start: str | None = None,
    end: str | None = None,
    ticker_factory: Callable[[str], TickerLike] | None = None,
    use_cache: bool = True,
    cache: StockDataCache | None = None,
    return_download_info: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def load_stock_data(
    ticker: str,
    period: str = "1mo",
    start: str | None = None,
    end: str | None = None,
    ticker_factory: Callable[[str], TickerLike] | None = None,
    use_cache: bool = True,
    cache: StockDataCache | None = None,
    return_download_info: Literal[True] = ...,
) -> tuple[pd.DataFrame, bool]: ...


def load_stock_data(
    ticker: str,
    period: str = "1mo",
    start: str | None = None,
    end: str | None = None,
    ticker_factory: Callable[[str], TickerLike] | None = None,
    use_cache: bool = True,
    cache: StockDataCache | None = None,
    return_download_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, bool]:
    """Load stock data for a given ticker using yfinance with caching.

    This function first checks the cache for existing data, then downloads
    only the missing date ranges to minimize API calls.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL", "7203.T").
        period (str, optional): Data period to download (e.g., "1mo", "1y", "max"). Defaults to "1mo".
            Ignored if start/end are provided.
        start (str | None, optional): Start date string (YYYY-MM-DD) or datetime. Defaults to None.
        end (str | None, optional): End date string (YYYY-MM-DD) or datetime. Defaults to None.
        ticker_factory (Callable[[str], TickerLike] | None, optional): Factory for creating ticker clients.
            Defaults to None, which lazily creates ``yfinance.Ticker``.
        use_cache (bool, optional): Whether to use caching. Defaults to True.
        cache (StockDataCache | None, optional): Cache instance to use.
            Defaults to None, which uses the global cache.
        return_download_info (bool, optional): If True, returns a tuple of (DataFrame, downloaded)
            where downloaded is True if data was fetched from yfinance. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the stock price history.
        tuple[pd.DataFrame, bool]: If return_download_info is True, returns (data, downloaded).
    """
    if not ticker or not ticker.strip():
        raise ValueError("Ticker must be a non-empty string.")

    # Get cache instance
    cache = (cache or get_global_cache()) if use_cache else None

    # Determine requested date range
    today = datetime.now()
    if start is not None:
        requested_start = datetime.strptime(start, "%Y-%m-%d") if isinstance(start, str) else start
    else:
        days = _parse_period_to_days(period)
        requested_start = today - timedelta(days=days)

    requested_end = (datetime.strptime(end, "%Y-%m-%d") if isinstance(end, str) else end) if end is not None else today

    LOGGER.info("Loading data for %s (Start: %s, End: %s).", ticker, requested_start.date(), requested_end.date())

    # Check cache and calculate missing ranges
    if cache:
        cached_start, cached_end = cache.get_date_range(ticker)

        if cached_start and cached_end:
            LOGGER.debug(
                "Cache for %s: %s to %s",
                ticker,
                cached_start.date(),
                cached_end.date(),
            )

            # Check if cache fully covers requested range
            if cached_start <= requested_start and cached_end >= requested_end:
                LOGGER.info("Using cached data for %s (fully covered)", ticker)
                cached_data = cache.get(ticker)
                if cached_data is not None:
                    # Filter to requested range
                    # Normalize index to timezone-naive for comparison
                    index = cached_data.index
                    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
                        index = index.tz_localize(None)
                    mask = (index >= requested_start) & (index <= requested_end)
                    result_df = cast(pd.DataFrame, cached_data[mask])
                    if return_download_info:
                        return result_df, False
                    return result_df

        missing_ranges = _calculate_missing_ranges(
            requested_start,
            requested_end,
            cached_start,
            cached_end,
        )
    else:
        missing_ranges = [(requested_start, requested_end)]

    # Download missing data
    ticker_factory = ticker_factory or _default_ticker_factory
    ticker_obj = ticker_factory(ticker)

    all_new_data: list[pd.DataFrame] = []
    for range_start, range_end in missing_ranges:
        LOGGER.info(
            "Downloading %s: %s to %s",
            ticker,
            range_start.date(),
            range_end.date(),
        )
        df = ticker_obj.history(
            start=range_start.strftime("%Y-%m-%d"),
            end=(range_end + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        if not df.empty:
            all_new_data.append(df)

    # Combine new data
    if all_new_data:
        new_data = pd.concat(all_new_data)
        new_data = new_data[~new_data.index.duplicated(keep="last")]
        new_data = new_data.sort_index()
    else:
        new_data = pd.DataFrame()

    # Merge with cache
    if cache:
        merged = cache.merge_and_update(ticker, new_data)
        # Filter to requested range
        if not merged.empty:
            # Normalize index to timezone-naive for comparison
            index = merged.index
            if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
                index = index.tz_localize(None)
            mask = (index >= requested_start) & (index <= requested_end)  # type: ignore[operator]
            result = merged[mask]
        else:
            result = merged
    else:
        result = new_data

    if result.empty:
        LOGGER.warning("No data found for %s.", ticker)

    # downloaded is True if we fetched any data from yfinance
    downloaded = len(all_new_data) > 0
    if return_download_info:
        return cast(pd.DataFrame, result), downloaded
    return cast(pd.DataFrame, result)
