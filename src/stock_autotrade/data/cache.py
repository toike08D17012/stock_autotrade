"""Stock data caching module.

This module provides caching functionality for stock data to reduce
API calls and speed up repeated data requests.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "stock_autotrade"


class StockDataCache:
    """File-based cache for stock price data.

    The cache stores data as Parquet files, which provide efficient
    columnar storage with compression and fast read performance.

    Attributes:
        cache_dir: Directory where cache files are stored.
        enabled: Whether caching is enabled.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize the cache.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.cache/stock_autotrade.
            enabled: Whether to enable caching. Defaults to True.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.debug("Cache directory: %s", self.cache_dir)

    def _get_cache_path(self, ticker: str) -> Path:
        """Get the cache file path for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Path to the cache file.
        """
        # Sanitize ticker for filename (replace special characters)
        safe_ticker = ticker.replace(".", "_").replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_ticker}.parquet"

    def get(self, ticker: str) -> pd.DataFrame | None:
        """Get cached data for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Cached DataFrame or None if not found.
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(ticker)
        if not cache_path.exists():
            LOGGER.debug("Cache miss for %s", ticker)
            return None

        try:
            df = pd.read_parquet(cache_path)
            LOGGER.debug("Cache hit for %s: %d rows", ticker, len(df))
            return df
        except Exception as e:
            LOGGER.warning("Failed to read cache for %s: %s", ticker, e)
            return None

    def put(self, ticker: str, data: pd.DataFrame) -> None:
        """Store data in cache.

        Args:
            ticker: Stock ticker symbol.
            data: DataFrame to cache.
        """
        if not self.enabled or data.empty:
            return

        cache_path = self._get_cache_path(ticker)
        try:
            data.to_parquet(cache_path, compression="snappy")
            LOGGER.debug("Cached %d rows for %s", len(data), ticker)
        except Exception as e:
            LOGGER.warning("Failed to cache data for %s: %s", ticker, e)

    def get_date_range(self, ticker: str) -> tuple[datetime | None, datetime | None]:
        """Get the date range of cached data.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Tuple of (start_date, end_date) or (None, None) if not cached.
            Returned datetimes are timezone-naive for consistent comparisons.
        """
        cached = self.get(ticker)
        if cached is None or cached.empty:
            return None, None

        # Ensure index is datetime
        if not isinstance(cached.index, pd.DatetimeIndex):
            return None, None

        start_ts = cached.index.min()
        end_ts = cached.index.max()

        # Convert to timezone-naive datetime for consistent comparisons
        start = start_ts.tz_localize(None).to_pydatetime() if start_ts.tzinfo else start_ts.to_pydatetime()
        end = end_ts.tz_localize(None).to_pydatetime() if end_ts.tzinfo else end_ts.to_pydatetime()
        return start, end

    def merge_and_update(self, ticker: str, new_data: pd.DataFrame) -> pd.DataFrame:
        """Merge new data with cached data and update cache.

        Args:
            ticker: Stock ticker symbol.
            new_data: New DataFrame to merge.

        Returns:
            Merged DataFrame with all available data.
        """
        if new_data.empty:
            cached = self.get(ticker)
            return cached if cached is not None else new_data

        cached = self.get(ticker)
        if cached is None or cached.empty:
            self.put(ticker, new_data)
            return new_data

        # Merge cached and new data, removing duplicates
        merged = pd.concat([cached, new_data])
        merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.sort_index()

        self.put(ticker, merged)
        return merged

    def clear(self, ticker: str | None = None) -> None:
        """Clear cache for a specific ticker or all tickers.

        Args:
            ticker: Ticker to clear, or None to clear all.
        """
        if ticker:
            cache_path = self._get_cache_path(ticker)
            if cache_path.exists():
                cache_path.unlink()
                LOGGER.info("Cleared cache for %s", ticker)
        else:
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            LOGGER.info("Cleared all cache files")

    def get_cache_info(self) -> dict[str, dict[str, str | int]]:
        """Get information about cached data.

        Returns:
            Dictionary with cache info for each ticker.
        """
        info: dict[str, dict[str, str | int]] = {}
        for cache_file in self.cache_dir.glob("*.parquet"):
            ticker = cache_file.stem.replace("_", ".")
            try:
                df = pd.read_parquet(cache_file)
                start, end = None, None
                if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
                    start = df.index.min().strftime("%Y-%m-%d")
                    end = df.index.max().strftime("%Y-%m-%d")
                info[ticker] = {
                    "rows": len(df),
                    "start": start or "N/A",
                    "end": end or "N/A",
                    "size_bytes": cache_file.stat().st_size,
                }
            except Exception:
                pass
        return info


# Global cache instance
_global_cache: StockDataCache | None = None


def get_global_cache(cache_dir: str | Path | None = None, enabled: bool = True) -> StockDataCache:
    """Get or create the global cache instance.

    Args:
        cache_dir: Directory for cache files.
        enabled: Whether caching is enabled.

    Returns:
        StockDataCache instance.
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = StockDataCache(cache_dir=cache_dir, enabled=enabled)
    return _global_cache


def set_global_cache(cache: StockDataCache | None) -> None:
    """Set the global cache instance.

    Args:
        cache: Cache instance or None to disable.
    """
    global _global_cache
    _global_cache = cache
