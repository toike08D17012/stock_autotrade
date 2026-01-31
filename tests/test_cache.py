"""Tests for the stock data cache module."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_autotrade.data.cache import StockDataCache, get_global_cache, set_global_cache


class TestStockDataCache:
    """Tests for StockDataCache class."""

    def test_cache_disabled(self, tmp_path: Path) -> None:
        """Test that disabled cache returns None."""
        cache = StockDataCache(cache_dir=tmp_path, enabled=False)

        # Should return None even when trying to put data
        df = pd.DataFrame({"Close": [100, 101, 102]})
        cache.put("TEST", df)
        assert cache.get("TEST") is None

    def test_cache_put_and_get(self, tmp_path: Path) -> None:
        """Test basic put and get operations."""
        cache = StockDataCache(cache_dir=tmp_path, enabled=True)

        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates)

        cache.put("TEST", df)
        cached = cache.get("TEST")

        assert cached is not None
        assert len(cached) == 5
        assert "Close" in cached.columns

    def test_cache_miss(self, tmp_path: Path) -> None:
        """Test that non-existent ticker returns None."""
        cache = StockDataCache(cache_dir=tmp_path, enabled=True)
        assert cache.get("NONEXISTENT") is None

    def test_get_date_range(self, tmp_path: Path) -> None:
        """Test getting date range from cached data."""
        cache = StockDataCache(cache_dir=tmp_path, enabled=True)

        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates)
        cache.put("TEST", df)

        start, end = cache.get_date_range("TEST")

        assert start is not None
        assert end is not None
        assert start.date() == datetime(2023, 1, 1).date()
        assert end.date() == datetime(2023, 1, 5).date()

    def test_get_date_range_empty_cache(self, tmp_path: Path) -> None:
        """Test getting date range from non-existent cache."""
        cache = StockDataCache(cache_dir=tmp_path, enabled=True)

        start, end = cache.get_date_range("NONEXISTENT")

        assert start is None
        assert end is None

    def test_merge_and_update(self, tmp_path: Path) -> None:
        """Test merging new data with cached data."""
        cache = StockDataCache(cache_dir=tmp_path, enabled=True)

        # Initial data
        dates1 = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df1 = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates1)
        cache.put("TEST", df1)

        # New data with overlap and extension
        dates2 = pd.date_range(start="2023-01-03", periods=5, freq="D")
        df2 = pd.DataFrame({"Close": [200, 201, 202, 203, 204]}, index=dates2)

        merged = cache.merge_and_update("TEST", df2)

        # Should have 7 unique days (Jan 1-7)
        assert len(merged) == 7
        # Overlapping data should use new values
        assert merged.loc["2023-01-03", "Close"] == 200

    def test_merge_with_no_cache(self, tmp_path: Path) -> None:
        """Test merging when no cache exists."""
        cache = StockDataCache(cache_dir=tmp_path, enabled=True)

        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({"Close": [100, 101, 102]}, index=dates)

        merged = cache.merge_and_update("TEST", df)

        assert len(merged) == 3
        # Should also be cached now
        assert cache.get("TEST") is not None

    def test_clear_single_ticker(self, tmp_path: Path) -> None:
        """Test clearing cache for a single ticker."""
        cache = StockDataCache(cache_dir=tmp_path, enabled=True)

        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({"Close": [100, 101, 102]}, index=dates)

        cache.put("TEST1", df)
        cache.put("TEST2", df)

        cache.clear("TEST1")

        assert cache.get("TEST1") is None
        assert cache.get("TEST2") is not None

    def test_clear_all(self, tmp_path: Path) -> None:
        """Test clearing all cache."""
        cache = StockDataCache(cache_dir=tmp_path, enabled=True)

        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({"Close": [100, 101, 102]}, index=dates)

        cache.put("TEST1", df)
        cache.put("TEST2", df)

        cache.clear()

        assert cache.get("TEST1") is None
        assert cache.get("TEST2") is None

    def test_get_cache_info(self, tmp_path: Path) -> None:
        """Test getting cache info."""
        cache = StockDataCache(cache_dir=tmp_path, enabled=True)

        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates)
        cache.put("TEST_T", df)

        info = cache.get_cache_info()

        # Ticker name is converted (underscore to dot)
        assert "TEST.T" in info
        assert info["TEST.T"]["rows"] == 5
        assert info["TEST.T"]["start"] == "2023-01-01"
        assert info["TEST.T"]["end"] == "2023-01-05"

    def test_ticker_name_sanitization(self, tmp_path: Path) -> None:
        """Test that ticker names with special characters are handled."""
        cache = StockDataCache(cache_dir=tmp_path, enabled=True)

        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({"Close": [100, 101, 102]}, index=dates)

        # Ticker with special characters
        cache.put("7203.T", df)

        # Should be able to retrieve it
        cached = cache.get("7203.T")
        assert cached is not None
        assert len(cached) == 3


class TestGlobalCache:
    """Tests for global cache functions."""

    def test_get_and_set_global_cache(self, tmp_path: Path) -> None:
        """Test setting and getting global cache."""
        cache = StockDataCache(cache_dir=tmp_path, enabled=True)
        set_global_cache(cache)

        retrieved = get_global_cache()
        assert retrieved is cache

        # Reset for other tests
        set_global_cache(None)

    def test_get_global_cache_creates_default(self) -> None:
        """Test that get_global_cache creates a default cache."""
        set_global_cache(None)  # Reset

        cache = get_global_cache()
        assert cache is not None
        assert cache.enabled is True

        # Reset for other tests
        set_global_cache(None)
