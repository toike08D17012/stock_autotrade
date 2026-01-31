import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest  # type: ignore[import-not-found]

from stock_autotrade.data.loader import load_stock_data


def test_load_stock_data_uses_period_when_no_date_range() -> None:
    """Uses period argument when no date range is provided."""
    expected = pd.DataFrame({"Close": [1.0, 2.0]})
    ticker_mock = MagicMock()
    ticker_mock.history.return_value = expected
    ticker_factory = MagicMock(return_value=ticker_mock)

    result = load_stock_data("AAPL", period="5d", start=None, end=None, ticker_factory=ticker_factory)

    assert result is expected
    ticker_factory.assert_called_once_with("AAPL")
    ticker_mock.history.assert_called_once_with(period="5d")


def test_load_stock_data_uses_date_range_when_start_provided() -> None:
    """Uses start/end arguments when start is provided."""
    expected = pd.DataFrame({"Close": [1.0]})
    ticker_mock = MagicMock()
    ticker_mock.history.return_value = expected
    ticker_factory = MagicMock(return_value=ticker_mock)

    result = load_stock_data("AAPL", start="2024-01-01", end="2024-01-10", ticker_factory=ticker_factory)

    assert result is expected
    ticker_factory.assert_called_once_with("AAPL")
    ticker_mock.history.assert_called_once_with(start="2024-01-01", end="2024-01-10")


def test_load_stock_data_uses_date_range_when_only_end_provided() -> None:
    """Uses start/end arguments when only end is provided."""
    expected = pd.DataFrame({"Close": [1.0]})
    ticker_mock = MagicMock()
    ticker_mock.history.return_value = expected
    ticker_factory = MagicMock(return_value=ticker_mock)

    result = load_stock_data("AAPL", end="2024-01-10", ticker_factory=ticker_factory)

    assert result is expected
    ticker_factory.assert_called_once_with("AAPL")
    ticker_mock.history.assert_called_once_with(start=None, end="2024-01-10")


def test_load_stock_data_logs_warning_when_empty(caplog: pytest.LogCaptureFixture) -> None:
    """Logs a warning when data is empty."""
    expected = pd.DataFrame()
    ticker_mock = MagicMock()
    ticker_mock.history.return_value = expected
    ticker_factory = MagicMock(return_value=ticker_mock)

    with caplog.at_level(logging.WARNING, logger="src.data.loader"):
        result = load_stock_data("AAPL", ticker_factory=ticker_factory)

    assert result is expected
    assert any("No data found" in message for message in caplog.messages)


def test_load_stock_data_rejects_blank_ticker() -> None:
    """Rejects blank tickers."""
    with pytest.raises(ValueError):
        load_stock_data(" ")
