import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest  # type: ignore[import-not-found]

import stock_autotrade.data.loader as loader_module
from stock_autotrade.data.loader import load_stock_data


def test_load_stock_data_uses_period_when_no_date_range() -> None:
    """Uses period argument when no date range is provided."""
    dates = pd.date_range(start="2024-01-01", periods=2, freq="D")
    expected = pd.DataFrame({"Close": [1.0, 2.0]}, index=dates)
    ticker_mock = MagicMock()
    ticker_mock.history.return_value = expected
    ticker_factory = MagicMock(return_value=ticker_mock)

    result = load_stock_data("AAPL", period="5d", start=None, end=None, ticker_factory=ticker_factory, use_cache=False)

    assert len(result) == 2
    ticker_factory.assert_called_once_with("AAPL")


def test_load_stock_data_uses_date_range_when_start_provided() -> None:
    """Uses start/end arguments when start is provided."""
    dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
    expected = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
    ticker_mock = MagicMock()
    ticker_mock.history.return_value = expected
    ticker_factory = MagicMock(return_value=ticker_mock)

    result = load_stock_data(
        "AAPL", start="2024-01-01", end="2024-01-10", ticker_factory=ticker_factory, use_cache=False
    )

    assert len(result) == 5
    ticker_factory.assert_called_once_with("AAPL")


def test_load_stock_data_uses_date_range_when_only_end_provided() -> None:
    """Uses start/end arguments when only end is provided."""
    dates = pd.date_range(start="2024-01-01", periods=3, freq="D")
    expected = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=dates)
    ticker_mock = MagicMock()
    ticker_mock.history.return_value = expected
    ticker_factory = MagicMock(return_value=ticker_mock)

    result = load_stock_data("AAPL", end="2024-01-10", ticker_factory=ticker_factory, use_cache=False)

    assert len(result) == 3
    ticker_factory.assert_called_once_with("AAPL")


def test_load_stock_data_logs_warning_when_empty(caplog: pytest.LogCaptureFixture) -> None:
    """Logs a warning when data is empty."""
    expected = pd.DataFrame()
    ticker_mock = MagicMock()
    ticker_mock.history.return_value = expected
    ticker_factory = MagicMock(return_value=ticker_mock)

    with caplog.at_level(logging.WARNING, logger="stock_autotrade.data.loader"):
        result = load_stock_data("AAPL", ticker_factory=ticker_factory, use_cache=False)

    assert result.empty
    assert any("No data found" in message for message in caplog.messages)


def test_load_stock_data_rejects_blank_ticker() -> None:
    """Rejects blank tickers."""
    with pytest.raises(ValueError):
        load_stock_data(" ")


def test_load_stock_data_uses_lazy_default_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use lazy default ticker factory when no custom factory is provided."""
    dates = pd.date_range(start="2024-01-01", periods=2, freq="D")
    expected = pd.DataFrame({"Close": [1.0, 2.0]}, index=dates)
    ticker_mock = MagicMock()
    ticker_mock.history.return_value = expected

    def fake_default_factory(_: str) -> MagicMock:
        return ticker_mock

    monkeypatch.setattr(loader_module, "_default_ticker_factory", fake_default_factory)

    result = load_stock_data("AAPL", use_cache=False)

    assert len(result) == 2
    ticker_mock.history.assert_called_once()
