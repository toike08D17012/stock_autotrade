"""Tests for the stock screener module."""

from datetime import datetime
from unittest.mock import MagicMock, PropertyMock

import pandas as pd
import pytest

from stock_autotrade.data.screener import (
    ScreeningCriteria,
    ScreeningResult,
    StockMetrics,
    StockScreener,
    get_nikkei225_tickers,
    get_sp500_sample_tickers,
)


@pytest.fixture
def sample_stock_data() -> pd.DataFrame:
    """Create sample stock data for testing."""
    dates = pd.date_range(start="2025-01-01", periods=30, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(30)],
            "High": [105.0 + i for i in range(30)],
            "Low": [95.0 + i for i in range(30)],
            "Close": [102.0 + i for i in range(30)],
            "Volume": [1000000 + i * 10000 for i in range(30)],
        },
        index=dates,
    )


@pytest.fixture
def screener_with_mock(sample_stock_data: pd.DataFrame) -> tuple[StockScreener, MagicMock]:
    """Create a StockScreener with a mocked ticker factory."""
    ticker_mock = MagicMock()
    ticker_mock.history.return_value = sample_stock_data
    type(ticker_mock).info = PropertyMock(return_value={"marketCap": 1_000_000_000})
    ticker_factory = MagicMock(return_value=ticker_mock)
    screener = StockScreener(ticker_factory=ticker_factory)
    return screener, ticker_factory


class TestScreeningCriteria:
    """Tests for ScreeningCriteria dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        criteria = ScreeningCriteria()
        assert criteria.min_price is None
        assert criteria.max_price is None
        assert criteria.avg_price_days == 20
        assert criteria.price_range_days == 20
        assert criteria.volatility_days == 20

    def test_custom_values(self) -> None:
        """Test custom values are set correctly."""
        criteria = ScreeningCriteria(
            min_price=100.0,
            max_price=500.0,
            min_avg_price=150.0,
            avg_price_days=30,
        )
        assert criteria.min_price == 100.0
        assert criteria.max_price == 500.0
        assert criteria.min_avg_price == 150.0
        assert criteria.avg_price_days == 30


class TestStockMetrics:
    """Tests for StockMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        metrics = StockMetrics(ticker="AAPL")
        assert metrics.ticker == "AAPL"
        assert metrics.latest_price is None
        assert metrics.avg_price is None
        assert metrics.market_cap is None

    def test_custom_values(self) -> None:
        """Test custom values are set correctly."""
        metrics = StockMetrics(
            ticker="AAPL",
            latest_price=150.0,
            avg_price=145.0,
            low_price=140.0,
            high_price=160.0,
        )
        assert metrics.ticker == "AAPL"
        assert metrics.latest_price == 150.0
        assert metrics.avg_price == 145.0
        assert metrics.low_price == 140.0
        assert metrics.high_price == 160.0


class TestScreeningResult:
    """Tests for ScreeningResult dataclass."""

    def test_get_passed_tickers(self) -> None:
        """Test get_passed_tickers returns correct ticker list."""
        result = ScreeningResult()
        result.passed.append(StockMetrics(ticker="AAPL"))
        result.passed.append(StockMetrics(ticker="GOOGL"))
        result.failed.append(StockMetrics(ticker="MSFT"))

        tickers = result.get_passed_tickers()
        assert tickers == ["AAPL", "GOOGL"]

    def test_empty_result(self) -> None:
        """Test empty result returns empty list."""
        result = ScreeningResult()
        assert result.get_passed_tickers() == []


class TestStockScreener:
    """Tests for StockScreener class."""

    def test_calculate_metrics_returns_correct_values(
        self, screener_with_mock: tuple[StockScreener, MagicMock], sample_stock_data: pd.DataFrame
    ) -> None:
        """Test calculate_metrics returns correct values."""
        screener, ticker_factory = screener_with_mock

        metrics = screener.calculate_metrics("AAPL", days=30, avg_days=20)

        ticker_factory.assert_called_once_with("AAPL")
        assert metrics.ticker == "AAPL"
        assert metrics.latest_price == sample_stock_data["Close"].iloc[-1]
        assert metrics.avg_price is not None
        assert metrics.low_price is not None
        assert metrics.high_price is not None
        assert metrics.avg_volume is not None

    def test_calculate_metrics_raises_on_empty_data(self) -> None:
        """Test calculate_metrics raises ValueError when no data is available."""
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = pd.DataFrame()
        ticker_factory = MagicMock(return_value=ticker_mock)
        screener = StockScreener(ticker_factory=ticker_factory)

        with pytest.raises(ValueError, match="No data available"):
            screener.calculate_metrics("INVALID")

    def test_check_criteria_passes_when_all_met(self) -> None:
        """Test check_criteria returns True when all criteria are met."""
        screener = StockScreener()
        metrics = StockMetrics(
            ticker="AAPL",
            latest_price=150.0,
            avg_price=145.0,
            low_price=140.0,
            high_price=160.0,
            avg_volume=1_000_000.0,
            market_cap=2_500_000_000_000.0,
            volatility=0.02,
        )
        criteria = ScreeningCriteria(
            min_price=100.0,
            max_price=200.0,
            min_avg_price=100.0,
            max_avg_price=200.0,
            min_low_price=130.0,
            max_high_price=170.0,
            min_volume=500_000,
            min_market_cap=1_000_000_000_000.0,
        )

        assert screener.check_criteria(metrics, criteria) is True

    def test_check_criteria_fails_when_price_below_min(self) -> None:
        """Test check_criteria returns False when price is below minimum."""
        screener = StockScreener()
        metrics = StockMetrics(ticker="AAPL", latest_price=50.0)
        criteria = ScreeningCriteria(min_price=100.0)

        assert screener.check_criteria(metrics, criteria) is False

    def test_check_criteria_fails_when_price_above_max(self) -> None:
        """Test check_criteria returns False when price is above maximum."""
        screener = StockScreener()
        metrics = StockMetrics(ticker="AAPL", latest_price=250.0)
        criteria = ScreeningCriteria(max_price=200.0)

        assert screener.check_criteria(metrics, criteria) is False

    def test_check_criteria_fails_when_avg_price_out_of_range(self) -> None:
        """Test check_criteria returns False when avg price is out of range."""
        screener = StockScreener()
        metrics = StockMetrics(ticker="AAPL", avg_price=50.0)
        criteria = ScreeningCriteria(min_avg_price=100.0)

        assert screener.check_criteria(metrics, criteria) is False

    def test_check_criteria_fails_when_low_price_below_min(self) -> None:
        """Test check_criteria returns False when low price is below minimum."""
        screener = StockScreener()
        metrics = StockMetrics(ticker="AAPL", low_price=90.0)
        criteria = ScreeningCriteria(min_low_price=100.0)

        assert screener.check_criteria(metrics, criteria) is False

    def test_check_criteria_fails_when_high_price_above_max(self) -> None:
        """Test check_criteria returns False when high price is above maximum."""
        screener = StockScreener()
        metrics = StockMetrics(ticker="AAPL", high_price=250.0)
        criteria = ScreeningCriteria(max_high_price=200.0)

        assert screener.check_criteria(metrics, criteria) is False

    def test_check_criteria_fails_when_volume_below_min(self) -> None:
        """Test check_criteria returns False when volume is below minimum."""
        screener = StockScreener()
        metrics = StockMetrics(ticker="AAPL", avg_volume=100_000.0)
        criteria = ScreeningCriteria(min_volume=500_000)

        assert screener.check_criteria(metrics, criteria) is False

    def test_check_criteria_fails_when_market_cap_out_of_range(self) -> None:
        """Test check_criteria returns False when market cap is out of range."""
        screener = StockScreener()
        metrics = StockMetrics(ticker="AAPL", market_cap=500_000_000.0)
        criteria = ScreeningCriteria(min_market_cap=1_000_000_000.0)

        assert screener.check_criteria(metrics, criteria) is False

    def test_check_criteria_fails_when_volatility_out_of_range(self) -> None:
        """Test check_criteria returns False when volatility is out of range."""
        screener = StockScreener()
        metrics = StockMetrics(ticker="AAPL", volatility=0.05)
        criteria = ScreeningCriteria(max_volatility=0.03)

        assert screener.check_criteria(metrics, criteria) is False

    def test_check_criteria_passes_with_none_values(self) -> None:
        """Test check_criteria handles None values correctly."""
        screener = StockScreener()
        metrics = StockMetrics(ticker="AAPL", latest_price=None)
        criteria = ScreeningCriteria(min_price=100.0)

        # Should pass since we can't compare None to a number
        assert screener.check_criteria(metrics, criteria) is True

    def test_screen_returns_correct_results(self, screener_with_mock: tuple[StockScreener, MagicMock]) -> None:
        """Test screen returns correct screening results."""
        screener, _ = screener_with_mock
        criteria = ScreeningCriteria(min_price=100.0, max_price=200.0)

        result = screener.screen(["AAPL", "GOOGL"], criteria, days=30)

        assert isinstance(result, ScreeningResult)
        assert len(result.passed) == 2  # Both should pass with sample data
        assert len(result.failed) == 0
        assert len(result.errors) == 0

    def test_screen_handles_errors_gracefully(self) -> None:
        """Test screen handles errors without crashing."""
        ticker_mock = MagicMock()
        ticker_mock.history.side_effect = Exception("API Error")
        ticker_factory = MagicMock(return_value=ticker_mock)
        screener = StockScreener(ticker_factory=ticker_factory)
        criteria = ScreeningCriteria()

        result = screener.screen(["AAPL"], criteria)

        assert len(result.passed) == 0
        assert len(result.failed) == 0
        assert len(result.errors) == 1
        assert result.errors[0][0] == "AAPL"
        assert "API Error" in result.errors[0][1]

    def test_to_dataframe_converts_metrics(self) -> None:
        """Test to_dataframe converts metrics list to DataFrame."""
        screener = StockScreener()
        metrics_list = [
            StockMetrics(
                ticker="AAPL",
                latest_price=150.0,
                avg_price=145.0,
                low_price=140.0,
                high_price=160.0,
                avg_volume=1_000_000.0,
                market_cap=2_500_000_000_000.0,
                volatility=0.02,
                data_start_date=datetime(2025, 1, 1),
                data_end_date=datetime(2025, 1, 30),
            ),
            StockMetrics(
                ticker="GOOGL",
                latest_price=180.0,
                avg_price=175.0,
            ),
        ]

        df = screener.to_dataframe(metrics_list)

        assert len(df) == 2
        assert "ticker" in df.columns
        assert "latest_price" in df.columns
        assert df.iloc[0]["ticker"] == "AAPL"
        assert df.iloc[0]["latest_price"] == 150.0
        assert df.iloc[1]["ticker"] == "GOOGL"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_nikkei225_tickers_returns_list(self) -> None:
        """Test get_nikkei225_tickers returns a non-empty list."""
        tickers = get_nikkei225_tickers()
        assert isinstance(tickers, list)
        assert len(tickers) > 0
        assert all(t.endswith(".T") for t in tickers)

    def test_get_sp500_sample_tickers_returns_list(self) -> None:
        """Test get_sp500_sample_tickers returns a non-empty list."""
        tickers = get_sp500_sample_tickers()
        assert isinstance(tickers, list)
        assert len(tickers) > 0
        assert "AAPL" in tickers
