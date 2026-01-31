"""Tests for the exceptions module."""

import pytest

from stock_autotrade.exceptions import (
    BacktestError,
    DataError,
    DataNotFoundError,
    InsufficientDataError,
    InvalidParameterError,
    InvalidTickerError,
    NoValidResultsError,
    OptimizationError,
    StockAutotradeError,
    StrategyError,
)


class TestExceptionHierarchy:
    """Test the exception class hierarchy."""

    def test_base_exception(self) -> None:
        """Test that StockAutotradeError is the base class."""
        exc = StockAutotradeError("test error")
        assert str(exc) == "test error"
        assert isinstance(exc, Exception)

    def test_data_error_hierarchy(self) -> None:
        """Test DataError hierarchy."""
        exc = DataError("data error")
        assert isinstance(exc, StockAutotradeError)

    def test_backtest_error_hierarchy(self) -> None:
        """Test BacktestError hierarchy."""
        exc = BacktestError("backtest error")
        assert isinstance(exc, StockAutotradeError)

    def test_strategy_error_hierarchy(self) -> None:
        """Test StrategyError hierarchy."""
        exc = StrategyError("strategy error")
        assert isinstance(exc, StockAutotradeError)

    def test_optimization_error_hierarchy(self) -> None:
        """Test OptimizationError hierarchy."""
        exc = OptimizationError("optimization error")
        assert isinstance(exc, StockAutotradeError)


class TestDataNotFoundError:
    """Tests for DataNotFoundError."""

    def test_default_message(self) -> None:
        """Test default error message."""
        exc = DataNotFoundError("AAPL")
        assert exc.ticker == "AAPL"
        assert "AAPL" in str(exc)
        assert "No data found" in str(exc)

    def test_custom_message(self) -> None:
        """Test custom error message."""
        exc = DataNotFoundError("AAPL", "Custom error message")
        assert exc.ticker == "AAPL"
        assert str(exc) == "Custom error message"

    def test_inheritance(self) -> None:
        """Test inheritance chain."""
        exc = DataNotFoundError("AAPL")
        assert isinstance(exc, DataError)
        assert isinstance(exc, StockAutotradeError)


class TestInvalidTickerError:
    """Tests for InvalidTickerError."""

    def test_default_message(self) -> None:
        """Test default error message."""
        exc = InvalidTickerError("")
        assert exc.ticker == ""
        assert "Invalid ticker" in str(exc)

    def test_custom_message(self) -> None:
        """Test custom error message."""
        exc = InvalidTickerError("BAD", "Ticker not recognized")
        assert exc.ticker == "BAD"
        assert str(exc) == "Ticker not recognized"


class TestInsufficientDataError:
    """Tests for InsufficientDataError."""

    def test_default_message(self) -> None:
        """Test default error message."""
        exc = InsufficientDataError(required=100, available=50)
        assert exc.required == 100
        assert exc.available == 50
        assert "100" in str(exc)
        assert "50" in str(exc)

    def test_custom_message(self) -> None:
        """Test custom error message."""
        exc = InsufficientDataError(required=100, available=50, message="Not enough data")
        assert str(exc) == "Not enough data"

    def test_inheritance(self) -> None:
        """Test inheritance chain."""
        exc = InsufficientDataError(100, 50)
        assert isinstance(exc, BacktestError)


class TestInvalidParameterError:
    """Tests for InvalidParameterError."""

    def test_default_message(self) -> None:
        """Test default error message."""
        exc = InvalidParameterError("window", -1)
        assert exc.param_name == "window"
        assert exc.value == -1
        assert "window" in str(exc)
        assert "-1" in str(exc)

    def test_custom_message(self) -> None:
        """Test custom error message."""
        exc = InvalidParameterError("window", -1, "Window must be positive")
        assert str(exc) == "Window must be positive"


class TestNoValidResultsError:
    """Tests for NoValidResultsError."""

    def test_basic_usage(self) -> None:
        """Test basic usage."""
        exc = NoValidResultsError("All parameter combinations failed")
        assert "failed" in str(exc)
        assert isinstance(exc, OptimizationError)


class TestExceptionRaising:
    """Test that exceptions can be raised and caught properly."""

    def test_raise_and_catch_data_error(self) -> None:
        """Test raising and catching DataError."""
        with pytest.raises(DataError):
            raise DataNotFoundError("AAPL")

    def test_raise_and_catch_base_error(self) -> None:
        """Test catching with base class."""
        with pytest.raises(StockAutotradeError):
            raise InvalidParameterError("test", 0)
