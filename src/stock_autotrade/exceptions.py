"""Custom exceptions for the stock autotrade framework.

This module defines specific exceptions for better error handling
and debugging throughout the application.
"""


class StockAutotradeError(Exception):
    """Base exception class for stock autotrade errors."""


class DataError(StockAutotradeError):
    """Exception raised for data-related errors."""


class DataNotFoundError(DataError):
    """Exception raised when requested data is not found."""

    def __init__(self, ticker: str, message: str | None = None) -> None:
        """Initialize DataNotFoundError.

        Args:
            ticker: The ticker symbol that was not found.
            message: Optional custom message.
        """
        self.ticker = ticker
        if message is None:
            message = f"No data found for ticker: {ticker}"
        super().__init__(message)


class InvalidTickerError(DataError):
    """Exception raised for invalid ticker symbols."""

    def __init__(self, ticker: str, message: str | None = None) -> None:
        """Initialize InvalidTickerError.

        Args:
            ticker: The invalid ticker symbol.
            message: Optional custom message.
        """
        self.ticker = ticker
        if message is None:
            message = f"Invalid ticker symbol: {ticker}"
        super().__init__(message)


class BacktestError(StockAutotradeError):
    """Exception raised for backtest-related errors."""


class InsufficientDataError(BacktestError):
    """Exception raised when there is insufficient data for backtesting."""

    def __init__(self, required: int, available: int, message: str | None = None) -> None:
        """Initialize InsufficientDataError.

        Args:
            required: Required number of data points.
            available: Available number of data points.
            message: Optional custom message.
        """
        self.required = required
        self.available = available
        if message is None:
            message = f"Insufficient data: required {required} periods, but only {available} available."
        super().__init__(message)


class InvalidParameterError(StockAutotradeError):
    """Exception raised for invalid parameter values."""

    def __init__(self, param_name: str, value: object, message: str | None = None) -> None:
        """Initialize InvalidParameterError.

        Args:
            param_name: Name of the invalid parameter.
            value: The invalid value.
            message: Optional custom message.
        """
        self.param_name = param_name
        self.value = value
        if message is None:
            message = f"Invalid value for parameter '{param_name}': {value}"
        super().__init__(message)


class StrategyError(StockAutotradeError):
    """Exception raised for strategy-related errors."""


class OptimizationError(StockAutotradeError):
    """Exception raised for optimization-related errors."""


class NoValidResultsError(OptimizationError):
    """Exception raised when optimization produces no valid results."""
