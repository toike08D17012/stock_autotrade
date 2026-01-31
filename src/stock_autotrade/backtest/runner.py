"""High-level backtest runner."""

from collections.abc import Callable

import pandas as pd

from stock_autotrade.backtest.engine import BacktestResult, run_backtest
from stock_autotrade.data.loader import load_stock_data
from stock_autotrade.strategy.simple_moving_average import generate_signals


def run_strategy_backtest(
    ticker: str,
    signal_generator: Callable[[pd.Series], pd.Series],
    period: str = "6mo",
    start: str | None = None,
    end: str | None = None,
    initial_cash: float = 1_000_000.0,
    fee_rate: float = 0.0,
    trade_size: float = 1.0,
    max_position: float = 1.0,
) -> BacktestResult:
    """Load data and run a backtest with a given signal generator.

    Args:
        ticker (str): Ticker symbol to download.
        signal_generator (Callable[[pd.Series], pd.Series]): Strategy signal generator.
        period (str, optional): Download period. Defaults to "6mo".
        start (str | None, optional): Start date string (YYYY-MM-DD). Defaults to None.
        end (str | None, optional): End date string (YYYY-MM-DD). Defaults to None.
        initial_cash (float, optional): Starting capital. Defaults to 1_000_000.0.
        fee_rate (float, optional): Fee applied per position change. Defaults to 0.0.
        trade_size (float, optional): Trade size per step (shares). Defaults to 1.0.
        max_position (float, optional): Maximum position size (shares). Defaults to 1.0.

    Returns:
        BacktestResult: Backtest outputs.
    """
    data = load_stock_data(ticker=ticker, period=period, start=start, end=end)
    if "Close" not in data.columns:
        raise ValueError("Downloaded data must contain Close prices.")

    prices = data["Close"].dropna()
    if prices.empty:
        raise ValueError("Close prices are empty after dropping missing values.")

    signals = signal_generator(prices)
    return run_backtest(
        prices=prices,
        signals=signals,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        trade_size=trade_size,
        max_position=max_position,
    )


def run_simple_ma_backtest(
    ticker: str,
    period: str = "6mo",
    start: str | None = None,
    end: str | None = None,
    short_window: int = 5,
    long_window: int = 20,
    initial_cash: float = 1_000_000.0,
    fee_rate: float = 0.0,
    trade_size: float = 1.0,
    max_position: float = 1.0,
) -> BacktestResult:
    """Load data and run a simple moving average backtest.

    Args:
        ticker (str): Ticker symbol to download.
        period (str, optional): Download period. Defaults to "6mo".
        start (str | None, optional): Start date string (YYYY-MM-DD). Defaults to None.
        end (str | None, optional): End date string (YYYY-MM-DD). Defaults to None.
        short_window (int, optional): Short moving average window. Defaults to 5.
        long_window (int, optional): Long moving average window. Defaults to 20.
        initial_cash (float, optional): Starting capital. Defaults to 1_000_000.0.
        fee_rate (float, optional): Fee applied per position change. Defaults to 0.0.
        trade_size (float, optional): Trade size per step (shares). Defaults to 1.0.
        max_position (float, optional): Maximum position size (shares). Defaults to 1.0.

    Returns:
        BacktestResult: Backtest outputs.
    """

    def _signals(prices: pd.Series) -> pd.Series:
        return generate_signals(prices, short_window=short_window, long_window=long_window)

    return run_strategy_backtest(
        ticker=ticker,
        signal_generator=_signals,
        period=period,
        start=start,
        end=end,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        trade_size=trade_size,
        max_position=max_position,
    )


def summarize_result(result: BacktestResult) -> pd.DataFrame:
    """Summarize backtest result into a single-row DataFrame.

    Args:
        result (BacktestResult): Backtest result instance.

    Returns:
        pd.DataFrame: Summary statistics.
    """
    summary = {
        "final_equity": float(result.equity_curve.iloc[-1]),
        "total_return": float(result.total_return),
        "max_drawdown": float(_max_drawdown(result.equity_curve)),
    }
    return pd.DataFrame([summary])


def _max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown from equity curve.

    Args:
        equity_curve (pd.Series): Equity curve.

    Returns:
        float: Maximum drawdown as a negative number.
    """
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())
