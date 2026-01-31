import pandas as pd
import pytest  # type: ignore[import-not-found]

from stock_autotrade.backtest.engine import run_backtest
from stock_autotrade.strategy.simple_moving_average import generate_signals


def test_generate_signals_turns_long_after_long_window() -> None:
    """Signals should turn long once both moving averages are available on rising prices."""
    prices = pd.Series([1, 2, 3, 4, 5, 6])
    signals = generate_signals(prices, short_window=2, long_window=3)

    assert (signals.iloc[:2] == 0).all()
    assert (signals.iloc[2:] == 1).all()


def test_run_backtest_long_only_grows_equity() -> None:
    """Equity should grow with rising prices when long one share."""
    prices = pd.Series([100.0, 101.0, 102.0, 103.0])
    signals = pd.Series([1, 1, 1, 1])

    result = run_backtest(prices=prices, signals=signals, initial_cash=1000.0, trade_size=1.0, max_position=1.0)

    assert result.equity_curve.iloc[-1] == pytest.approx(1002.0, rel=1e-6)


def test_run_backtest_applies_trade_size_and_max_position() -> None:
    """Positions should ramp up by trade size until max position is reached."""
    prices = pd.Series([100.0, 100.0, 100.0, 100.0])
    signals = pd.Series([1, 1, 1, 1])

    result = run_backtest(prices=prices, signals=signals, trade_size=2.0, max_position=4.0)

    expected_positions = pd.Series([0.0, 2.0, 4.0, 4.0])
    pd.testing.assert_series_equal(result.positions.reset_index(drop=True), expected_positions)
    assert result.equity_curve.iloc[-1] == pytest.approx(1_000_000.0, rel=1e-6)
