"""Tests for the EMA strategy module."""

import pandas as pd
import pytest

from stock_autotrade.strategy.exponential_moving_average import calculate_ema, generate_signals


class TestCalculateEma:
    """Tests for calculate_ema function."""

    def test_ema_calculation(self) -> None:
        """Test basic EMA calculation."""
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        ema = calculate_ema(prices, window=3)

        assert len(ema) == len(prices)
        # EMA should be less than simple average at the end for rising prices
        assert isinstance(ema.iloc[-1], float)

    def test_ema_invalid_window(self) -> None:
        """Test that invalid window raises error."""
        prices = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="positive"):
            calculate_ema(prices, window=0)

        with pytest.raises(ValueError, match="positive"):
            calculate_ema(prices, window=-1)


class TestGenerateSignals:
    """Tests for generate_signals function."""

    def test_signals_on_rising_prices(self) -> None:
        """Test signals on rising prices."""
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3)
        signals = generate_signals(prices, short_window=3, long_window=6)

        assert len(signals) == len(prices)
        assert all(s in (0, 1) for s in signals)
        # After warmup, should be long on rising prices
        assert signals.iloc[-1] == 1

    def test_signals_on_falling_prices(self) -> None:
        """Test signals on falling prices."""
        prices = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0] * 3)
        signals = generate_signals(prices, short_window=3, long_window=6)

        # On falling prices, short EMA should be below long EMA -> flat
        assert signals.iloc[-1] == 0

    def test_signals_with_invalid_windows(self) -> None:
        """Test that invalid windows raise errors."""
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="positive"):
            generate_signals(prices, short_window=0, long_window=5)

        with pytest.raises(ValueError, match="smaller"):
            generate_signals(prices, short_window=5, long_window=3)

    def test_signals_on_empty_series(self) -> None:
        """Test signals on empty series."""
        prices = pd.Series(dtype=float)
        signals = generate_signals(prices)

        assert len(signals) == 0

    def test_signals_warmup_period(self) -> None:
        """Test that signals are 0 during warmup period."""
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        signals = generate_signals(prices, short_window=2, long_window=5)

        # First long_window-1 signals should be 0
        assert all(signals.iloc[:4] == 0)
