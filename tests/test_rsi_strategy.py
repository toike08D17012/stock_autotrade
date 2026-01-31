"""Tests for the RSI strategy module."""

import pandas as pd
import pytest

from stock_autotrade.strategy.rsi import calculate_rsi, generate_mean_reversion_signals, generate_signals


class TestCalculateRsi:
    """Tests for calculate_rsi function."""

    def test_rsi_range(self) -> None:
        """Test that RSI values are in valid range."""
        prices = pd.Series([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 3)
        rsi = calculate_rsi(prices, window=14)

        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert all(0 <= v <= 100 for v in valid_rsi)

    def test_rsi_on_rising_prices(self) -> None:
        """Test RSI on consistently rising prices."""
        # Create steadily rising prices
        prices = pd.Series([100.0 + i for i in range(30)])
        rsi = calculate_rsi(prices, window=14)

        # RSI should be high (near 100) for consistently rising prices
        assert rsi.iloc[-1] > 70

    def test_rsi_on_falling_prices(self) -> None:
        """Test RSI on consistently falling prices."""
        # Create steadily falling prices
        prices = pd.Series([100.0 - i * 0.5 for i in range(30)])
        rsi = calculate_rsi(prices, window=14)

        # RSI should be low (near 0) for consistently falling prices
        assert rsi.iloc[-1] < 30

    def test_rsi_invalid_window(self) -> None:
        """Test that invalid window raises error."""
        prices = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="positive"):
            calculate_rsi(prices, window=0)

    def test_rsi_empty_series(self) -> None:
        """Test RSI on empty series."""
        prices = pd.Series(dtype=float)
        rsi = calculate_rsi(prices)
        assert len(rsi) == 0


class TestGenerateSignals:
    """Tests for generate_signals function."""

    def test_signals_format(self) -> None:
        """Test that signals are 0 or 1."""
        prices = pd.Series([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 5)
        signals = generate_signals(prices)

        assert all(s in (0, 1) for s in signals)

    def test_buy_signal_on_oversold(self) -> None:
        """Test buy signal when RSI is oversold."""
        # Create falling then rising prices to trigger oversold
        falling = [100.0 - i for i in range(20)]
        rising = [80.0 + i for i in range(10)]
        prices = pd.Series(falling + rising)

        signals = generate_signals(prices, oversold=30.0, overbought=70.0)

        # Should have some buy signals after oversold condition
        assert signals.iloc[-1] == 1 or signals.sum() > 0

    def test_invalid_parameters(self) -> None:
        """Test that invalid parameters raise errors."""
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="positive"):
            generate_signals(prices, window=0)

        with pytest.raises(ValueError, match="Oversold must be less"):
            generate_signals(prices, oversold=70.0, overbought=30.0)

    def test_empty_series(self) -> None:
        """Test signals on empty series."""
        prices = pd.Series(dtype=float)
        signals = generate_signals(prices)
        assert len(signals) == 0


class TestGenerateMeanReversionSignals:
    """Tests for generate_mean_reversion_signals function."""

    def test_mean_reversion_signals_format(self) -> None:
        """Test that mean reversion signals are 0 or 1."""
        prices = pd.Series([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 5)
        signals = generate_mean_reversion_signals(prices)

        assert all(s in (0, 1) for s in signals)

    def test_invalid_parameters(self) -> None:
        """Test that invalid parameters raise errors."""
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="positive"):
            generate_mean_reversion_signals(prices, window=0)

        with pytest.raises(ValueError, match="Oversold must be less"):
            generate_mean_reversion_signals(prices, oversold=80.0, overbought=20.0)

    def test_empty_series(self) -> None:
        """Test signals on empty series."""
        prices = pd.Series(dtype=float)
        signals = generate_mean_reversion_signals(prices)
        assert len(signals) == 0
