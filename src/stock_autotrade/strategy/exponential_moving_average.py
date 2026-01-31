"""Exponential Moving Average (EMA) crossover strategy.

This strategy generates buy signals when the short-term EMA crosses
above the long-term EMA, and sell signals when it crosses below.
"""

import pandas as pd


def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average.

    Args:
        prices: Price series indexed by time.
        window: Window size for EMA calculation.

    Returns:
        pd.Series: EMA values.

    Raises:
        ValueError: If window is not positive.
    """
    if window <= 0:
        raise ValueError("Window size must be positive.")
    return prices.ewm(span=window, adjust=False).mean()


def generate_signals(
    prices: pd.Series,
    short_window: int = 12,
    long_window: int = 26,
) -> pd.Series:
    """Generate long-only signals based on EMA crossover.

    Args:
        prices: Price series indexed by time.
        short_window: Window size for short EMA. Defaults to 12.
        long_window: Window size for long EMA. Defaults to 26.

    Returns:
        pd.Series: Signal series where 1 means long and 0 means flat.

    Raises:
        ValueError: If window sizes are invalid.
    """
    if short_window <= 0 or long_window <= 0:
        raise ValueError("Window sizes must be positive integers.")
    if short_window >= long_window:
        raise ValueError("Short window must be smaller than long window.")
    if prices.empty:
        return pd.Series(dtype=int, index=prices.index)

    short_ema = calculate_ema(prices, short_window)
    long_ema = calculate_ema(prices, long_window)

    # Signal when short EMA is above long EMA
    signals = (short_ema > long_ema).astype(int)

    # Require at least long_window periods before generating signals
    signals.iloc[: long_window - 1] = 0

    return signals
