"""Simple moving average crossover strategy."""

import pandas as pd


def generate_signals(prices: pd.Series, short_window: int = 5, long_window: int = 20) -> pd.Series:
    """Generate long-only signals based on moving average crossover.

    Args:
        prices (pd.Series): Price series indexed by time.
        short_window (int, optional): Window size for short moving average. Defaults to 5.
        long_window (int, optional): Window size for long moving average. Defaults to 20.

    Returns:
        pd.Series: Signal series where 1 means long and 0 means flat.
    """
    if short_window <= 0 or long_window <= 0:
        raise ValueError("Window sizes must be positive integers.")
    if short_window >= long_window:
        raise ValueError("Short window must be smaller than long window.")
    if prices.empty:
        return pd.Series(dtype=int, index=prices.index)

    short_ma = prices.rolling(window=short_window, min_periods=short_window).mean()
    long_ma = prices.rolling(window=long_window, min_periods=long_window).mean()
    signals = (short_ma > long_ma).astype(int)
    return signals.fillna(0).astype(int)
