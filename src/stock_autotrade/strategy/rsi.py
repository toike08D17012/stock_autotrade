"""Relative Strength Index (RSI) based trading strategy.

This strategy generates signals based on RSI overbought/oversold levels.
Buy when RSI is below oversold threshold, sell when above overbought threshold.
"""

import pandas as pd


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss

    Args:
        prices: Price series indexed by time.
        window: Lookback window for RSI calculation. Defaults to 14.

    Returns:
        pd.Series: RSI values (0-100 scale).

    Raises:
        ValueError: If window is not positive.
    """
    if window <= 0:
        raise ValueError("Window size must be positive.")
    if prices.empty:
        return pd.Series(dtype=float, index=prices.index)

    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    # Calculate average gains and losses using exponential moving average
    avg_gains = gains.ewm(com=window - 1, min_periods=window).mean()
    avg_losses = losses.ewm(com=window - 1, min_periods=window).mean()

    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    # Handle division by zero (when avg_losses is 0)
    rsi = rsi.fillna(50.0)

    return rsi


def generate_signals(
    prices: pd.Series,
    window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.Series:
    """Generate signals based on RSI levels.

    Buy signal (1) when RSI crosses below oversold level.
    Sell signal (0) when RSI crosses above overbought level.
    Hold previous signal otherwise.

    Args:
        prices: Price series indexed by time.
        window: RSI calculation window. Defaults to 14.
        oversold: RSI level below which to generate buy signal. Defaults to 30.
        overbought: RSI level above which to generate sell signal. Defaults to 70.

    Returns:
        pd.Series: Signal series where 1 means long and 0 means flat.

    Raises:
        ValueError: If parameters are invalid.
    """
    if window <= 0:
        raise ValueError("Window must be positive.")
    if not 0 <= oversold < overbought <= 100:
        raise ValueError("Oversold must be less than overbought, both between 0 and 100.")
    if prices.empty:
        return pd.Series(dtype=int, index=prices.index)

    rsi = calculate_rsi(prices, window)

    # Generate signals based on RSI levels
    signals = pd.Series(index=prices.index, dtype=int)
    signals.iloc[0] = 0  # Start flat

    position = 0
    for i in range(1, len(rsi)):
        if pd.isna(rsi.iloc[i]):
            signals.iloc[i] = position
        elif rsi.iloc[i] < oversold:
            position = 1  # Buy signal
            signals.iloc[i] = position
        elif rsi.iloc[i] > overbought:
            position = 0  # Sell signal
            signals.iloc[i] = position
        else:
            signals.iloc[i] = position  # Hold

    return signals


def generate_mean_reversion_signals(
    prices: pd.Series,
    window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.Series:
    """Generate mean reversion signals based on RSI.

    This is an alternative interpretation where:
    - Buy when RSI is oversold (expecting price to rise)
    - Sell when RSI is overbought (expecting price to fall)

    Unlike the standard signals, this exits when RSI returns to neutral.

    Args:
        prices: Price series indexed by time.
        window: RSI calculation window. Defaults to 14.
        oversold: RSI level for buy signal. Defaults to 30.
        overbought: RSI level for sell signal. Defaults to 70.

    Returns:
        pd.Series: Signal series where 1 means long and 0 means flat.
    """
    if window <= 0:
        raise ValueError("Window must be positive.")
    if not 0 <= oversold < overbought <= 100:
        raise ValueError("Oversold must be less than overbought, both between 0 and 100.")
    if prices.empty:
        return pd.Series(dtype=int, index=prices.index)

    rsi = calculate_rsi(prices, window)
    midpoint = (oversold + overbought) / 2

    signals = pd.Series(index=prices.index, dtype=int)
    signals.iloc[0] = 0

    position = 0
    for i in range(1, len(rsi)):
        if pd.isna(rsi.iloc[i]):
            signals.iloc[i] = position
        elif rsi.iloc[i] < oversold and position == 0:
            position = 1  # Enter long
            signals.iloc[i] = position
        elif position == 1 and rsi.iloc[i] > midpoint:
            position = 0  # Exit when RSI recovers
            signals.iloc[i] = position
        else:
            signals.iloc[i] = position

    return signals
