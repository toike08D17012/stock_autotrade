"""Trading strategies module.

This module provides various trading strategy implementations:
- Simple Moving Average (SMA) crossover strategy
- Exponential Moving Average (EMA) crossover strategy
- Relative Strength Index (RSI) strategy
- Parameter optimization tools

Available strategies:
    simple_moving_average: SMA crossover signal generator
    exponential_moving_average: EMA crossover signal generator
    rsi: RSI-based signal generator

Submodules:
    optimization: Parameter optimization for strategies
"""

from stock_autotrade.strategy import exponential_moving_average, rsi, simple_moving_average
from stock_autotrade.strategy.simple_moving_average import generate_signals


__all__ = [
    "exponential_moving_average",
    "generate_signals",
    "rsi",
    "simple_moving_average",
]
