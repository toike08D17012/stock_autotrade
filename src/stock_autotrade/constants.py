"""Constants and default configuration values for the stock autotrade framework.

This module centralizes all constants and default values used throughout
the application to ensure consistency and ease of maintenance.
"""

# Trading days per year (used for annualization)
TRADING_DAYS_PER_YEAR: float = 252.0

# Default backtest parameters
DEFAULT_INITIAL_CASH: float = 1_000_000.0
DEFAULT_FEE_RATE: float = 0.001  # 0.1% per trade
DEFAULT_TRADE_SIZE: float = 100.0
DEFAULT_MAX_POSITION: float = 1.0

# Default strategy parameters
DEFAULT_SMA_SHORT_WINDOW: int = 5
DEFAULT_SMA_LONG_WINDOW: int = 20
DEFAULT_EMA_SHORT_WINDOW: int = 12
DEFAULT_EMA_LONG_WINDOW: int = 26
DEFAULT_RSI_WINDOW: int = 14
DEFAULT_RSI_OVERSOLD: float = 30.0
DEFAULT_RSI_OVERBOUGHT: float = 70.0

# Data loading defaults
DEFAULT_DATA_PERIOD: str = "1y"

# Optimization defaults
DEFAULT_N_JOBS: int = 1
DEFAULT_VERBOSE: int = 0

# Grid search parameter ranges
GRID_SHORT_WINDOW_MIN: int = 3
GRID_SHORT_WINDOW_MAX: int = 20
GRID_SHORT_WINDOW_STEP: int = 1
GRID_LONG_WINDOW_MIN: int = 10
GRID_LONG_WINDOW_MAX: int = 60
GRID_LONG_WINDOW_STEP: int = 5

# Performance metric thresholds
MIN_SHARPE_RATIO: float = 0.5
MAX_DRAWDOWN_THRESHOLD: float = -0.20  # -20%
MIN_WIN_RATE: float = 0.45

# Logging
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
