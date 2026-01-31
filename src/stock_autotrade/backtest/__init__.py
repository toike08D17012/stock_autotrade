"""Backtesting components for evaluating trading strategies.

This module provides:
- BacktestResult: Container for backtest outputs
- run_backtest: Core backtesting engine
- run_strategy_backtest: High-level backtest runner
- run_simple_ma_backtest: Convenience function for SMA strategy
- summarize_result: Generate backtest summary statistics
- Performance metrics calculation functions

Example:
    >>> from backtest import run_backtest, BacktestResult
    >>> result = run_backtest(prices, signals)
    >>> print(result.total_return)
"""

from stock_autotrade.backtest.engine import BacktestResult, run_backtest
from stock_autotrade.backtest.metrics import (
    calculate_all_metrics,
    calculate_annualized_return,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_return,
    calculate_volatility,
    calculate_win_rate,
    count_trades,
)
from stock_autotrade.backtest.runner import run_simple_ma_backtest, run_strategy_backtest, summarize_result


__all__ = [
    "BacktestResult",
    "calculate_all_metrics",
    "calculate_annualized_return",
    "calculate_calmar_ratio",
    "calculate_max_drawdown",
    "calculate_profit_factor",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_total_return",
    "calculate_volatility",
    "calculate_win_rate",
    "count_trades",
    "run_backtest",
    "run_simple_ma_backtest",
    "run_strategy_backtest",
    "summarize_result",
]
