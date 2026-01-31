"""Objective functions for strategy optimization.

This module provides reusable objective functions that evaluate backtest results
and return a single score for optimization. Each function can be used with any
optimizer implementation.
"""

from collections.abc import Callable

import numpy as np

from stock_autotrade.backtest.engine import BacktestResult


# Type alias for objective functions
ObjectiveFunction = Callable[[BacktestResult], float]


def maximize_return(result: BacktestResult) -> float:
    """Objective function that maximizes total return.

    Args:
        result: Backtest result to evaluate.

    Returns:
        float: Total return (higher is better).
    """
    return result.total_return


def minimize_drawdown(result: BacktestResult) -> float:
    """Objective function that minimizes maximum drawdown.

    Note: Returns negative of max drawdown so that maximizing this score
    corresponds to minimizing drawdown.

    Args:
        result: Backtest result to evaluate.

    Returns:
        float: Negative of maximum drawdown (higher is better, i.e., less drawdown).
    """
    running_max = result.equity_curve.cummax()
    drawdown = result.equity_curve / running_max - 1.0
    max_dd = float(drawdown.min())
    # Return negative so that "higher is better" semantics are preserved
    return -max_dd


def maximize_sharpe(result: BacktestResult, risk_free_rate: float = 0.0, annualization_factor: float = 252.0) -> float:
    """Objective function that maximizes Sharpe ratio.

    Args:
        result: Backtest result to evaluate.
        risk_free_rate: Annual risk-free rate. Defaults to 0.0.
        annualization_factor: Number of trading days per year. Defaults to 252.0.

    Returns:
        float: Annualized Sharpe ratio (higher is better).
    """
    returns = result.strategy_returns
    if returns.empty or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / annualization_factor
    mean_return = float(excess_returns.mean())
    std_return = float(returns.std())

    if std_return == 0:
        return 0.0

    sharpe = (mean_return / std_return) * np.sqrt(annualization_factor)
    return float(sharpe)


def calmar_ratio(result: BacktestResult, annualization_factor: float = 252.0) -> float:
    """Objective function that maximizes Calmar ratio (return / max drawdown).

    Args:
        result: Backtest result to evaluate.
        annualization_factor: Number of trading days per year. Defaults to 252.0.

    Returns:
        float: Calmar ratio (higher is better).
    """
    returns = result.strategy_returns
    if returns.empty:
        return 0.0

    annualized_return = float(returns.mean()) * annualization_factor
    max_dd = abs(minimize_drawdown(result))  # Get positive drawdown value

    if max_dd == 0:
        return 0.0 if annualized_return <= 0 else float("inf")

    return annualized_return / max_dd


def create_combined_objective(
    weights: dict[str, float] | None = None,
) -> ObjectiveFunction:
    """Create a combined objective function with weighted metrics.

    Args:
        weights: Dictionary mapping metric names to weights.
            Supported metrics: 'return', 'sharpe', 'drawdown', 'calmar'.
            Defaults to equal weights for return and sharpe.

    Returns:
        ObjectiveFunction: Combined objective function.
    """
    if weights is None:
        weights = {"return": 0.5, "sharpe": 0.5}

    metric_fns: dict[str, ObjectiveFunction] = {
        "return": maximize_return,
        "sharpe": maximize_sharpe,
        "drawdown": minimize_drawdown,
        "calmar": calmar_ratio,
    }

    def combined(result: BacktestResult) -> float:
        score = 0.0
        for metric_name, weight in weights.items():
            if metric_name in metric_fns:
                score += weight * metric_fns[metric_name](result)
        return score

    return combined


def compute_all_metrics(result: BacktestResult) -> dict[str, float]:
    """Compute all available metrics for a backtest result.

    Args:
        result: Backtest result to evaluate.

    Returns:
        dict[str, float]: Dictionary of metric names to values.
    """
    return {
        "total_return": maximize_return(result),
        "sharpe_ratio": maximize_sharpe(result),
        "max_drawdown": -minimize_drawdown(result),  # Return as negative value
        "calmar_ratio": calmar_ratio(result),
        "final_equity": float(result.equity_curve.iloc[-1]),
        "num_trades": int((result.positions.diff().abs() > 0).sum()),
    }
