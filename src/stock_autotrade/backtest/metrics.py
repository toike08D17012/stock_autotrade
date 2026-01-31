"""Performance metrics for evaluating trading strategy results.

This module provides functions to calculate various performance metrics
commonly used in quantitative finance and portfolio analysis.
"""

import numpy as np
import pandas as pd


def calculate_total_return(equity_curve: pd.Series) -> float:
    """Calculate total return from equity curve.

    Args:
        equity_curve: Series of portfolio values over time.

    Returns:
        Total return as a decimal (e.g., 0.10 for 10% return).
    """
    if equity_curve.empty:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)


def calculate_annualized_return(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    """Calculate annualized return from a series of returns.

    Args:
        returns: Series of periodic returns.
        periods_per_year: Number of trading periods per year. Defaults to 252 (trading days).

    Returns:
        Annualized return as a decimal.
    """
    if returns.empty:
        return 0.0
    cumulative_return: float = (1 + returns).prod()  # type: ignore[assignment]
    total_return = cumulative_return - 1
    n_periods = len(returns)
    annualized = (1 + total_return) ** (periods_per_year / n_periods) - 1
    return float(annualized)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown from equity curve.

    The maximum drawdown is the largest peak-to-trough decline in the
    portfolio value.

    Args:
        equity_curve: Series of portfolio values over time.

    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.20 for 20% drawdown).
    """
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Calculate annualized Sharpe ratio.

    The Sharpe ratio measures risk-adjusted return, calculated as the
    excess return over the risk-free rate divided by the standard deviation.

    Args:
        returns: Series of periodic returns.
        risk_free_rate: Annual risk-free rate. Defaults to 0.0.
        periods_per_year: Number of trading periods per year. Defaults to 252.

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if standard deviation is zero.
    """
    if returns.empty:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    mean_excess = float(excess_returns.mean())
    std_return = float(returns.std())

    if std_return == 0:
        return 0.0

    sharpe = (mean_excess / std_return) * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Calculate annualized Sortino ratio.

    The Sortino ratio is similar to the Sharpe ratio but only considers
    downside volatility, making it more appropriate for asymmetric return
    distributions.

    Args:
        returns: Series of periodic returns.
        risk_free_rate: Annual risk-free rate. Defaults to 0.0.
        periods_per_year: Number of trading periods per year. Defaults to 252.

    Returns:
        Annualized Sortino ratio. Returns 0.0 if downside deviation is zero.
    """
    if returns.empty:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    mean_excess = float(excess_returns.mean())

    # Calculate downside deviation (std of negative returns only)
    downside_returns = returns[returns < 0]
    if downside_returns.empty:
        return float("inf") if mean_excess > 0 else 0.0

    downside_std = float(downside_returns.std())
    if downside_std == 0:
        return 0.0

    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)
    return float(sortino)


def calculate_calmar_ratio(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: float = 252.0,
) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown).

    The Calmar ratio measures return relative to the worst drawdown,
    providing insight into the risk of large losses.

    Args:
        returns: Series of periodic returns.
        equity_curve: Series of portfolio values over time.
        periods_per_year: Number of trading periods per year. Defaults to 252.

    Returns:
        Calmar ratio. Returns 0.0 if max drawdown is zero.
    """
    ann_return = calculate_annualized_return(returns, periods_per_year)
    max_dd = abs(calculate_max_drawdown(equity_curve))

    if max_dd == 0:
        return 0.0 if ann_return <= 0 else float("inf")

    return ann_return / max_dd


def calculate_win_rate(positions: pd.Series, returns: pd.Series) -> float:
    """Calculate win rate (percentage of profitable trades).

    Args:
        positions: Series of position sizes over time.
        returns: Series of periodic returns.

    Returns:
        Win rate as a decimal (e.g., 0.55 for 55% win rate).
    """
    # Identify trades (when position changes)
    position_changes = positions.diff().fillna(0)
    trade_entries = position_changes != 0

    if not trade_entries.any():
        return 0.0

    # Calculate returns at trade entry points
    trade_returns = returns[trade_entries]
    if trade_returns.empty:
        return 0.0

    winning_trades = (trade_returns > 0).sum()
    total_trades = len(trade_returns)

    return float(winning_trades / total_trades) if total_trades > 0 else 0.0


def calculate_profit_factor(positions: pd.Series, returns: pd.Series) -> float:
    """Calculate profit factor (gross profit / gross loss).

    A profit factor greater than 1 indicates a profitable strategy.

    Args:
        positions: Series of position sizes over time.
        returns: Series of periodic returns.

    Returns:
        Profit factor. Returns infinity if there are no losing trades.
    """
    # Filter returns when we have a position
    active_returns = returns[positions != 0]

    if active_returns.empty:
        return 0.0

    gross_profit = float(active_returns[active_returns > 0].sum())
    gross_loss = abs(float(active_returns[active_returns < 0].sum()))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_volatility(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    """Calculate annualized volatility (standard deviation of returns).

    Args:
        returns: Series of periodic returns.
        periods_per_year: Number of trading periods per year. Defaults to 252.

    Returns:
        Annualized volatility as a decimal.
    """
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(periods_per_year))


def count_trades(positions: pd.Series) -> int:
    """Count the number of trades (position changes).

    Args:
        positions: Series of position sizes over time.

    Returns:
        Number of trades executed.
    """
    return int((positions.diff().abs() > 0).sum())


def calculate_all_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    positions: pd.Series,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Calculate all available performance metrics.

    Args:
        equity_curve: Series of portfolio values over time.
        returns: Series of periodic returns.
        positions: Series of position sizes over time.
        periods_per_year: Number of trading periods per year. Defaults to 252.
        risk_free_rate: Annual risk-free rate. Defaults to 0.0.

    Returns:
        Dictionary containing all calculated metrics.
    """
    return {
        "total_return": calculate_total_return(equity_curve),
        "annualized_return": calculate_annualized_return(returns, periods_per_year),
        "max_drawdown": calculate_max_drawdown(equity_curve),
        "sharpe_ratio": calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        "calmar_ratio": calculate_calmar_ratio(returns, equity_curve, periods_per_year),
        "volatility": calculate_volatility(returns, periods_per_year),
        "win_rate": calculate_win_rate(positions, returns),
        "profit_factor": calculate_profit_factor(positions, returns),
        "num_trades": float(count_trades(positions)),
        "final_equity": float(equity_curve.iloc[-1]) if not equity_curve.empty else 0.0,
    }
