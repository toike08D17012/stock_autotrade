"""Parameter optimization module for trading strategies."""

from .base import OptimizationResult, Optimizer, ParameterSpace
from .grid_search import GridSearchOptimizer
from .objective import (
    ObjectiveFunction,
    maximize_return,
    maximize_sharpe,
    minimize_drawdown,
)


__all__ = [
    "GridSearchOptimizer",
    "ObjectiveFunction",
    "OptimizationResult",
    "Optimizer",
    "ParameterSpace",
    "maximize_return",
    "maximize_sharpe",
    "minimize_drawdown",
]
