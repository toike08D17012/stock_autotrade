"""Base classes and protocols for parameter optimization.

This module defines the abstract interfaces that all optimizers must implement.
This allows swapping optimization strategies (e.g., grid search, Bayesian optimization)
without changing the calling code.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# Type alias for objective functions
ObjectiveFnType = Callable[[dict[str, Any]], float]


@dataclass(frozen=True)
class ParameterSpace:
    """Definition of the parameter search space.

    Attributes:
        name: Parameter name.
        values: List of candidate values for grid search.
        low: Lower bound for continuous optimization (e.g., Bayesian).
        high: Upper bound for continuous optimization.
        step: Step size for discrete optimization within [low, high].
        param_type: Type of parameter ('int', 'float', 'categorical').
    """

    name: str
    values: list[Any] | None = None
    low: float | None = None
    high: float | None = None
    step: float | None = None
    param_type: str = "int"

    def __post_init__(self) -> None:
        """Validate parameter space definition."""
        if self.values is None and (self.low is None or self.high is None):
            raise ValueError(f"Parameter '{self.name}' must have either 'values' or 'low'/'high' bounds.")

    def get_grid_values(self) -> list[Any]:
        """Return list of values for grid search.

        Returns:
            list[Any]: List of candidate values.
        """
        if self.values is not None:
            return list(self.values)

        if self.low is None or self.high is None:
            raise ValueError(f"Cannot generate grid values for '{self.name}': bounds not set.")

        step = self.step if self.step is not None else 1
        result: list[Any] = []
        current = self.low
        while current <= self.high:
            if self.param_type == "int":
                result.append(int(current))
            else:
                result.append(current)
            current += step
        return result


@dataclass
class OptimizationResult:
    """Container for optimization results.

    Attributes:
        best_params: Dictionary of best parameter values.
        best_score: Best objective score achieved.
        all_results: DataFrame with all evaluated parameter combinations.
        metadata: Additional metadata about the optimization run.
    """

    best_params: dict[str, Any]
    best_score: float
    all_results: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def top_n(self, n: int = 10, ascending: bool = False) -> pd.DataFrame:
        """Return top N results sorted by score.

        Args:
            n: Number of results to return.
            ascending: Sort order. False for maximization, True for minimization.

        Returns:
            pd.DataFrame: Top N results.
        """
        return self.all_results.sort_values("score", ascending=ascending).head(n)


class Optimizer(ABC):
    """Abstract base class for parameter optimizers.

    Subclasses must implement the `optimize` method. This allows easy swapping
    between different optimization strategies (grid search, random search,
    Bayesian optimization, etc.).
    """

    @abstractmethod
    def optimize(
        self,
        objective_fn: ObjectiveFnType,
        param_spaces: list[ParameterSpace],
        **kwargs: Any,
    ) -> OptimizationResult:
        """Run optimization to find best parameters.

        Args:
            objective_fn: Function that takes parameters and returns a score.
            param_spaces: List of parameter space definitions.
            **kwargs: Additional optimizer-specific arguments.

        Returns:
            OptimizationResult: Optimization results.
        """
        pass
