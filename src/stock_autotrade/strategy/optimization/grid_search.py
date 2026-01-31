"""Grid search optimizer implementation.

This module provides a grid search optimizer that evaluates all combinations
of parameter values. It supports parallel execution for improved performance.
"""

import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import pandas as pd

from .base import ObjectiveFnType, OptimizationResult, Optimizer, ParameterSpace


logger = logging.getLogger(__name__)


class GridSearchOptimizer(Optimizer):
    """Grid search optimizer that evaluates all parameter combinations.

    This optimizer performs an exhaustive search over a specified parameter grid.
    It supports parallel execution using multiprocessing.

    Attributes:
        n_jobs: Number of parallel workers. -1 uses all available CPUs.
        verbose: Verbosity level for logging.
    """

    def __init__(self, n_jobs: int = 1, verbose: int = 0) -> None:
        """Initialize the grid search optimizer.

        Args:
            n_jobs: Number of parallel workers. 1 for sequential, -1 for all CPUs.
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
        """
        self.n_jobs = n_jobs
        self.verbose = verbose

    def optimize(
        self,
        objective_fn: ObjectiveFnType,
        param_spaces: list[ParameterSpace],
        **kwargs: Any,
    ) -> OptimizationResult:
        """Run grid search optimization.

        Args:
            objective_fn: Function that takes a parameter dict and returns a score.
            param_spaces: List of parameter space definitions.
            **kwargs: Additional arguments (unused in grid search).

        Returns:
            OptimizationResult: Optimization results with all evaluated combinations.
        """
        param_grid = self._build_param_grid(param_spaces)
        total_combinations = len(param_grid)

        if self.verbose >= 1:
            logger.info("Starting grid search with %d combinations", total_combinations)

        results = self._evaluate_grid(objective_fn, param_grid)
        results_df = pd.DataFrame(results)

        if results_df.empty:
            raise ValueError("No valid results from grid search.")

        best_idx = results_df["score"].idxmax()
        best_row = results_df.loc[best_idx]
        best_params = {ps.name: best_row[ps.name] for ps in param_spaces}
        best_score = float(best_row["score"])

        if self.verbose >= 1:
            logger.info("Best score: %.4f with params: %s", best_score, best_params)

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results_df,
            metadata={
                "optimizer": "GridSearchOptimizer",
                "total_combinations": total_combinations,
                "n_jobs": self.n_jobs,
            },
        )

    def _build_param_grid(self, param_spaces: list[ParameterSpace]) -> list[dict[str, Any]]:
        """Build list of all parameter combinations.

        Args:
            param_spaces: List of parameter space definitions.

        Returns:
            list[dict[str, Any]]: List of parameter dictionaries.
        """
        param_names = [ps.name for ps in param_spaces]
        param_values = [ps.get_grid_values() for ps in param_spaces]

        combinations = list(itertools.product(*param_values))
        return [dict(zip(param_names, combo, strict=True)) for combo in combinations]

    def _evaluate_grid(
        self,
        objective_fn: ObjectiveFnType,
        param_grid: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Evaluate objective function for all parameter combinations.

        Args:
            objective_fn: Objective function to evaluate.
            param_grid: List of parameter dictionaries.

        Returns:
            list[dict[str, Any]]: List of results with params and scores.
        """
        if self.n_jobs == 1:
            return self._evaluate_sequential(objective_fn, param_grid)
        else:
            return self._evaluate_parallel(objective_fn, param_grid)

    def _evaluate_sequential(
        self,
        objective_fn: ObjectiveFnType,
        param_grid: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Evaluate grid sequentially.

        Args:
            objective_fn: Objective function to evaluate.
            param_grid: List of parameter dictionaries.

        Returns:
            list[dict[str, Any]]: List of results.
        """
        results: list[dict[str, Any]] = []
        total = len(param_grid)

        for i, params in enumerate(param_grid):
            try:
                score = objective_fn(params)
                result = {**params, "score": score}
                results.append(result)

                if self.verbose >= 2:
                    logger.info("[%d/%d] params=%s score=%.4f", i + 1, total, params, score)
                elif self.verbose >= 1 and (i + 1) % max(1, total // 10) == 0:
                    logger.info("Progress: %d/%d (%.1f%%)", i + 1, total, 100 * (i + 1) / total)

            except Exception as e:
                logger.warning("Failed to evaluate params %s: %s", params, e)
                results.append({**params, "score": float("-inf")})

        return results

    def _evaluate_parallel(
        self,
        objective_fn: ObjectiveFnType,
        param_grid: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Evaluate grid in parallel using multiprocessing.

        Args:
            objective_fn: Objective function to evaluate.
            param_grid: List of parameter dictionaries.

        Returns:
            list[dict[str, Any]]: List of results.
        """
        n_workers = self.n_jobs if self.n_jobs > 0 else None
        results: list[dict[str, Any]] = []
        total = len(param_grid)
        completed = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_params = {executor.submit(_evaluate_single, objective_fn, p): p for p in param_grid}

            for future in as_completed(future_to_params):
                params = future_to_params[future]
                completed += 1

                try:
                    score = future.result()
                    results.append({**params, "score": score})

                    if self.verbose >= 2:
                        logger.info("[%d/%d] params=%s score=%.4f", completed, total, params, score)
                    elif self.verbose >= 1 and completed % max(1, total // 10) == 0:
                        logger.info("Progress: %d/%d (%.1f%%)", completed, total, 100 * completed / total)

                except Exception as e:
                    logger.warning("Failed to evaluate params %s: %s", params, e)
                    results.append({**params, "score": float("-inf")})

        return results


def _evaluate_single(objective_fn: ObjectiveFnType, params: dict[str, Any]) -> float:
    """Evaluate a single parameter combination (for parallel execution).

    Args:
        objective_fn: Objective function to evaluate.
        params: Parameter dictionary.

    Returns:
        float: Objective score.
    """
    return objective_fn(params)


def grid_search_simple_ma(
    ticker: str,
    short_windows: list[int] | range,
    long_windows: list[int] | range,
    period: str = "1y",
    start: str | None = None,
    end: str | None = None,
    initial_cash: float = 1_000_000.0,
    fee_rate: float = 0.0,
    trade_size: float = 100.0,
    max_position: float = 1.0,
    objective: str = "return",
    n_jobs: int = 1,
    verbose: int = 0,
) -> OptimizationResult:
    """Convenience function for grid search on Simple MA strategy.

    This function provides a simple interface for running grid search
    on the Simple Moving Average crossover strategy without needing
    to set up the optimizer manually.

    Args:
        ticker: Stock ticker symbol.
        short_windows: List or range of short window values to test.
        long_windows: List or range of long window values to test.
        period: Data download period. Defaults to "1y".
        start: Start date (YYYY-MM-DD). Defaults to None.
        end: End date (YYYY-MM-DD). Defaults to None.
        initial_cash: Starting capital. Defaults to 1_000_000.0.
        fee_rate: Trading fee rate. Defaults to 0.0.
        trade_size: Trade size per step. Defaults to 100.0.
        max_position: Maximum position size. Defaults to 1.0.
        objective: Objective to optimize ('return', 'sharpe', 'drawdown', 'calmar').
        n_jobs: Number of parallel workers. Defaults to 1.
        verbose: Verbosity level. Defaults to 0.

    Returns:
        OptimizationResult: Grid search results.
    """
    from stock_autotrade.backtest.engine import BacktestResult
    from stock_autotrade.backtest.runner import run_simple_ma_backtest
    from stock_autotrade.strategy.optimization.objective import (
        ObjectiveFunction,
        calmar_ratio,
        maximize_return,
        maximize_sharpe,
        minimize_drawdown,
    )

    # Select objective function
    objective_fns: dict[str, ObjectiveFunction] = {
        "return": maximize_return,
        "sharpe": maximize_sharpe,
        "drawdown": minimize_drawdown,
        "calmar": calmar_ratio,
    }
    obj_fn: ObjectiveFunction = objective_fns.get(objective, maximize_return)

    def evaluate(params: dict[str, Any]) -> float:
        """Evaluate a single parameter combination."""
        short_window = params["short_window"]
        long_window = params["long_window"]

        # Skip invalid combinations
        if short_window >= long_window:
            return float("-inf")

        result: BacktestResult = run_simple_ma_backtest(
            ticker=ticker,
            period=period,
            start=start,
            end=end,
            short_window=short_window,
            long_window=long_window,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            trade_size=trade_size,
            max_position=max_position,
        )
        return float(obj_fn(result))

    param_spaces = [
        ParameterSpace(name="short_window", values=list(short_windows)),
        ParameterSpace(name="long_window", values=list(long_windows)),
    ]

    optimizer = GridSearchOptimizer(n_jobs=n_jobs, verbose=verbose)
    return optimizer.optimize(evaluate, param_spaces)
