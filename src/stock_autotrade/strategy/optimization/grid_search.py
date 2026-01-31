"""Grid search optimizer implementation.

This module provides a grid search optimizer that evaluates all combinations
of parameter values. It supports parallel execution for improved performance.
"""

import itertools
import logging
import queue
import threading
import time
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


class _SimpleMAEvaluator:
    """Callable evaluator for Simple MA strategy that can be pickled.

    This class wraps the evaluation logic so it can be serialized
    and used with multiprocessing. It uses pre-downloaded data to avoid
    rate limiting issues with yfinance during parallel execution.
    """

    def __init__(
        self,
        ticker_data: dict[str, pd.Series],
        initial_cash: float,
        fee_rate: float,
        trade_size: float,
        max_position: float,
        objective: str,
        aggregation: str,
    ) -> None:
        """Initialize the evaluator with backtest configuration.

        Args:
            ticker_data: Dictionary mapping ticker symbols to their price series.
            initial_cash: Starting capital.
            fee_rate: Trading fee rate.
            trade_size: Trade size per step.
            max_position: Maximum position size.
            objective: Objective to optimize.
            aggregation: Aggregation method for multiple tickers.
        """
        self.ticker_data = ticker_data
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.trade_size = trade_size
        self.max_position = max_position
        self.objective = objective
        self.aggregation = aggregation

    def __call__(self, params: dict[str, Any]) -> float:
        """Evaluate a single parameter combination across all tickers.

        Args:
            params: Parameter dictionary with 'short_window' and 'long_window'.

        Returns:
            float: Aggregated objective score.
        """
        import statistics

        from stock_autotrade.backtest.engine import run_backtest
        from stock_autotrade.strategy.optimization.objective import (
            calmar_ratio,
            maximize_return,
            maximize_sharpe,
            minimize_drawdown,
        )
        from stock_autotrade.strategy.simple_moving_average import generate_signals

        short_window = params["short_window"]
        long_window = params["long_window"]

        # Skip invalid combinations
        if short_window >= long_window:
            return float("-inf")

        # Select objective function
        from stock_autotrade.strategy.optimization.objective import ObjectiveFunction

        objective_fns: dict[str, ObjectiveFunction] = {
            "return": maximize_return,
            "sharpe": maximize_sharpe,
            "drawdown": minimize_drawdown,
            "calmar": calmar_ratio,
        }
        obj_fn: ObjectiveFunction = objective_fns.get(self.objective, maximize_return)

        # Select aggregation function
        from collections.abc import Callable

        aggregation_fns: dict[str, Callable[[list[float]], float]] = {
            "mean": statistics.mean,
            "min": min,
            "median": statistics.median,
        }
        agg_fn: Callable[[list[float]], float] = aggregation_fns.get(self.aggregation, statistics.mean)

        scores: list[float] = []
        for ticker, prices in self.ticker_data.items():
            try:
                signals = generate_signals(prices, short_window=short_window, long_window=long_window)
                result = run_backtest(
                    prices=prices,
                    signals=signals,
                    initial_cash=self.initial_cash,
                    fee_rate=self.fee_rate,
                    trade_size=self.trade_size,
                    max_position=self.max_position,
                )
                scores.append(float(obj_fn(result)))
            except Exception as e:
                logger.warning("Failed to evaluate ticker %s: %s", ticker, e)
                scores.append(float("-inf"))

        # If all tickers failed, return -inf
        valid_scores = [s for s in scores if s > float("-inf")]
        if not valid_scores:
            return float("-inf")

        return float(agg_fn(valid_scores))


def grid_search_simple_ma(
    ticker: str | list[str],
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
    aggregation: str = "mean",
    download_delay: float = 1.0,
    streaming: bool = False,
) -> OptimizationResult:
    """Convenience function for grid search on Simple MA strategy.

    This function provides a simple interface for running grid search
    on the Simple Moving Average crossover strategy without needing
    to set up the optimizer manually.

    Supports multiple tickers - when multiple tickers are provided,
    the objective score for each parameter combination is aggregated
    across all tickers using the specified aggregation method.

    Data is pre-downloaded sequentially before parallel optimization
    to avoid rate limiting issues with yfinance.

    Args:
        ticker: Stock ticker symbol or list of ticker symbols.
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
        aggregation: Aggregation method for multiple tickers ('mean', 'min', 'median').
            Defaults to "mean".
        download_delay: Delay in seconds between data downloads to avoid rate limiting.
            Defaults to 1.0.
        streaming: If True, use streaming mode where grid search starts as soon as
            each ticker's data is downloaded. This is faster for large ticker lists.
            Defaults to False.

    Returns:
        OptimizationResult: Grid search results.
    """
    # Normalize ticker to list
    tickers: list[str] = [ticker] if isinstance(ticker, str) else list(ticker)

    # Use streaming mode for faster processing with large ticker lists
    if streaming and len(tickers) > 1:
        return _grid_search_simple_ma_streaming(
            tickers=tickers,
            short_windows=short_windows,
            long_windows=long_windows,
            period=period,
            start=start,
            end=end,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            trade_size=trade_size,
            max_position=max_position,
            objective=objective,
            n_jobs=n_jobs,
            verbose=verbose,
            aggregation=aggregation,
            download_delay=download_delay,
        )

    # Original batch mode
    from datetime import datetime, timedelta

    from stock_autotrade.data.loader import load_stock_data

    # Use yesterday as end date if not specified (today's data typically doesn't exist)
    effective_end = end
    if effective_end is None:
        yesterday = datetime.now() - timedelta(days=1)
        effective_end = yesterday.strftime("%Y-%m-%d")
        logger.debug("Using yesterday as end date: %s", effective_end)

    # Pre-download all ticker data sequentially to avoid rate limiting
    logger.info("Pre-downloading data for %d tickers...", len(tickers))
    ticker_data: dict[str, pd.Series] = {}
    failed_tickers: list[str] = []

    for i, t in enumerate(tickers):
        try:
            load_result = load_stock_data(
                ticker=t, period=period, start=start, end=effective_end, return_download_info=True
            )
            data, downloaded = load_result
            if "Close" in data.columns:
                prices = data["Close"].dropna()
                if not prices.empty:
                    ticker_data[t] = prices
                    if verbose >= 1:
                        cache_status = "" if downloaded else " (cached)"
                        logger.info(
                            "[%d/%d] Downloaded %s: %d data points%s",
                            i + 1,
                            len(tickers),
                            t,
                            len(prices),
                            cache_status,
                        )
                else:
                    logger.warning("[%d/%d] %s: Empty price data", i + 1, len(tickers), t)
                    failed_tickers.append(t)
            else:
                logger.warning("[%d/%d] %s: No Close column", i + 1, len(tickers), t)
                failed_tickers.append(t)

            # Delay only if we actually downloaded from yfinance
            if downloaded and i < len(tickers) - 1:
                time.sleep(download_delay)
        except Exception as e:
            logger.warning("[%d/%d] Failed to download %s: %s", i + 1, len(tickers), t, e)
            failed_tickers.append(t)

    if not ticker_data:
        raise ValueError("Failed to download data for all tickers.")

    logger.info(
        "Downloaded data for %d/%d tickers successfully (failed: %d)",
        len(ticker_data),
        len(tickers),
        len(failed_tickers),
    )

    # Create a picklable evaluator with pre-downloaded data
    evaluate = _SimpleMAEvaluator(
        ticker_data=ticker_data,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        trade_size=trade_size,
        max_position=max_position,
        objective=objective,
        aggregation=aggregation,
    )

    param_spaces = [
        ParameterSpace(name="short_window", values=list(short_windows)),
        ParameterSpace(name="long_window", values=list(long_windows)),
    ]

    optimizer = GridSearchOptimizer(n_jobs=n_jobs, verbose=verbose)
    result = optimizer.optimize(evaluate, param_spaces)

    # Add tickers and aggregation info to metadata
    result.metadata["tickers"] = list(ticker_data.keys())
    result.metadata["failed_tickers"] = failed_tickers
    result.metadata["aggregation"] = aggregation

    return result


class _SingleTickerEvaluator:
    """Callable evaluator for a single ticker that can be pickled."""

    def __init__(
        self,
        prices: pd.Series,
        ticker: str,
        initial_cash: float,
        fee_rate: float,
        trade_size: float,
        max_position: float,
        objective: str,
    ) -> None:
        """Initialize the evaluator for a single ticker.

        Args:
            prices: Price series for the ticker.
            ticker: Ticker symbol (for logging).
            initial_cash: Starting capital.
            fee_rate: Trading fee rate.
            trade_size: Trade size per step.
            max_position: Maximum position size.
            objective: Objective to optimize.
        """
        self.prices = prices
        self.ticker = ticker
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.trade_size = trade_size
        self.max_position = max_position
        self.objective = objective

    def __call__(self, params: dict[str, Any]) -> float:
        """Evaluate a single parameter combination for this ticker.

        Args:
            params: Parameter dictionary with 'short_window' and 'long_window'.

        Returns:
            float: Objective score.
        """
        from stock_autotrade.backtest.engine import run_backtest
        from stock_autotrade.strategy.optimization.objective import (
            ObjectiveFunction,
            calmar_ratio,
            maximize_return,
            maximize_sharpe,
            minimize_drawdown,
        )
        from stock_autotrade.strategy.simple_moving_average import generate_signals

        short_window = params["short_window"]
        long_window = params["long_window"]

        # Skip invalid combinations
        if short_window >= long_window:
            return float("-inf")

        objective_fns: dict[str, ObjectiveFunction] = {
            "return": maximize_return,
            "sharpe": maximize_sharpe,
            "drawdown": minimize_drawdown,
            "calmar": calmar_ratio,
        }
        obj_fn: ObjectiveFunction = objective_fns.get(self.objective, maximize_return)

        try:
            signals = generate_signals(self.prices, short_window=short_window, long_window=long_window)
            result = run_backtest(
                prices=self.prices,
                signals=signals,
                initial_cash=self.initial_cash,
                fee_rate=self.fee_rate,
                trade_size=self.trade_size,
                max_position=self.max_position,
            )
            return float(obj_fn(result))
        except Exception as e:
            logger.warning("Failed to evaluate ticker %s with params %s: %s", self.ticker, params, e)
            return float("-inf")


def _grid_search_simple_ma_streaming(
    tickers: list[str],
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
    aggregation: str = "mean",
    download_delay: float = 1.0,
) -> OptimizationResult:
    """Streaming grid search that starts optimization as data becomes available.

    This function downloads data and runs grid search in parallel. As each
    ticker's data is downloaded, the grid search for that ticker starts
    immediately, without waiting for all tickers to finish downloading.

    Args:
        tickers: List of ticker symbols.
        short_windows: List or range of short window values to test.
        long_windows: List or range of long window values to test.
        period: Data download period.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        initial_cash: Starting capital.
        fee_rate: Trading fee rate.
        trade_size: Trade size per step.
        max_position: Maximum position size.
        objective: Objective to optimize.
        n_jobs: Number of parallel workers.
        verbose: Verbosity level.
        aggregation: Aggregation method for multiple tickers.
        download_delay: Delay in seconds between data downloads.

    Returns:
        OptimizationResult: Grid search results.
    """
    import statistics
    from collections.abc import Callable
    from datetime import datetime, timedelta

    from stock_autotrade.data.loader import load_stock_data

    # Use yesterday as end date if not specified (today's data typically doesn't exist)
    effective_end = end
    if effective_end is None:
        yesterday = datetime.now() - timedelta(days=1)
        effective_end = yesterday.strftime("%Y-%m-%d")
        logger.debug("Using yesterday as end date: %s", effective_end)

    # Build parameter grid
    param_names = ["short_window", "long_window"]
    param_values = [list(short_windows), list(long_windows)]
    combinations = list(itertools.product(*param_values))
    param_grid = [dict(zip(param_names, combo, strict=True)) for combo in combinations]

    # Filter out invalid combinations (short >= long)
    valid_param_grid = [p for p in param_grid if p["short_window"] < p["long_window"]]
    total_combinations = len(valid_param_grid)

    logger.info(
        "Starting streaming grid search: %d tickers x %d valid param combinations",
        len(tickers),
        total_combinations,
    )

    # Store results per ticker: {ticker: {param_tuple: score}}
    ticker_results: dict[str, dict[tuple[int, int], float]] = {}
    ticker_results_lock = threading.Lock()

    # Track completed tickers and failures
    completed_tickers: list[str] = []
    failed_tickers: list[str] = []
    status_lock = threading.Lock()

    # Queue for tickers ready to be processed
    ticker_queue: queue.Queue[tuple[str, pd.Series] | None] = queue.Queue()

    def download_worker() -> None:
        """Worker thread that downloads ticker data and puts it in the queue."""
        for i, t in enumerate(tickers):
            downloaded = False
            try:
                result = load_stock_data(
                    ticker=t, period=period, start=start, end=effective_end, return_download_info=True
                )
                data, downloaded = result
                if "Close" in data.columns:
                    prices = data["Close"].dropna()
                    if not prices.empty:
                        ticker_queue.put((t, prices))
                        if verbose >= 1:
                            cache_status = "" if downloaded else " (cached)"
                            logger.info(
                                "[Download %d/%d] %s: %d data points%s",
                                i + 1,
                                len(tickers),
                                t,
                                len(prices),
                                cache_status,
                            )
                    else:
                        logger.warning("[Download %d/%d] %s: Empty price data", i + 1, len(tickers), t)
                        with status_lock:
                            failed_tickers.append(t)
                else:
                    logger.warning("[Download %d/%d] %s: No Close column", i + 1, len(tickers), t)
                    with status_lock:
                        failed_tickers.append(t)
            except Exception as e:
                logger.warning("[Download %d/%d] Failed to download %s: %s", i + 1, len(tickers), t, e)
                with status_lock:
                    failed_tickers.append(t)

            # Delay only if we actually downloaded from yfinance
            if downloaded and i < len(tickers) - 1:
                time.sleep(download_delay)

        # Signal that downloading is complete
        ticker_queue.put(None)

    def process_ticker(ticker: str, prices: pd.Series) -> None:
        """Process a single ticker's grid search."""
        evaluator = _SingleTickerEvaluator(
            prices=prices,
            ticker=ticker,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            trade_size=trade_size,
            max_position=max_position,
            objective=objective,
        )

        results: dict[tuple[int, int], float] = {}

        # Run grid search for this ticker
        if n_jobs == 1:
            # Sequential evaluation
            for params in valid_param_grid:
                score = evaluator(params)
                key = (params["short_window"], params["long_window"])
                results[key] = score
        else:
            # Parallel evaluation
            n_workers = n_jobs if n_jobs > 0 else None
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_params = {executor.submit(_evaluate_single, evaluator, p): p for p in valid_param_grid}
                for future in as_completed(future_to_params):
                    params = future_to_params[future]
                    try:
                        score = future.result()
                    except Exception as e:
                        logger.warning("Failed to evaluate %s with %s: %s", ticker, params, e)
                        score = float("-inf")
                    key = (params["short_window"], params["long_window"])
                    results[key] = score

        with ticker_results_lock:
            ticker_results[ticker] = results
        with status_lock:
            completed_tickers.append(ticker)

        if verbose >= 1:
            logger.info("[GridSearch] Completed %s (%d tickers done)", ticker, len(completed_tickers))

    # Start download thread
    download_thread = threading.Thread(target=download_worker, daemon=True)
    download_thread.start()

    # Process tickers as they become available
    process_threads: list[threading.Thread] = []
    while True:
        item = ticker_queue.get()
        if item is None:
            # Download complete
            break

        ticker, prices = item
        # Start processing this ticker in a new thread
        thread = threading.Thread(target=process_ticker, args=(ticker, prices), daemon=True)
        thread.start()
        process_threads.append(thread)

    # Wait for all processing threads to complete
    for thread in process_threads:
        thread.join()

    # Wait for download thread to complete
    download_thread.join()

    if not ticker_results:
        raise ValueError("Failed to process any tickers.")

    logger.info(
        "Streaming grid search complete: %d/%d tickers processed (failed: %d)",
        len(ticker_results),
        len(tickers),
        len(failed_tickers),
    )

    # Aggregate results across tickers
    aggregation_fns: dict[str, Callable[[list[float]], float]] = {
        "mean": statistics.mean,
        "min": min,
        "median": statistics.median,
    }
    agg_fn: Callable[[list[float]], float] = aggregation_fns.get(aggregation, statistics.mean)

    # Combine results: for each param combo, aggregate scores across tickers
    aggregated_results: list[dict[str, Any]] = []
    for params in valid_param_grid:
        key = (params["short_window"], params["long_window"])
        scores: list[float] = []
        for ticker in ticker_results:
            score = ticker_results[ticker].get(key, float("-inf"))
            if score > float("-inf"):
                scores.append(score)

        agg_score = agg_fn(scores) if scores else float("-inf")

        aggregated_results.append({**params, "score": agg_score})

    results_df = pd.DataFrame(aggregated_results)

    if results_df.empty or results_df["score"].max() == float("-inf"):
        raise ValueError("No valid results from streaming grid search.")

    best_idx = results_df["score"].idxmax()
    best_row = results_df.loc[best_idx]
    best_params = {"short_window": best_row["short_window"], "long_window": best_row["long_window"]}
    best_score = float(best_row["score"])

    if verbose >= 1:
        logger.info("Best score: %.4f with params: %s", best_score, best_params)

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_results=results_df,
        metadata={
            "optimizer": "GridSearchOptimizer",
            "mode": "streaming",
            "total_combinations": total_combinations,
            "n_jobs": n_jobs,
            "tickers": completed_tickers,
            "failed_tickers": failed_tickers,
            "aggregation": aggregation,
        },
    )
