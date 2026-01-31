"""Tests for parameter optimization module."""

import pandas as pd
import pytest

from stock_autotrade.backtest.engine import BacktestResult
from stock_autotrade.strategy.optimization import (
    GridSearchOptimizer,
    OptimizationResult,
    ParameterSpace,
)
from stock_autotrade.strategy.optimization.objective import (
    calmar_ratio,
    compute_all_metrics,
    create_combined_objective,
    maximize_return,
    maximize_sharpe,
    minimize_drawdown,
)


# Fixtures


@pytest.fixture
def sample_equity_curve() -> pd.Series:
    """Create a sample equity curve for testing."""
    return pd.Series([100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 110.0])


@pytest.fixture
def sample_backtest_result(sample_equity_curve: pd.Series) -> BacktestResult:
    """Create a sample backtest result for testing."""
    equity = sample_equity_curve
    returns = equity.pct_change().fillna(0.0)
    positions = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    cash = pd.Series([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 110.0])
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    return BacktestResult(
        equity_curve=equity,
        strategy_returns=returns,
        positions=positions,
        cash=cash,
        total_return=total_return,
    )


# ParameterSpace Tests


class TestParameterSpace:
    """Tests for ParameterSpace class."""

    def test_create_with_values(self) -> None:
        """Test creating parameter space with explicit values."""
        ps = ParameterSpace(name="short_window", values=[5, 10, 15, 20])
        assert ps.name == "short_window"
        assert ps.get_grid_values() == [5, 10, 15, 20]

    def test_create_with_bounds(self) -> None:
        """Test creating parameter space with low/high bounds."""
        ps = ParameterSpace(name="window", low=5, high=15, step=5, param_type="int")
        assert ps.get_grid_values() == [5, 10, 15]

    def test_create_with_bounds_float(self) -> None:
        """Test creating parameter space with float bounds."""
        ps = ParameterSpace(name="threshold", low=0.0, high=0.2, step=0.1, param_type="float")
        values = ps.get_grid_values()
        assert len(values) == 3
        assert abs(values[0] - 0.0) < 1e-9
        assert abs(values[1] - 0.1) < 1e-9
        assert abs(values[2] - 0.2) < 1e-9

    def test_invalid_parameter_space(self) -> None:
        """Test that invalid parameter space raises error."""
        with pytest.raises(ValueError, match="must have either"):
            ParameterSpace(name="invalid")


# OptimizationResult Tests


class TestOptimizationResult:
    """Tests for OptimizationResult class."""

    def test_top_n(self) -> None:
        """Test top_n method returns correct results."""
        df = pd.DataFrame(
            {
                "param": [1, 2, 3, 4, 5],
                "score": [0.1, 0.5, 0.3, 0.8, 0.2],
            }
        )
        result = OptimizationResult(
            best_params={"param": 4},
            best_score=0.8,
            all_results=df,
        )

        top3 = result.top_n(3, ascending=False)
        assert len(top3) == 3
        assert top3.iloc[0]["score"] == 0.8
        assert top3.iloc[1]["score"] == 0.5
        assert top3.iloc[2]["score"] == 0.3


# Objective Function Tests


class TestObjectiveFunctions:
    """Tests for objective functions."""

    def test_maximize_return(self, sample_backtest_result: BacktestResult) -> None:
        """Test maximize_return returns total return."""
        score = maximize_return(sample_backtest_result)
        expected = 0.1  # 110/100 - 1
        assert abs(score - expected) < 1e-9

    def test_minimize_drawdown(self, sample_backtest_result: BacktestResult) -> None:
        """Test minimize_drawdown returns negative of max drawdown."""
        score = minimize_drawdown(sample_backtest_result)
        # Drawdown: 105 -> 103 = -1.9%
        assert score > 0  # Negative of negative drawdown is positive

    def test_maximize_sharpe(self, sample_backtest_result: BacktestResult) -> None:
        """Test maximize_sharpe computes Sharpe ratio."""
        score = maximize_sharpe(sample_backtest_result)
        assert isinstance(score, float)

    def test_maximize_sharpe_zero_std(self) -> None:
        """Test maximize_sharpe returns 0 when std is 0."""
        equity = pd.Series([100.0, 100.0, 100.0])
        result = BacktestResult(
            equity_curve=equity,
            strategy_returns=pd.Series([0.0, 0.0, 0.0]),
            positions=pd.Series([0.0, 0.0, 0.0]),
            cash=equity,
            total_return=0.0,
        )
        score = maximize_sharpe(result)
        assert score == 0.0

    def test_calmar_ratio(self, sample_backtest_result: BacktestResult) -> None:
        """Test calmar_ratio computes correctly."""
        score = calmar_ratio(sample_backtest_result)
        assert isinstance(score, float)

    def test_compute_all_metrics(self, sample_backtest_result: BacktestResult) -> None:
        """Test compute_all_metrics returns all expected metrics."""
        metrics = compute_all_metrics(sample_backtest_result)
        expected_keys = {"total_return", "sharpe_ratio", "max_drawdown", "calmar_ratio", "final_equity", "num_trades"}
        assert set(metrics.keys()) == expected_keys

    def test_create_combined_objective(self, sample_backtest_result: BacktestResult) -> None:
        """Test create_combined_objective creates working function."""
        combined = create_combined_objective({"return": 0.7, "sharpe": 0.3})
        score = combined(sample_backtest_result)
        assert isinstance(score, float)


# GridSearchOptimizer Tests


class TestGridSearchOptimizer:
    """Tests for GridSearchOptimizer class."""

    def test_simple_grid_search(self) -> None:
        """Test basic grid search functionality."""

        def objective(params: dict) -> float:
            x = int(params["x"])
            y = int(params["y"])
            # Simple quadratic with maximum at x=5, y=10
            return float(-((x - 5) ** 2) - (y - 10) ** 2)

        param_spaces = [
            ParameterSpace(name="x", values=[1, 3, 5, 7, 9]),
            ParameterSpace(name="y", values=[5, 10, 15]),
        ]

        optimizer = GridSearchOptimizer(n_jobs=1)
        result = optimizer.optimize(objective, param_spaces)

        assert result.best_params["x"] == 5
        assert result.best_params["y"] == 10
        assert result.best_score == 0.0
        assert len(result.all_results) == 15  # 5 * 3 combinations

    def test_grid_search_with_bounds(self) -> None:
        """Test grid search with parameter bounds."""

        def objective(params: dict) -> float:
            return float(-abs(int(params["x"]) - 7))

        param_spaces = [
            ParameterSpace(name="x", low=1, high=10, step=2, param_type="int"),
        ]

        optimizer = GridSearchOptimizer()
        result = optimizer.optimize(objective, param_spaces)

        assert result.best_params["x"] == 7
        assert result.best_score == 0.0

    def test_grid_search_handles_exceptions(self) -> None:
        """Test that grid search handles objective function exceptions."""
        call_count = 0

        def flaky_objective(params: dict) -> float:
            nonlocal call_count
            call_count += 1
            if params["x"] == 2:
                raise ValueError("Simulated error")
            return float(params["x"])

        param_spaces = [
            ParameterSpace(name="x", values=[1, 2, 3]),
        ]

        optimizer = GridSearchOptimizer()
        result = optimizer.optimize(flaky_objective, param_spaces)

        # Should still complete and find best among valid results
        assert result.best_params["x"] == 3
        assert call_count == 3

    def test_grid_search_metadata(self) -> None:
        """Test that grid search includes metadata."""

        def objective(params: dict) -> float:
            return float(params["x"])

        param_spaces = [ParameterSpace(name="x", values=[1, 2, 3])]

        optimizer = GridSearchOptimizer(n_jobs=1)
        result = optimizer.optimize(objective, param_spaces)

        assert "optimizer" in result.metadata
        assert result.metadata["optimizer"] == "GridSearchOptimizer"
        assert result.metadata["total_combinations"] == 3
