"""Tests for the metrics module."""

import pandas as pd
import pytest

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


@pytest.fixture
def sample_equity_curve() -> pd.Series:
    """Create a sample equity curve for testing."""
    return pd.Series([100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 110.0])


@pytest.fixture
def sample_returns(sample_equity_curve: pd.Series) -> pd.Series:
    """Create sample returns from equity curve."""
    return sample_equity_curve.pct_change().fillna(0.0)


@pytest.fixture
def sample_positions() -> pd.Series:
    """Create sample positions for testing."""
    return pd.Series([0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0])


class TestTotalReturn:
    """Tests for calculate_total_return."""

    def test_positive_return(self, sample_equity_curve: pd.Series) -> None:
        """Test with positive return."""
        result = calculate_total_return(sample_equity_curve)
        expected = 0.1  # (110 - 100) / 100
        assert abs(result - expected) < 1e-9

    def test_negative_return(self) -> None:
        """Test with negative return."""
        equity = pd.Series([100.0, 95.0, 90.0])
        result = calculate_total_return(equity)
        expected = -0.1
        assert abs(result - expected) < 1e-9

    def test_empty_series(self) -> None:
        """Test with empty series."""
        equity = pd.Series(dtype=float)
        result = calculate_total_return(equity)
        assert result == 0.0


class TestAnnualizedReturn:
    """Tests for calculate_annualized_return."""

    def test_positive_return(self, sample_returns: pd.Series) -> None:
        """Test annualized return calculation."""
        result = calculate_annualized_return(sample_returns)
        assert isinstance(result, float)
        assert result > 0

    def test_empty_series(self) -> None:
        """Test with empty series."""
        returns = pd.Series(dtype=float)
        result = calculate_annualized_return(returns)
        assert result == 0.0


class TestMaxDrawdown:
    """Tests for calculate_max_drawdown."""

    def test_drawdown_exists(self, sample_equity_curve: pd.Series) -> None:
        """Test max drawdown calculation with drawdowns."""
        result = calculate_max_drawdown(sample_equity_curve)
        # Drawdown from 105 to 103 = -1.9%
        assert result < 0
        assert result > -1.0

    def test_no_drawdown(self) -> None:
        """Test with monotonically increasing equity."""
        equity = pd.Series([100.0, 101.0, 102.0, 103.0])
        result = calculate_max_drawdown(equity)
        assert result == 0.0

    def test_empty_series(self) -> None:
        """Test with empty series."""
        equity = pd.Series(dtype=float)
        result = calculate_max_drawdown(equity)
        assert result == 0.0


class TestSharpeRatio:
    """Tests for calculate_sharpe_ratio."""

    def test_positive_sharpe(self, sample_returns: pd.Series) -> None:
        """Test Sharpe ratio calculation."""
        result = calculate_sharpe_ratio(sample_returns)
        assert isinstance(result, float)

    def test_zero_std_returns_zero(self) -> None:
        """Test that zero std deviation returns zero."""
        returns = pd.Series([0.0, 0.0, 0.0, 0.0])
        result = calculate_sharpe_ratio(returns)
        assert result == 0.0

    def test_empty_series(self) -> None:
        """Test with empty series."""
        returns = pd.Series(dtype=float)
        result = calculate_sharpe_ratio(returns)
        assert result == 0.0


class TestSortinoRatio:
    """Tests for calculate_sortino_ratio."""

    def test_with_downside_volatility(self, sample_returns: pd.Series) -> None:
        """Test Sortino ratio calculation."""
        result = calculate_sortino_ratio(sample_returns)
        assert isinstance(result, float)

    def test_no_negative_returns(self) -> None:
        """Test with no negative returns."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.03])
        result = calculate_sortino_ratio(returns)
        assert result == float("inf")

    def test_empty_series(self) -> None:
        """Test with empty series."""
        returns = pd.Series(dtype=float)
        result = calculate_sortino_ratio(returns)
        assert result == 0.0


class TestCalmarRatio:
    """Tests for calculate_calmar_ratio."""

    def test_calmar_calculation(self, sample_returns: pd.Series, sample_equity_curve: pd.Series) -> None:
        """Test Calmar ratio calculation."""
        result = calculate_calmar_ratio(sample_returns, sample_equity_curve)
        assert isinstance(result, float)

    def test_no_drawdown_returns_inf(self) -> None:
        """Test with no drawdown returns inf for positive returns."""
        equity = pd.Series([100.0, 101.0, 102.0, 103.0])
        returns = equity.pct_change().fillna(0.0)
        result = calculate_calmar_ratio(returns, equity)
        assert result == float("inf")


class TestVolatility:
    """Tests for calculate_volatility."""

    def test_volatility_calculation(self, sample_returns: pd.Series) -> None:
        """Test volatility calculation."""
        result = calculate_volatility(sample_returns)
        assert isinstance(result, float)
        assert result >= 0

    def test_empty_series(self) -> None:
        """Test with empty series."""
        returns = pd.Series(dtype=float)
        result = calculate_volatility(returns)
        assert result == 0.0


class TestWinRate:
    """Tests for calculate_win_rate."""

    def test_win_rate_calculation(self, sample_positions: pd.Series, sample_returns: pd.Series) -> None:
        """Test win rate calculation."""
        result = calculate_win_rate(sample_positions, sample_returns)
        assert 0.0 <= result <= 1.0

    def test_no_trades(self) -> None:
        """Test with no trades."""
        positions = pd.Series([0.0, 0.0, 0.0, 0.0])
        returns = pd.Series([0.01, 0.02, -0.01, 0.01])
        result = calculate_win_rate(positions, returns)
        assert result == 0.0


class TestProfitFactor:
    """Tests for calculate_profit_factor."""

    def test_profit_factor_calculation(self, sample_positions: pd.Series, sample_returns: pd.Series) -> None:
        """Test profit factor calculation."""
        result = calculate_profit_factor(sample_positions, sample_returns)
        assert isinstance(result, float)

    def test_no_losses_returns_inf(self) -> None:
        """Test with no losses returns inf."""
        positions = pd.Series([1.0, 1.0, 1.0, 1.0])
        returns = pd.Series([0.01, 0.02, 0.01, 0.03])
        result = calculate_profit_factor(positions, returns)
        assert result == float("inf")


class TestCountTrades:
    """Tests for count_trades."""

    def test_count_trades(self, sample_positions: pd.Series) -> None:
        """Test trade counting."""
        result = count_trades(sample_positions)
        # Changes: 0->1, 1->0, 0->1, 1->0 = 4 trades
        assert result == 4

    def test_no_trades(self) -> None:
        """Test with no trades."""
        positions = pd.Series([0.0, 0.0, 0.0, 0.0])
        result = count_trades(positions)
        assert result == 0


class TestAllMetrics:
    """Tests for calculate_all_metrics."""

    def test_returns_all_metrics(
        self,
        sample_equity_curve: pd.Series,
        sample_returns: pd.Series,
        sample_positions: pd.Series,
    ) -> None:
        """Test that all metrics are returned."""
        result = calculate_all_metrics(sample_equity_curve, sample_returns, sample_positions)

        expected_keys = {
            "total_return",
            "annualized_return",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "volatility",
            "win_rate",
            "profit_factor",
            "num_trades",
            "final_equity",
        }
        assert set(result.keys()) == expected_keys
        assert all(isinstance(v, float) for v in result.values())
