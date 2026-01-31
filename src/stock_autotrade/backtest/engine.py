"""Backtest engine for simple long-only strategies."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Container for backtest outputs."""

    equity_curve: pd.Series
    strategy_returns: pd.Series
    positions: pd.Series
    cash: pd.Series
    total_return: float


def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    initial_cash: float = 1_000_000.0,
    fee_rate: float = 0.0,
    trade_size: float = 1.0,
    max_position: float = 1.0,
) -> BacktestResult:
    """Run a long-only backtest based on signals with explicit trades.

    Args:
        prices (pd.Series): Price series indexed by time.
        signals (pd.Series): Signal series where 1 means long and 0 means flat.
        initial_cash (float, optional): Starting capital. Defaults to 1_000_000.0.
        fee_rate (float, optional): Fee applied per trade notional (e.g., 0.001 = 0.1%). Defaults to 0.0.
        trade_size (float, optional): Trade size per step (shares). Defaults to 1.0.
        max_position (float, optional): Maximum position size (shares). Defaults to 1.0.

    Returns:
        BacktestResult: Backtest outputs including equity curve and returns.
    """
    if prices.empty:
        raise ValueError("Prices must not be empty.")
    if initial_cash <= 0:
        raise ValueError("Initial cash must be positive.")
    if fee_rate < 0:
        raise ValueError("Fee rate must be non-negative.")
    if trade_size <= 0:
        raise ValueError("Trade size must be positive.")
    if max_position < 0:
        raise ValueError("Max position must be non-negative.")

    aligned_signals = signals.reindex(prices.index).fillna(0).astype(int)
    target_positions: pd.Series = aligned_signals.shift(1).fillna(0).astype(float) * max_position

    positions, cash = _simulate_trades(
        prices=prices,
        target_positions=target_positions,
        initial_cash=initial_cash,
        trade_size=trade_size,
        fee_rate=fee_rate,
    )
    equity_curve = cash + positions * prices
    strategy_returns = equity_curve.pct_change().fillna(0.0)
    total_return = float(equity_curve.iloc[-1] / initial_cash - 1.0)

    return BacktestResult(
        equity_curve=equity_curve,
        strategy_returns=strategy_returns,
        positions=positions,
        cash=cash,
        total_return=total_return,
    )


def _simulate_trades(
    prices: pd.Series,
    target_positions: pd.Series,
    initial_cash: float,
    trade_size: float,
    fee_rate: float,
) -> tuple[pd.Series, pd.Series]:
    """Simulate buy/sell trades to move toward target positions.

    Args:
        prices (pd.Series): Price series indexed by time.
        target_positions (pd.Series): Desired positions over time (shares).
        initial_cash (float): Starting cash balance.
        trade_size (float): Maximum trade size per step (shares).
        fee_rate (float): Fee rate applied to trade notional.

    Returns:
        tuple[pd.Series, pd.Series]: Positions and cash series.
    """
    positions = []
    cash = []
    current_position = 0.0
    current_cash = initial_cash

    for price, target in zip(prices.tolist(), target_positions.tolist(), strict=True):
        delta = target - current_position
        if delta != 0:
            trade_shares = min(abs(delta), trade_size)
            trade_shares = trade_shares if delta > 0 else -trade_shares
            trade_notional = trade_shares * price
            fee = abs(trade_notional) * fee_rate
            projected_cash = current_cash - trade_notional - fee
            if projected_cash < 0:
                max_affordable = (current_cash / (price * (1 + fee_rate))) if price > 0 else 0.0
                trade_shares = max(0.0, min(trade_shares, max_affordable))
                trade_notional = trade_shares * price
                fee = abs(trade_notional) * fee_rate
                projected_cash = current_cash - trade_notional - fee
            current_cash = projected_cash
            current_position += trade_shares

        positions.append(current_position)
        cash.append(current_cash)

    return pd.Series(positions, index=target_positions.index), pd.Series(cash, index=target_positions.index)
