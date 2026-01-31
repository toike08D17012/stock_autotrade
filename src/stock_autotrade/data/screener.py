"""Stock screener module for filtering stocks based on various criteria.

This module provides functionality to screen stocks based on price ranges,
moving averages, and other technical indicators using yfinance data.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


LOGGER = logging.getLogger(__name__)


@dataclass
class ScreeningCriteria:
    """Criteria for stock screening.

    Attributes:
        min_price: Minimum stock price (latest close). None means no lower bound.
        max_price: Maximum stock price (latest close). None means no upper bound.
        min_avg_price: Minimum n-day average price. None means no lower bound.
        max_avg_price: Maximum n-day average price. None means no upper bound.
        avg_price_days: Number of days for calculating average price. Defaults to 20.
        min_low_price: Minimum n-day low price. None means no lower bound.
        max_low_price: Maximum n-day low price. None means no upper bound.
        min_high_price: Minimum n-day high price. None means no lower bound.
        max_high_price: Maximum n-day high price. None means no upper bound.
        price_range_days: Number of days for calculating min/max price. Defaults to 20.
        min_volume: Minimum average daily volume. None means no lower bound.
        min_market_cap: Minimum market capitalization. None means no lower bound.
        max_market_cap: Maximum market capitalization. None means no upper bound.
        min_volatility: Minimum daily return volatility (std dev). None means no lower bound.
        max_volatility: Maximum daily return volatility (std dev). None means no upper bound.
        volatility_days: Number of days for calculating volatility. Defaults to 20.
    """

    min_price: float | None = None
    max_price: float | None = None
    min_avg_price: float | None = None
    max_avg_price: float | None = None
    avg_price_days: int = 20
    min_low_price: float | None = None
    max_low_price: float | None = None
    min_high_price: float | None = None
    max_high_price: float | None = None
    price_range_days: int = 20
    min_volume: int | None = None
    min_market_cap: float | None = None
    max_market_cap: float | None = None
    min_volatility: float | None = None
    max_volatility: float | None = None
    volatility_days: int = 20


@dataclass
class StockMetrics:
    """Calculated metrics for a single stock.

    Attributes:
        ticker: The stock ticker symbol.
        latest_price: The most recent closing price.
        avg_price: The n-day average closing price.
        low_price: The n-day lowest price.
        high_price: The n-day highest price.
        avg_volume: The average daily trading volume.
        market_cap: The market capitalization.
        volatility: The daily return volatility (standard deviation).
        data_start_date: The start date of available data.
        data_end_date: The end date of available data.
    """

    ticker: str
    latest_price: float | None = None
    avg_price: float | None = None
    low_price: float | None = None
    high_price: float | None = None
    avg_volume: float | None = None
    market_cap: float | None = None
    volatility: float | None = None
    data_start_date: datetime | None = None
    data_end_date: datetime | None = None


@dataclass
class ScreeningResult:
    """Result of stock screening.

    Attributes:
        passed: List of stocks that passed all criteria.
        failed: List of stocks that failed one or more criteria.
        errors: List of tickers that encountered errors during screening.
    """

    passed: list[StockMetrics] = field(default_factory=list)
    failed: list[StockMetrics] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    def get_passed_tickers(self) -> list[str]:
        """Return list of tickers that passed screening.

        Returns:
            List of ticker symbols that passed all criteria.
        """
        return [m.ticker for m in self.passed]


class StockScreener:
    """Stock screener for filtering stocks based on various criteria.

    This class provides methods to screen stocks using yfinance data
    based on price ranges, moving averages, volume, and other metrics.

    Attributes:
        ticker_factory: Factory function for creating yfinance Ticker objects.
    """

    def __init__(self, ticker_factory: Callable[[str], yf.Ticker] | None = None) -> None:
        """Initialize the stock screener.

        Args:
            ticker_factory: Optional factory for creating Ticker objects.
                Defaults to yfinance.Ticker.
        """
        self._ticker_factory = ticker_factory or yf.Ticker

    def calculate_metrics(
        self,
        ticker: str,
        days: int = 60,
        avg_days: int | None = None,
        price_range_days: int | None = None,
        volatility_days: int | None = None,
    ) -> StockMetrics:
        """Calculate metrics for a single stock.

        Args:
            ticker: The stock ticker symbol.
            days: Number of days of historical data to fetch. Defaults to 60.
            avg_days: Number of days for average price calculation.
                Defaults to None (uses 20 days).
            price_range_days: Number of days for high/low price calculation.
                Defaults to None (uses 20 days).
            volatility_days: Number of days for volatility calculation.
                Defaults to None (uses 20 days).

        Returns:
            StockMetrics object containing calculated metrics.

        Raises:
            ValueError: If no data is available for the ticker.
        """
        avg_days = avg_days or 20
        price_range_days = price_range_days or 20
        volatility_days = volatility_days or 20

        # Fetch enough data to calculate all metrics
        required_days = max(avg_days, price_range_days, volatility_days, days)
        end_date = datetime.now()
        # Add buffer days for weekends and holidays
        start_date = end_date - timedelta(days=int(required_days * 1.5) + 10)

        LOGGER.debug("Fetching data for %s from %s to %s", ticker, start_date, end_date)

        ticker_obj = self._ticker_factory(ticker)
        df = ticker_obj.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

        if df.empty:
            raise ValueError(f"No data available for ticker: {ticker}")

        metrics = StockMetrics(ticker=ticker)

        # Latest price
        metrics.latest_price = float(df["Close"].iloc[-1])

        # Average price (n-day)
        if len(df) >= avg_days:
            metrics.avg_price = float(df["Close"].tail(avg_days).mean())
        else:
            metrics.avg_price = float(df["Close"].mean())
            LOGGER.warning(
                "Not enough data for %d-day average for %s. Using %d days instead.", avg_days, ticker, len(df)
            )

        # Low/High price (n-day)
        range_data = df.tail(price_range_days) if len(df) >= price_range_days else df
        metrics.low_price = float(range_data["Low"].min())
        metrics.high_price = float(range_data["High"].max())

        # Average volume
        metrics.avg_volume = float(df["Volume"].mean())

        # Volatility (standard deviation of daily returns)
        if len(df) >= 2:
            returns = df["Close"].pct_change().dropna()
            if len(returns) >= volatility_days:
                metrics.volatility = float(returns.tail(volatility_days).std())
            else:
                metrics.volatility = float(returns.std())
        else:
            metrics.volatility = None

        # Market cap (from yfinance info if available)
        try:
            info = ticker_obj.info
            metrics.market_cap = info.get("marketCap")
        except Exception:
            LOGGER.debug("Could not fetch market cap for %s", ticker)
            metrics.market_cap = None

        # Data date range
        metrics.data_start_date = df.index[0].to_pydatetime()
        metrics.data_end_date = df.index[-1].to_pydatetime()

        return metrics

    def _check_range(
        self,
        ticker: str,
        value: float | None,
        min_val: float | int | None,
        max_val: float | int | None,
        name: str,
        fmt: str = "%.2f",
    ) -> bool:
        """Check if a value is within the specified range.

        Args:
            ticker: The stock ticker symbol for logging.
            value: The value to check.
            min_val: Minimum allowed value, or None for no lower bound.
            max_val: Maximum allowed value, or None for no upper bound.
            name: The name of the metric for logging.
            fmt: Format string for logging the value.

        Returns:
            True if value is within range or any value is None, False otherwise.
        """
        if min_val is not None and value is not None and value < min_val:
            LOGGER.debug(f"%s: {name} {fmt} < min_{name} {fmt}", ticker, value, min_val)
            return False
        if max_val is not None and value is not None and value > max_val:
            LOGGER.debug(f"%s: {name} {fmt} > max_{name} {fmt}", ticker, value, max_val)
            return False
        return True

    def check_criteria(self, metrics: StockMetrics, criteria: ScreeningCriteria) -> bool:
        """Check if stock metrics meet the screening criteria.

        Args:
            metrics: The calculated stock metrics.
            criteria: The screening criteria to check against.

        Returns:
            True if all criteria are met, False otherwise.
        """
        checks = [
            # Latest price range
            self._check_range(
                metrics.ticker, metrics.latest_price, criteria.min_price, criteria.max_price, "latest_price"
            ),
            # Average price range
            self._check_range(
                metrics.ticker, metrics.avg_price, criteria.min_avg_price, criteria.max_avg_price, "avg_price"
            ),
            # Low price range
            self._check_range(
                metrics.ticker, metrics.low_price, criteria.min_low_price, criteria.max_low_price, "low_price"
            ),
            # High price range
            self._check_range(
                metrics.ticker, metrics.high_price, criteria.min_high_price, criteria.max_high_price, "high_price"
            ),
            # Volume (only min check)
            self._check_range(metrics.ticker, metrics.avg_volume, criteria.min_volume, None, "avg_volume", "%.0f"),
            # Market cap range
            self._check_range(
                metrics.ticker,
                metrics.market_cap,
                criteria.min_market_cap,
                criteria.max_market_cap,
                "market_cap",
                "%.0f",
            ),
            # Volatility range
            self._check_range(
                metrics.ticker,
                metrics.volatility,
                criteria.min_volatility,
                criteria.max_volatility,
                "volatility",
                "%.4f",
            ),
        ]

        return all(checks)

    def screen(
        self,
        tickers: list[str],
        criteria: ScreeningCriteria,
        days: int = 60,
    ) -> ScreeningResult:
        """Screen multiple stocks against the given criteria.

        Args:
            tickers: List of stock ticker symbols to screen.
            criteria: The screening criteria to apply.
            days: Number of days of historical data to fetch. Defaults to 60.

        Returns:
            ScreeningResult containing passed, failed, and error lists.
        """
        result = ScreeningResult()

        for ticker in tickers:
            try:
                LOGGER.info("Screening %s...", ticker)
                metrics = self.calculate_metrics(
                    ticker,
                    days=days,
                    avg_days=criteria.avg_price_days,
                    price_range_days=criteria.price_range_days,
                    volatility_days=criteria.volatility_days,
                )

                if self.check_criteria(metrics, criteria):
                    result.passed.append(metrics)
                    LOGGER.info("%s passed screening.", ticker)
                else:
                    result.failed.append(metrics)
                    LOGGER.info("%s failed screening.", ticker)

            except Exception as e:
                LOGGER.warning("Error screening %s: %s", ticker, str(e))
                result.errors.append((ticker, str(e)))

        LOGGER.info(
            "Screening complete. Passed: %d, Failed: %d, Errors: %d",
            len(result.passed),
            len(result.failed),
            len(result.errors),
        )

        return result

    def to_dataframe(self, metrics_list: list[StockMetrics]) -> pd.DataFrame:
        """Convert a list of StockMetrics to a pandas DataFrame.

        Args:
            metrics_list: List of StockMetrics objects.

        Returns:
            DataFrame with one row per stock.
        """
        data = []
        for m in metrics_list:
            data.append(
                {
                    "ticker": m.ticker,
                    "latest_price": m.latest_price,
                    "avg_price": m.avg_price,
                    "low_price": m.low_price,
                    "high_price": m.high_price,
                    "avg_volume": m.avg_volume,
                    "market_cap": m.market_cap,
                    "volatility": m.volatility,
                    "data_start": m.data_start_date,
                    "data_end": m.data_end_date,
                }
            )
        return pd.DataFrame(data)


def get_nikkei225_tickers() -> list[str]:
    """Return a list of Nikkei 225 component stock tickers.

    Returns:
        List of ticker symbols for Nikkei 225 stocks (with .T suffix).

    Note:
        This is a static list and may not reflect the latest index composition.
        For the most up-to-date list, consider fetching from an external source.
    """
    # Major Nikkei 225 components (subset - commonly traded large caps)
    # Full list would need to be updated periodically
    return [
        "7203.T",  # Toyota
        "6758.T",  # Sony
        "9984.T",  # SoftBank
        "6861.T",  # Keyence
        "8306.T",  # MUFG
        "9433.T",  # KDDI
        "6098.T",  # Recruit
        "4063.T",  # Shin-Etsu Chemical
        "6367.T",  # Daikin
        "6954.T",  # Fanuc
        "8035.T",  # Tokyo Electron
        "4519.T",  # Chugai Pharmaceutical
        "7974.T",  # Nintendo
        "6501.T",  # Hitachi
        "4502.T",  # Takeda
        "6902.T",  # Denso
        "7267.T",  # Honda
        "8058.T",  # Mitsubishi Corp
        "7751.T",  # Canon
        "6857.T",  # Advantest
    ]


def get_sp500_sample_tickers() -> list[str]:
    """Return a sample list of S&P 500 component stock tickers.

    Returns:
        List of ticker symbols for major S&P 500 stocks.

    Note:
        This is a sample list of major components, not the full index.
    """
    return [
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL",  # Alphabet
        "AMZN",  # Amazon
        "NVDA",  # NVIDIA
        "META",  # Meta
        "TSLA",  # Tesla
        "BRK-B",  # Berkshire Hathaway
        "UNH",  # UnitedHealth
        "JNJ",  # Johnson & Johnson
        "JPM",  # JPMorgan
        "V",  # Visa
        "PG",  # Procter & Gamble
        "XOM",  # Exxon Mobil
        "HD",  # Home Depot
        "MA",  # Mastercard
        "CVX",  # Chevron
        "MRK",  # Merck
        "ABBV",  # AbbVie
        "PEP",  # PepsiCo
    ]
