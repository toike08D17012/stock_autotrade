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
        List of ticker symbols for all Nikkei 225 stocks (with .T suffix).

    Note:
        This is a static list based on the composition as of January 2026.
        The actual index composition may change over time due to regular reviews.
        For the most up-to-date list, consider fetching from an external source.
        Source: https://indexes.nikkei.co.jp/nkave/index/component?idx=nk225
    """
    return [
        # 医薬品 (Pharmaceuticals) - 9 stocks
        "4151.T",  # Kyowa Kirin
        "4502.T",  # Takeda Pharmaceutical
        "4503.T",  # Astellas Pharma
        "4506.T",  # Sumitomo Pharma
        "4507.T",  # Shionogi
        "4519.T",  # Chugai Pharmaceutical
        "4523.T",  # Eisai
        "4568.T",  # Daiichi Sankyo
        "4578.T",  # Otsuka Holdings
        # 電気機器 (Electrical Equipment) - 28 stocks
        "4062.T",  # Ibiden
        "6479.T",  # Minebea Mitsumi
        "6501.T",  # Hitachi
        "6503.T",  # Mitsubishi Electric
        "6504.T",  # Fuji Electric
        "6506.T",  # Yaskawa Electric
        "6526.T",  # SociNext
        "6645.T",  # Omron
        "6674.T",  # GS Yuasa
        "6701.T",  # NEC
        "6702.T",  # Fujitsu
        "6723.T",  # Renesas Electronics
        "6724.T",  # Seiko Epson
        "6752.T",  # Panasonic Holdings
        "6753.T",  # Sharp
        "6758.T",  # Sony Group
        "6762.T",  # TDK
        "6770.T",  # Alps Alpine
        "6841.T",  # Yokogawa Electric
        "6857.T",  # Advantest
        "6861.T",  # Keyence
        "6902.T",  # Denso
        "6920.T",  # Lasertec
        "6952.T",  # Casio Computer
        "6954.T",  # Fanuc
        "6963.T",  # Rohm
        "6971.T",  # Kyocera
        "6976.T",  # Taiyo Yuden
        "6981.T",  # Murata Manufacturing
        "7735.T",  # Screen Holdings
        "7751.T",  # Canon
        "7752.T",  # Ricoh
        "8035.T",  # Tokyo Electron
        # 自動車 (Automotive) - 12 stocks
        "7201.T",  # Nissan Motor
        "7202.T",  # Isuzu Motors
        "7203.T",  # Toyota Motor
        "7205.T",  # Hino Motors
        "7211.T",  # Mitsubishi Motors
        "7261.T",  # Mazda Motor
        "7267.T",  # Honda Motor
        "7269.T",  # Suzuki Motor
        "7270.T",  # Subaru
        "7272.T",  # Yamaha Motor
        # 精密機器 (Precision Instruments) - 6 stocks
        "4543.T",  # Terumo
        "4902.T",  # Konica Minolta
        "6146.T",  # Disco
        "7731.T",  # Nikon
        "7733.T",  # Olympus
        "7741.T",  # Hoya
        # 通信 (Telecommunications) - 4 stocks
        "9432.T",  # NTT
        "9433.T",  # KDDI
        "9434.T",  # SoftBank Corp
        "9984.T",  # SoftBank Group
        # 銀行 (Banking) - 11 stocks
        "5831.T",  # Shizuoka Financial Group
        "7186.T",  # Yokohama Financial Group
        "8304.T",  # Aozora Bank
        "8306.T",  # Mitsubishi UFJ Financial Group
        "8308.T",  # Resona Holdings
        "8309.T",  # Sumitomo Mitsui Trust Group
        "8316.T",  # Sumitomo Mitsui Financial Group
        "8331.T",  # Chiba Bank
        "8354.T",  # Fukuoka Financial Group
        "8411.T",  # Mizuho Financial Group
        # その他金融 (Other Finance) - 3 stocks
        "8253.T",  # Credit Saison
        "8591.T",  # Orix
        "8697.T",  # Japan Exchange Group
        # 証券 (Securities) - 2 stocks
        "8601.T",  # Daiwa Securities Group
        "8604.T",  # Nomura Holdings
        # 保険 (Insurance) - 5 stocks
        "8630.T",  # Sompo Holdings
        "8725.T",  # MS&AD Insurance Group
        "8750.T",  # Dai-ichi Life Holdings
        "8766.T",  # Tokio Marine Holdings
        "8795.T",  # T&D Holdings
        # 水産 (Fishery) - 1 stock
        "1332.T",  # Nissui
        # 食品 (Food) - 10 stocks
        "2002.T",  # Nisshin Seifun Group
        "2269.T",  # Meiji Holdings
        "2282.T",  # NH Foods
        "2501.T",  # Sapporo Holdings
        "2502.T",  # Asahi Group Holdings
        "2503.T",  # Kirin Holdings
        "2801.T",  # Kikkoman
        "2802.T",  # Ajinomoto
        "2871.T",  # Nichirei
        "2914.T",  # Japan Tobacco
        # 小売業 (Retail) - 10 stocks
        "3086.T",  # J. Front Retailing
        "3092.T",  # ZOZO
        "3099.T",  # Isetan Mitsukoshi Holdings
        "3382.T",  # Seven & i Holdings
        "7453.T",  # Ryohin Keikaku (MUJI)
        "8233.T",  # Takashimaya
        "8252.T",  # Marui Group
        "8267.T",  # Aeon
        "9843.T",  # Nitori Holdings
        "9983.T",  # Fast Retailing (Uniqlo)
        # サービス (Services) - 17 stocks
        "2413.T",  # M3
        "2432.T",  # DeNA
        "3659.T",  # Nexon
        "3697.T",  # SHIFT
        "4307.T",  # Nomura Research Institute
        "4324.T",  # Dentsu Group
        "4385.T",  # Mercari
        "4661.T",  # Oriental Land (Disney)
        "4689.T",  # LINE Yahoo
        "4704.T",  # Trend Micro
        "4751.T",  # CyberAgent
        "4755.T",  # Rakuten Group
        "6098.T",  # Recruit Holdings
        "6178.T",  # Japan Post Holdings
        "6532.T",  # BayCurrent
        "7974.T",  # Nintendo
        "9602.T",  # Toho
        "9735.T",  # Secom
        "9766.T",  # Konami Group
        # 鉱業 (Mining) - 1 stock
        "1605.T",  # INPEX
        # 繊維 (Textiles) - 2 stocks
        "3401.T",  # Teijin
        "3402.T",  # Toray Industries
        # パルプ・紙 (Pulp & Paper) - 1 stock
        "3861.T",  # Oji Holdings
        # 化学 (Chemicals) - 17 stocks
        "3405.T",  # Kuraray
        "3407.T",  # Asahi Kasei
        "4004.T",  # Resonac Holdings
        "4005.T",  # Sumitomo Chemical
        "4021.T",  # Nissan Chemical
        "4042.T",  # Tosoh
        "4043.T",  # Tokuyama
        "4061.T",  # Denka
        "4063.T",  # Shin-Etsu Chemical
        "4183.T",  # Mitsui Chemicals
        "4188.T",  # Mitsubishi Chemical Group
        "4208.T",  # UBE
        "4452.T",  # Kao
        "4901.T",  # Fujifilm Holdings
        "4911.T",  # Shiseido
        "6988.T",  # Nitto Denko
        # 石油 (Oil) - 2 stocks
        "5019.T",  # Idemitsu Kosan
        "5020.T",  # ENEOS Holdings
        # ゴム (Rubber) - 2 stocks
        "5101.T",  # Yokohama Rubber
        "5108.T",  # Bridgestone
        # 窯業 (Glass & Ceramics) - 6 stocks
        "5201.T",  # AGC
        "5214.T",  # Nippon Electric Glass
        "5233.T",  # Taiheiyo Cement
        "5301.T",  # Tokai Carbon
        "5332.T",  # TOTO
        "5333.T",  # NGK Insulators
        # 鉄鋼 (Steel) - 3 stocks
        "5401.T",  # Nippon Steel
        "5406.T",  # Kobe Steel
        "5411.T",  # JFE Holdings
        # 非鉄・金属 (Non-ferrous Metals) - 8 stocks
        "3436.T",  # SUMCO
        "5706.T",  # Mitsui Mining & Smelting
        "5711.T",  # Mitsubishi Materials
        "5713.T",  # Sumitomo Metal Mining
        "5714.T",  # DOWA Holdings
        "5801.T",  # Furukawa Electric
        "5802.T",  # Sumitomo Electric Industries
        "5803.T",  # Fujikura
        # 商社 (Trading Companies) - 7 stocks
        "2768.T",  # Sojitz
        "8001.T",  # Itochu
        "8002.T",  # Marubeni
        "8015.T",  # Toyota Tsusho
        "8031.T",  # Mitsui & Co
        "8053.T",  # Sumitomo Corp
        "8058.T",  # Mitsubishi Corp
        # 建設 (Construction) - 9 stocks
        "1721.T",  # Comsys Holdings
        "1801.T",  # Taisei Corp
        "1802.T",  # Obayashi Corp
        "1803.T",  # Shimizu Corp
        "1808.T",  # Haseko Corp
        "1812.T",  # Kajima Corp
        "1925.T",  # Daiwa House Industry
        "1928.T",  # Sekisui House
        "1963.T",  # JGC Holdings
        # 機械 (Machinery) - 16 stocks
        "5631.T",  # Japan Steel Works
        "6103.T",  # Okuma
        "6113.T",  # Amada
        "6273.T",  # SMC
        "6301.T",  # Komatsu
        "6302.T",  # Sumitomo Heavy Industries
        "6305.T",  # Hitachi Construction Machinery
        "6326.T",  # Kubota
        "6361.T",  # Ebara
        "6367.T",  # Daikin Industries
        "6471.T",  # NSK
        "6472.T",  # NTN
        "6473.T",  # JTEKT
        "7004.T",  # Kanadevia (formerly Hitachi Zosen)
        "7011.T",  # Mitsubishi Heavy Industries
        "7013.T",  # IHI
        # 造船 (Shipbuilding) - 1 stock
        "7012.T",  # Kawasaki Heavy Industries
        # その他製造 (Other Manufacturing) - 4 stocks
        "7832.T",  # Bandai Namco Holdings
        "7911.T",  # Toppan Holdings
        "7912.T",  # Dai Nippon Printing
        "7951.T",  # Yamaha
        # 不動産 (Real Estate) - 5 stocks
        "3289.T",  # Tokyu Fudosan Holdings
        "8801.T",  # Mitsui Fudosan
        "8802.T",  # Mitsubishi Estate
        "8804.T",  # Tokyo Tatemono
        "8830.T",  # Sumitomo Realty & Development
        # 鉄道・バス (Railway & Bus) - 8 stocks
        "9001.T",  # Tobu Railway
        "9005.T",  # Tokyu Corp
        "9007.T",  # Odakyu Electric Railway
        "9008.T",  # Keio Corp
        "9009.T",  # Keisei Electric Railway
        "9020.T",  # East Japan Railway
        "9021.T",  # West Japan Railway
        "9022.T",  # Central Japan Railway
        # 陸運 (Land Transportation) - 2 stocks
        "9064.T",  # Yamato Holdings
        "9147.T",  # Nippon Express Holdings
        # 海運 (Marine Transportation) - 3 stocks
        "9101.T",  # Nippon Yusen (NYK)
        "9104.T",  # Mitsui O.S.K. Lines
        "9107.T",  # Kawasaki Kisen Kaisha
        # 空運 (Air Transportation) - 2 stocks
        "9201.T",  # Japan Airlines
        "9202.T",  # ANA Holdings
        # 電力 (Electric Power) - 3 stocks
        "9501.T",  # Tokyo Electric Power
        "9502.T",  # Chubu Electric Power
        "9503.T",  # Kansai Electric Power
        # ガス (Gas) - 2 stocks
        "9531.T",  # Tokyo Gas
        "9532.T",  # Osaka Gas
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
