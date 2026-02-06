"""Protocol definitions for ticker client interfaces."""

from collections.abc import Mapping
from typing import Protocol

import pandas as pd


class TickerLike(Protocol):
    """Protocol for ticker clients that can fetch historical price data."""

    def history(self, start: str, end: str) -> pd.DataFrame:
        """Fetch historical OHLCV data for a date range."""


class TickerWithInfoLike(TickerLike, Protocol):
    """Protocol for ticker clients that also expose metadata via ``info``."""

    @property
    def info(self) -> Mapping[str, object]:
        """Return metadata such as market capitalization."""
