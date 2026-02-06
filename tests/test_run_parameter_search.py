"""Tests for the parameter search script."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.run_parameter_search import _load_tickers_from_file


def test_load_tickers_from_file_strips_and_filters_empty_values(tmp_path: Path) -> None:
    """Load tickers from CSV while stripping whitespace and skipping empty values."""
    csv_path = tmp_path / "tickers.csv"
    pd.DataFrame({"ticker": [" 7203.T ", None, "", "   ", "6758.T", 9984]}).to_csv(csv_path, index=False)

    tickers = _load_tickers_from_file(str(csv_path), "ticker")

    assert tickers == ["7203.T", "6758.T", "9984"]


def test_load_tickers_from_file_raises_on_missing_column(tmp_path: Path) -> None:
    """Raise ValueError when the target column is not present."""
    csv_path = tmp_path / "tickers.csv"
    pd.DataFrame({"symbol": ["AAPL", "MSFT"]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Column 'ticker' not found"):
        _load_tickers_from_file(str(csv_path), "ticker")
