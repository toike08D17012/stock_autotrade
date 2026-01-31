"""Tests for the stock screener script."""

import json
import pickle

# Add src to path for imports
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scripts.run_stock_screener import save_results_to_file
from stock_autotrade.data.screener import ScreeningResult, StockMetrics, StockScreener


@pytest.fixture
def sample_result() -> ScreeningResult:
    """Create a sample screening result for testing."""
    result = ScreeningResult()
    result.passed = [
        StockMetrics(
            ticker="AAPL",
            latest_price=150.0,
            avg_price=145.0,
            low_price=140.0,
            high_price=160.0,
            avg_volume=50_000_000.0,
            market_cap=2_500_000_000_000.0,
            volatility=0.02,
        ),
        StockMetrics(
            ticker="GOOGL",
            latest_price=180.0,
            avg_price=175.0,
        ),
    ]
    result.failed = [StockMetrics(ticker="MSFT", latest_price=50.0)]
    result.errors = [("INVALID", "No data available")]
    return result


class TestSaveResultsToFile:
    """Tests for save_results_to_file function."""

    def test_save_to_csv(self, sample_result: ScreeningResult) -> None:
        """Test saving results to CSV format."""
        screener = StockScreener()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            save_results_to_file(screener, sample_result, output_path)

            # Verify file was created and has content
            df = pd.read_csv(output_path)
            assert len(df) == 2
            assert "ticker" in df.columns
            assert df.iloc[0]["ticker"] == "AAPL"
        finally:
            Path(output_path).unlink()

    def test_save_to_json(self, sample_result: ScreeningResult) -> None:
        """Test saving results to JSON format."""
        screener = StockScreener()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            save_results_to_file(screener, sample_result, output_path)

            # Verify file was created and has correct structure
            with open(output_path, encoding="utf-8") as f:
                data = json.load(f)

            assert "timestamp" in data
            assert "summary" in data
            assert data["summary"]["passed"] == 2
            assert data["summary"]["failed"] == 1
            assert data["summary"]["errors"] == 1
            assert len(data["passed_stocks"]) == 2
            assert data["passed_stocks"][0]["ticker"] == "AAPL"
            assert len(data["errors"]) == 1
        finally:
            Path(output_path).unlink()

    def test_save_to_pickle(self, sample_result: ScreeningResult) -> None:
        """Test saving results to Pickle format."""
        screener = StockScreener()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            output_path = f.name

        try:
            save_results_to_file(screener, sample_result, output_path)

            # Verify file was created and can be loaded
            with open(output_path, "rb") as f:
                loaded_result = pickle.load(f)

            assert isinstance(loaded_result, ScreeningResult)
            assert len(loaded_result.passed) == 2
            assert len(loaded_result.failed) == 1
            assert len(loaded_result.errors) == 1
        finally:
            Path(output_path).unlink()

    def test_save_to_excel_without_openpyxl(self, sample_result: ScreeningResult) -> None:
        """Test saving to Excel without openpyxl raises ImportError."""
        screener = StockScreener()

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            output_path = f.name

        try:
            # Mock pandas.DataFrame.to_excel to raise ImportError
            with (
                patch("pandas.DataFrame.to_excel", side_effect=ImportError("No module named 'openpyxl'")),
                pytest.raises(ImportError),
            ):
                save_results_to_file(screener, sample_result, output_path)
        finally:
            # Clean up if file was created
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_save_unknown_extension_defaults_to_csv(self, sample_result: ScreeningResult) -> None:
        """Test that unknown extensions default to CSV format."""
        screener = StockScreener()

        with tempfile.NamedTemporaryFile(suffix=".unknown", delete=False) as f:
            output_path = f.name

        try:
            save_results_to_file(screener, sample_result, output_path)

            # Verify file was saved as CSV despite .unknown extension
            df = pd.read_csv(output_path)
            assert len(df) == 2
            assert "ticker" in df.columns
        finally:
            Path(output_path).unlink()

    def test_save_empty_result(self) -> None:
        """Test saving empty results doesn't crash."""
        screener = StockScreener()
        empty_result = ScreeningResult()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            save_results_to_file(screener, empty_result, output_path)

            # Verify empty file was created (may be empty or have headers only)
            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size >= 0
        finally:
            Path(output_path).unlink()
