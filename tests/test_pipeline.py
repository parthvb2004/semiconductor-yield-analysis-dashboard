"""
test_pipeline.py
Lightweight smoke tests for the end-to-end SECOM pipeline.
Run with: pytest tests/test_pipeline.py
"""

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import run_pipeline

PROCESSED_DIR = Path("data/processed")


@pytest.fixture(scope="module", autouse=True)
def run_pipeline_once():
    """Run the full pipeline once before the tests in this module."""
    run_pipeline()


def test_clean_dataset_created():
    """The cleaned SECOM CSV should exist and contain no nulls."""
    path = PROCESSED_DIR / "secom_clean.csv"
    assert path.exists(), "secom_clean.csv was not created"
    df = pd.read_csv(path)
    assert df.isna().sum().sum() == 0, "Cleaned dataset contains unexpected nulls"
    assert df["Label"].isin([0, 1]).all(), "Label column contains unexpected values"


def test_master_dataset_created():
    """The merged master dataset should exist and include key metadata columns."""
    path = PROCESSED_DIR / "secom_master.csv"
    assert path.exists(), "secom_master.csv was not created"
    df = pd.read_csv(path)
    for col in ["Wafer_ID", "Lot_ID", "Machine_ID", "Operator_ID", "Shift"]:
        assert col in df.columns, f"Missing expected metadata column: {col}"


def test_kpi_tables_created():
    """All 3 KPI CSVs should exist and yield/failure percentages should sum to 100."""
    overall = pd.read_csv(PROCESSED_DIR / "kpi_overall_yield.csv")
    assert (overall["Yield_Pct"] + overall["Failure_Rate_Pct"]).round(2).eq(100.0).all()

    by_machine = pd.read_csv(PROCESSED_DIR / "kpi_yield_by_machine.csv")
    assert not by_machine.empty, "Yield by machine table is empty"

    by_shift = pd.read_csv(PROCESSED_DIR / "kpi_yield_by_shift.csv")
    assert not by_shift.empty, "Yield by shift table is empty"


def test_feature_shortlist_created():
    """The feature selection output should exist and be ranked descending by importance."""
    path = PROCESSED_DIR / "top_features.csv"
    assert path.exists(), "top_features.csv was not created"
    df = pd.read_csv(path)
    assert df["Importance"].is_monotonic_decreasing, "Feature shortlist is not sorted by importance"