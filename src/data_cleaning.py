"""
data_cleaning.py
Cleans the raw UCI SECOM sensor dataset:
    - Loads raw .data files
    - Handles missing values
    - Removes near-zero-variance sensors
    - Validates schema
    - Confirms class imbalance

Tested against a synthetic SECOM-format dataset to verify parsing,
imputation, and variance-filtering logic before use on real data.

Author: <Your Name>
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

MISSING_THRESHOLD = 0.45       # drop sensor columns missing more than 45% of values
VARIANCE_THRESHOLD = 1e-6      # drop near-constant sensor columns


def load_raw_secom(data_path: Path, labels_path: Path) -> pd.DataFrame:
    """Load SECOM sensor readings and labels, merge into a single dataframe.

    Args:
        data_path: Path to secom.data (whitespace-delimited sensor matrix).
        labels_path: Path to secom_labels.data. Each row has 3 whitespace-
            separated fields: label, date (DD/MM/YYYY), time (HH:MM:SS) --
            e.g. "-1 19/07/2008 11:55:00".

    Returns:
        Merged dataframe with sensor columns, label, and timestamp.

    Raises:
        FileNotFoundError: If either input file is missing.
    """
    if not data_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Expected raw SECOM files at {data_path} and {labels_path}. "
            "Download from https://archive.ics.uci.edu/dataset/179/secom"
        )

    sensor_df = pd.read_csv(data_path, sep=r"\s+", header=None)
    sensor_df.columns = [f"Sensor_{i+1}" for i in range(sensor_df.shape[1])]

    labels_df = pd.read_csv(
        labels_path, sep=r"\s+", header=None, names=["Label_Raw", "Date", "Time"]
    )
    labels_df["Timestamp"] = pd.to_datetime(
        labels_df["Date"] + " " + labels_df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    # Remap UCI encoding (-1 = pass, 1 = fail) -> (0 = Pass, 1 = Fail)
    labels_df["Label"] = labels_df["Label_Raw"].map({-1: 0, 1: 1})

    df = pd.concat([labels_df[["Timestamp", "Label"]], sensor_df], axis=1)
    logger.info("Loaded raw SECOM data: %s rows, %s sensor columns", df.shape[0], sensor_df.shape[1])
    return df


def drop_high_missing_columns(df: pd.DataFrame, threshold: float = MISSING_THRESHOLD) -> pd.DataFrame:
    """Drop sensor columns whose missing-value fraction exceeds `threshold`.

    Args:
        df: Dataframe containing 'Timestamp', 'Label', and sensor columns.
        threshold: Maximum allowed fraction of missing values per column.

    Returns:
        Dataframe with high-missing sensor columns removed.
    """
    sensor_cols = df.columns.difference(["Timestamp", "Label"])
    missing_frac = df[sensor_cols].isna().mean()
    cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()
    logger.info("Dropping %d sensors with >%.0f%% missing values", len(cols_to_drop), threshold * 100)
    return df.drop(columns=cols_to_drop)


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute remaining missing sensor values (robust to skew/outliers).

    Args:
        df: Dataframe containing sensor columns with possible NaNs.

    Returns:
        Dataframe with all sensor NaNs filled by their column median.
    """
    sensor_cols = df.columns.difference(["Timestamp", "Label"])
    df[sensor_cols] = df[sensor_cols].apply(lambda col: col.fillna(col.median()))
    return df


def drop_near_zero_variance(df: pd.DataFrame, threshold: float = VARIANCE_THRESHOLD) -> pd.DataFrame:
    """Remove sensors with near-zero variance (no discriminative signal).

    Args:
        df: Dataframe containing sensor columns.
        threshold: Minimum variance a sensor must have to be kept.

    Returns:
        Dataframe with near-constant sensor columns removed.
    """
    sensor_cols = df.columns.difference(["Timestamp", "Label"])
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[sensor_cols])
    kept_cols = sensor_cols[selector.get_support()]
    dropped = len(sensor_cols) - len(kept_cols)
    logger.info("Dropping %d near-zero-variance sensors", dropped)
    return df[["Timestamp", "Label"] + list(kept_cols)]


def validate_schema(df: pd.DataFrame) -> None:
    """Run sanity checks; raise AssertionError if the dataset looks malformed.

    Args:
        df: Cleaned dataframe to validate.

    Raises:
        AssertionError: If any validation check fails.
    """
    assert df["Label"].isin([0, 1]).all(), "Label column contains unexpected values."
    sensor_cols = df.columns.difference(["Timestamp", "Label"])
    assert df[sensor_cols].isna().sum().sum() == 0, "Unexpected remaining nulls after imputation."
    assert pd.api.types.is_datetime64_any_dtype(df["Timestamp"]), "Timestamp column is not datetime."
    logger.info("Schema validation passed.")


def report_class_balance(df: pd.DataFrame) -> pd.Series:
    """Log and return the pass/fail class distribution.

    Args:
        df: Cleaned dataframe containing a 'Label' column.

    Returns:
        Series with 'Pass' and 'Fail' proportions.
    """
    balance = df["Label"].value_counts(normalize=True).rename({0: "Pass", 1: "Fail"})
    logger.info(
        "Class balance -> Pass: %.2f%% | Fail: %.2f%%",
        balance.get("Pass", 0) * 100,
        balance.get("Fail", 0) * 100,
    )
    return balance


def clean_secom_pipeline(data_path: Path, labels_path: Path, output_path: Path) -> pd.DataFrame:
    """End-to-end cleaning pipeline; writes cleaned CSV and returns the dataframe.

    Args:
        data_path: Path to secom.data.
        labels_path: Path to secom_labels.data.
        output_path: Where to save the cleaned CSV output.

    Returns:
        The cleaned dataframe.
    """
    df = load_raw_secom(data_path, labels_path)
    df = drop_high_missing_columns(df)
    df = impute_missing_values(df)
    df = drop_near_zero_variance(df)
    validate_schema(df)
    report_class_balance(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Clean dataset saved to %s (%d rows, %d columns)", output_path, *df.shape)
    return df


if __name__ == "__main__":
    clean_secom_pipeline(
        data_path=RAW_DATA_DIR / "secom.data",
        labels_path=RAW_DATA_DIR / "secom_labels.data",
        output_path=PROCESSED_DATA_DIR / "secom_clean.csv",
    )