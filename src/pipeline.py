"""
pipeline.py
Orchestrates the full SECOM preprocessing pipeline end-to-end:
    raw data -> cleaned data -> synthetic metadata -> master dataset -> KPI tables -> feature shortlist

Run with:
    python -m src.pipeline

Author: <Your Name>
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data_cleaning import clean_secom_pipeline
from src.feature_selection import select_features
from src.kpi_calculations import calculate_yield
from src.synthetic_metadata import build_master_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def run_pipeline() -> None:
    """Execute the full preprocessing pipeline, stage by stage, with error handling."""
    try:
        logger.info("STAGE 1/4 -- Cleaning raw SECOM data")
        clean_secom_pipeline(
            data_path=RAW_DIR / "secom.data",
            labels_path=RAW_DIR / "secom_labels.data",
            output_path=PROCESSED_DIR / "secom_clean.csv",
        )

        logger.info("STAGE 2/4 -- Generating synthetic metadata & building master dataset")
        master_df = build_master_dataset(
            clean_secom_path=str(PROCESSED_DIR / "secom_clean.csv"),
            output_path=str(PROCESSED_DIR / "secom_master.csv"),
        )

        logger.info("STAGE 3/4 -- Calculating KPI tables")
        overall_yield = calculate_yield(master_df)
        yield_by_machine = calculate_yield(master_df, group_by="Machine_ID")
        yield_by_shift = calculate_yield(master_df, group_by="Shift")
        overall_yield.to_csv(PROCESSED_DIR / "kpi_overall_yield.csv", index=False)
        yield_by_machine.to_csv(PROCESSED_DIR / "kpi_yield_by_machine.csv", index=False)
        yield_by_shift.to_csv(PROCESSED_DIR / "kpi_yield_by_shift.csv", index=False)

        logger.info("STAGE 4/4 -- Running feature selection")
        clean_df = pd.read_csv(PROCESSED_DIR / "secom_clean.csv")
        top_features = select_features(clean_df)
        top_features.to_csv(PROCESSED_DIR / "top_features.csv", index=False)

        logger.info("Pipeline completed successfully. All outputs in %s", PROCESSED_DIR)

    except FileNotFoundError as exc:
        logger.error(
            "Missing input file -- check data/raw/ contains secom.data and secom_labels.data: %s",
            exc,
        )
        raise
    except Exception:
        logger.exception("Pipeline failed unexpectedly.")
        raise


if __name__ == "__main__":
    run_pipeline()