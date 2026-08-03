"""
synthetic_metadata.py
Generates realistic manufacturing metadata to enrich the cleaned SECOM dataset,
simulating a Manufacturing Execution System (MES) export.

Author: <Your Name>
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

MACHINES = [f"ETCH-{i:02d}" for i in range(1, 5)] + \
           [f"CMP-{i:02d}" for i in range(1, 4)] + \
           [f"LITHO-{i:02d}" for i in range(1, 4)]
MACHINE_WEIGHTS = rng.dirichlet(np.ones(len(MACHINES)) * 3)  # uneven but realistic utilization

OPERATORS = [f"OP-{i:03d}" for i in range(1, 21)]
PRODUCT_LINES = ["Logic-A", "Logic-B", "Memory-X", "Memory-Y"]
LINE_LOCATIONS = ["Fab1-BayA", "Fab1-BayB", "Fab1-BayC"]

MACHINE_TO_STEP = {
    **{m: "Etch" for m in MACHINES if m.startswith("ETCH")},
    **{m: "CMP" for m in MACHINES if m.startswith("CMP")},
    **{m: "Litho" for m in MACHINES if m.startswith("LITHO")},
}

SHIFT_HOURS = {
    "Day": set(range(6, 14)),
    "Swing": set(range(14, 22)),
    "Night": set(range(22, 24)) | set(range(0, 6)),
}


def _assign_shift(hour: int) -> str:
    """Map an hour-of-day (0-23) to a shift label."""
    for shift, hours in SHIFT_HOURS.items():
        if hour in hours:
            return shift
    return "Day"


def generate_metadata(df: pd.DataFrame, lot_size: int = 25) -> pd.DataFrame:
    """Generate synthetic manufacturing metadata aligned row-for-row with `df`.

    Args:
        df: Cleaned SECOM dataframe with a 'Timestamp' column.
        lot_size: Number of wafers grouped into each lot.

    Returns:
        Dataframe of metadata columns, aligned to df's chronological order.
    """
    n = len(df)
    df_sorted = df.sort_values("Timestamp").reset_index(drop=True)

    wafer_ids = [f"WFR-{i+1:06d}" for i in range(n)]
    lot_ids = [f"LOT-{(i // lot_size) + 1:05d}" for i in range(n)]

    machine_ids = rng.choice(MACHINES, size=n, p=MACHINE_WEIGHTS)
    process_steps = [MACHINE_TO_STEP[m] for m in machine_ids]
    operator_ids = rng.choice(OPERATORS, size=n)
    product_lines = rng.choice(PRODUCT_LINES, size=n, p=[0.35, 0.25, 0.25, 0.15])
    line_locations = rng.choice(LINE_LOCATIONS, size=n)
    shifts = df_sorted["Timestamp"].dt.hour.apply(_assign_shift)

    # Simulate cycle time: log-normal (most runs short, some long), clipped to a realistic range
    cycle_minutes = rng.lognormal(mean=3.2, sigma=0.5, size=n)
    cycle_minutes = np.clip(cycle_minutes, 10, 180)
    start_times = df_sorted["Timestamp"]
    end_times = start_times + pd.to_timedelta(cycle_minutes, unit="m")

    rework_flag = rng.random(n) < 0.04  # ~4% rework rate

    metadata = pd.DataFrame({
        "Wafer_ID": wafer_ids,
        "Lot_ID": lot_ids,
        "Machine_ID": machine_ids,
        "Process_Step": process_steps,
        "Operator_ID": operator_ids,
        "Shift": shifts.values,
        "Product_Line": product_lines,
        "Line_Location": line_locations,
        "Wafer_Start_Time": start_times.values,
        "Wafer_End_Time": end_times.values,
        "Cycle_Time_Minutes": cycle_minutes.round(1),
        "Rework_Flag": rework_flag,
    })

    logger.info("Generated synthetic metadata for %d wafers across %d lots", n, metadata["Lot_ID"].nunique())
    return metadata, df_sorted


def build_master_dataset(clean_secom_path: str, output_path: str) -> pd.DataFrame:
    """Merge cleaned SECOM data with generated metadata into a single master table.

    Args:
        clean_secom_path: Path to the cleaned SECOM CSV (from Phase 4).
        output_path: Where to save the merged master CSV.

    Returns:
        The merged master dataframe.
    """
    df = pd.read_csv(clean_secom_path, parse_dates=["Timestamp"])
    metadata, df_sorted = generate_metadata(df)
    master = pd.concat([metadata, df_sorted.drop(columns=["Timestamp"])], axis=1)
    master.to_csv(output_path, index=False)
    logger.info("Master dataset saved to %s (%d rows, %d columns)", output_path, *master.shape)
    return master


if __name__ == "__main__":
    build_master_dataset(
        clean_secom_path="data/processed/secom_clean.csv",
        output_path="data/processed/secom_master.csv",
    )