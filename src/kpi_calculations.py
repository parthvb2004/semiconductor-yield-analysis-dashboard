"""kpi_calculations.py — Manufacturing yield KPI functions.

Author: <Your Name>
"""

from __future__ import annotations

import pandas as pd


def calculate_yield(df: pd.DataFrame, group_by: str | list[str] | None = None) -> pd.DataFrame:
    """Calculate Yield % and Failure Rate %, optionally grouped by dimension(s).

    Args:
        df: Dataframe containing a binary 'Label' column (0 = Pass, 1 = Fail).
        group_by: Column name(s) to group by (e.g. 'Machine_ID', ['Shift', 'Lot_ID']).

    Returns:
        Dataframe with Total_Wafers, Passed, Failed, Yield_Pct, Failure_Rate_Pct.
    """
    if group_by:
        grouped = df.groupby(group_by)
        result = grouped["Label"].agg(
            Total_Wafers="count",
            Failed=lambda s: (s == 1).sum(),
        )
        result["Passed"] = result["Total_Wafers"] - result["Failed"]
        result["Yield_Pct"] = (result["Passed"] / result["Total_Wafers"] * 100).round(2)
        result["Failure_Rate_Pct"] = (result["Failed"] / result["Total_Wafers"] * 100).round(2)
        return result.reset_index()

    # Ungrouped (overall) case -- return a single clean summary row
    total = len(df)
    failed = int((df["Label"] == 1).sum())
    passed = total - failed
    return pd.DataFrame({
        "Total_Wafers": [total],
        "Failed": [failed],
        "Passed": [passed],
        "Yield_Pct": [round(passed / total * 100, 2)],
        "Failure_Rate_Pct": [round(failed / total * 100, 2)],
    })


def calculate_rolling_yield(df: pd.DataFrame, date_col: str = "Timestamp", window: str = "7D") -> pd.DataFrame:
    """Calculate a rolling yield % over a time window (e.g., '7D', '30D').

    Args:
        df: Dataframe with a datetime column and a binary 'Label' column.
        date_col: Name of the datetime column to index/sort on.
        window: Pandas offset alias for the rolling window size.

    Returns:
        Dataframe with the date column and 'Rolling_Yield_Pct'.
    """
    ts = df.set_index(date_col).sort_index()
    ts["Pass_Flag"] = (ts["Label"] == 0).astype(int)
    rolling = ts["Pass_Flag"].rolling(window).mean() * 100
    return rolling.rename("Rolling_Yield_Pct").reset_index()


def calculate_cost_of_quality(df: pd.DataFrame, cost_per_failed_wafer: float = 1500.0) -> float:
    """Estimate simulated cost of poor quality from failed wafer count.

    Args:
        df: Dataframe containing a binary 'Label' column.
        cost_per_failed_wafer: Assumed simulated cost per failed wafer, in dollars.

    Returns:
        Total simulated cost of poor quality.
    """
    failed = (df["Label"] == 1).sum()
    return round(failed * cost_per_failed_wafer, 2)


def calculate_sensor_excursion_rate(df: pd.DataFrame, sensor: str, n_sigma: float = 3.0) -> float:
    """% of runs where `sensor` falls outside +/- n_sigma of its own mean (SPC-style control limit).

    Args:
        df: Dataframe containing the sensor column.
        sensor: Name of the sensor column to evaluate.
        n_sigma: Number of standard deviations defining the control limit.

    Returns:
        Percentage of runs falling outside the control limit.
    """
    mean, std = df[sensor].mean(), df[sensor].std()
    lower, upper = mean - n_sigma * std, mean + n_sigma * std
    excursions = ((df[sensor] < lower) | (df[sensor] > upper)).sum()
    return round(excursions / len(df) * 100, 2)