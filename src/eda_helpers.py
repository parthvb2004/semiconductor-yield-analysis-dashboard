"""
eda_helpers.py
Reusable EDA plotting and analysis functions for the cleaned SECOM dataset.

Author: <Your Name>
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_class_balance(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Bar chart of Pass vs Fail proportions."""
    ax = ax or plt.gca()
    counts = df["Label"].map({0: "Pass", 1: "Fail"}).value_counts(normalize=True) * 100
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, ax=ax,
                palette=["#2E8B7C", "#D96C4A"], legend=False)
    ax.set_ylabel("% of Wafers")
    ax.set_title("Class Balance — Pass vs Fail")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center")
    return ax


def plot_sensor_pass_fail(df: pd.DataFrame, sensor: str, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot a single sensor's distribution split by Pass/Fail label."""
    ax = ax or plt.gca()
    sns.kdeplot(
        data=df, x=sensor, hue=df["Label"].map({0: "Pass", 1: "Fail"}),
        fill=True, common_norm=False, alpha=0.4, ax=ax,
        palette={"Pass": "#2E8B7C", "Fail": "#D96C4A"},
    )
    ax.set_title(f"{sensor} — Distribution by Outcome")
    return ax


def top_variance_correlation_heatmap(df: pd.DataFrame, n: int = 30, ax: plt.Axes | None = None) -> plt.Axes:
    """Correlation heatmap for the n highest-variance sensors (full matrix is unreadable)."""
    ax = ax or plt.gca()
    sensor_cols = df.columns.difference(["Timestamp", "Label"])
    top_cols = df[sensor_cols].var().sort_values(ascending=False).head(n).index
    corr = df[top_cols].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title(f"Correlation — Top {n} Highest-Variance Sensors")
    return ax


def plot_failure_rate_over_time(df: pd.DataFrame, freq: str = "D", ax: plt.Axes | None = None) -> plt.Axes:
    """Line chart of failure rate % over time, resampled at `freq`."""
    ax = ax or plt.gca()
    ts = df.set_index("Timestamp").sort_index()
    failure_rate = ts["Label"].resample(freq).mean() * 100
    failure_rate.plot(ax=ax, color="#D96C4A", marker="o")
    ax.set_ylabel("Failure Rate %")
    ax.set_title(f"Failure Rate Over Time (resampled: {freq})")
    return ax


def plot_pca_scatter(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """2D PCA scatter of sensor readings, colored by Pass/Fail."""
    ax = ax or plt.gca()
    sensor_cols = df.columns.difference(["Timestamp", "Label"])
    X_scaled = StandardScaler().fit_transform(df[sensor_cols])
    components = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    labels = df["Label"].map({0: "Pass", 1: "Fail"})
    sns.scatterplot(
        x=components[:, 0], y=components[:, 1], hue=labels, alpha=0.6, ax=ax,
        palette={"Pass": "#2E8B7C", "Fail": "#D96C4A"},
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA — 2D Sensor Space by Outcome")
    return ax


def find_most_separable_sensors(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Rank sensors by absolute difference in mean value between Pass and Fail groups
    (a quick, simple separability signal ahead of formal feature selection in Phase 6).
    """
    sensor_cols = df.columns.difference(["Timestamp", "Label"])
    pass_means = df.loc[df["Label"] == 0, sensor_cols].mean()
    fail_means = df.loc[df["Label"] == 1, sensor_cols].mean()
    pooled_std = df[sensor_cols].std().replace(0, 1e-9)
    separation = ((fail_means - pass_means).abs() / pooled_std).sort_values(ascending=False)
    return separation.head(top_n).rename("Standardized_Mean_Gap").reset_index().rename(columns={"index": "Sensor"})