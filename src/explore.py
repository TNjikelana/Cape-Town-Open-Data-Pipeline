"""Stage 3: exploration and visualisation for the System Energy dataset.

Takes the output of ``transform.transform_data`` and:
    1. prints a written summary (descriptive statistics and key findings), then
    2. draws seven charts, each answering one question about the data.

    Chart 1  Trend            How has total output changed since 2017?
    Chart 2  Seasonality      Does output follow a seasonal cycle?
    Chart 3  Growth check     Does the supplied growth column match our own year-on-year calculation?
    Chart 4  Distributions    What did the power transform do to skewed data?
    Chart 5  Correlations     Which measures move together?
    Chart 6  Energy mix       How much output comes from sources other than Eskom?
    Chart 7  Annual trend     How does average output compare across complete years?

Each chart is saved as a PNG and, by default, shown on screen. Close a chart's
window to move on to the next one.
"""

from itertools import combinations
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TOTAL = "Total_System_OUT_GWh"
BASE_COLUMNS = (
    "Eskom_SPS_GWh",
    "SPS_Import_GWh",
    "SPS_OUT_GWh",
    "AGT_OUT_GWh",
    "RGT_OUT_GWh",
    TOTAL,
)
REQUIRED_COLUMNS = (
    "Date", "Month", "Year", "Season", "Rolling_12M_GWh", "System_Energy_Growth",
    "YoY_Change_Pct", "Non_Eskom_Share_Pct", "flag_growth_yoy_mismatch",
    "flag_growth_placeholder", *BASE_COLUMNS,
)

SEASON_ORDER = ["Summer", "Autumn", "Winter", "Spring"]
SEASON_COLOURS = {
    "Summer": "#F4A261",
    "Autumn": "#B08968",
    "Winter": "#4C78A8",
    "Spring": "#6BAA75",
}
MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
LINE_COLOUR = "#4C78A8"
ACCENT_COLOUR = "#E4572E"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _new_figure(width: float = 11, height: float = 6):
    return plt.subplots(figsize=(width, height), constrained_layout=True)


def _finish(fig, path: Path, show: bool) -> None:
    """Save the chart, optionally show it on screen, then free its memory."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path.name}")
    if show:
        plt.show()
    plt.close(fig)


def _label(column: str) -> str:
    """Readable axis label from a column name, e.g. 'Eskom_SPS_GWh' -> 'Eskom SPS'."""
    return column.removesuffix("_GWh").replace("_", " ")


def _year_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# --------------------------------------------------------------------------
# Written summary
# --------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    """Descriptive statistics and key findings, computed from the data."""
    line = "=" * 64
    print(f"\n{line}\nEXPLORATION SUMMARY\n{line}")
    print(
        f"Period: {df['Date'].min():%B %Y} to {df['Date'].max():%B %Y} "
        f"({len(df)} monthly records)\n"
    )

    stats = df[list(BASE_COLUMNS)].describe().T[["mean", "std", "min", "max"]]
    stats["skew"] = df[list(BASE_COLUMNS)].skew()
    stats.index = [_label(name) for name in stats.index]
    print("Descriptive statistics (GWh):")
    print(stats.round(3).to_string())

    print("\nKey findings:")
    monthly = df.groupby("Month")[TOTAL].mean()
    peak, low = monthly.idxmax(), monthly.idxmin()
    print(
        f"  - Seasonality: average output peaks in {MONTH_LABELS[peak - 1]} "
        f"({monthly[peak]:.0f} GWh) and is lowest in {MONTH_LABELS[low - 1]} "
        f"({monthly[low]:.0f} GWh)."
    )
    seasonal = df.groupby("Season")[TOTAL].mean()
    gap = (seasonal["Winter"] / seasonal["Summer"] - 1) * 100
    print(f"  - Winter months average {gap:.1f}% more output than summer months.")

    yearly = df.groupby("Year")[TOTAL].agg(["mean", "size"])
    full_years = yearly[yearly["size"] == 12]["mean"]
    if len(full_years) >= 2:
        first, last = full_years.index[0], full_years.index[-1]
        change = (full_years[last] / full_years[first] - 1) * 100
        print(
            f"  - Trend: average monthly output went from {full_years[first]:.0f} GWh "
            f"in {first} to {full_years[last]:.0f} GWh in {last} ({change:+.1f}%), "
            f"comparing full calendar years only."
        )

    mismatches = int(df["flag_growth_yoy_mismatch"].sum())
    placeholders = int(df["flag_growth_placeholder"].sum())
    print(
        f"  - Data quality: the supplied growth column disagrees with our recomputed "
        f"year-on-year change in {mismatches} months, {placeholders} of which are the "
        f"constant -1.00% placeholder."
    )

    correlations = df[list(BASE_COLUMNS)].corr()[TOTAL].drop(TOTAL)
    print("  - Correlation of each measure with total output:")
    for name, value in correlations.sort_values(ascending=False).items():
        print(f"      {_label(name):<12} r = {value:+.3f}")

    # Eskom and the total are almost the same series, so look at the other pairs.
    matrix = df[list(BASE_COLUMNS)].corr()
    other_pairs = [
        pair for pair in combinations(BASE_COLUMNS, 2)
        if set(pair) != {"Eskom_SPS_GWh", TOTAL}
    ]
    first, second = max(other_pairs, key=lambda pair: abs(matrix.loc[pair]))
    print(
        f"  - Strongest relationship between the other measures: {_label(first)} "
        f"and {_label(second)} (r = {matrix.loc[first, second]:+.2f})."
    )

    transformed = [c for c in df.columns if c.endswith("_YJ")]
    for column in transformed:
        original = column.removesuffix("_YJ")
        print(
            f"  - Power transform: {_label(original)} skewness "
            f"{df[original].skew():.2f} -> {df[column].skew():.2f}."
        )
    print(line)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def plot_trend(df: pd.DataFrame):
    """Chart 1: total output with its 12-month rolling average."""
    fig, ax = _new_figure()
    ax.plot(df["Date"], df[TOTAL], color=LINE_COLOUR, alpha=0.55, label="Monthly output")
    ax.plot(
        df["Date"], df["Rolling_12M_GWh"], color=ACCENT_COLOUR, linewidth=3,
        label="12-month rolling average",
    )
    ax.set_title("Total system output over time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total system output (GWh)")
    _year_axis(ax)
    ax.legend(loc="upper right")
    return fig


def plot_seasonality(df: pd.DataFrame):
    """Chart 2: output by calendar month, coloured by season."""
    fig, ax = _new_figure()
    sns.boxplot(
        data=df, x="Month", y=TOTAL, hue="Season", hue_order=SEASON_ORDER,
        palette=SEASON_COLOURS, dodge=False, ax=ax,
    )
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_title("Output by month: a clear winter peak")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total system output (GWh)")
    ax.legend(title="Season", loc="upper right")
    return fig


def plot_growth_check(df: pd.DataFrame):
    """Chart 3: supplied growth vs. recomputed year-on-year change."""
    fig, ax = _new_figure()
    ax.axhline(0, color="grey", linewidth=1)

    # Merge runs of consecutive flagged months into single shaded bands.
    flagged = df["flag_growth_yoy_mismatch"]
    run_id = flagged.ne(flagged.shift()).cumsum()
    labelled = False
    for _, run in df[flagged].groupby(run_id[flagged]):
        ax.axvspan(
            run["Date"].min() - pd.Timedelta(days=15),
            run["Date"].max() + pd.Timedelta(days=15),
            color=ACCENT_COLOUR, alpha=0.15, linewidth=0,
            label=None if labelled else "Disagree by more than 0.1 points",
        )
        labelled = True

    ax.plot(df["Date"], df["System_Energy_Growth"], color=ACCENT_COLOUR, label="Supplied growth")
    ax.plot(
        df["Date"], df["YoY_Change_Pct"], color=LINE_COLOUR, linewidth=2,
        label="Recomputed year-on-year change",
    )
    ax.set_title("Supplied growth vs. recomputed year-on-year change")
    ax.set_xlabel("Year")
    ax.set_ylabel("Change vs. same month a year earlier (%)")
    _year_axis(ax)
    ax.legend(loc="upper left")
    return fig


def plot_distributions(df: pd.DataFrame):
    """Chart 4: skewed columns before and after the Yeo-Johnson transform."""
    pairs = [(c.removesuffix("_YJ"), c) for c in df.columns if c.endswith("_YJ")]
    if not pairs:
        return None
    fig, axes = plt.subplots(
        len(pairs), 2, figsize=(11, 4.2 * len(pairs)), constrained_layout=True,
        squeeze=False,
    )
    for row, (original, transformed) in zip(axes, pairs):
        for ax, column, stage, colour in (
            (row[0], original, "Before", ACCENT_COLOUR),
            (row[1], transformed, "After", LINE_COLOUR),
        ):
            sns.histplot(df[column], bins=20, kde=True, color=colour, ax=ax)
            ax.set_title(f"{_label(original)}: {stage} (skewness {df[column].skew():.2f})")
            ax.set_xlabel("GWh" if stage == "Before" else "Transformed value")
            ax.set_ylabel("Number of months")
    fig.suptitle("Effect of the Yeo-Johnson transform on skewed columns", fontsize=15)
    return fig


def plot_correlations(df: pd.DataFrame):
    """Chart 5: correlation heatmap of the main measures."""
    columns = [*BASE_COLUMNS, "Non_Eskom_Share_Pct", "System_Energy_Growth"]
    corr = df[columns].corr()
    corr.index = corr.columns = [_label(c) for c in corr.columns]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = _new_figure(10, 8)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
        square=True, linewidths=0.5, cbar_kws={"label": "Correlation (r)"}, ax=ax,
    )
    ax.grid(False)
    ax.set_title("Which measures move together?")
    return fig


def plot_non_eskom_share(df: pd.DataFrame):
    """Chart 6: monthly share of total output from non-Eskom sources."""
    fig, ax = _new_figure()
    ax.plot(
        df["Date"],
        df["Non_Eskom_Share_Pct"],
        color=LINE_COLOUR,
        linewidth=2,
        label="Monthly non-Eskom share",
    )
    ax.axhline(
        df["Non_Eskom_Share_Pct"].mean(),
        color=ACCENT_COLOUR,
        linestyle="--",
        label=f"Period average ({df['Non_Eskom_Share_Pct'].mean():.1f}%)",
    )
    ax.set_title("Share of system output from non-Eskom sources")
    ax.set_xlabel("Year")
    ax.set_ylabel("Non-Eskom share of total output (%)")
    ax.set_ylim(bottom=0)
    _year_axis(ax)
    ax.legend(loc="upper right")
    return fig


def plot_annual_average(df: pd.DataFrame):
    """Chart 7: average monthly output for complete calendar years only."""
    yearly = df.groupby("Year")[TOTAL].agg(["mean", "count"])
    complete_years = yearly.loc[yearly["count"].eq(12)]
    if complete_years.empty:
        return None

    fig, ax = _new_figure()
    bars = ax.bar(
        complete_years.index.astype(str),
        complete_years["mean"],
        color=LINE_COLOUR,
    )
    ax.bar_label(bars, fmt="%.0f", padding=3)

    partial_years = yearly.index[yearly["count"].ne(12)].astype(str).tolist()
    excluded = ", ".join(partial_years) if partial_years else "none"
    ax.set_title("Average monthly system output by complete calendar year")
    ax.set_xlabel(f"Year (partial years excluded: {excluded})")
    ax.set_ylabel("Average monthly total output (GWh)")
    ax.set_ylim(bottom=0)
    return fig


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run_exploration(df: pd.DataFrame, figures_dir: Path, show: bool = True) -> None:
    """Print the summary, then save (and optionally show) the seven charts."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {', '.join(missing)}. "
            "Run the transformation stage first."
        )

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.15)

    print_summary(df)

    print("\nCharts:")
    charts = (
        ("01_trend.png", plot_trend),
        ("02_seasonality.png", plot_seasonality),
        ("03_growth_check.png", plot_growth_check),
        ("04_distributions.png", plot_distributions),
        ("05_correlations.png", plot_correlations),
        ("06_non_eskom_share.png", plot_non_eskom_share),
        ("07_annual_average.png", plot_annual_average),
    )
    for filename, plot in charts:
        fig = plot(df)
        if fig is None:
            print(f"  Skipped {filename} (nothing to show)")
            continue
        _finish(fig, figures_dir / filename, show)