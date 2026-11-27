#!/usr/bin/env python3
"""
=====================================================================
EXPLORATORY DATA ANALYSIS (EDA)  (easy-to-read version)
=====================================================================
Reads : data/processed/social_media_stress_augmented.csv  (1,500 rows)
Saves : every chart to reports/figures/

NOTE: the EDA analyses the SAME augmented dataset the models use
(500 real + 1,000 synthetic bootstrap rows), so the insights, the
decision tree and the regression all line up on one dataset.

HOW TO RUN (from the project root):
    python3 scripts/eda.py

WHAT THIS SCRIPT DOES
---------------------
EDA means "getting to know the data". It answers questions like:
  * How is each variable distributed? (histograms, boxplots)
  * Are there missing values? (counts + row-level report)
  * Are the variables skewed? (skewness)
  * Do variables move together? (scatter plots, correlation heatmap)
  * Are there outliers? (IQR scan)

Every section prints its findings in plain English and saves its
charts as PNG files so they can be pasted into the final report.
=====================================================================
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files instead of opening windows

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------
# File locations
# ---------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
AUG_FILE = PROJECT / "data" / "processed" / "social_media_stress_augmented.csv"
FIG_DIR = PROJECT / "reports" / "figures"

# ---------------------------------------------------------------
# One consistent, professional look for every chart
# ---------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})

NAVY = "#1F4E79"
TEAL = "#2E86AB"
ORANGE = "#E67E22"
GREEN = "#27AE60"
RED = "#C0392B"
PURPLE = "#8E44AD"
PALETTE = [NAVY, TEAL, ORANGE, GREEN, RED, PURPLE]

NUMERIC_COLS = [
    "age",
    "daily_screen_time_hours",
    "sleep_quality",
    "stress_level",
    "days_without_social_media",
    "exercise_frequency_per_week",
    "happiness_index",
]

CATEGORICAL_COLS = ["gender", "social_media_platform", "screen_time_group"]


def styled_title(ax, title, subtitle=None):
    """Add a bold title plus a small grey subtitle under it."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                ha="center", fontsize=9, color="#595959", style="italic")


def save_chart(fig, name):
    """Save one chart to reports/figures/ and close it."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    saved -> reports/figures/{name}")


# ================================================================
# SECTION 1 - OVERVIEW
# ================================================================
def overview(df):
    print("=" * 60)
    print("SECTION 1 - OVERVIEW")
    print("=" * 60)
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print("\nColumn types:")
    print(df.dtypes.to_string())
    print("\nNumeric summary:")
    print(df.describe().round(2).to_string())
    return df


# ================================================================
# SECTION 2 - MISSING VALUES
# ================================================================
def missing_values_heatmap(df):
    """
    Count empty cells and report WHICH rows have them.
    The heatmap is only drawn when gaps actually exist - on a fully
    clean dataset an all-white heatmap carries no information, so we
    print the check result as text instead.
    """
    print("\n" + "=" * 60)
    print("SECTION 2 - MISSING VALUES")
    print("=" * 60)
    total_missing = df.isna().sum().sum()
    print(f"Total missing values in the dataset: {total_missing}")

    # Which exact rows have gaps? Check every cell, one by one.
    found_any = False
    for index, row in df.iterrows():
        for col in df.columns:
            if pd.isna(row[col]):          # is this cell empty?
                found_any = True
                if "user_id" in df.columns:
                    label = row["user_id"]
                else:
                    label = index
                print(f"  {label}: missing in '{col}'")
    if not found_any:
        print("No rows have missing values - every cell is filled.")

    if total_missing > 0:
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(df.isna(), cbar=False, yticklabels=False,
                    cmap=["#1F4E79", "#E67E22"], ax=ax)
        styled_title(ax, "Missing values map", "blue = filled, orange = missing")
        save_chart(fig, "eda_missing_values.png")
    else:
        print("(heatmap skipped - nothing to show on a clean dataset)")
    return df


# ================================================================
# SECTION 3 - UNIVARIATE ANALYSIS (one variable at a time)
# ================================================================
def univariate_numeric(df):
    """
    For every numeric column: a histogram with mean/median markers
    next to a boxplot with its quartiles labelled.
    """
    print("\n" + "=" * 60)
    print("SECTION 3 - UNIVARIATE (numeric columns)")
    print("=" * 60)
    for col in NUMERIC_COLS:
        mean = df[col].mean()
        median = df[col].median()
        skew = df[col].skew()
        q1, q3 = df[col].quantile([0.25, 0.75])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                       gridspec_kw={"width_ratios": [1.4, 1]})
        # histogram with density curve and mean/median markers
        sns.histplot(df[col], kde=True, color=NAVY, alpha=0.85,
                     edgecolor="white", linewidth=0.5, ax=ax1)
        ax1.axvline(mean, color=ORANGE, ls="--", lw=1.6, label=f"mean {mean:.2f}")
        ax1.axvline(median, color=GREEN, ls=":", lw=1.8, label=f"median {median:.2f}")
        ax1.legend(loc="upper right", fontsize=9, frameon=False)
        styled_title(ax1, f"{col}", f"distribution · skewness {skew:+.2f}")
        ax1.set_xlabel(col)

        # boxplot with quartiles annotated
        sns.boxplot(x=df[col], color=TEAL, width=0.35, ax=ax2)
        for value, label, color in [(q1, f"Q1 {q1:.1f}", "white"),
                                    (median, f"median {median:.1f}", GREEN),
                                    (q3, f"Q3 {q3:.1f}", "white")]:
            ax2.annotate(label, xy=(value, -0.32), xytext=(value, -0.44),
                         ha="center", fontsize=8.5, color=color)
        styled_title(ax2, f"{col}", "boxplot: middle 50% and spread")
        ax2.set_xlabel(col)

        save_chart(fig, f"eda_{col}.png")

        # print the key numbers in plain English
        print(f"  {col}: mean {mean:.2f}, median {median:.2f}, skewness {skew:+.2f}")
        if skew > 1:
            print(f"      -> strongly right-skewed (long tail of high values)")
        elif skew < -1:
            print(f"      -> strongly left-skewed (long tail of low values)")
        else:
            print(f"      -> roughly symmetric or mildly skewed")
    return df


def univariate_categorical(df):
    """Count each category, label the bars, draw a sorted bar chart."""
    print("\n" + "=" * 60)
    print("SECTION 3b - UNIVARIATE (categorical columns)")
    print("=" * 60)
    for col in CATEGORICAL_COLS:
        counts = df[col].value_counts()
        print(f"\n  {col} counts:")
        for label, n in counts.items():
            print(f"    {label}: {n} ({n / len(df) * 100:.1f}%)")

        fig, ax = plt.subplots(figsize=(8, 4.4))
        bars = sns.barplot(x=counts.values, y=counts.index,
                           color=NAVY, alpha=0.9, ax=ax)
        for bar, n in zip(ax.patches, counts.values):
            ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
                    f"{n}  ({n / len(df) * 100:.0f}%)", va="center",
                    fontsize=9.5, color="#404040")
        ax.set_xlim(0, max(counts.values) * 1.18)
        styled_title(ax, f"{col}", "counts and share of the 1,500 users")
        ax.set_xlabel("users")
        ax.set_ylabel("")
        save_chart(fig, f"eda_{col}_counts.png")
    return df


# ================================================================
# SECTION 4 - BIVARIATE ANALYSIS (two variables together)
# ================================================================
def bivariate(df):
    """
    Scatter plots with a fitted trend line for two numeric variables,
    boxplots (with group means) to compare across groups.
    """
    print("\n" + "=" * 60)
    print("SECTION 4 - BIVARIATE (relationships)")
    print("=" * 60)

    # one colour per platform (fixed, colourblind-friendly order)
    platforms = sorted(df["social_media_platform"].unique())
    platform_colors = dict(zip(platforms, PALETTE))

    scatter_pairs = [
        ("daily_screen_time_hours", "stress_level", "social_media_platform",
         "Screen time vs stress", "coloured by platform"),
        ("daily_screen_time_hours", "sleep_quality", None,
         "Screen time vs sleep quality", "with fitted trend line"),
        ("exercise_frequency_per_week", "happiness_index", None,
         "Exercise vs happiness", "with fitted trend line"),
    ]
    for x_col, y_col, hue, title, subtitle in scatter_pairs:
        fig, ax = plt.subplots(figsize=(8, 5))
        if hue:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue,
                            palette=platform_colors, alpha=0.6, s=45,
                            edgecolor="none", ax=ax)
            sns.regplot(data=df, x=x_col, y=y_col, scatter=False,
                        color=RED, line_kws={"lw": 2}, ax=ax)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                      fontsize=8, title=None, frameon=False)
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col, color=TEAL,
                            alpha=0.55, s=45, edgecolor="none", ax=ax)
            sns.regplot(data=df, x=x_col, y=y_col, scatter=False,
                        color=RED, line_kws={"lw": 2}, ax=ax)
        corr = df[x_col].corr(df[y_col])
        styled_title(ax, title, f"{subtitle} · r = {corr:+.2f}")
        save_chart(fig, f"eda_{x_col}_vs_{y_col}.png")
        print(f"  {x_col} vs {y_col}: correlation = {corr:+.2f}")

    # Boxplots across groups, with the group means annotated
    group_pairs = [
        ("screen_time_group", "stress_level", "Stress by screen-time group"),
        ("social_media_platform", "sleep_quality", "Sleep quality by platform"),
        ("gender", "stress_level", "Stress by gender"),
    ]
    for group_col, value_col, title in group_pairs:
        groups = sorted(df[group_col].unique())
        fig, ax = plt.subplots(figsize=(8, 4.6))
        sns.boxplot(data=df, x=group_col, y=value_col, order=groups,
                    palette=PALETTE[:len(groups)], width=0.55, ax=ax)
        means = df.groupby(group_col)[value_col].mean()
        for i, g in enumerate(groups):
            ax.text(i, means[g] + 0.18, f"μ {means[g]:.2f}", ha="center",
                    fontsize=9.5, color="#404040", fontweight="bold")
        ax.set_ylim(bottom=df[value_col].min() - 0.9)
        styled_title(ax, title, "boxes = middle 50% · μ = group mean")
        plt.xticks(rotation=20)
        save_chart(fig, f"eda_{value_col}_by_{group_col}.png")
        print(f"\n  Mean {value_col} by {group_col}:")
        for label, mean in df.groupby(group_col)[value_col].mean().round(2).items():
            print(f"    {label}: {mean}")
    return df


# ================================================================
# SECTION 5 - CORRELATION
# ================================================================
def correlations(df):
    """
    Correlation measures how two numbers move together:
      +1 = move together, -1 = move opposite, 0 = no link.
    """
    print("\n" + "=" * 60)
    print("SECTION 5 - CORRELATION")
    print("=" * 60)
    corr = df[NUMERIC_COLS].corr()

    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.6, linecolor="white",
                cbar_kws={"shrink": 0.8, "label": "correlation"}, ax=ax)
    styled_title(ax, "Correlation matrix", "red = opposites · blue = move together")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    save_chart(fig, "eda_correlation_heatmap.png")

    # Find the strongest links (ignore a variable with itself)
    print("\n  Strongest relationships:")
    pairs = []
    for i in range(len(NUMERIC_COLS)):
        for j in range(i + 1, len(NUMERIC_COLS)):
            col_a, col_b = NUMERIC_COLS[i], NUMERIC_COLS[j]
            pairs.append((abs(corr.loc[col_a, col_b]), col_a, col_b,
                          corr.loc[col_a, col_b]))
    for strength, col_a, col_b, value in sorted(pairs, reverse=True)[:5]:
        direction = "positive" if value > 0 else "negative"
        print(f"    {col_a} <-> {col_b}: {value:+.2f} ({direction})")
    return df


# ================================================================
# SECTION 6 - OUTLIER SCAN
# ================================================================
def outlier_scan(df):
    """
    Same IQR rule as the cleaning script: report values beyond
    1.5 x IQR from the quartiles. We look, we do not delete.
    """
    print("\n" + "=" * 60)
    print("SECTION 6 - OUTLIER SCAN (IQR rule)")
    print("=" * 60)
    for col in NUMERIC_COLS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low_fence = q1 - 1.5 * iqr
        high_fence = q3 + 1.5 * iqr
        n_out = ((df[col] < low_fence) | (df[col] > high_fence)).sum()
        print(f"  {col}: {n_out} potential outlier(s) "
              f"(outside {low_fence:.1f} - {high_fence:.1f})")
    return df


# ================================================================
# MAIN
# ================================================================
def main():
    print("STARTING THE EDA")
    df = pd.read_csv(AUG_FILE)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from the augmented file.\n")

    df = overview(df)
    df = missing_values_heatmap(df)
    df = univariate_numeric(df)
    df = univariate_categorical(df)
    df = bivariate(df)
    df = correlations(df)
    df = outlier_scan(df)

    print("\nEDA FINISHED - all charts are in reports/figures/")


if __name__ == "__main__":
    main()
