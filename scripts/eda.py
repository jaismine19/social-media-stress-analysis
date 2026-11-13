#!/usr/bin/env python3
"""
=====================================================================
EXPLORATORY DATA ANALYSIS (EDA)  (easy-to-read version)
=====================================================================
Reads : data/processed/social_media_stress_clean.csv
Saves : every chart to reports/figures/

HOW TO RUN (from the project root):
    python3 scripts/eda.py

WHAT THIS SCRIPT DOES
---------------------
EDA means "getting to know the data". It answers questions like:
  * How is each variable distributed? (histograms, boxplots)
  * Are there missing values? (heatmap)
  * Are the variables skewed? (skewness / kurtosis)
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
CLEAN_FILE = PROJECT / "data" / "processed" / "social_media_stress_clean.csv"
FIG_DIR = PROJECT / "reports" / "figures"

sns.set_theme(style="whitegrid")  # nicer chart style

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


def save_chart(fig, name):
    """Save one chart to reports/figures/ and close it."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=120)
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
# SECTION 2 - MISSING VALUES HEATMAP
# ================================================================
def missing_values_heatmap(df):
    """
    Draw a heatmap where each cell is one value: white = present,
    coloured = missing. On live data this shows WHERE the gaps are.
    """
    print("\n" + "=" * 60)
    print("SECTION 2 - MISSING VALUES")
    print("=" * 60)
    total_missing = df.isna().sum().sum()
    print(f"Total missing values in the dataset: {total_missing}")

    # Which exact rows have gaps?
    rows_with_gaps = df[df.isna().any(axis=1)]
    if rows_with_gaps.empty:
        print("No rows have missing values - every cell is filled.")
    else:
        print(f"\nRows with missing values ({len(rows_with_gaps)} found):")
        for idx, row in rows_with_gaps.iterrows():
            cols = [c for c in df.columns if pd.isna(row[c])]
            label = row["user_id"] if "user_id" in df.columns else f"row {idx}"
            print(f"  {label}: missing in {cols}")

    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, ax=ax)
    ax.set_title("Missing values heatmap (white = present, coloured = missing)")
    save_chart(fig, "eda_missing_values.png")
    return df


# ================================================================
# SECTION 3 - UNIVARIATE ANALYSIS (one variable at a time)
# ================================================================
def univariate_numeric(df):
    """
    For every numeric column: a histogram (shape of the data) next
    to a boxplot (centre, spread and outliers).
    """
    print("\n" + "=" * 60)
    print("SECTION 3 - UNIVARIATE (numeric columns)")
    print("=" * 60)
    for col in NUMERIC_COLS:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        sns.histplot(df[col], kde=True, ax=ax1)
        ax1.set_title(f"{col} - distribution")
        sns.boxplot(x=df[col], ax=ax2)
        ax2.set_title(f"{col} - boxplot")
        save_chart(fig, f"eda_{col}.png")

        # print the key numbers in plain English
        mean = df[col].mean()
        median = df[col].median()
        skew = df[col].skew()
        print(f"  {col}: mean {mean:.2f}, median {median:.2f}, skewness {skew:+.2f}")
        if skew > 1:
            print(f"      -> strongly right-skewed (long tail of high values)")
        elif skew < -1:
            print(f"      -> strongly left-skewed (long tail of low values)")
        else:
            print(f"      -> roughly symmetric or mildly skewed")
    return df


def univariate_categorical(df):
    """Count each category and draw a bar chart."""
    print("\n" + "=" * 60)
    print("SECTION 3b - UNIVARIATE (categorical columns)")
    print("=" * 60)
    for col in CATEGORICAL_COLS:
        counts = df[col].value_counts()
        print(f"\n  {col} counts:")
        for label, n in counts.items():
            print(f"    {label}: {n} ({n / len(df) * 100:.1f}%)")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x=col, order=counts.index, ax=ax)
        ax.set_title(f"{col} - counts")
        plt.xticks(rotation=20)
        save_chart(fig, f"eda_{col}_counts.png")
    return df


# ================================================================
# SECTION 4 - BIVARIATE ANALYSIS (two variables together)
# ================================================================
def bivariate(df):
    """
    Scatter plots for two numeric variables, boxplots to compare a
    numeric variable across groups.
    """
    print("\n" + "=" * 60)
    print("SECTION 4 - BIVARIATE (relationships)")
    print("=" * 60)

    # Scatter plots
    scatter_pairs = [
        ("daily_screen_time_hours", "stress_level", "social_media_platform",
         "Screen time vs stress (coloured by platform)"),
        ("daily_screen_time_hours", "sleep_quality", None,
         "Screen time vs sleep quality"),
        ("exercise_frequency_per_week", "happiness_index", None,
         "Exercise vs happiness"),
    ]
    for x_col, y_col, hue, title in scatter_pairs:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, alpha=0.6, ax=ax)
        ax.set_title(title)
        if hue:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        save_chart(fig, f"eda_{x_col}_vs_{y_col}.png")
        corr = df[x_col].corr(df[y_col])
        print(f"  {x_col} vs {y_col}: correlation = {corr:+.2f}")

    # Boxplots across groups
    group_pairs = [
        ("screen_time_group", "stress_level", "Stress by screen-time group"),
        ("social_media_platform", "sleep_quality", "Sleep quality by platform"),
        ("gender", "stress_level", "Stress by gender"),
    ]
    for group_col, value_col, title in group_pairs:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x=group_col, y=value_col, ax=ax)
        ax.set_title(title)
        plt.xticks(rotation=20)
        save_chart(fig, f"eda_{value_col}_by_{group_col}.png")
        # print group means
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

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=ax)
    ax.set_title("Correlation heatmap (numeric columns)")
    save_chart(fig, "eda_correlation_heatmap.png")

    # Find the strongest links (ignore a variable with itself)
    print("\n  Strongest relationships:")
    pairs = []
    for i in range(len(NUMERIC_COLS)):
        for j in range(i + 1, len(NUMERIC_COLS)):
            col_a, col_b = NUMERIC_COLS[i], NUMERIC_COLS[j]
            pairs.append((abs(corr.loc[col_a, col_b]), col_a, col_b, corr.loc[col_a, col_b]))
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
    df = pd.read_csv(CLEAN_FILE)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from the cleaned file.\n")

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
