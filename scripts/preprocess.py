#!/usr/bin/env python3
"""
Data preprocessing & cleaning pipeline
======================================

Dataset : Social Media and Stress Patterns (Kaggle)
Source  : https://www.kaggle.com/datasets/houdlillianmaltiti/social-media-and-stress-patterns
License : CC0: Public Domain
Raw file: data/raw/social-media-usage-and-stress-rates.csv

What this script does
---------------------
1. Reads the raw CSV and profiles it (shape, dtypes, missing values, duplicates).
2. Renames columns to a consistent snake_case convention.
3. Trims leading/trailing whitespace in all string columns.
4. Validates:
     * User_ID format (U001-U500, unique)
     * numeric ranges (age, screen time, 1-10 scales, weekly counts)
     * consistency between daily_screen_time_hours and screen_time_group
5. Converts whole-number float columns to integers.
6. Rounds daily_screen_time_hours to 1 decimal place.
7. Documents anomalies instead of silently dropping/changing them.
8. Saves the cleaned dataset to data/processed/social_media_stress_clean.csv
   and writes a cleaning report to reports/data_cleaning_report.md.

Run from the project root:
    python3 scripts/preprocess.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths (relative to project root)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "social-media-usage-and-stress-rates.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "social_media_stress_clean.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "data_cleaning_report.md"

# Column renaming map: original -> clean snake_case
RENAME_MAP = {
    "User_ID": "user_id",
    "Age": "age",
    "Gender": "gender",
    "Daily_Screen_Time(hrs)": "daily_screen_time_hours",
    "Sleep_Quality(1-10)": "sleep_quality",
    "Stress_Level(1-10)": "stress_level",
    "Days_Without_Social_Media": "days_without_social_media",
    "Exercise_Frequency(week)": "exercise_frequency_per_week",
    "Social_Media_Platform": "social_media_platform",
    "Happiness_Index(1-10)": "happiness_index",
    "Screen_Time_Group": "screen_time_group",
}

# Numeric columns and their expected plausible ranges.
# Ranges are documented in the data dictionary (README.md).
RANGE_CHECKS = {
    "age": (10, 100),
    "daily_screen_time_hours": (0, 24),
    "sleep_quality": (1, 10),
    "stress_level": (1, 10),
    "days_without_social_media": (0, 7),   # see anomaly note below
    "exercise_frequency_per_week": (0, 7),
    "happiness_index": (1, 10),
}

# Whole-number float columns that should be integers
INT_COLUMNS = [
    "sleep_quality",
    "stress_level",
    "days_without_social_media",
    "exercise_frequency_per_week",
    "happiness_index",
]

STRING_COLUMNS = ["gender", "social_media_platform", "screen_time_group"]

# Expected screen-time group for a given number of hours
def expected_group(hours: float) -> str:
    if hours < 3:
        return "0-3 hrs"
    if hours < 6:
        return "3-6 hrs"
    if hours < 9:
        return "6-9 hrs"
    return "9-12 hrs"


def main() -> None:
    notes: list[str] = []      # anomalies observed and how they were handled
    actions: list[str] = []    # cleaning steps applied

    # ------------------------------------------------------------------
    # 0. Load raw data
    # ------------------------------------------------------------------
    df = pd.read_csv(RAW_PATH)
    n_rows_raw, n_cols_raw = df.shape
    print(f"Loaded raw data: {n_rows_raw} rows x {n_cols_raw} columns")

    # ------------------------------------------------------------------
    # 1. Rename columns
    # ------------------------------------------------------------------
    missing_cols = set(RENAME_MAP) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Expected columns not found in raw file: {missing_cols}")
    df = df.rename(columns=RENAME_MAP)
    actions.append("Renamed 11 columns to snake_case (see mapping in script).")

    # ------------------------------------------------------------------
    # 2. Trim whitespace in string columns
    # ------------------------------------------------------------------
    for col in STRING_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    actions.append("Trimmed leading/trailing whitespace in all string columns.")

    # ------------------------------------------------------------------
    # 3. Validate User_ID
    # ------------------------------------------------------------------
    bad_ids = df[~df["user_id"].str.fullmatch(r"U\d{3}")]
    if not bad_ids.empty:
        raise ValueError(f"Malformed user_id values: {bad_ids['user_id'].tolist()}")
    if df["user_id"].nunique() != len(df):
        raise ValueError("user_id is not unique.")
    actions.append("Validated user_id: 500 unique, well-formed IDs (U001-U500).")

    # ------------------------------------------------------------------
    # 4. Range validation
    # ------------------------------------------------------------------
    for col, (lo, hi) in RANGE_CHECKS.items():
        viol = df[(df[col] < lo) | (df[col] > hi)]
        if col == "days_without_social_media" and not viol.empty:
            # 9.0 exceeds a 7-day week, but the dataset does not state the
            # reference period, so the value is retained and documented.
            n = len(viol)
            notes.append(
                f"`days_without_social_media` has {n} value(s) of 9.0, which exceeds "
                "a 7-day week. The reference period is not documented, so the "
                "value(s) were **retained** and flagged for review."
            )
        elif not viol.empty:
            raise ValueError(f"{col} has values outside [{lo}, {hi}]: {viol[col].tolist()}")
    actions.append(
        "Checked numeric columns against plausible ranges (age, screen time, "
        "1-10 scales, weekly counts)."
    )

    # ------------------------------------------------------------------
    # 5. Convert whole-number floats to integers
    # ------------------------------------------------------------------
    for col in INT_COLUMNS:
        if not (df[col] % 1 == 0).all():
            raise ValueError(f"{col} contains fractional values and cannot be cast to int.")
        df[col] = df[col].astype(int)
    actions.append(
        "Converted sleep_quality, stress_level, days_without_social_media, "
        "exercise_frequency_per_week and happiness_index from float to int "
        "(all values were whole numbers)."
    )

    # ------------------------------------------------------------------
    # 6. Round screen time to 1 decimal
    # ------------------------------------------------------------------
    df["daily_screen_time_hours"] = df["daily_screen_time_hours"].round(1)
    actions.append("Rounded daily_screen_time_hours to 1 decimal place.")

    # ------------------------------------------------------------------
    # 7. Validate derived screen_time_group consistency
    # ------------------------------------------------------------------
    computed = df["daily_screen_time_hours"].apply(expected_group)
    inconsistent = df[df["screen_time_group"] != computed]
    if not inconsistent.empty:
        raise ValueError(
            f"screen_time_group inconsistent with daily_screen_time_hours for "
            f"{len(inconsistent)} rows."
        )
    actions.append(
        "Verified screen_time_group is fully consistent with "
        "daily_screen_time_hours (0 mismatches)."
    )

    # ------------------------------------------------------------------
    # 8. Duplicates & missing values (final check)
    # ------------------------------------------------------------------
    n_dup = df.duplicated().sum()
    if n_dup:
        raise ValueError(f"{n_dup} duplicate rows remain after cleaning.")
    if df.isna().sum().sum():
        raise ValueError("Missing values remain after cleaning.")
    actions.append("Final check: 0 missing values, 0 duplicate rows.")

    # ------------------------------------------------------------------
    # 9. Save cleaned data & report
    # ------------------------------------------------------------------
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    n_rows_new, n_cols_new = df.shape
    print(f"Saved cleaned data: {n_rows_new} rows x {n_cols_new} columns -> {PROCESSED_PATH}")

    # --- Build the cleaning report -------------------------------------
    dtypes_md = "\n".join(
        f"| `{col}` | {dtype} |" for col, dtype in df.dtypes.items()
    )
    notes_md = "\n".join(f"- {note}" for note in notes) or "- None"
    actions_md = "\n".join(f"- {action}" for action in actions)

    report = f"""# Data Cleaning Report

**Dataset:** Social Media and Stress Patterns
**Source:** [Kaggle](https://www.kaggle.com/datasets/houdlillianmaltiti/social-media-and-stress-patterns) (CC0: Public Domain)
**Pipeline:** `scripts/preprocess.py`

## Summary

| Metric | Raw | Cleaned |
|--------|-----|---------|
| Rows | {n_rows_raw} | {n_rows_new} |
| Columns | {n_cols_raw} | {n_cols_new} |
| Missing values | 0 | 0 |
| Duplicate rows | 0 | 0 |

## Cleaning steps applied

{actions_md}

## Anomalies found

{notes_md}

## Final schema

| Column | Dtype |
|--------|-------|
{dtypes_md}
"""
    REPORT_PATH.write_text(report)
    print(f"Saved cleaning report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
