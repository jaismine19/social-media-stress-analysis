#!/usr/bin/env python3
"""
=====================================================================
DATA PREPROCESSING & CLEANING  (easy-to-read version)
=====================================================================
Dataset : Social Media and Stress Patterns (Kaggle, CC0 license)
Raw file: data/raw/social-media-usage-and-stress-rates.csv
Output  : data/processed/social_media_stress_clean.csv

HOW TO RUN (from the project root):
    python3 scripts/preprocess.py

WHAT THIS SCRIPT DOES
---------------------
It cleans the data step by step. Every step is a small function with
plain-English comments, and the script PRINTS what it does and why.

The steps are written as if the data were "live" (real, messy) data:

  Step  1  Load the raw file
  Step  2  Profile the data (know it before you change it)
  Step  3  Rename columns to a clean naming style
  Step  4  Remove stray spaces in text columns
  Step  5  Missing values policy (median for numbers, 'Unknown' for text)
  Step  6  Duplicate rows policy (drop exact duplicates)
  Step  7  Validate the ID column
  Step  8  Check every number is in a plausible range
  Step  9  Detect outliers (IQR rule) - report, do not drop blindly
  Step 10  Fix column types (7.0 -> 7)
  Step 11  Check the derived column matches its source column
  Step 12  Final safety checks + save

WHY NO OUTLIERS ARE DROPPED
---------------------------
Outliers are REPORTED, not deleted. A value can only be removed when
there is a real reason (typo, sensor error). Here the "outliers" are
plausible values (e.g. someone exercising 7 times a week), so we keep
them and let the report speak.
=====================================================================
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------
# File locations. Path(__file__) = this script, so the paths work
# no matter which folder you run the script from.
# ---------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT / "data" / "raw" / "social-media-usage-and-stress-rates.csv"
CLEAN_FILE = PROJECT / "data" / "processed" / "social_media_stress_clean.csv"

# ---------------------------------------------------------------
# The rename table: old (messy) name -> new (clean) name.
# snake_case = lowercase words joined with underscores.
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Plausible range for every numeric column.
# 1-10 scales, 0-7 days a week, age 10-100, screen time 0-24.
# ---------------------------------------------------------------
RANGES = {
    "age": (10, 100),
    "daily_screen_time_hours": (0, 24),
    "sleep_quality": (1, 10),
    "stress_level": (1, 10),
    "days_without_social_media": (0, 7),
    "exercise_frequency_per_week": (0, 7),
    "happiness_index": (1, 10),
}

# Columns that are whole numbers stored as floats -> should be integers
INT_COLUMNS = [
    "sleep_quality",
    "stress_level",
    "days_without_social_media",
    "exercise_frequency_per_week",
    "happiness_index",
]

# Text columns that may contain stray spaces
TEXT_COLUMNS = ["gender", "social_media_platform", "screen_time_group"]


# ================================================================
# STEP 1 - LOAD
# ================================================================
def load_data():
    """Read the raw CSV file."""
    df = pd.read_csv(RAW_FILE)
    print(f"STEP 1 - Loaded {df.shape[0]} rows and {df.shape[1]} columns from the raw file.")
    return df


# ================================================================
# STEP 2 - PROFILE (get to know the data BEFORE changing anything)
# ================================================================
def profile_data(df):
    """Print basic facts about the data. Always do this first."""
    print("\n" + "=" * 60)
    print("STEP 2 - DATA PROFILE (before cleaning)")
    print("=" * 60)
    print("Column types:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nUnique values per column:")
    print(df.nunique().to_string())
    print("\nNumeric summary (count, mean, min, max, quartiles):")
    print(df.describe().round(2).to_string())


# ================================================================
# STEP 3 - RENAME COLUMNS
# ================================================================
def rename_columns(df):
    """Give every column a clean snake_case name."""
    df = df.rename(columns=RENAME_MAP)
    print("\nSTEP 3 - Renamed columns:")
    for old, new in RENAME_MAP.items():
        print(f"    {old}  ->  {new}")
    return df


# ================================================================
# STEP 4 - CLEAN TEXT COLUMNS
# ================================================================
def clean_text_columns(df):
    """Remove spaces before/after text. 'Male ' becomes 'Male'."""
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    print("\nSTEP 4 - Trimmed stray spaces in:", ", ".join(TEXT_COLUMNS))
    return df


# ================================================================
# STEP 5 - MISSING VALUES POLICY
# ================================================================
def handle_missing_values(df):
    """
    Live data usually has gaps (empty cells). Our policy:
      * numeric column -> fill with the MEDIAN
        (the median is not thrown off by extreme values)
      * text column    -> fill with "Unknown"
    Nothing is dropped and nothing is filled with zero by accident.
    If there are no gaps, nothing happens.
    """
    print("\nSTEP 5 - Missing values policy")
    missing = df.isna().sum()
    if missing.sum() == 0:
        print("    No missing values found - nothing to do.")
        return df

    for col in df.columns:
        if missing[col] == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
            print(f"    {col}: {missing[col]} missing -> filled with median {fill_value}")
        else:
            fill_value = "Unknown"
            print(f"    {col}: {missing[col]} missing -> filled with 'Unknown'")
        df[col] = df[col].fillna(fill_value)
    return df


# ================================================================
# STEP 6 - DUPLICATES POLICY
# ================================================================
def handle_duplicates(df):
    """
    Two identical rows are almost always the same record twice.
    Keep the first copy, drop the rest.
    """
    print("\nSTEP 6 - Duplicate rows policy")
    n_dups = df.duplicated().sum()
    if n_dups == 0:
        print("    No duplicate rows found.")
    else:
        print(f"    Removed {n_dups} duplicate rows (kept the first copy).")
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
    return df


# ================================================================
# STEP 7 - VALIDATE ID COLUMN
# ================================================================
def validate_user_ids(df):
    """
    Every ID must look like U001, U002, ... (letter U + 3 digits)
    and every ID must appear only once.
    """
    print("\nSTEP 7 - Validate user IDs")
    # r"U\d{3}" means: U followed by exactly 3 digits
    valid = df["user_id"].str.fullmatch(r"U\d{3}").fillna(False)
    bad_ids = df.loc[~valid, "user_id"].tolist()
    if bad_ids:
        raise ValueError(f"Malformed user IDs found: {bad_ids}")
    if df["user_id"].nunique() != len(df):
        raise ValueError("user_id is not unique!")
    print(f"    All {len(df)} IDs are valid and unique (U001-U{len(df):03d}).")


# ================================================================
# STEP 8 - RANGE CHECKS
# ================================================================
def check_ranges(df):
    """
    Every number must sit in a plausible range, e.g. a 1-10 rating
    cannot be 15. Values outside the range are kept but reported -
    we never "fix" a number we do not understand.
    """
    print("\nSTEP 8 - Range checks")
    for col, (low, high) in RANGES.items():
        bad = df[(df[col] < low) | (df[col] > high)]
        if len(bad) == 0:
            print(f"    {col}: all values inside {low}-{high}  ✓")
        else:
            values = sorted(bad[col].unique().tolist())
            print(f"    {col}: {len(bad)} value(s) outside {low}-{high}: {values}  (kept, flagged in report)")
    return df


# ================================================================
# STEP 9 - OUTLIER DETECTION (IQR rule)
# ================================================================
def outliers_iqr(df, col):
    """
    IQR rule: a value is suspicious if it lies more than 1.5 x IQR
    below the 25th percentile or above the 75th percentile.
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr
    return df[(df[col] < low_fence) | (df[col] > high_fence)], low_fence, high_fence


def check_outliers(df):
    """
    Detect outliers so the analyst can decide. We REPORT them; we do
    NOT drop them. Dropping needs a real reason (typo, broken sensor).
    """
    print("\nSTEP 9 - Outlier scan (IQR rule, report only)")
    for col in ["age", "daily_screen_time_hours", "exercise_frequency_per_week"]:
        out, low_fence, high_fence = outliers_iqr(df, col)
        if len(out) == 0:
            print(f"    {col}: no outliers (fences {low_fence:.2f} - {high_fence:.2f})")
        else:
            values = sorted(out[col].unique().tolist())
            print(f"    {col}: {len(out)} potential outlier(s) {values} (kept - values are plausible)")
    return df


# ================================================================
# STEP 10 - FIX COLUMN TYPES
# ================================================================
def fix_types(df):
    """
    Whole-number floats should be integers: 7.0 -> 7.
    Screen time is a decimal, so it stays a float (rounded to 1 dp).
    """
    print("\nSTEP 10 - Fix column types")
    for col in INT_COLUMNS:
        # only convert if every value is a whole number
        if (df[col] % 1 == 0).all():
            df[col] = df[col].astype(int)
            print(f"    {col}: float -> integer")
        else:
            print(f"    {col}: has fractions, kept as float")
    df["daily_screen_time_hours"] = df["daily_screen_time_hours"].round(1)
    print("    daily_screen_time_hours: rounded to 1 decimal place")
    return df


# ================================================================
# STEP 11 - CHECK THE DERIVED COLUMN
# ================================================================
def expected_group(hours):
    """Which label SHOULD a given number of hours carry?"""
    if hours < 3:
        return "0-3 hrs"
    if hours < 6:
        return "3-6 hrs"
    if hours < 9:
        return "6-9 hrs"
    return "9-12 hrs"


def check_derived_column(df):
    """
    screen_time_group is supposed to be a copy of the screen-time
    hours in label form. Check the two tell the same story.
    """
    print("\nSTEP 11 - Check screen_time_group against screen-time hours")
    actual = df["daily_screen_time_hours"].apply(expected_group)
    n_mismatch = (df["screen_time_group"] != actual).sum()
    if n_mismatch == 0:
        print("    Perfect match - 0 mismatches. The label column is trustworthy.")
    else:
        print(f"    WARNING: {n_mismatch} rows where the label does not match the hours!")
    return df


# ================================================================
# STEP 12 - FINAL CHECKS + SAVE
# ================================================================
def final_checks(df):
    """Last safety net before the data leaves this script."""
    print("\nSTEP 12 - Final checks")
    assert df.isna().sum().sum() == 0, "Missing values remain!"
    assert df.duplicated().sum() == 0, "Duplicate rows remain!"
    print("    No missing values, no duplicates - the data is clean.")


def save_data(df):
    """Write the cleaned dataset."""
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False)
    print(f"\nSaved cleaned data ({df.shape[0]} rows x {df.shape[1]} columns) -> {CLEAN_FILE}")


# ================================================================
# MAIN - run every step in order
# ================================================================
def main():
    print("STARTING THE CLEANING PIPELINE")
    df = load_data()
    profile_data(df)
    df = rename_columns(df)
    df = clean_text_columns(df)
    df = handle_missing_values(df)
    df = handle_duplicates(df)
    validate_user_ids(df)
    df = check_ranges(df)
    df = check_outliers(df)
    df = fix_types(df)
    df = check_derived_column(df)
    final_checks(df)
    save_data(df)
    print("PIPELINE FINISHED")


if __name__ == "__main__":
    main()
