#!/usr/bin/env python3
"""
=====================================================================
DATA AUGMENTATION  (easy-to-read version)
=====================================================================
Purpose : grow the dataset beyond 1000 rows for modelling,
          as requested by the instructor.
Reads   : data/processed/social_media_stress_clean.csv   (500 real rows)
Writes  : data/processed/social_media_stress_augmented.csv (1500 rows)

HOW IT WORKS (in plain words)
-----------------------------
We create 1000 NEW rows by copying the PATTERNS of the real 500:
  * words (gender, platform)  -> drawn with the same proportions
  * whole-number scores       -> a real answer + a tiny random
                                 nudge, then rounded and clipped
                                 to the valid range (1-10 etc.)
  * screen time (decimal)     -> same idea, 1 decimal kept
  * the screen-time label     -> RE-COMPUTED from the hours
                                 (never copied - it is derived)

This technique is called "non-parametric bootstrap with jitter":
we resample from what we observed instead of inventing numbers
from nothing.

HONESTY NOTE
------------
The synthetic rows follow the real distributions; they do not
contain new information about real people. The EDA and all
conclusions stay anchored on the original 500 records. The
augmented file exists so the models train on a larger sample,
as the instructor asked.
=====================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# File locations
# ---------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
REAL_FILE = PROJECT / "data" / "processed" / "social_media_stress_clean.csv"
AUG_FILE = PROJECT / "data" / "processed" / "social_media_stress_augmented.csv"

# How many new rows to create (500 real + 1000 synthetic = 1500)
N_SYNTHETIC = 1000


def expected_group(hours):
    """Which label SHOULD a given number of hours carry?"""
    if hours < 3:
        return "0-3 hrs"
    if hours < 6:
        return "3-6 hrs"
    if hours < 9:
        return "6-9 hrs"
    return "9-12 hrs"


def make_synthetic(real, n):
    """
    Build n new rows that follow the real data's patterns.

    KEY IDEA: we copy a whole real ROW as a template, then nudge a few
    numbers slightly. Copying whole rows keeps the relationships intact
    (e.g. heavy phone users still tend to have high stress). Drawing
    each column on its own would destroy those links.
    """
    rng = np.random.default_rng(42)

    # 1. Pick n random REAL rows as templates (with replacement)
    new = real.sample(n=n, replace=True, random_state=42).reset_index(drop=True)

    # IDs continue after U500
    new["user_id"] = [f"U{500 + i + 1:03d}" for i in range(n)]

    # 2. Small nudges so the new rows are not exact copies
    score_columns = {
        "age": (16, 49),
        "sleep_quality": (1, 10),
        "stress_level": (1, 10),
        "days_without_social_media": (0, 7),
        "exercise_frequency_per_week": (0, 7),
        "happiness_index": (1, 10),
    }
    for col, (low, high) in score_columns.items():
        nudge = rng.normal(0, 0.5, size=n)          # tiny random nudge
        new[col] = np.clip(np.round(new[col].astype(float) + nudge),
                           low, high).astype(int)

    # 3. Screen time: decimal, 1 decimal place
    base = new["daily_screen_time_hours"].astype(float)
    new["daily_screen_time_hours"] = np.round(
        np.clip(base + rng.normal(0, 0.2, size=n), 1.0, 12.0), 1)

    # 4. The label column is DERIVED from the hours - recompute it,
    #    never copy it.
    new["screen_time_group"] = new["daily_screen_time_hours"].apply(expected_group)

    return new


def main():
    real = pd.read_csv(REAL_FILE)
    print(f"Real rows loaded: {len(real)}")

    synthetic = make_synthetic(real, N_SYNTHETIC)
    combined = pd.concat([real, synthetic], ignore_index=True)

    # Keep the exact same column order as the real file
    combined = combined[real.columns]

    # Safety checks before saving
    assert combined.duplicated("user_id").sum() == 0, "Duplicate IDs!"
    assert combined.isna().sum().sum() == 0, "Missing values!"
    assert (combined["screen_time_group"] ==
            combined["daily_screen_time_hours"].apply(expected_group)).all(), \
        "Label column does not match hours!"

    combined.to_csv(AUG_FILE, index=False)
    print(f"Synthetic rows created: {len(synthetic)}")
    print(f"Saved {len(combined)} rows (real + synthetic) -> {AUG_FILE}")

    # Quick before/after comparison of the important patterns
    print("\nPattern check (real vs augmented):")
    for col in ["daily_screen_time_hours", "stress_level", "sleep_quality"]:
        print(f"  {col}: mean {real[col].mean():.2f} -> {combined[col].mean():.2f}")


if __name__ == "__main__":
    main()
