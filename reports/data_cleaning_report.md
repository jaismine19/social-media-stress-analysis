# Data Cleaning Report

**Dataset:** Social Media and Stress Patterns
**Source:** [Kaggle](https://www.kaggle.com/datasets/houdlillianmaltiti/social-media-and-stress-patterns) (CC0: Public Domain)
**Pipeline:** `scripts/preprocess.py`

## Summary

| Metric | Raw | Cleaned |
|--------|-----|---------|
| Rows | 500 | 500 |
| Columns | 11 | 11 |
| Missing values | 0 | 0 |
| Duplicate rows | 0 | 0 |

## Cleaning steps applied

- Renamed 11 columns to snake_case (see mapping in script).
- Trimmed leading/trailing whitespace in all string columns.
- Validated user_id: 500 unique, well-formed IDs (U001-U500).
- Checked numeric columns against plausible ranges (age, screen time, 1-10 scales, weekly counts).
- Converted sleep_quality, stress_level, days_without_social_media, exercise_frequency_per_week and happiness_index from float to int (all values were whole numbers).
- Rounded daily_screen_time_hours to 1 decimal place.
- Verified screen_time_group is fully consistent with daily_screen_time_hours (0 mismatches).
- Final check: 0 missing values, 0 duplicate rows.

## Anomalies found

- `days_without_social_media` has 1 value(s) of 9.0, which exceeds a 7-day week. The reference period is not documented, so the value(s) were **retained** and flagged for review.

## Final schema

| Column | Dtype |
|--------|-------|
| `user_id` | str |
| `age` | int64 |
| `gender` | str |
| `daily_screen_time_hours` | float64 |
| `sleep_quality` | int64 |
| `stress_level` | int64 |
| `days_without_social_media` | int64 |
| `exercise_frequency_per_week` | int64 |
| `social_media_platform` | str |
| `happiness_index` | int64 |
| `screen_time_group` | str |
