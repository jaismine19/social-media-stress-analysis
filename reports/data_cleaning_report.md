# Data Cleaning Report

**Dataset:** Social Media and Stress Patterns
**Source:** [Kaggle](https://www.kaggle.com/datasets/houdlillianmaltiti/social-media-and-stress-patterns) (CC0: Public Domain)
**Pipeline:** `scripts/preprocess.py`

## Summary

| Metric | Raw | Cleaned |
|--------|-----|---------|
| Rows | 500 | 500 (nothing dropped) |
| Columns | 11 | 11 |
| Missing values | 0 | 0 |
| Duplicate rows | 0 | 0 |

## Cleaning steps applied

1. **Load & profile** — read the raw CSV and print column types, missing counts, duplicates, unique values and summary statistics *before* changing anything.
2. **Rename columns** — all 11 headers changed to a consistent snake_case convention (`Daily_Screen_Time(hrs)` → `daily_screen_time_hours`).
3. **Trim text** — stray spaces removed from gender, platform and screen-time group.
4. **Row-level missing-value report** — every cell is checked one by one; any gap would be reported as "which user, which column". (None found.)
5. **Missing-value policy** — median fill for numbers, "Unknown" for text. (Not triggered — 0 gaps.)
6. **Duplicate policy** — exact duplicate rows dropped, keep-first. (Not triggered — 0 duplicates.)
7. **ID validation** — every `user_id` matches the U + 3-digits pattern and is unique (U001–U500).
8. **Range checks** — every numeric column checked against plausible bounds (1–10 scales, 0–24 h, 0–7 counts). One value flagged: `days_without_social_media = 9`.
9. **Outlier scan (IQR rule)** — values beyond 1.5×IQR reported (screen time 10.8 h ×2, exercise 7×/week ×2). All retained — plausible real answers, not errors.
10. **Type fixes** — whole-number floats converted to integers (7.0 → 7); screen time rounded to 1 decimal.
11. **Derived-column check** — `screen_time_group` verified against the hours (0 mismatches).
12. **Final safety checks + save** — asserts for zero missing/duplicates, then writes `data/processed/social_media_stress_clean.csv`.

## Anomalies found (and kept)

- **`days_without_social_media = 9`** for user U450: exceeds a 7-day week, but the dataset does not document the reference period, so the value was retained and flagged rather than silently changed.
- **Outliers:** 10.8 h daily screen time (×2) and 7 exercise sessions/week (×2) are flagged by the IQR rule but retained — they are plausible extreme habits, not data errors.

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

## Downstream datasets

- `data/processed/social_media_stress_clean.csv` — the cleaned real data (500 rows); used by the EDA.
- `data/processed/social_media_stress_augmented.csv` — 1,500 rows (500 real + 1,000 synthetic bootstrap rows); used by the two models. See `reports/data_augmentation_report.md`.
