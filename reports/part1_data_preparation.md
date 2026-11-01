# Part 1 — Data Collection, Preprocessing & Cleaning

| | |
|---|---|
| **Owner** | Member 1 |
| **Status** | ✅ Completed |
| **Repo** | https://github.com/jaismine19/social-media-stress-analysis |

## What was done

### 1. Data collection
- Dataset: **Social Media and Stress Patterns** (500 users × 11 attributes)
- Source: [Kaggle](https://www.kaggle.com/datasets/houdlillianmaltiti/social-media-and-stress-patterns)
- License: CC0 (public domain) — free to use for coursework
- Original file preserved untouched at `data/raw/social-media-usage-and-stress-rates.csv`

### 2. Preprocessing & cleaning
Implemented as a reproducible script: `scripts/preprocess.py`

| Step | Action |
|------|--------|
| 1 | Renamed all 11 columns to snake_case |
| 2 | Trimmed whitespace from string columns |
| 3 | Validated user IDs (500 unique, well-formed) |
| 4 | Range-checked all numeric columns |
| 5 | Converted whole-number float columns to `int` |
| 6 | Rounded screen time to 1 decimal place |
| 7 | Verified `screen_time_group` matches screen-time hours (0 mismatches) |
| 8 | Final checks: 0 missing values, 0 duplicate rows — all 500 rows retained |

### 3. Anomaly handled
- One row has `days_without_social_media = 9` (exceeds a 7-day week). The dataset does not document the reference period, so the value was **retained** and flagged in the cleaning report rather than silently modified.

## Files produced

| File | Purpose |
|------|---------|
| `data/processed/social_media_stress_clean.csv` | Cleaned dataset (used by all other parts) |
| `scripts/preprocess.py` | Reproducible cleaning pipeline |
| `reports/data_cleaning_report.md` | Full record of checks, changes, anomalies |
| `data/data_dictionary.csv` | Column names, types, descriptions, ranges |
| `data/raw/README.md` | Dataset source & license attribution |
| `README.md` | Project overview & data dictionary |

## Reproducing the work

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/preprocess.py
```

## Key facts about the data (for the group)

- 500 users, ages 16–49 (mean ≈ 33)
- Mean daily screen time ≈ 5.5 h (min 1.0, max 10.8)
- Mean stress ≈ 6.6, mean sleep quality ≈ 6.3, mean happiness ≈ 8.4 (all on 1–10 scales)
- Platforms: TikTok (95), X/Twitter (88), LinkedIn (87), Facebook (81), YouTube (75), Instagram (74)
- Gender: Male (248), Female (229), Other (23)
- Screen-time groups: 3-6 hrs (249), 6-9 hrs (201), 0-3 hrs (37), 9-12 hrs (13)
