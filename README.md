# Social Media and Stress Patterns — Data Analysis

A fundamentals-of-data-analytics project exploring how digital habits relate to stress and wellbeing. This repository contains the raw dataset, a reproducible data-cleaning pipeline, the cleaned dataset, and a data-quality report.

## Dataset

| Attribute | Value |
|-----------|-------|
| Title | Social Media and Stress Patterns |
| Source | [Kaggle — houdlillianmaltiti/social-media-and-stress-patterns](https://www.kaggle.com/datasets/houdlillianmaltiti/social-media-and-stress-patterns) |
| License | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) |
| Rows / Columns | 500 × 11 |
| Description | 500 users across age groups; each record captures daily screen time, sleep quality, stress level, days without social media, exercise frequency, preferred platform, and a happiness index. |

## Repository structure

```
social-media-stress-analysis/
├── README.md                       # You are here
├── requirements.txt                # Python dependencies
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── social-media-usage-and-stress-rates.csv   # Original, untouched
│   │   └── README.md                # Source & license details
│   ├── processed/
│   │   └── social_media_stress_clean.csv              # Cleaned dataset
│   └── data_dictionary.csv          # Column documentation
├── notebooks/                      # Analysis notebooks
├── scripts/
│   ├── preprocess.py               # Cleaning pipeline (easy-to-read, live-data ready)
│   ├── eda.py                      # Full EDA in Python: distributions, relationships, outliers
│   ├── eda.R                       # Full EDA in R (base R + corrplot) — Part 2 deliverable
│   ├── decision_tree.py            # Decision tree classification (stress classes)
│   └── regression.py               # Regression: predict stress score + clustering
├── reports/
│   ├── data_cleaning_report.md     # Checks, changes, and anomalies
│   ├── part1_data_preparation.md   # Data preparation summary
│   └── figures/                    # Charts produced by eda.py and the models
└── presentation/                   # Final deck
```

## Data dictionary (cleaned dataset)

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Unique respondent ID (U001–U500) |
| `age` | int | Age in years (16–49) |
| `gender` | string | Male / Female / Other |
| `daily_screen_time_hours` | float | Daily screen time in hours (1.0–10.8) |
| `sleep_quality` | int | Self-rated sleep quality (1–10) |
| `stress_level` | int | Self-rated stress level (1–10) |
| `days_without_social_media` | int | Days without social media (0–9; see report note) |
| `exercise_frequency_per_week` | int | Exercise sessions per week (0–7) |
| `social_media_platform` | string | Preferred platform (Facebook, Instagram, LinkedIn, TikTok, X (Twitter), YouTube) |
| `happiness_index` | int | Self-rated happiness (1–10; observed range 4–10) |
| `screen_time_group` | string | Derived screen-time bucket (0-3, 3-6, 6-9, 9-12 hrs) |

## Getting started

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Reproduce the cleaned dataset & report
python3 scripts/preprocess.py
```

The cleaning pipeline reads `data/raw/`, performs the documented steps, and writes `data/processed/social_media_stress_clean.csv`.

```bash
# 4. Run the full analysis suite (each script prints plain-English
#    explanations of what it does and why)
python3 scripts/eda.py              # EDA + charts -> reports/figures/
python3 scripts/decision_tree.py    # decision tree + result tables
python3 scripts/regression.py       # regression models + clustering

# EDA is also available in R (requires R + the corrplot package):
Rscript scripts/eda.R               # R EDA + charts (r_*.png) -> reports/figures/
```

## Cleaning summary

- Renamed columns to snake_case
- Trimmed whitespace from string columns
- Validated user IDs, numeric ranges, and the consistency of `screen_time_group`
- Missing-value policy (median for numbers, `Unknown` for text) and duplicate removal — applied if needed, as with live data
- Outlier detection via the IQR rule — reported, not blindly dropped
- Converted whole-number float columns to integer types
- Rounded screen time to 1 decimal place
- No rows dropped: 500 rows retained (0 missing, 0 duplicates)
- One anomaly documented: a single `days_without_social_media = 9` value was retained (reference period unspecified) and flagged in the report

## Next steps

- **Exploratory data analysis:** distributions, scatter plots, group comparisons, correlation heatmap
- **Statistical analysis:** correlation tests, ANOVA / Kruskal-Wallis, heavy vs light user comparison
- **Predictive modeling:** predict stress level, feature importance, user segmentation
- **Final report & presentation:** synthesis of findings and recommendations

## License

The dataset is in the public domain ([CC0](https://creativecommons.org/publicdomain/zero/1.0/)). All code in this repository is MIT licensed.
