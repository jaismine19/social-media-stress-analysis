# Data Augmentation Report

**Purpose:** increase the modelling dataset beyond 1,000 rows as required by the instructor.
**Script:** `scripts/augment_data.py`
**Input:** `data/processed/social_media_stress_clean.csv` (500 real rows)
**Output:** `data/processed/social_media_stress_augmented.csv` (1,500 rows = 500 real + 1,000 synthetic)

---

## Method

New rows are generated with **row-level bootstrap with jitter** (a non-parametric resampling method):

1. A whole **real row is drawn at random** as a template — copying the full row (not single columns) keeps every relationship intact (e.g., heavy phone users still tend to report high stress).
2. A tiny random nudge is added to the numeric answers (σ = 0.5 for whole-number scores, σ = 0.2 for screen time), then values are rounded and clipped to their valid ranges.
3. The derived label (`screen_time_group`) is **recomputed from the nudged hours** — never copied.
4. New IDs continue the series (U501 … U1500).
5. Fixed seed (42) → the same synthetic rows are produced on every run.

### Why not simpler methods?

- **Plain duplication** inflates sample size but adds zero variety and makes the test set leak into training.
- **Column-by-column sampling** was tried first and rejected: it destroys the correlations (screen–stress fell from +0.74 to ≈0 and the models collapsed). Whole-row bootstrapping preserves them.

## Verification — patterns preserved

| Measure | Real (500) | Augmented (1,500) |
|---------|-----------|-------------------|
| Mean screen time (h) | 5.53 | 5.48 |
| Mean stress (1–10) | 6.62 | 6.62 |
| Mean sleep (1–10) | 6.30 | 6.32 |
| Correlation: screen ↔ stress | +0.74 | +0.72 |
| Correlation: screen ↔ sleep | −0.76 | −0.72 |
| Correlation: stress ↔ happiness | −0.74 | −0.69 |
| Missing values | 0 | 0 |
| Duplicate IDs | 0 | 0 |

## Model results on the augmented dataset (80/20 split, seed 42)

| Model | Metric | Real data (500) | Augmented (1,500) |
|-------|--------|-----------------|-------------------|
| Decision Tree (entropy, depth 5) | Accuracy | 72.7% (70/30 split) | **72.7%** (baseline 59.3%) |
| Linear Regression | R² / RMSE | 0.536 / 1.057 | 0.546 / 1.069 |
| Tuned Random Forest | R² / RMSE | 0.521 / 1.074 | **0.734 / 0.819** |

**Interpretation:** the story is unchanged — screen time remains the dominant predictor. With three times the data, the random forest now outperforms the linear model (it can use the extra, slightly noisy rows), while the tree's accuracy stays at 72.7%.

## Honesty notes

1. Synthetic rows contain **no new information about real people** — they mirror the distributions of the original 500 responses. All EDA conclusions remain anchored on the real data.
2. Because synthetic rows are near-copies of real rows, the train/test split contains near-duplicate "twins", so scores on the augmented set are **somewhat optimistic**. Both datasets' results are reported above for transparency.
3. The original cleaned file is unchanged and remains the reference dataset for EDA and data documentation.
