# Part 2 — Exploratory Data Analysis (EDA) Summary

**Tool:** Python (pandas, matplotlib, seaborn) — script at `scripts/eda.py`
**Data:** `data/processed/social_media_stress_clean.csv` (500 rows × 11 columns)
**Charts:** `reports/figures/eda_*.png` (18 charts)

---

## 1. Overview

- 500 users, 11 columns, **0 missing values, 0 duplicate rows**
- All numeric columns sit in their expected ranges (verified in Part 1)

## 2. Univariate findings

| Variable | Mean | Median | Skewness | Takeaway |
|----------|------|--------|----------|----------|
| Age | 33.0 | 34 | ~0 | Ages 16–49, fairly balanced |
| Daily screen time (h) | 5.53 | 5.6 | mild | Most users 3–9 hrs; a few extremes (max 10.8) |
| Sleep quality (1–10) | 6.30 | 6 | mild negative | Slightly more good sleepers |
| Stress level (1–10) | 6.62 | 7 | mild negative | Most users rate themselves 6–8 |
| Days without social media | 3.13 | 3 | ~0 | 0–9 days (one 9-day value flagged in Part 1) |
| Exercise per week | 2.45 | 2 | mild positive | Most exercise 0–3 times a week |
| Happiness index (1–10) | 8.38 | 9 | strong negative | Mostly happy — observed range 4–10 |

**Categories:**
- Platforms: TikTok 95, X (Twitter) 88, LinkedIn 87, Facebook 81, YouTube 75, Instagram 74
- Gender: Male 248, Female 229, Other 23
- Screen-time groups: 3-6 hrs (249), 6-9 hrs (201), 0-3 hrs (37), 9-12 hrs (13)

## 3. Bivariate findings (the heart of the analysis)

**Correlation heatmap — top relationships:**

| Pair | Correlation | Meaning |
|------|-------------|---------|
| Screen time ↔ Sleep quality | **−0.76** | More screen time, worse sleep |
| Screen time ↔ Stress | **+0.74** | More screen time, more stress |
| Stress ↔ Happiness | **−0.74** | More stress, less happiness |
| Screen time ↔ Happiness | −0.71 | More screen time, less happiness |
| Sleep ↔ Happiness | +0.68 | Better sleep, more happiness |

**Group comparisons:**
- **Stress by screen-time group** is monotonic: 0-3 hrs → 4.4, 3-6 hrs → 6.0, 6-9 hrs → 7.6, 9-12 hrs → 9.2
- **Sleep by platform:** LinkedIn users sleep best (6.55), Instagram users worst (5.91)
- **Stress by gender:** essentially identical (Female 6.60, Male 6.63, Other 6.61) — the story is in habits, not demographics
- Age is essentially uncorrelated with everything (|r| ≤ 0.06)

## 4. Outliers

IQR scan flags only a handful of plausible extremes (screen time 10.8 h ×2, exercise 7×/week ×2) — kept and noted, none removed.

## 5. Interesting facts for Members 3 & 4

1. Screen time and stress are strongly linked (+0.74): heavier users report more stress.
2. Screen time and sleep move opposite (−0.76): more screen, worse sleep.
3. Stress rises in steps with each screen-time group: 4.4 → 6.0 → 7.6 → 9.2.
4. Happiness mirrors stress almost perfectly (−0.74).
5. Gender and age barely matter — lifestyle habits carry the signal.
6. Only 43 users (8.6%) are Low-stress — expect class imbalance in the decision tree.

## 6. What this means for the project

The EDA gives Members 3 & 4 a clear hypothesis to model: **screen time is the strongest single predictor of stress**, with sleep quality close behind. The relationships are roughly linear, which suggests even simple models (linear regression, shallow trees) should work well.
