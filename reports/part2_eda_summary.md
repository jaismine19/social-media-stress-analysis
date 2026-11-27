# Part 2 — Exploratory Data Analysis (EDA) Summary

**Tool:** Python (pandas, matplotlib, seaborn) — script at `scripts/eda.py`
**Data:** `data/processed/social_media_stress_augmented.csv` (1,500 rows: 500 real + 1,000 synthetic)
**Charts:** `reports/figures/eda_*.png` (17 charts)

---

## 1. Overview

- 1,500 rows, 11 columns, **0 missing values, 0 duplicate rows**
- The EDA now runs on the same augmented dataset the models use, so the insights and the models line up on one dataset

## 2. Univariate findings

| Variable | Mean | Median | Skewness | Takeaway |
|----------|------|--------|----------|----------|
| Age | 33.4 | 34 | ~0 | Ages 16–49, balanced |
| Daily screen time (h) | 5.48 | 5.55 | ~0 | Most users 3–9 hrs; a few extremes (max 10.8) |
| Sleep quality (1–10) | 6.32 | 6 | mild | Slightly more good sleepers |
| Stress level (1–10) | 6.62 | 7 | mild | Most users rate themselves 6–8 |
| Days without social media | 3.18 | 3 | ~0 | 0–9 days |
| Exercise per week | 2.43 | 2 | mild positive | Most exercise 0–3 times a week |
| Happiness index (1–10) | 8.38 | 9 | left-skewed | Mostly happy — range 4–10 |

**Categories:**
- Platforms: LinkedIn 275, X (Twitter) 266, TikTok 262, Facebook 248, Instagram 233, YouTube 216
- Gender: Male 722 (48.1%), Female 711 (47.4%), Other 67 (4.5%)
- Screen-time groups: 3-6 hrs (762, 50.8%), 6-9 hrs (584, 38.9%), 0-3 hrs (109, 7.3%), 9-12 hrs (45, 3.0%)

## 3. Bivariate findings (the heart of the analysis)

**Correlation heatmap — top relationships:**

| Pair | Correlation | Meaning |
|------|-------------|---------|
| Screen time ↔ Sleep quality | **−0.72** | More screen time, worse sleep |
| Screen time ↔ Stress | **+0.72** | More screen time, more stress |
| Stress ↔ Happiness | −0.69 | More stress, less happiness |
| Screen time ↔ Happiness | −0.68 | More screen time, less happiness |
| Sleep ↔ Happiness | +0.61 | Better sleep, more happiness |

**Group comparisons:**
- **Stress by screen-time group** is monotonic: 0-3 hrs → 4.3, 3-6 hrs → 6.0, 6-9 hrs → 7.6, 9-12 hrs → 9.1
- **Sleep by platform:** LinkedIn users sleep best (6.53), Instagram users worst (5.94)
- **Stress by gender:** essentially identical (Female 6.64, Male 6.62, Other 6.48) — the story is in habits, not demographics
- Age is essentially uncorrelated with everything (|r| ≤ 0.06)

## 4. Outliers

IQR scan flags a small number of plausible extremes (screen time > 10.4 h ×5, exercise 7×/week ×6) — kept and noted, none removed.

## 5. Interesting facts for the models

1. Screen time and stress are strongly linked (+0.72): heavier users report more stress.
2. Screen time and sleep move opposite (−0.72): more screen, worse sleep.
3. Stress rises in steps with each screen-time group: 4.3 → 6.0 → 7.6 → 9.1.
4. Happiness mirrors stress (−0.69).
5. Gender and age barely matter — lifestyle habits carry the signal.
6. Only 146 users (9.7%) are Low-stress — expect class imbalance in the decision tree.

## 6. Conclusions

Based on the exploratory analysis alone, we draw six conclusions:

1. **Screen time is the single strongest signal in the dataset.** It shows the two largest associations of any pair: +0.72 with stress and −0.72 with sleep quality.
2. **The screen-time → stress relationship is monotonic (a dose–response staircase).** Mean stress rises with every usage group: 4.3 → 6.0 → 7.6 → 9.1 — every extra hour counts.
3. **Sleep looks like the middle link in the chain.** Screen ↔ sleep (−0.72), sleep ↔ happiness (+0.61) and stress ↔ happiness (−0.69) suggest: more screen → worse sleep → more stress → less happiness.
4. **Habits matter; demographics do not.** Age and gender are essentially uncorrelated with stress (|r| ≤ 0.06), and stress is nearly identical across genders.
5. **Exercise is a candidate protective factor.** Its simple correlation with stress is weak, but group patterns hint it matters in combination with heavy screen use — tested in the decision tree.
6. **The data is well-behaved for modelling, with one known difficulty.** Relationships are approximately linear (|r| ≈ 0.7), so simple models should be competitive; but only 9.7% of users are Low-stress — class imbalance to handle with weighted metrics.

**Implication for the project:** these conclusions became the design brief for Parts 3 and 4 — feature selection (habits only), the expected winner (screen time), the expected weakness (Low-stress recall), and the modelling strategy (simple models first). The decision tree (72.7% accuracy) and the linear regression (R² = 0.55, +0.64 per hour of screen time) confirmed every one of these expectations.
