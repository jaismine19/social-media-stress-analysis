# Part 4 — Linear Regression Summary

**Tool:** Python (scikit-learn) — script at `scripts/regression.py`
**Data:** `data/processed/social_media_stress_augmented.csv` (1,500 rows: 500 real + 1,000 synthetic)
**Split:** 80/20 (1,200 training / 300 test), fixed seed 42

---

## Objective and method

The project uses exactly two models: a decision tree for classification and a **single linear regression** for numeric prediction. We predicted the self-reported stress score (1–10) from seven habit/demographic features: age, gender, daily screen time, sleep quality, days without social media, exercise frequency, and preferred platform. Excluded: user ID (identifier), happiness index (circular — another wellbeing outcome), and the derived screen-time group label.

Preprocessing: one-hot encoding for the categorical features, standardization for the numeric ones. Validation: 5-fold cross-validation on the training split only; the 20% test set was used exactly once.

## Results (held-out test set)

| Metric | Value |
|--------|-------|
| RMSE | **1.069** stress points |
| R² | **0.546** |
| 5-fold CV RMSE | 1.091 ± 0.016 |

The model's typical guess is off by about **one point** on the 1–10 scale, and it explains roughly **half** of the variance between users.

## Interpretable coefficients (plain units)

The interpretation model is refit without scaling so each coefficient means "change in predicted stress per 1 unit of the feature":

| Feature | Coefficient |
|---------|-------------|
| Intercept | +3.15 |
| daily_screen_time_hours | **+0.64** |
| gender = Other | +0.15 |
| platform = LinkedIn | +0.08 |
| exercise_frequency_per_week | +0.08 |
| platform = X (Twitter) | +0.06 |
| platform = TikTok | +0.06 |
| platform = YouTube | +0.06 |
| gender = Male | −0.05 |
| sleep_quality | −0.05 |

**Reading:** every extra hour of screen time adds about +0.64 to the predicted stress score; better sleep slightly reduces it; gender and platform effects are small.

## Interpretation

1. Screen time is the dominant feature of the regression — consistent with the EDA (r = +0.74) and the decision tree's importance (81%).
2. The linear fit captures the near-linear dose–response pattern found in the EDA (stress 4.4 → 6.0 → 7.6 → 9.2 across usage groups).
3. An R² of 0.55 from seven questions alone shows habits carry real signal — and that unmeasured factors (life events, personality, self-report noise) make up the rest.

## Limitations

- Observational, self-reported data — coefficients describe associations, not causes.
- The augmented rows are synthetic (bootstrap with jitter); near-duplicates across the split make scores slightly optimistic. EDA conclusions remain anchored on the 500 real records.
- Low-stress users are under-represented (class imbalance noted in the EDA).

## Files exported

- `reports/regression_metrics.csv` — RMSE, R², CV scores
- `reports/regression_coefficients.csv` — full coefficient table
- `reports/figures/regression_actual_vs_predicted.png`
- `reports/figures/regression_coefficients.png`
