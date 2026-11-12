# Member 4 — Regression Modeling and User Segmentation

## Objective and methods
We predicted numerical stress scores using age, gender, daily screen time, sleep quality,
days without social media, exercise frequency, and preferred social media platform.
We excluded the user ID, derived screen-time category, and happiness outcome.
The cleaned dataset contains 500 observations. We used an 80/20 train/test split
(random_state=42), one-hot encoding, standardized numeric predictors, five-fold
cross-validation on training data, and GridSearchCV for Random Forest.

## Regression results (held-out test data)
| Model               |   RMSE |     R2 |
|:--------------------|-------:|-------:|
| Linear Regression   | 1.0572 | 0.536  |
| Random Forest       | 1.0874 | 0.509  |
| Tuned Random Forest | 1.0743 | 0.5208 |

The lowest test RMSE was obtained by **Linear Regression**. Lower RMSE and higher R²
indicate better predictive performance; test scores should not be interpreted as causation.

## Five-fold cross-validation (training set)
| Model             |   CV mean RMSE |   CV std RMSE |
|:------------------|---------------:|--------------:|
| Linear Regression |         1.054  |        0.06   |
| Random Forest     |         1.1108 |        0.0598 |

## Tuned Random Forest
Best GridSearchCV parameters: `{'model__max_depth': None, 'model__min_samples_leaf': 3, 'model__n_estimators': 200}`. Best training-CV RMSE:
1.0815.

## Feature importance
The highest test permutation-importance predictor was **daily_screen_time_hours**.
Permutation importance describes model reliance; it does not prove the predictor
causes changes in stress. The ranked details are in `member4_feature_importance.csv`.

## Happiness prediction (stretch)
| Model             |   RMSE |     R2 |
|:------------------|-------:|-------:|
| Linear Regression | 1.0591 | 0.5278 |
| Random Forest     | 1.1166 | 0.4752 |

## User segmentation (stretch)
K-Means silhouette scores: k=3: 0.1357, k=4: 0.1338.
We selected k=3 by the higher score of the tested options. Behavior/demographic
variables defined the clusters; stress and happiness were used only to profile them.
- Cluster 0 (Middle Screen Time): 177 users, 5.75 hours screen time, stress 6.74, sleep 6.07, exercise 2.91, happiness 8.40.
- Cluster 1 (Higher Screen Time): 163 users, 6.99 hours screen time, stress 7.65, sleep 5.06, exercise 2.02, happiness 7.25.
- Cluster 2 (Lower Screen Time): 160 users, 3.80 hours screen time, stress 5.43, sleep 7.82, exercise 2.38, happiness 9.51.

These labels summarize relative averages, not distinct clinical categories. Low silhouette
scores indicate overlapping segments.

## Conclusions and recommendations
- Compare Linear Regression and tuned Random Forest using held-out RMSE and R².
- Screen time and other variables can help predict stress, but associations are not causal.
- Use user segments as exploratory descriptions, not definitive categories.
- To improve reliability, collect larger, more representative samples and validate externally.

## Limitations
1. Only 500 observations; the sample may not represent all users.
2. Self-reported lifestyle and wellbeing measurements may contain bias.
3. Unmeasured variables can affect stress; observational results do not prove causation.
4. Model scores can vary on other train/test samples, and clusters may overlap.

## Files exported
- `reports/member4_model_comparison.csv`
- `reports/member4_cross_validation.csv`
- `reports/member4_feature_importance.csv`
- `reports/member4_happiness_comparison.csv`
- `reports/member4_cluster_profiles.csv`
- `reports/figures/member4_feature_importance.png`
- `reports/figures/member4_actual_vs_predicted.png`
- `reports/figures/member4_cluster_profiles.png`

## References
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Pandas documentation: https://pandas.pydata.org/docs/
- Dataset provenance: see the project's README and data dictionary for its original source.
