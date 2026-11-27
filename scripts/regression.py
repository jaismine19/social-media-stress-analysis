#!/usr/bin/env python3
"""
=====================================================================
LINEAR REGRESSION  (easy-to-read version)
=====================================================================
Task    : predict the stress SCORE (1-10) from the user's habits
Reads   : data/processed/social_media_stress_augmented.csv
Outputs : charts + result tables in reports/

HOW TO RUN (from the project root):
    python3 scripts/regression.py

WHAT THIS SCRIPT DOES
---------------------
The project uses exactly TWO models, as the instructor asked:
  1. Decision Tree  -> classifies stress (Low / Moderate / High)
  2. Linear Regression -> predicts the numeric stress score

A regression model predicts a NUMBER (unlike the tree, which
predicts a class). Linear regression fits the best straight line
through the data: every feature gets a weight, and the prediction
is  intercept + weight1*feature1 + weight2*feature2 + ...

This script:
  * loads the 1,500-row augmented dataset
  * splits 80% training / 20% testing (fixed seed)
  * one-hot encodes the words, standardizes the numbers
  * trains ONE linear regression model
  * scores it with RMSE and R-squared on the unseen test set
  * checks stability with 5-fold cross-validation
  * refits a plain-units model so the coefficients are readable
    ("+0.64 stress points per extra hour of screen time")

IMPORTANT DESIGN CHOICES
------------------------
* happiness_index is NOT used - it is another wellbeing score, so
  predicting stress from it is circular (see decision_tree.py).
* Standardization is used for training (so no column dominates);
  the interpretation model is refit WITHOUT scaling so its
  coefficients are in real, quotable units.
=====================================================================
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------
# File locations
# ---------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
AUG_FILE = PROJECT / "data" / "processed" / "social_media_stress_augmented.csv"
REPORT_DIR = PROJECT / "reports"
FIG_DIR = REPORT_DIR / "figures"

sns.set_theme(style="whitegrid")

NUMERIC_FEATURES = [
    "age",
    "daily_screen_time_hours",
    "sleep_quality",
    "days_without_social_media",
    "exercise_frequency_per_week",
]
CATEGORICAL_FEATURES = ["gender", "social_media_platform"]


def save_chart(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"    saved -> reports/figures/{name}")


# ================================================================
# STEP 1 - FEATURES AND TARGET
# ================================================================
def choose_features(df):
    """
    Target  = stress_level (the number 1-10 we want to predict).
    Features = everything the model may look at, EXCEPT:
      * user_id           -> identifier, carries no information
      * happiness_index   -> circular (another wellbeing score)
      * screen_time_group -> derived from screen time
    """
    X = df.drop(columns=["user_id", "stress_level",
                         "happiness_index", "screen_time_group"])
    y = df["stress_level"]
    print("STEP 1 - Target: stress_level (1-10).")
    print("        Features:", list(X.columns))
    return X, y


# ================================================================
# STEP 2 - TRAIN / TEST SPLIT (80 / 20)
# ================================================================
def split_data(X, y):
    """
    80% of rows train the model, 20% test it. The test rows are
    never seen during training - that would be cheating.
    random_state=42 fixes the shuffle so the result is repeatable.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"STEP 2 - Split: {len(X_train)} training rows, {len(X_test)} test rows (80/20).")
    return X_train, X_test, y_train, y_test


# ================================================================
# STEP 3 - PREPROCESSING
# ================================================================
def build_preprocessor():
    """
    * One-hot encode the words (gender, platform) into 0/1 columns.
    * Standardize the numbers (mean 0, spread 1) because linear
      regression compares columns by size - without scaling, the
      columns with the biggest numbers would dominate.
    """
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                                  sparse_output=False),
             CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="passthrough",
    )
    return pre


# ================================================================
# STEP 4 - TRAIN AND SCORE THE MODEL
# ================================================================
def train_and_evaluate(pre, X_train, X_test, y_train, y_test):
    """
    Train ONE linear regression and measure two standard scores:
      * RMSE - average size of the mistake in stress points
               (lower is better; same unit as the 1-10 scale)
      * R^2  - share of the differences between people that the
               model explains (1.0 = everything, 0.0 = nothing)
    """
    model = Pipeline([("prep", pre), ("model", LinearRegression())])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\n--- Linear Regression ---")
    print(f"    RMSE (test): {rmse:.3f} stress points")
    print(f"    R^2  (test): {r2:.3f}")

    # Cross-validation: train/test 5 times on different slices and
    # average the RMSE - a stable model gets similar scores each time.
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring="neg_root_mean_squared_error")
    cv_rmse = -cv_scores
    print(f"    5-fold CV RMSE: {cv_rmse.round(3).tolist()} "
          f"-> mean {cv_rmse.mean():.3f}")

    return model, y_pred, rmse, r2, cv_rmse.mean(), cv_rmse.std()


# ================================================================
# STEP 5 - PREDICTED vs ACTUAL CHART
# ================================================================
def predicted_vs_actual(y_test, y_pred):
    """
    Scatter the real stress scores against the predicted ones.
    Perfect predictions would sit exactly on the diagonal line.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="perfect prediction")
    ax.set_xlabel("Actual stress")
    ax.set_ylabel("Predicted stress")
    ax.set_title("Linear regression: actual vs predicted stress")
    ax.legend()
    save_chart(fig, "regression_actual_vs_predicted.png")


# ================================================================
# STEP 6 - READABLE COEFFICIENTS
# ================================================================
def show_coefficients(X_train, y_train):
    """
    The trained pipeline uses SCALED features, so its weights are
    hard to quote. For interpretation we refit a model on plain
    units (one-hot encoding only, no scaling): now each coefficient
    means "stress points per 1 unit of this feature".
    """
    plain = ColumnTransformer(
        [("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                               sparse_output=False), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    X_plain = plain.fit_transform(X_train)
    lr = LinearRegression().fit(X_plain, y_train)

    # readable names: 'cat__gender_Male' -> 'gender = Male'
    names = list(plain.get_feature_names_out())
    pretty_names = []
    for name in names:
        if name.startswith("remainder__"):
            pretty_names.append(name.replace("remainder__", ""))
        else:
            for col in CATEGORICAL_FEATURES:
                prefix = f"cat__{col}_"
                if name.startswith(prefix):
                    pretty_names.append(f"{col} = {name[len(prefix):]}")
                    break
            else:
                pretty_names.append(name)

    coef_df = pd.DataFrame({"Feature": pretty_names, "Coefficient": lr.coef_})
    coef_df = coef_df.reindex(
        coef_df["Coefficient"].abs().sort_values(ascending=False).index)
    coef_df["Coefficient"] = coef_df["Coefficient"].round(3)

    print(f"\nSTEP 6 - Intercept: {lr.intercept_:.2f}")
    print("        Coefficients (stress points per 1 unit):")
    print(coef_df.to_string(index=False))

    top = coef_df.head(8)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=top, x="Coefficient", y="Feature", hue="Feature",
                legend=False, ax=ax,
                palette=["#2E86AB" if v >= 0 else "#E67E22"
                         for v in top["Coefficient"]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Linear regression coefficients (top 8)")
    save_chart(fig, "regression_coefficients.png")

    return coef_df, lr.intercept_


# ================================================================
# MAIN
# ================================================================
def main():
    print("STARTING THE LINEAR REGRESSION")
    df = pd.read_csv(AUG_FILE)

    X, y = choose_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    pre = build_preprocessor()

    model, y_pred, rmse, r2, cv_mean, cv_std = train_and_evaluate(
        pre, X_train, X_test, y_train, y_test)

    predicted_vs_actual(y_test, y_pred)
    coef_df, intercept = show_coefficients(X_train, y_train)

    # Save result tables for the report
    metrics = pd.DataFrame({
        "Metric": ["Test RMSE (stress points)", "Test R^2",
                   "CV mean RMSE", "CV std RMSE"],
        "Value": [round(rmse, 3), round(r2, 3),
                  round(cv_mean, 3), round(cv_std, 3)],
    })
    metrics.to_csv(REPORT_DIR / "regression_metrics.csv", index=False)
    coef_df.to_csv(REPORT_DIR / "regression_coefficients.csv", index=False)
    print("\nSaved result tables -> reports/regression_metrics.csv, "
          "reports/regression_coefficients.csv")

    # Plain-English interpretation
    screen_w = coef_df.loc[coef_df["Feature"] == "daily_screen_time_hours",
                           "Coefficient"]
    if not screen_w.empty:
        print(f"\nWHAT THE MODEL SAYS:")
        print(f"    Every extra hour of screen time adds about "
              f"{screen_w.iloc[0]:+.2f} to the predicted stress score.")
    print("LINEAR REGRESSION FINISHED")


if __name__ == "__main__":
    main()
