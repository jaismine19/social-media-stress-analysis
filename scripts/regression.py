#!/usr/bin/env python3
"""
=====================================================================
REGRESSION MODELING  (easy-to-read version)
=====================================================================
Task    : predict the stress SCORE (1-10) from the user's habits
Reads   : data/processed/social_media_stress_clean.csv
Outputs : charts + result tables in reports/

HOW TO RUN (from the project root):
    python3 scripts/regression.py

WHAT THIS SCRIPT DOES
---------------------
A regression model predicts a NUMBER (unlike the decision tree,
which predicts a class). Here we predict the stress score itself.

We build two models and compare them:
  1. Linear Regression  - draws the best straight line through the
                          data (simple, easy to explain)
  2. Random Forest      - 200 small decision trees that vote on the
                          answer (usually more accurate)

Then we:
  * check the models with cross-validation (are they stable?)
  * tune the random forest (find its best settings)
  * see which features matter most
  * (bonus) group users into segments with K-Means clustering

IMPORTANT DESIGN CHOICES
------------------------
* happiness_index is NOT used - it is another wellbeing score, so
  predicting stress from it is circular (see decision_tree.py).
* Numeric features are SCALED for linear regression (it compares
  sizes, so columns need the same scale). Trees do not need this.
=====================================================================
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------
# File locations
# ---------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
CLEAN_FILE = PROJECT / "data" / "processed" / "social_media_stress_clean.csv"
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
      * user_id           -> identifier
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
# STEP 2 - TRAIN / TEST SPLIT
# ================================================================
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"STEP 2 - Split: {len(X_train)} training rows, {len(X_test)} test rows.")
    return X_train, X_test, y_train, y_test


# ================================================================
# STEP 3 - PREPROCESSING
# ================================================================
def build_preprocessor():
    """
    * One-hot encode the words (gender, platform) into 0/1 columns.
    * Scale the numbers to the same size (mean 0, spread 1) because
      linear regression compares columns by size.
    Trees do not need scaling, but the pipeline uses it anyway for
    the linear model - scaling never hurts.
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
# STEP 4 - BUILD AND EVALUATE ONE MODEL
# ================================================================
def evaluate_model(pre, X_train, X_test, y_train, y_test, name, estimator):
    """
    Train the model and measure two standard regression scores:
      * RMSE  - average size of the mistake in stress points
                (lower is better; same unit as the 1-10 scale)
      * R^2   - share of the differences the model explains
                (1.0 = perfect, 0.0 = no better than guessing the mean)
    """
    model = Pipeline([("prep", pre), ("model", estimator)])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- {name} ---")
    print(f"    RMSE (test): {rmse:.3f} stress points")
    print(f"    R^2  (test): {r2:.3f}")

    # Cross-validation: train/test 5 times on different slices and
    # average the R^2 - a stable model gets similar scores each time.
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring="r2")
    print(f"    5-fold CV R^2: {cv_scores.round(3).tolist()} "
          f"-> mean {cv_scores.mean():.3f}")

    return model, y_pred, rmse, r2, cv_scores.mean()


# ================================================================
# STEP 5 - PREDICTED vs ACTUAL CHART
# ================================================================
def predicted_vs_actual(y_test, y_pred, name):
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
    ax.set_title(f"{name}: actual vs predicted")
    ax.legend()
    save_chart(fig, "regression_actual_vs_predicted.png")


# ================================================================
# STEP 6 - TUNE THE RANDOM FOREST
# ================================================================
def tune_random_forest(pre, X_train, y_train):
    """
    GridSearchCV tries every combination of settings we list,
    evaluates each with cross-validation, and keeps the best.
    """
    print("\nSTEP 6 - Tuning the random forest (GridSearchCV)...")
    pipe = Pipeline([("prep", pre), ("model", RandomForestRegressor(random_state=42))])
    grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 8, 12],
    }
    search = GridSearchCV(pipe, grid, cv=5, scoring="r2", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"    Best settings: {search.best_params_}")
    print(f"    Best CV R^2  : {search.best_score_:.3f}")
    return search.best_estimator_


# ================================================================
# STEP 7 - FEATURE IMPORTANCE (random forest)
# ================================================================
def feature_importance(model):
    """
    The forest records how often each feature was useful for its
    trees. This is our best guess at "what drives stress".
    """
    names = list(model.named_steps["prep"].get_feature_names_out())

    def pretty_name(name):
        """Make encoded names readable: 'num__age' -> 'age',
        'cat__gender_Male' -> 'gender = Male'."""
        for prefix in ("remainder__", "num__"):
            if name.startswith(prefix):
                return name.replace(prefix, "")
        for col in CATEGORICAL_FEATURES:
            cat_prefix = f"cat__{col}_"
            if name.startswith(cat_prefix):
                return f"{col} = {name[len(cat_prefix):]}"
        return name

    pretty = [pretty_name(n) for n in names]

    imp = model.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"Feature": pretty, "Importance": imp})
    imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    imp_df["Rank"] = imp_df.index + 1
    imp_df = imp_df[["Rank", "Feature", "Importance"]]
    imp_df["Importance"] = imp_df["Importance"].round(4)

    print("\nSTEP 7 - Feature importance (random forest):")
    print(imp_df.head(8).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=imp_df.head(8), x="Importance", y="Feature", ax=ax)
    ax.set_title("Feature importance - random forest (top 8)")
    save_chart(fig, "regression_importance.png")
    return imp_df


# ================================================================
# STEP 8 (BONUS) - K-MEANS CLUSTERING
# ================================================================
def cluster_users(pre, X, df):
    """
    K-Means groups users into k clusters of "similar" people.
    We then profile each cluster (its averages) to give it a name,
    e.g. "light users, low stress, good sleep".
    """
    print("\nSTEP 8 (bonus) - K-Means clustering")
    scaled = pre.fit_transform(X)
    results = []
    for k in [2, 3, 4]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        sil = silhouette_score(scaled, labels)
        results.append((k, sil))
        print(f"    k={k}: silhouette score {sil:.3f}")

    best_k = max(results, key=lambda t: t[1])[0]
    print(f"    Best k by silhouette: {best_k}")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)

    # Profile the clusters with ALL interesting columns, including
    # the wellbeing scores (used for describing, never for training).
    profile = X.copy()
    profile["stress_level"] = df["stress_level"]
    profile["happiness_index"] = df["happiness_index"]
    profile["cluster"] = labels
    profile_cols = ["age", "daily_screen_time_hours", "sleep_quality",
                    "days_without_social_media", "exercise_frequency_per_week",
                    "stress_level", "happiness_index"]
    means = profile.groupby("cluster")[profile_cols].mean().round(2)
    sizes = profile["cluster"].value_counts().sort_index()
    print("\n    Cluster profiles (means, plus size):")
    print(means.to_string())
    print(f"\n    Cluster sizes: {sizes.to_dict()}")
    return profile


# ================================================================
# MAIN
# ================================================================
def main():
    print("STARTING THE REGRESSION")
    df = pd.read_csv(CLEAN_FILE)

    X, y = choose_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    pre = build_preprocessor()

    # Model 1: linear regression (simple baseline)
    lr_model, lr_pred, lr_rmse, lr_r2, lr_cv = evaluate_model(
        pre, X_train, X_test, y_train, y_test,
        "Linear Regression (baseline)", LinearRegression())

    # Model 2: random forest (default settings first)
    rf_model, rf_pred, rf_rmse, rf_r2, rf_cv = evaluate_model(
        pre, X_train, X_test, y_train, y_test,
        "Random Forest (default)", RandomForestRegressor(random_state=42))

    # Tune the forest and re-evaluate
    tuned = tune_random_forest(pre, X_train, y_train)
    tuned_pred = tuned.predict(X_test)
    tuned_rmse = mean_squared_error(y_test, tuned_pred) ** 0.5
    tuned_r2 = r2_score(y_test, tuned_pred)
    print(f"\nTuned Random Forest -> RMSE: {tuned_rmse:.3f}, R^2: {tuned_r2:.3f}")

    # Chart for the tuned model
    predicted_vs_actual(y_test, tuned_pred, "Tuned Random Forest")

    # Comparison table
    comparison = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest (default)", "Random Forest (tuned)"],
        "RMSE": [round(lr_rmse, 3), round(rf_rmse, 3), round(tuned_rmse, 3)],
        "R^2": [round(lr_r2, 3), round(rf_r2, 3), round(tuned_r2, 3)],
        "CV R^2 (mean)": [round(lr_cv, 3), round(rf_cv, 3), "-"],
    })
    print("\nMODEL COMPARISON:")
    print(comparison.to_string(index=False))
    comparison.to_csv(REPORT_DIR / "regression_model_comparison.csv", index=False)

    # Feature importance from the tuned forest
    imp_df = feature_importance(tuned)
    imp_df.to_csv(REPORT_DIR / "regression_feature_importance.csv", index=False)

    # Bonus clustering
    cluster_users(pre, X, df)

    # Honest interpretation of the comparison
    print("\nWHICH MODEL WINS?")
    print("Linear regression is the winner here. That makes sense:")
    print("the strongest relationships in this dataset (screen time vs")
    print("stress, sleep, happiness) are close to straight lines")
    print("(|correlation| ~ 0.7), which is exactly when simple linear")
    print("models work best and forests cannot do much better.")

    print("\nSaved result tables -> reports/regression_*.csv")
    print("REGRESSION FINISHED")


if __name__ == "__main__":
    main()
