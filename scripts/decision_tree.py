#!/usr/bin/env python3
"""
=====================================================================
DECISION TREE CLASSIFICATION  (easy-to-read version)
=====================================================================
Task    : classify each user's stress as Low / Moderate / High
Reads   : data/processed/social_media_stress_clean.csv
Outputs : charts + result tables in reports/

HOW TO RUN (from the project root):
    python3 scripts/decision_tree.py

WHAT THIS SCRIPT DOES
---------------------
A decision tree asks a chain of yes/no questions about the data
(e.g. "is screen time above 5.75 hours?") until it can guess the
stress class. We build TWO trees, compare them, keep the better one,
draw it, and explain which questions matter most.

IMPORTANT DESIGN CHOICES
------------------------
1. The target has THREE classes:
      Low      = stress 1-4
      Moderate = stress 5-7
      High     = stress 8-10
2. happiness_index is NOT used as a feature. Happiness is another
   wellbeing score, so predicting stress from it is circular
   ("stressed people are unhappy") and inflates accuracy.
3. screen_time_group is NOT used - it is derived from screen time.
=====================================================================
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# ---------------------------------------------------------------
# File locations
# ---------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
CLEAN_FILE = PROJECT / "data" / "processed" / "social_media_stress_clean.csv"
REPORT_DIR = PROJECT / "reports"
FIG_DIR = REPORT_DIR / "figures"

sns.set_theme(style="whitegrid")


def save_chart(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"    saved -> reports/figures/{name}")


# ================================================================
# STEP 1 - CREATE THE TARGET (the thing we want to predict)
# ================================================================
def create_target(df):
    """
    Turn the 1-10 stress score into three classes.
    A normal if/elif/else - easy to read and change.
    """
    def stress_class(x):
        if x <= 4:
            return "Low"
        elif x <= 7:
            return "Moderate"
        else:
            return "High"

    df["stress_category"] = df["stress_level"].apply(stress_class)
    print("STEP 1 - Target created. Class sizes:")
    for label, n in df["stress_category"].value_counts().items():
        print(f"    {label}: {n} users ({n / len(df) * 100:.1f}%)")
    print("    NOTE: Low is a small class -> we will use weighted metrics.")
    return df


# ================================================================
# STEP 2 - CHOOSE FEATURES
# ================================================================
def choose_features(df):
    """
    Features = the columns the model is allowed to look at.
    We remove:
      * user_id            -> identifier, carries no information
      * stress_level       -> the raw score the target came from
      * stress_category    -> the target itself
      * happiness_index    -> circular (see the note at the top)
      * screen_time_group  -> derived from screen time
    """
    X = df.drop(columns=["user_id", "stress_level", "stress_category",
                         "happiness_index", "screen_time_group"])
    y = df["stress_category"]
    print("\nSTEP 2 - Features chosen:", list(X.columns))
    return X, y


# ================================================================
# STEP 3 - TRAIN / TEST SPLIT
# ================================================================
def split_data(X, y):
    """
    70% of rows train the model, 30% test it. The model must never
    see the test rows during training - that would be cheating.
    random_state=42 fixes the shuffle so the result is repeatable.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    print(f"\nSTEP 3 - Split: {len(X_train)} training rows, {len(X_test)} test rows.")
    return X_train, X_test, y_train, y_test


# ================================================================
# STEP 4 - PREPROCESSING FOR THE MODEL
# ================================================================
def build_preprocessor():
    """
    gender and platform are words. Models need numbers, so we turn
    each category into its own 0/1 column ("one-hot encoding").
    Example: platform = [TikTok, YouTube, TikTok]
             becomes  platform_TikTok = [1, 0, 1]
                       platform_YouTube = [0, 1, 0]
    drop="first" removes one redundant column.
    handle_unknown="ignore" survives unseen categories in live data.
    """
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                                  sparse_output=False),
             ["gender", "social_media_platform"]),
        ],
        remainder="passthrough",  # numeric columns pass through unchanged
    )
    return pre


# ================================================================
# STEP 5 - BUILD AND EVALUATE ONE TREE
# ================================================================
def build_and_evaluate(pre, X_train, X_test, y_train, y_test, name,
                       criterion, max_depth):
    """
    Build one decision tree inside a Pipeline, train it, test it,
    and print all the standard scores.
    """
    model = Pipeline([
        ("prep", pre),
        ("tree", DecisionTreeClassifier(criterion=criterion,
                                        max_depth=max_depth,
                                        random_state=42)),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n--- {name} (criterion={criterion}, max_depth={max_depth}) ---")
    print(f"    Train accuracy : {model.score(X_train, y_train):.4f}")
    print(f"    Test accuracy  : {acc:.4f}")
    print(f"    Precision      : {prec:.4f}   (weighted)")
    print(f"    Recall         : {rec:.4f}   (weighted)")
    print(f"    F1 score       : {f1:.4f}   (weighted)")
    print("\n    Per-class report:")
    print("    " + classification_report(y_test, y_pred).replace("\n", "\n    "))

    return model, y_pred, acc, prec, rec, f1


# ================================================================
# STEP 6 - BASELINE COMPARISON
# ================================================================
def majority_baseline(y_test):
    """
    What would we score if we did NOTHING clever and always guessed
    the most common class ("Moderate")? The model must beat this.
    """
    baseline = y_test.value_counts(normalize=True).max()
    print(f"\nSTEP 6 - Majority-class baseline: always predicting 'Moderate' "
          f"scores {baseline:.4f}.")
    return baseline


# ================================================================
# STEP 7 - CONFUSION MATRIX (where do we make mistakes?)
# ================================================================
def show_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_table = pd.DataFrame(
        cm,
        index=["Actual High", "Actual Low", "Actual Moderate"],
        columns=["Pred High", "Pred Low", "Pred Moderate"],
    )
    print("\nSTEP 7 - Confusion matrix (rows = truth, columns = guesses):")
    print(cm_table.to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm_table, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion matrix - test set")
    save_chart(fig, "decision_tree_confusion_matrix.png")
    return cm_table


# ================================================================
# STEP 8 - DRAW THE TREE
# ================================================================
def draw_tree(model, title):
    feature_names = list(model.named_steps["prep"].get_feature_names_out())

    def pretty_name(name):
        """Make encoded names readable: 'cat__gender_Male' -> 'gender = Male'."""
        if name.startswith("remainder__"):
            return name.replace("remainder__", "")
        for col in ["gender", "social_media_platform"]:
            prefix = f"cat__{col}_"
            if name.startswith(prefix):
                return f"{col} = {name[len(prefix):]}"
        return name

    pretty = [pretty_name(n) for n in feature_names]

    fig, ax = plt.subplots(figsize=(22, 12))
    plot_tree(model.named_steps["tree"], feature_names=pretty,
              class_names=["High", "Low", "Moderate"],
              filled=True, rounded=True, fontsize=9, ax=ax)
    ax.set_title(title)
    save_chart(fig, "decision_tree_tree.png")

    rules = export_text(model.named_steps["tree"], feature_names=pretty)
    (REPORT_DIR / "decision_tree_rules.txt").write_text(rules)
    print("    tree rules saved -> reports/decision_tree_rules.txt")
    return pretty


# ================================================================
# STEP 9 - FEATURE IMPORTANCE
# ================================================================
def feature_importance(model, pretty_names):
    """
    Importance = how much each feature contributed to the tree's
    questions. Higher = the tree relies on it more.
    """
    importance = model.named_steps["tree"].feature_importances_
    imp_df = pd.DataFrame({"Feature": pretty_names,
                           "Importance": importance})
    imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    imp_df["Rank"] = imp_df.index + 1
    imp_df = imp_df[["Rank", "Feature", "Importance"]]
    imp_df["Importance"] = imp_df["Importance"].round(4)

    print("\nSTEP 9 - Feature importance:")
    print(imp_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
    ax.set_title("Feature importance - decision tree")
    save_chart(fig, "decision_tree_importance.png")
    return imp_df


# ================================================================
# MAIN
# ================================================================
def main():
    print("STARTING THE DECISION TREE")
    df = pd.read_csv(CLEAN_FILE)

    df = create_target(df)
    X, y = choose_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    pre = build_preprocessor()

    # Model A: gini, depth 4
    model_a, y_pred_a, *_ = build_and_evaluate(
        pre, X_train, X_test, y_train, y_test,
        "Model A", criterion="gini", max_depth=4)

    # Model B: entropy, depth 5
    model_b, y_pred_b, *_ = build_and_evaluate(
        pre, X_train, X_test, y_train, y_test,
        "Model B", criterion="entropy", max_depth=5)

    baseline = majority_baseline(y_test)

    # Pick the winner by F1 (best balance of precision and recall)
    f1_a = f1_score(y_test, y_pred_a, average="weighted")
    f1_b = f1_score(y_test, y_pred_b, average="weighted")
    if f1_b >= f1_a:
        best, y_pred_best, best_name = model_b, y_pred_b, "Model B (entropy, depth 5)"
    else:
        best, y_pred_best, best_name = model_a, y_pred_a, "Model A (gini, depth 4)"
    print(f"\nWINNER: {best_name} (higher weighted F1)")

    cm_table = show_confusion_matrix(y_test, y_pred_best)
    pretty = draw_tree(best, f"Decision tree - {best_name}")
    imp_df = feature_importance(best, pretty)

    # Save result tables for the report
    summary = pd.DataFrame({
        "Metric": ["Test accuracy", "Weighted F1", "Majority baseline"],
        "Value": [round(accuracy_score(y_test, y_pred_best), 4),
                  round(f1_score(y_test, y_pred_best, average="weighted"), 4),
                  round(baseline, 4)],
    })
    summary.to_csv(REPORT_DIR / "decision_tree_summary.csv", index=False)
    cm_table.to_csv(REPORT_DIR / "decision_tree_confusion_matrix.csv")
    imp_df.to_csv(REPORT_DIR / "decision_tree_feature_importance.csv", index=False)
    print("\nSaved result tables -> reports/decision_tree_*.csv")
    print("DECISION TREE FINISHED")


if __name__ == "__main__":
    main()
