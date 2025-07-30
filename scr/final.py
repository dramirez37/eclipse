# ------------------------------------------------------------------
# Script: Eclipse Software Bugs with Walk-Forward Optimization (Python)
# Author: David Ramirez
# Date: July 28, 2025
# Description: This script uses a stacking ensemble model (XGBoost,
#              LightGBM, Logistic Regression) with a full suite of
#              advanced, dynamically generated features to predict
#              bug resolution status. It uses TF-IDF for text
#              vectorization and walk-forward validation.
# ------------------------------------------------------------------

# 1. SETUP & LOAD LIBRARIES
# ==================================================================
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack, csr_matrix
import time
import os
import gc
import matplotlib.pyplot as plt
import csv
import re
import json
import joblib

# --- Configuration ---
pd.options.mode.chained_assignment = None


# 2. DATA LOADING
# ==================================================================
print("Loading data with robust parser...")
data_path = "data"


def robust_csv_reader(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        num_columns = len(header)
        data = []
        for i, row in enumerate(reader):
            if len(row) > num_columns:
                data.append(row[:num_columns])
            elif len(row) < num_columns:
                data.append(row + [""] * (num_columns - len(row)))
            else:
                data.append(row)
    return pd.DataFrame(data, columns=header)


try:
    all_dfs = {
        name: robust_csv_reader(os.path.join(data_path, f"{name}.csv"))
        for name in [
            "assigned_to",
            "bug_status",
            "component",
            "op_sys",
            "priority",
            "resolution",
            "severity",
            "cc",
            "short_desc",
            "version",
            "product",
            "reports",
        ]
    }
except FileNotFoundError as e:
    print(f"Error: Could not find a necessary data file: {e.filename}")
    print("Please ensure all 12 source CSV files are in the specified data directory.")
    exit()

# Convert timestamp columns to numeric before creating datetime objects
for name, df in all_dfs.items():
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    if "when" in df.columns:
        numeric_when = pd.to_numeric(df["when"], errors="coerce")
        df["when"] = pd.to_datetime(numeric_when, unit="s", errors="coerce")
    if "opening" in df.columns:
        numeric_opening = pd.to_numeric(df["opening"], errors="coerce")
        df["opening"] = pd.to_datetime(numeric_opening, unit="s", errors="coerce")
    all_dfs[name] = df.dropna(subset=["id"])
    all_dfs[name]["id"] = all_dfs[name]["id"].astype(np.int64)

(
    assigned_to,
    bug_status,
    component,
    op_sys,
    priority,
    resolution,
    severity,
    cc,
    short_desc,
    version,
    product,
    reports,
) = all_dfs.values()

print("Data loading complete.")


# 3. FEATURE ENGINEERING
# ==================================================================
print("Starting initial feature engineering...")
final_resolution = (
    resolution.sort_values("when")
    .groupby("id")["what"]
    .last()
    .str.lower()
    .reset_index()
    .rename(columns={"what": "final_resolution"})
)


def get_initial_attribute(df, col_name):
    return (
        df.sort_values("when")
        .groupby("id")["what"]
        .first()
        .reset_index()
        .rename(columns={"what": f"{col_name}_initial"})
    )


priority_initial = get_initial_attribute(priority, "priority")
severity_initial = get_initial_attribute(severity, "severity")
component_initial = get_initial_attribute(component, "component")
assigned_to_initial = get_initial_attribute(assigned_to, "assigned_to")
short_desc_first = (
    short_desc.sort_values("when")
    .groupby("id")["what"]
    .first()
    .reset_index()
    .rename(columns={"what": "short_desc"})
)

cc_count = (
    cc.groupby("id")["what"]
    .nunique()
    .reset_index()
    .rename(columns={"what": "cc_count"})
)
reassignment_counts = (
    assigned_to.groupby("id")["what"]
    .nunique()
    .apply(lambda n: n - 1)
    .reset_index()
    .rename(columns={"what": "reassignment_count"})
)
reopening_counts = (
    bug_status[bug_status["what"] == "REOPENED"]
    .groupby("id")
    .size()
    .reset_index(name="reopening_count")
)

print("Calculating Time-in-Status features...")
status_changes = bug_status.sort_values(["id", "when"])
status_changes["next_when"] = status_changes.groupby("id")["when"].shift(-1)
last_timestamp = status_changes["when"].max()
status_changes["next_when"].fillna(last_timestamp, inplace=True)
status_changes["duration"] = (
    status_changes["next_when"] - status_changes["when"]
).dt.total_seconds() / 3600
time_in_status = status_changes.pivot_table(
    index="id", columns="what", values="duration", aggfunc="sum"
).reset_index()
time_in_status.columns = [
    f"duration_in_{col.strip().upper()}" for col in time_in_status.columns
]
time_in_status.rename(columns={"duration_in_ID": "id"}, inplace=True)


# --- Assemble Master Dataset ---
print("Assembling master dataset...")
master_data = reports[["id", "opening", "reporter"]]
feature_dfs = [
    final_resolution,
    short_desc_first,
    priority_initial,
    severity_initial,
    component_initial,
    assigned_to_initial,
    cc_count,
    reassignment_counts,
    reopening_counts,
    time_in_status,
]
for df in feature_dfs:
    master_data = pd.merge(master_data, df, on="id", how="left")

# --- Filter and Clean ---
final_data = master_data[
    master_data["final_resolution"].isin(
        ["fixed", "invalid", "wontfix", "duplicate", "worksforme"]
    )
].copy()
final_data.sort_values("opening", inplace=True)
final_data.reset_index(drop=True, inplace=True)

# Fill NaNs for base features
final_data.fillna(
    {"cc_count": 0, "short_desc": "", "reassignment_count": 0, "reopening_count": 0},
    inplace=True,
)
# Fill NaNs for new duration features
for col in final_data.columns:
    if "duration_in_" in col:
        final_data[col] = final_data[col].fillna(0)
final_data["target_binary"] = (final_data["final_resolution"] == "fixed").astype(int)


# 4. ADVANCED & DYNAMIC FEATURE ENGINEERING
# ==================================================================
print("Calculating advanced time-series and text features...")
reporter_history = {}
assignee_history = {}
component_history = {}
advanced_features = []
for index, row in final_data.iterrows():
    reporter = str(row["reporter"])
    assignee = str(row["assigned_to_initial"])
    component_name = str(row["component_initial"])
    reporter_history.setdefault(reporter, {"total": 0, "fixed": 0})
    assignee_history.setdefault(assignee, {"total": 0, "fixed": 0})
    component_history.setdefault(component_name, {"total": 0, "fixed": 0})
    reporter_rate = (
        (reporter_history[reporter]["fixed"] / reporter_history[reporter]["total"])
        if reporter_history[reporter]["total"] > 0
        else 0
    )
    assignee_rate = (
        (assignee_history[assignee]["fixed"] / assignee_history[assignee]["total"])
        if assignee_history[assignee]["total"] > 0
        else 0
    )
    component_rate = (
        (
            component_history[component_name]["fixed"]
            / component_history[component_name]["total"]
        )
        if component_history[component_name]["total"] > 0
        else 0
    )
    advanced_features.append(
        {
            "reporter_success_rate": reporter_rate,
            "assignee_success_rate": assignee_rate,
            "component_fix_rate": component_rate,
        }
    )
    is_fixed = 1 if row["final_resolution"] == "fixed" else 0
    reporter_history[reporter]["total"] += 1
    reporter_history[reporter]["fixed"] += is_fixed
    assignee_history[assignee]["total"] += 1
    assignee_history[assignee]["fixed"] += is_fixed
    component_history[component_name]["total"] += 1
    component_history[component_name]["fixed"] += is_fixed
final_data = pd.concat(
    [final_data, pd.DataFrame(advanced_features, index=final_data.index)], axis=1
)

final_data["desc_length"] = final_data["short_desc"].str.len().fillna(0)
keywords = ["crash", "fail", "error", "npe", "exception", "patch", "block"]
for keyword in keywords:
    final_data[f"has_keyword_{keyword}"] = (
        final_data["short_desc"]
        .str.contains(keyword, case=False, regex=False)
        .astype(int)
    )
severity_map = {
    "trivial": 1,
    "minor": 2,
    "normal": 3,
    "major": 4,
    "critical": 5,
    "blocker": 6,
}
final_data["severity_numeric"] = (
    final_data["severity_initial"].map(severity_map).fillna(3)
)
final_data["comp_rate_x_reassign"] = (
    final_data["component_fix_rate"] * final_data["reassignment_count"]
)
final_data["opening_wday"] = final_data["opening"].dt.dayofweek
print("All feature engineering complete.")


# 5. WALK-FORWARD VALIDATION WITH STACKING ENSEMBLE
# ==================================================================
n_splits = 13
split_size = len(final_data) // n_splits
auc_scores = []

base_cols = [
    "cc_count",
    "reassignment_count",
    "reopening_count",
    "reporter_success_rate",
    "assignee_success_rate",
    "component_fix_rate",
    "desc_length",
    "severity_numeric",
    "comp_rate_x_reassign",
    "opening_wday",
]
keyword_cols = [f"has_keyword_{k}" for k in keywords]
duration_cols = [col for col in final_data.columns if "duration_in_" in col]
pagerank_cols = ["reporter_pagerank", "assignee_pagerank", "pagerank_diff"]
structured_feature_cols = base_cols + keyword_cols + duration_cols + pagerank_cols

print(f"\nStarting Walk-Forward Validation with {n_splits} splits...")

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

for i in range(1, n_splits):
    train_end_idx = i * split_size
    validation_start_idx = i * split_size
    validation_end_idx = (i + 1) * split_size

    train_data = final_data.iloc[:train_end_idx].copy()
    validation_data = final_data.iloc[validation_start_idx:validation_end_idx].copy()

    if len(validation_data) == 0:
        continue

    print(f"\n--- Fold {i+1}/{n_splits} ---")
    print(
        f"Training on {len(train_data)} samples, validating on {len(validation_data)} samples."
    )

    edges = train_data[["reporter", "assigned_to_initial"]].dropna()
    g = nx.from_pandas_edgelist(
        edges, "reporter", "assigned_to_initial", create_using=nx.DiGraph()
    )
    pagerank_scores = nx.pagerank(g, alpha=0.85)
    train_data["reporter_pagerank"] = (
        train_data["reporter"].map(pagerank_scores).fillna(0)
    )
    train_data["assignee_pagerank"] = (
        train_data["assigned_to_initial"].map(pagerank_scores).fillna(0)
    )
    validation_data["reporter_pagerank"] = (
        validation_data["reporter"].map(pagerank_scores).fillna(0)
    )
    validation_data["assignee_pagerank"] = (
        validation_data["assigned_to_initial"].map(pagerank_scores).fillna(0)
    )
    train_data["pagerank_diff"] = (
        train_data["reporter_pagerank"] - train_data["assignee_pagerank"]
    )
    validation_data["pagerank_diff"] = (
        validation_data["reporter_pagerank"] - validation_data["assignee_pagerank"]
    )

    # --- Prepare data for model ---
    vectorizer = TfidfVectorizer(
        max_features=500, stop_words="english", ngram_range=(1, 2)
    )
    X_train_text = vectorizer.fit_transform(train_data["short_desc"])
    X_validation_text = vectorizer.transform(validation_data["short_desc"])

    X_train_structured = csr_matrix(train_data[structured_feature_cols].values)
    X_validation_structured = csr_matrix(
        validation_data[structured_feature_cols].values
    )
    X_train_final = hstack([X_train_structured, X_train_text])
    X_validation_final = hstack([X_validation_structured, X_validation_text])
    y_train = train_data["target_binary"]
    y_validation = validation_data["target_binary"]

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_final, y_train)

    print("Training Level 1 models...")
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        random_state=42,
        n_jobs=-1,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
    )

    xgb_model.fit(
        X_train_res,
        y_train_res,
        eval_set=[(X_validation_final, y_validation)],
        verbose=False,
    )
    lgb_model.fit(
        X_train_res,
        y_train_res,
        eval_set=[(X_validation_final, y_validation)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    print("Training Level 2 meta-model...")
    xgb_preds_val = xgb_model.predict_proba(X_validation_final)[:, 1]
    lgb_preds_val = lgb_model.predict_proba(X_validation_final)[:, 1]
    xgb_preds_train = xgb_model.predict_proba(X_train_res)[:, 1]
    lgb_preds_train = lgb_model.predict_proba(X_train_res)[:, 1]

    stacked_features_train = np.column_stack((xgb_preds_train, lgb_preds_train))
    stacked_features_validation = np.column_stack((xgb_preds_val, lgb_preds_val))

    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(stacked_features_train, y_train_res)

    final_pred_probs = meta_model.predict_proba(stacked_features_validation)[:, 1]
    auc_val = roc_auc_score(y_validation, final_pred_probs)
    auc_scores.append(auc_val)
    print(f"Fold {i+1} AUC: {auc_val:.4f}")

    if i == n_splits - 1:
        print("Final fold complete. Saving model and feature artifacts...")
        # 1. Save the trained XGBoost model
        joblib.dump(xgb_model, os.path.join(output_folder, "final_xgb_model.joblib"))

        # 2. Save the complete list of feature names
        feature_names = (
            structured_feature_cols + vectorizer.get_feature_names_out().tolist()
        )
        joblib.dump(feature_names, os.path.join(output_folder, "feature_names.joblib"))

        # 3. Save predictions and true labels for the confusion matrix
        preds_for_cm = {
            "y_true": y_validation.tolist(),
            "predictions": final_pred_probs.tolist(),
        }
        with open(os.path.join(output_folder, "final_predictions.json"), "w") as f:
            json.dump(preds_for_cm, f)

    # Plot and save ROC curve for the current fold
    fpr, tpr, _ = roc_curve(y_validation, final_pred_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"Stacking Ensemble (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Fold {i+1}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plot_path = os.path.join(output_folder, f"roc_plot_fold_{i+1}.svg")
    plt.savefig(plot_path)

    plt.close()
    gc.collect()

# 6. FINAL RESULTS
# ==================================================================
if auc_scores:
    file_path = os.path.join(output_folder, "auc_scores.json")
    with open(file_path, "w") as f:
        json.dump(auc_scores, f)

    print("\n--- Overall Walk-Forward Validation Results ---")
    print(f"Average Stacking Ensemble AUC: {np.mean(auc_scores):.4f}")
    print(f"Standard Deviation of AUC: {np.std(auc_scores):.4f}")
else:
    print("\nNo validation folds were processed.")
print("\n--- Script execution finished ---")
