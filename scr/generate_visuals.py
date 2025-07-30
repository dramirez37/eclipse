import json
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_fold_performance(scores_file="output/auc_scores.json"):
    """
    Loads AUC scores from a JSON file and generates a bar chart of performance.
    """
    try:
        with open(scores_file, "r") as f:
            auc_scores = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {scores_file}.")
        print("Please run final.py first to generate the scores file.")
        return

    folds = range(2, len(auc_scores) + 2)
    mean_auc = np.mean(auc_scores)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12, 7))

    bars = plt.bar(folds, auc_scores, color="#2c7bb6", zorder=2)

    plt.axhline(
        y=mean_auc,
        color="#d7191c",
        linestyle="--",
        linewidth=2,
        label=f"Average AUC = {mean_auc:.4f}",
    )

    plt.xlabel("Validation Fold Number", fontsize=12)
    plt.ylabel("Area Under Curve (AUC)", fontsize=12)
    plt.title("Model Performance Across Walk-Forward Folds", fontsize=16, weight="bold")
    plt.xticks(folds)
    plt.ylim(0.7, 1.0)
    plt.legend(fontsize=12)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.005,
            f"{yval:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    output_filename = "output/fold_auc_performance.svg"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Successfully generated and saved '{output_filename}'")


def plot_feature_importance(
    model_path="output/final_xgb_model.joblib",
    features_path="output/feature_names.joblib",
):
    """
    Loads a trained model and its feature names to plot the
    top 15 most important features.
    """
    try:
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
    except FileNotFoundError:
        print("Error: Model or feature names file not found.")
        print("Please run final.py first to save these artifacts.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.nlargest(15).sort_values(ascending=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12, 8))

    bars = top_features.plot(kind="barh", color="#f4a582")
    plt.title(
        "Top 15 Most Important Features for Predicting Bug Fixes",
        fontsize=16,
        weight="bold",
    )
    plt.xlabel("Importance Score (XGBoost)", fontsize=12)
    plt.ylabel("Features", fontsize=12)

    plt.xlim(right=top_features.max() * 1.15)

    for bar in bars.containers[0]:
        plt.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            va="center",
            ha="left",
        )
    plt.tight_layout()
    output_filename = "output/feature_importance.svg"
    plt.savefig(output_filename, dpi=300)
    plt.close()

    print(f"Successfully generated and saved '{output_filename}'")


def plot_confusion_matrix(data_path="output/final_predictions.json"):
    """
    Loads true labels and predictions to generate a confusion matrix heatmap.
    """
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
        y_true = data["y_true"]
        final_preds = data["predictions"]
    except FileNotFoundError:
        print("Error: Predictions file not found.")
        print("Please run final.py first to save these artifacts.")
        return

    cm = confusion_matrix(y_true, (np.array(final_preds) > 0.5).astype(int))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Fixed", "Fixed"],
        yticklabels=["Not Fixed", "Fixed"],
        annot_kws={"size": 16},
    )

    plt.title("Model Prediction Performance (Final Fold)", fontsize=16, weight="bold")
    plt.xlabel("Predicted Outcome", fontsize=12)
    plt.ylabel("Actual Outcome", fontsize=12)

    output_filename = "output/confusion_matrix.svg"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Successfully generated and saved '{output_filename}'")


def plot_lift_chart(data_path="output/final_predictions.json"):
    """
    Loads predictions and true values to generate a cumulative gains (lift) chart.
    """
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
        y_true = np.array(data["y_true"])
        y_pred_proba = np.array(data["predictions"])
    except FileNotFoundError:
        print(f"Error: Predictions file not found: {data_path}")
        print("Please run final.py first to save these artifacts.")
        return

    results_df = pd.DataFrame({"y_true": y_true, "y_pred_proba": y_pred_proba})
    results_df = results_df.sort_values(by="y_pred_proba", ascending=False)

    total_positives = results_df["y_true"].sum()
    results_df["cumulative_gain"] = results_df["y_true"].cumsum() / total_positives
    results_df["percentage_of_data"] = np.arange(1, len(results_df) + 1) / len(
        results_df
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 8))
    plt.plot(
        results_df["percentage_of_data"],
        results_df["cumulative_gain"],
        color="#2c7bb6",
        lw=2,
        label="Model Gains",
    )
    plt.plot([0, 1], [0, 1], "k--", label="Random Selection (Baseline)")

    plt.title("Cumulative Gains Chart", fontsize=16, weight="bold")
    plt.xlabel("Percentage of Bugs Targeted (sorted by score)", fontsize=12)
    plt.ylabel('Percentage of "Fixed" Bugs Captured', fontsize=12)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter("{:.0%}".format))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter("{:.0%}".format))
    plt.legend(fontsize=12)
    plt.grid(True)

    point_20_percent = results_df.iloc[int(len(results_df) * 0.2) - 1]
    gain_at_20 = point_20_percent["cumulative_gain"]
    plt.plot([0.2], [gain_at_20], "ro")
    plt.annotate(
        f"Targeting top 20% of bugs\ncaptures {gain_at_20:.0%} of all fixes",
        xy=(0.2, gain_at_20),
        xytext=(0.25, gain_at_20 - 0.15),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=8),
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.6),
    )

    output_filename = "output/lift_chart.svg"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Successfully generated and saved '{output_filename}'")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    plot_fold_performance()
    plot_feature_importance()
    plot_confusion_matrix()
    plot_lift_chart()
