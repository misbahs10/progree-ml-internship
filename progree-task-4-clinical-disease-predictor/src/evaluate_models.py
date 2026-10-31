
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"


def create_output_directories():
    """Create directories for evaluation outputs."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_models(trained_models, X_test, y_test):
    """Evaluate models on untouched test data and save results."""

    create_output_directories()

    all_metrics = {}
    roc_data = {}
    pr_data = {}

    for model_name, model in trained_models.items():

        print("\n" + "=" * 55)
        print(f"TEST SET EVALUATION: {model_name}")
        print("=" * 55)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )
        roc_auc = roc_auc_score(y_test, y_proba)
        avg_precision = average_precision_score(
            y_test, y_proba
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "average_precision": float(avg_precision),
        }

        all_metrics[model_name] = metrics

        roc_data[model_name] = (y_test, y_proba)
        pr_data[model_name] = (y_test, y_proba)

        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Precision:         {precision:.4f}")
        print(f"Recall:            {recall:.4f}")
        print(f"F1-score:          {f1:.4f}")
        print(f"ROC-AUC:           {roc_auc:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")

        print("\nClassification Report:")

        print(
            classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=["Negative", "Positive"],
                zero_division=0,
            )
        )

        # Confusion matrix
        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        )

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )

        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()

        filename = (
            model_name.lower().replace(" ", "_")
            + "_confusion_matrix.png"
        )

        plt.savefig(
            FIGURES_DIR / filename,
            dpi=300,
        )
        plt.close()

    # Combined ROC curve
    plt.figure(figsize=(8, 6))

    for model_name, (actual, probabilities) in roc_data.items():

        fpr, tpr, _ = roc_curve(
            actual,
            probabilities,
        )

        auc_value = all_metrics[model_name]["roc_auc"]

        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC = {auc_value:.3f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "roc_curve_comparison.png",
        dpi=300,
    )
    plt.close()

    # Precision-Recall curve comparison
    plt.figure(figsize=(8, 6))

    for model_name, (actual, probabilities) in pr_data.items():

        precision_values, recall_values, _ = (
            precision_recall_curve(
                actual,
                probabilities,
            )
        )

        ap_value = all_metrics[model_name]["average_precision"]

        plt.plot(
            recall_values,
            precision_values,
            label=f"{model_name} (AP = {ap_value:.3f})",
        )

    # Baseline: positive-class prevalence
    baseline = float(y_test.mean())

    plt.axhline(
        y=baseline,
        linestyle="--",
        label=f"Baseline = {baseline:.3f}",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "precision_recall_curve_comparison.png",
        dpi=300,
    )
    plt.close()

    # Model comparison chart
    comparison_df = pd.DataFrame(
        all_metrics
    ).T

    comparison_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "average_precision",
    ]

    comparison_df[comparison_metrics].plot(
        kind="bar",
        figsize=(11, 6),
    )

    plt.title("Test Set Model Performance Comparison")
    plt.xlabel("Classifier")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(
        title="Metrics",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "model_comparison.png",
        dpi=300,
    )
    plt.close()

    # Save metrics as JSON
    metrics_path = METRICS_DIR / "test_metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(all_metrics, file, indent=4)

    print("\nEvaluation outputs saved:")
    print(f"- {metrics_path}")
    print("- outputs/figures/")

    return all_metrics
