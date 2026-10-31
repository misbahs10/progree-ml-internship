from pathlib import Path
import joblib
from src.data_loader import load_dataset, inspect_dataset
from src.train_models import (
    split_dataset,
    train_models,
    evaluate_cross_validation,
)
from src.evaluate_models import evaluate_models


def main():

    print("=" * 60)
    print("PROGREE INTERNSHIP - TASK 4")
    print("Clinical Disease Predictor Pipeline")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n[STEP 1] Loading dataset...")
    X, y = load_dataset()

    # Step 2: Inspect dataset
    print("\n[STEP 2] Inspecting dataset...")
    inspect_dataset(X, y)

    # Step 3: Split dataset
    print("\n[STEP 3] Preparing train/test split...")

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Step 4: Train and tune models
    print("\n[STEP 4] Training and tuning classifiers...")

    trained_models = train_models(
        X_train,
        y_train,
    )

    # Step 4.1: Save trained models
    print("\n[STEP 4.1] Saving trained models...")

    models_dir = Path("outputs/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in trained_models.items():

        file_name = model_name.lower().replace(" ", "_") + ".joblib"
        model_path = models_dir / file_name

        joblib.dump(model, model_path)

        print(f"Saved: {model_path}")

    # Step 5: Cross-validation
    print("\n[STEP 5] Evaluating models with cross-validation...")

    cv_results = evaluate_cross_validation(
        trained_models,
        X_train,
        y_train,
    )

    # Step 6: Test set evaluation
    print("\n[STEP 6] Evaluating models on untouched test data...")

    test_metrics = evaluate_models(
        trained_models,
        X_test,
        y_test,
    )

    # Step 7: Final comparison
    print("\n" + "=" * 60)
    print("FINAL TEST SET MODEL COMPARISON")
    print("=" * 60)

    for model_name, metrics in test_metrics.items():

        print(f"\n{model_name}")
        print("-" * 40)

        for metric_name, value in metrics.items():
            print(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED")
    print("=" * 60)

    print("\nTrained models:")
    for model_name in trained_models:
        print(f"- {model_name}")

    print("\nCross-validation results collected:")
    for model_name in cv_results:
        print(f"- {model_name}")

    print("\nGenerated outputs:")
    print("- outputs/metrics/test_metrics.json")
    print("- outputs/metrics/cross_validation_results.json")
    print("- outputs/metrics/cross_validation_fold_scores.csv")
    print("- outputs/figures/")
    print("- outputs/models/")
    print("\nTask 4 pipeline completed successfully.")


if __name__ == "__main__":
    main()
