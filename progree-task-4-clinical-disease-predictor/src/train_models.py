import json
from pathlib import Path
import pandas as pd

import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from sklearn.metrics import fbeta_score, make_scorer
from src.preprocessing import (
    prepare_features,
    build_preprocessor,
)


RANDOM_STATE = 42


def split_dataset(X, y):
    """Split data into stratified training and testing sets."""

    X = prepare_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTrain/Test Split")
    print("-" * 40)
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Training positive cases:", int(y_train.sum()))
    print("Testing positive cases:", int(y_test.sum()))

    return X_train, X_test, y_train, y_test



def get_models():
    """Define classifiers and hyperparameter grids."""

    models = {
        "SVM": {
            "classifier": SVC(
                probability=True,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "classifier__C": [0.1, 1, 10],
                "classifier__kernel": ["linear", "rbf"],
            },
        },

        "Random Forest": {
            "classifier": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5],
            },
        },

        "XGBoost": {
            "classifier": XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                scale_pos_weight=1,
            ),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [2, 3, 5],
                "classifier__learning_rate": [0.01, 0.1],
            },
        },
    }

    return models

def build_model_pipeline(classifier):
    """Create preprocessing + SMOTE + classifier pipeline."""

    preprocessing_steps = build_preprocessor()

    pipeline = ImbPipeline(
        steps=[
            *preprocessing_steps,
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", classifier),
        ]
    )

    return pipeline


def train_models(X_train, y_train):
    """Tune classifiers using recall-focused F2 scoring."""

    models = get_models()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    f2_scorer = make_scorer(
        fbeta_score,
        beta=2,
        zero_division=0,
    )

    trained_models = {}

    for model_name, config in models.items():

        print("\n" + "=" * 55)
        print(f"Training: {model_name}")
        print("=" * 55)

        pipeline = build_model_pipeline(config["classifier"])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=config["params"],
            scoring=f2_scorer,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )

        grid_search.fit(X_train, y_train)

        trained_models[model_name] = grid_search.best_estimator_

        print("\nBest parameters:")
        print(grid_search.best_params_)

        print(f"Best CV F2-score: {grid_search.best_score_:.4f}")

    return trained_models

def evaluate_cross_validation(trained_models, X_train, y_train):
    """Calculate CV metrics and save individual fold scores."""

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    output_dir = Path("outputs/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_results = {}
    json_results = {}
    csv_rows = []

    for model_name, model in trained_models.items():

        print("\n" + "=" * 55)
        print(f"Cross-validation: {model_name}")
        print("=" * 55)

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        cv_results[model_name] = scores
        json_results[model_name] = {}

        for metric in scoring:
            values = scores[f"test_{metric}"]

            json_results[model_name][metric] = {
                "fold_scores": values.tolist(),
                "mean": float(values.mean()),
                "std": float(values.std()),
            }

            for fold_number, score in enumerate(values, start=1):
                csv_rows.append({
                    "model": model_name,
                    "metric": metric,
                    "fold": fold_number,
                    "score": float(score),
                })

            print(f"\n{metric.upper()}: {np.round(values, 4)}")
            print(
                f"Mean: {values.mean():.4f} | "
                f"Std: {values.std():.4f}"
            )

    # Save individual fold scores and summary as JSON
    json_path = output_dir / "cross_validation_results.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(json_results, file, indent=4)

    # Save individual fold scores in CSV format
    csv_path = output_dir / "cross_validation_fold_scores.csv"

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print("\nCross-validation results saved:")
    print(f"- {json_path}")
    print(f"- {csv_path}")

    return cv_results