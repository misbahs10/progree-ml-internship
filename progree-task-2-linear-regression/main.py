
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

# -----------------------------
# 1. Project paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 2. Load dataset
# -----------------------------
print("Loading Ames Housing dataset...")

housing = fetch_openml(
    name="house_prices",
    version=1,
    as_frame=True,
)

X = housing.data.copy()
y = pd.to_numeric(housing.target, errors="coerce")

# Drop identifier column; it is not a meaningful house feature.
X = X.drop(columns=["Id"], errors="ignore")

# Remove rows with missing target values, if any.
valid_target = y.notna()
X = X.loc[valid_target].copy()
y = y.loc[valid_target].astype(float)

print(f"Dataset shape: {X.shape}")
print(f"Target: {housing.target.name}")

# -----------------------------
# 3. Identify feature types
# -----------------------------
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# -----------------------------
# 4. Preprocessing pipelines
# -----------------------------
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)

# -----------------------------
# 5. Full model pipeline
# -----------------------------
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge()),
    ]
)

# -----------------------------
# 6. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

# -----------------------------
# 7. Tune Ridge regularization
# -----------------------------
parameter_grid = {
    "regressor__alpha": [0.1, 1.0, 10.0, 100.0],
}

search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=parameter_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1,
)

print("Training and tuning Ridge Regression...")
search.fit(X_train, y_train)

best_model = search.best_estimator_

# -----------------------------
# 8. Evaluate on held-out test data
# -----------------------------
predictions = best_model.predict(X_test)

r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n===== MODEL RESULTS =====")
print(f"Best Ridge alpha: {search.best_params_['regressor__alpha']}")
print(f"Best CV R2: {search.best_score_:.4f}")
print(f"Test R2: {r2:.4f}")
print(f"Test RMSE: ${rmse:,.2f}")

# -----------------------------
# 9. Save metrics and predictions
# -----------------------------
metrics = {
    "dataset": "Ames Housing (OpenML house_prices)",
    "model": "Ridge Regression",
    "best_alpha": float(
        search.best_params_["regressor__alpha"]
    ),
    "best_cross_validation_r2": float(search.best_score_),
    "test_r2": float(r2),
    "test_rmse": float(rmse),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "numeric_features": int(len(numeric_features)),
    "categorical_features": int(len(categorical_features)),
}

with open(
    OUTPUT_DIR / "metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=4)

pd.DataFrame(
    {
        "actual_sale_price": y_test.to_numpy(),
        "predicted_sale_price": predictions,
    }
).to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False,
)

joblib.dump(best_model, OUTPUT_DIR / "ridge_model.joblib")

# -----------------------------
# 10. Actual vs predicted plot
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.7)

min_price = min(y_test.min(), predictions.min())
max_price = max(y_test.max(), predictions.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--",
)

plt.xlabel("Actual Sale Price ($)")
plt.ylabel("Predicted Sale Price ($)")
plt.title("Ames Housing: Actual vs Predicted")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=150,
)
plt.close()

# -----------------------------
# 11. Residual plot
# -----------------------------
residuals = y_test.to_numpy() - predictions

plt.figure(figsize=(8, 6))
plt.scatter(predictions, residuals, alpha=0.7)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Sale Price ($)")
plt.ylabel("Residual ($)")
plt.title("Ridge Regression: Residual Plot")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "residual_plot.png",
    dpi=150,
)
plt.close()

print("\nAll outputs saved in the outputs/ folder.")
print("Task 2 model pipeline completed successfully!")