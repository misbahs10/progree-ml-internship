
from ucimlrepo import fetch_ucirepo
import pandas as pd


TARGET_COLUMN = "Biopsy"

# Exclude other diagnostic outcomes to reduce target leakage.
LEAKAGE_COLUMNS = [
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "Dx",
    "Hinselmann",
    "Schiller",
    "Citology",
]


def load_dataset():
    """Load the UCI Cervical Cancer dataset and separate target."""

    print("Loading UCI Cervical Cancer dataset...")

    dataset = fetch_ucirepo(id=383)

    # UCI returns Biopsy inside features, not in targets.
    data = dataset.data.features.copy()

    # Handle question marks if present as missing-value markers.
    data = data.replace("?", pd.NA)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    # Separate target and predictors.
    y = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    X = data.drop(columns=[TARGET_COLUMN])

    # Remove other diagnostic outcome columns.
    X = X.drop(
        columns=LEAKAGE_COLUMNS,
        errors="ignore"
    )

    # Convert predictor columns to numeric where possible.
    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    # Remove rows where target is missing.
    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)

    print("\nDataset loaded successfully!")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target column: {TARGET_COLUMN}")

    return X, y


def inspect_dataset(X, y):
    """Inspect missing values and target class distribution."""

    print("\n" + "=" * 55)
    print("DATASET INFORMATION")
    print("=" * 55)

    print(f"\nTotal samples: {len(X)}")
    print(f"Total features: {X.shape[1]}")

    print("\nFeature data types:")
    print(X.dtypes.value_counts())

    print("\nMissing values per feature:")
    print(
        X.isnull()
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nTotal missing values:", X.isnull().sum().sum())

    print("\nTarget distribution:")
    counts = y.value_counts().sort_index()

    percentages = (
        y.value_counts(normalize=True).sort_index() * 100
    )

    distribution = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages
    })

    print(distribution)

    print("\nMinority class percentage:")
    print(f"{percentages.min():.2f}%")


if __name__ == "__main__":
    X, y = load_dataset()
    inspect_dataset(X, y)
