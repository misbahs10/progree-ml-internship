
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

HIGH_MISSING_COLUMNS = [
    "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis",
]


def prepare_features(X):
    X = X.copy()
    X = X.drop(columns=HIGH_MISSING_COLUMNS, errors="ignore")
    return X


def build_preprocessor():
    """Return individual preprocessing steps instead of a nested Pipeline."""

    return [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
