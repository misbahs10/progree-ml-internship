# Progree Internship — Task 2: Predictive Linear Regression Model Pipeline

## Overview

This project implements an end-to-end machine learning pipeline to predict residential house sale prices using the Ames Housing dataset. It uses Scikit-learn preprocessing pipelines and Ridge Regression to build and evaluate a predictive model.

## Objective

- Build a predictive Linear Regression pipeline using Scikit-learn.
- Handle missing numerical and categorical values.
- Apply feature scaling and categorical encoding.
- Train and tune a regularized Ridge Regression model.
- Evaluate performance using R² and Root Mean Squared Error (RMSE).

## Dataset

- **Name:** Ames Housing / House Prices
- **Source:** [OpenML — House Prices](https://www.openml.org/search?type=data&status=active&id=43926)
- **Samples:** 1,460
- **Original features:** 80
- **Target variable:** `SalePrice`

The `Id` column is removed because it is an identifier rather than a meaningful predictive feature. The remaining dataset contains 79 features, including numerical and categorical variables.

The dataset is loaded programmatically using Scikit-learn's `fetch_openml`.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Machine Learning Pipeline

1. Load the Ames Housing dataset.
2. Remove the identifier column and handle missing target values.
3. Split the data into training and testing sets using an 80/20 split.
4. Impute missing numerical values using the mean.
5. Standardize numerical features using `StandardScaler`.
6. Impute missing categorical values using the most frequent value.
7. Encode categorical features using `OneHotEncoder`.
8. Train Ridge Regression using a Scikit-learn `Pipeline`.
9. Select the regularization parameter (`alpha`) using five-fold cross-validation on the training set.
10. Evaluate the selected model on the held-out test set.
11. Save evaluation metrics, predictions, plots, and the trained model.

## Model Results

Results from the current run:

| Metric | Value |
|---|---:|
| Best Ridge alpha | 10.0 |
| Best cross-validation R² | 0.8166 |
| Test R² | 0.8780 |
| Test RMSE | $30,592.53 |
| Training samples | 1,168 |
| Test samples | 292 |

The test R² of 0.8780 indicates that the model explains approximately 87.8% of the variation in sale prices in the test set. The RMSE summarizes prediction error in the dataset's sale-price units.

## Output Files

The `outputs/` directory contains:

- `metrics.json` — evaluation metrics and model configuration.
- `predictions.csv` — actual and predicted sale prices.
- `ridge_model.joblib` — trained preprocessing and Ridge Regression pipeline.
- `actual_vs_predicted.png` — plot comparing actual and predicted prices.
- `residual_plot.png` — residual diagnostic plot.

## Setup and Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

Run the project:

```powershell
python main.py
```

An internet connection is required when the dataset is first fetched from OpenML.

## Notes and Limitations

- Results depend on the dataset, train/test split, and selected model configuration.
- RMSE is expressed in the sale-price units used by the dataset.
- The model is intended for educational purposes and should not be treated as a professional property valuation service.
- Actual outputs may change if the dataset, preprocessing, random seed, or model configuration changes.

## Internship

**Program:** Progree Remote Machine Learning Internship  
**Task:** Task 2 — Predictive Linear Regression Model Pipeline
