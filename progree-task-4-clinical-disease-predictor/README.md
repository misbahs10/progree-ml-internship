# Clinical Disease Predictor — High-Imbalance Classification

**Progree Remote Machine Learning Internship — Task 4**

## Project Overview

This project implements a machine learning pipeline for predicting cervical cancer biopsy outcomes using the UCI Cervical Cancer (Risk Factors) dataset.

The primary challenge is severe class imbalance: only 6.41% of the observations belong to the positive class. To address this, the project uses SMOTE oversampling within an imbalanced-learn pipeline and evaluates multiple classifiers with a recall-oriented F2 scoring strategy.

The project compares Support Vector Machine (SVM), Random Forest, and XGBoost classifiers using stratified cross-validation and an untouched test set.

> **Disclaimer:** This project is an educational machine learning experiment, not a clinically validated diagnostic system.

## Objectives

- Load and inspect a highly imbalanced clinical dataset.
- Handle missing values and prevent data leakage.
- Apply SMOTE to address class imbalance.
- Train and tune multiple classification algorithms.
- Prioritize recall using F2-score during hyperparameter tuning.
- Evaluate models through stratified 5-fold cross-validation.
- Generate classification metrics and visualizations.

## Dataset

**Dataset:** UCI Cervical Cancer (Risk Factors)  
**UCI Dataset ID:** 383  
**Target variable:** `Biopsy`

| Description | Value |
|---|---:|
| Total observations | 858 |
| Initial features | 28 |
| Negative cases | 803 |
| Positive cases | 55 |
| Negative class | 93.59% |
| Positive class | 6.41% |
| Missing values | 3,622 |

Diagnostic outcome columns were excluded to reduce target leakage. Two features with approximately 91.7% missing values were also removed.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib
- UCI Machine Learning Repository

## Machine Learning Workflow

1. Load the UCI dataset.
2. Inspect feature types, missing values, and target distribution.
3. Remove target leakage and highly incomplete features.
4. Split data using stratified train/test splitting.
5. Impute missing numerical values using median imputation.
6. Standardize numerical features.
7. Apply SMOTE to training folds within the model pipeline.
8. Train and tune SVM, Random Forest, and XGBoost.
9. Optimize hyperparameters using 5-fold stratified cross-validation and F2-score.
10. Evaluate the selected models on untouched test data.
11. Save models, metrics, cross-validation results, and visualizations.

## Models and Test Results

| Metric | SVM | Random Forest | XGBoost |
|---|---:|---:|---:|
| Accuracy | 0.8256 | 0.8314 | 0.7616 |
| Precision | 0.1724 | 0.0909 | 0.0833 |
| Recall | **0.4545** | 0.1818 | 0.2727 |
| F1-score | **0.2500** | 0.1212 | 0.1277 |
| ROC-AUC | 0.5878 | **0.6443** | 0.5692 |
| Average Precision | **0.1223** | 0.1022 | 0.1063 |

### Cross-Validation Summary

| Model | Mean Recall | Mean ROC-AUC |
|---|---:|---:|
| SVM | 0.4056 | 0.6154 |
| Random Forest | 0.2472 | 0.5995 |
| XGBoost | **0.4111** | **0.6488** |

### Best Hyperparameters

- **SVM:** C = 0.1, kernel = linear
- **Random Forest:** max_depth = 5, min_samples_split = 2, n_estimators = 100
- **XGBoost:** learning_rate = 0.01, max_depth = 3, n_estimators = 100

## Results and Observations

SVM achieved the highest test recall of 45.45%, detecting 5 out of 11 positive cases. It also achieved the highest test precision, F1-score, and Average Precision among the three models.

Random Forest achieved the highest test ROC-AUC of 0.6443 and accuracy of 83.14%, but its positive-class recall was only 18.18%.

XGBoost achieved the highest mean cross-validation recall and ROC-AUC, but its test-set performance was weaker than SVM on recall.

The results show that accuracy alone is not sufficient for evaluating this highly imbalanced classification problem.

## Project Structure

```text
progree-task-4-clinical-disease-predictor/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── utils.py
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
│
├── report/
│   └── task_4_clinical_disease_predictor_report.md
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Generated Outputs

### Figures

- `svm_confusion_matrix.png`
- `random_forest_confusion_matrix.png`
- `xgboost_confusion_matrix.png`
- `roc_curve_comparison.png`
- `precision_recall_curve_comparison.png`
- `model_comparison.png`

### Metrics

- `test_metrics.json`
- `cross_validation_results.json`
- `cross_validation_fold_scores.csv`

### Trained Models

- `svm.joblib`
- `random_forest.joblib`
- `xgboost.joblib`

## Installation and Execution

Clone the repository or navigate to the project directory.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete pipeline:

```bash
python main.py
```

## Limitations

- The dataset contains only 858 observations.
- There are only 55 positive samples in the full dataset.
- The test set contains only 11 positive cases.
- Positive-class precision and recall remain limited.
- Model performance requires validation on larger, independent datasets.
- The experiment does not establish clinical safety or diagnostic suitability.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for highly imbalanced clinical classification. SMOTE, stratified cross-validation, F2-based hyperparameter tuning, and multiple evaluation metrics were used to compare three classifiers.

SVM showed the strongest positive-class recall on the test set, while XGBoost achieved the strongest mean cross-validation recall and ROC-AUC.

Further work should investigate threshold selection, improved feature engineering, and validation using larger independent clinical datasets.

---

**Project:** Progree Remote Machine Learning Internship — Task 4  
**Focus:** High-Imbalance Clinical Disease Prediction