# High-Imbalance Clinical Disease Predictor Pipeline

**Progree Remote Machine Learning Internship — Task 4**  
**Project:** Clinical Disease Prediction using Machine Learning  
**Dataset:** UCI Cervical Cancer (Risk Factors)  
**Target Variable:** `Biopsy`

---

## 1. Abstract

This project develops an end-to-end machine learning pipeline for predicting cervical cancer biopsy outcomes using the UCI Cervical Cancer (Risk Factors) dataset. The main challenge is severe class imbalance: only 55 out of 858 observations (6.41%) belong to the positive class.

To address this issue, the project applies Synthetic Minority Over-sampling Technique (SMOTE) within an imbalanced-learn pipeline. Three classification algorithms—Support Vector Machine (SVM), Random Forest, and XGBoost—are trained and tuned using stratified five-fold cross-validation. Hyperparameter selection uses the F2-score to give greater importance to recall than precision.

The final models are evaluated on an untouched test set using accuracy, precision, recall, F1-score, ROC-AUC, and Average Precision. Confusion matrices, ROC curves, precision-recall curves, and a model comparison chart are also generated.

SVM achieves the highest test recall of 45.45%, while Random Forest achieves the highest test ROC-AUC of 0.6443. The results demonstrate the importance of class-imbalance-aware evaluation and show that further improvements are necessary before any clinical application could be considered.

## 2. Introduction

Machine learning classification problems involving medical data often contain imbalanced target classes. In such datasets, one class may have significantly more observations than another.

This imbalance can cause a classifier to favor the majority class. Consequently, a model may achieve high accuracy while failing to identify a substantial proportion of positive cases.

In a clinical prediction context, missing positive cases is an important concern. Therefore, this project focuses on evaluating models using recall, precision, F1-score, F2-score, ROC-AUC, and Average Precision rather than relying on accuracy alone.

### 2.1 Project Objectives

The main objectives are:

1. Load and analyze a highly imbalanced clinical dataset.
2. Inspect missing values and target-class distribution.
3. Reduce potential target leakage by excluding diagnostic outcome columns.
4. Apply appropriate preprocessing and SMOTE.
5. Train and tune SVM, Random Forest, and XGBoost classifiers.
6. Optimize hyperparameters using an F2-based scoring strategy.
7. Evaluate models using stratified five-fold cross-validation.
8. Compare final model performance on untouched test data.
9. Save trained models, evaluation metrics, and visualizations.

## 3. Dataset Description

The project uses the Cervical Cancer (Risk Factors) dataset from the UCI Machine Learning Repository.

**Dataset information:**

- Dataset name: Cervical Cancer (Risk Factors)
- UCI dataset ID: 383
- Target column: `Biopsy`
- Total observations: 858
- Features initially available after target-leakage exclusions: 28
- Target classes: 0 and 1

### 3.1 Target Distribution

The target variable is distributed as follows:

| Class | Meaning | Samples | Percentage |
|---|---|---:|---:|
| 0 | Negative | 803 | 93.59% |
| 1 | Positive | 55 | 6.41% |
| **Total** | | **858** | **100%** |

The positive class represents only 6.41% of the dataset. This confirms that the classification problem has substantial class imbalance.

### 3.2 Missing Values

The dataset contains 3,622 missing feature values.

The two most incomplete columns are:

| Feature | Missing Values |
|---|---:|
| STDs: Time since first diagnosis | 787 |
| STDs: Time since last diagnosis | 787 |

Each of these columns has approximately 91.7% missing values.

Other incomplete features include IUD, hormonal contraceptives, STD-related variables, number of pregnancies, and smoking-related variables.

### 3.3 Target Leakage Prevention

The data-loading stage excludes diagnostic outcome columns that could reveal information closely related to the biopsy outcome.

The excluded columns include:

- `Dx:Cancer`
- `Dx:CIN`
- `Dx:HPV`
- `Dx`
- `Hinselmann`
- `Schiller`
- `Citology`

The target column `Biopsy` is separated from the input features.

This step is important because including diagnostic outcomes as input variables could produce misleadingly optimistic model performance.

## 4. Data Preprocessing

Data preprocessing is performed before model training to prepare the features for classification.

### 4.1 Feature Selection

The two features with approximately 91.7% missing values are removed:

- `STDs: Time since first diagnosis`
- `STDs: Time since last diagnosis`

After removing these columns, 26 input features remain for model training.

### 4.2 Train-Test Split

The dataset is divided using stratified train-test splitting.

| Dataset | Samples | Positive Cases |
|---|---:|---:|
| Training | 686 | 44 |
| Testing | 172 | 11 |
| Total | 858 | 55 |

The split uses:

- Test size: 20%
- Random state: 42
- Stratification: Target variable

Stratification maintains approximately the same class distribution in both subsets.

The test set remains separate from model fitting and hyperparameter selection.

### 4.3 Missing-Value Imputation

`SimpleImputer` with median strategy is used to fill missing numerical feature values.

Median imputation is selected because it is less sensitive to extreme values than mean imputation.

### 4.4 Feature Scaling

`StandardScaler` is applied to standardize numerical features.

The standardization formula is:

\[
z = \frac{x-\mu}{\sigma}
\]

Where:

- \(x\) represents the original feature value.
- \(\mu\) represents the training mean.
- \(\sigma\) represents the training standard deviation.
- \(z\) represents the standardized value.

Scaling is particularly relevant for SVM because it is sensitive to feature magnitudes.

### 4.5 Handling Class Imbalance with SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) is used to address the imbalance between positive and negative classes.

Instead of simply duplicating minority-class observations, SMOTE generates synthetic minority-class samples using nearest-neighbor information.

SMOTE is placed inside the imbalanced-learn pipeline, after imputation and scaling.

This ensures that oversampling is performed on training data within each cross-validation fold rather than on the complete dataset before splitting.

The untouched test set is not oversampled.

## 5. Model Development and Hyperparameter Tuning

Three machine learning algorithms are evaluated.

### 5.1 Support Vector Machine (SVM)

SVM is a supervised learning algorithm that identifies a decision boundary separating different classes.

The experiment evaluates:

- C: 0.1, 1, 10
- Kernel: linear, RBF

Best parameters:

| Parameter | Selected Value |
|---|---|
| C | 0.1 |
| Kernel | linear |

Best cross-validation F2-score: **0.2938**

### 5.2 Random Forest

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

The experiment evaluates:

- Number of estimators: 100, 200
- Maximum depth: None, 5, 10
- Minimum samples split: 2, 5

Best parameters:

| Parameter | Selected Value |
|---|---|
| Number of estimators | 100 |
| Maximum depth | 5 |
| Minimum samples split | 2 |

Best cross-validation F2-score: **0.2186**

### 5.3 XGBoost

XGBoost is a gradient-boosting algorithm that builds an ensemble of decision trees sequentially.

The experiment evaluates:

- Number of estimators: 100, 200
- Maximum depth: 2, 3, 5
- Learning rate: 0.01, 0.1

Best parameters:

| Parameter | Selected Value |
|---|---|
| Number of estimators | 100 |
| Maximum depth | 3 |
| Learning rate | 0.01 |

Best cross-validation F2-score: **0.2863**

### 5.4 Hyperparameter Optimization

GridSearchCV is used to evaluate parameter combinations through five-fold stratified cross-validation.

The F2-score is selected as the optimization metric.

The general F-beta formula is:

\[
F_\beta =
(1+\beta^2)
\frac{Precision \times Recall}
{\beta^2 \times Precision + Recall}
\]

For F2-score, \(\beta=2\):

\[
F_2 =
5 \times
\frac{Precision \times Recall}
{4 \times Precision + Recall}
\]

F2 gives greater weight to recall than precision. This makes it suitable for an experiment where identifying positive cases is an important objective.

## 6. Cross-Validation Evaluation

After hyperparameter tuning, the selected models are evaluated using stratified five-fold cross-validation on the training dataset.

The evaluation records accuracy, precision, recall, F1-score, and ROC-AUC for each fold.

### 6.1 Cross-Validation Results

| Model | Mean Accuracy | Mean Precision | Mean Recall | Mean F1 | Mean ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVM | 0.8004 | 0.1537 | 0.4056 | 0.2157 | 0.6154 |
| Random Forest | 0.8658 | 0.1532 | 0.2472 | 0.1875 | 0.5995 |
| XGBoost | 0.7872 | 0.1355 | 0.4111 | 0.2008 | 0.6488 |

### 6.2 Cross-Validation Observations

- XGBoost achieves the highest mean cross-validation recall of 0.4111.
- XGBoost also achieves the highest mean cross-validation ROC-AUC of 0.6488.
- Random Forest achieves the highest mean accuracy of 0.8658.
- SVM achieves a mean recall of 0.4056, close to XGBoost.
- Precision remains low across all three classifiers.

The cross-validation results show that model performance varies across folds. This variation is important because the dataset contains relatively few positive observations.

## 7. Test Set Evaluation

The final selected models are evaluated on the untouched test set containing 172 observations, including 11 positive cases.

The following metrics are calculated:

**Accuracy**

\[
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
\]

**Precision**

\[
Precision = \frac{TP}{TP+FP}
\]

**Recall (Sensitivity)**

\[
Recall = \frac{TP}{TP+FN}
\]

**F1-score**

\[
F1 = 2 \times
\frac{Precision \times Recall}
{Precision+Recall}
\]

**ROC-AUC**

ROC-AUC measures how well a classifier ranks positive observations above negative observations across different classification thresholds.

**Average Precision**

Average Precision summarizes the precision-recall curve and is particularly informative when the positive class is rare.

### 7.1 Final Test Results

| Metric | SVM | Random Forest | XGBoost |
|---|---:|---:|---:|
| Accuracy | 0.8256 | 0.8314 | 0.7616 |
| Precision | 0.1724 | 0.0909 | 0.0833 |
| Recall | **0.4545** | 0.1818 | 0.2727 |
| F1-score | **0.2500** | 0.1212 | 0.1277 |
| ROC-AUC | 0.5878 | **0.6443** | 0.5692 |
| Average Precision | **0.1223** | 0.1022 | 0.1063 |

### 7.2 Model Comparison

**SVM**

SVM achieves the highest test recall of 45.45%. It identifies 5 of the 11 positive test observations.

It also achieves the highest test precision, F1-score, and Average Precision among the three models.

However, its ROC-AUC is 0.5878, indicating limited ranking performance in this experiment.

**Random Forest**

Random Forest achieves the highest test accuracy of 83.14% and the highest ROC-AUC of 0.6443.

However, its test recall is only 18.18%. This demonstrates why accuracy alone is not a sufficient measure for highly imbalanced classification.

**XGBoost**

XGBoost achieves a test recall of 27.27% and an Average Precision of 0.1063.

Its test performance is weaker than SVM on recall and Average Precision, although it performs better than Random Forest on recall.

### 7.3 Overall Interpretation

SVM is the strongest candidate when prioritizing positive-class recall on the current test set.

Random Forest achieves the highest ROC-AUC, while XGBoost achieves the highest mean cross-validation recall and ROC-AUC.

These differences indicate that model selection depends on the evaluation objective. No model demonstrates sufficiently strong performance to support clinical diagnostic use.

## 8. Visualizations and Generated Outputs

The pipeline generates six figures to support model evaluation.

### 8.1 Confusion Matrices

Separate confusion matrices are generated for:

- SVM
- Random Forest
- XGBoost

Each matrix shows true negatives, false positives, false negatives, and true positives.

These visualizations help assess the types of classification errors made by each model.

### 8.2 ROC Curve Comparison

The ROC curve comparison visualizes the relationship between the true positive rate and false positive rate at different classification thresholds.

The diagonal reference line represents random classification performance.

### 8.3 Precision-Recall Curve Comparison

The precision-recall curve displays the trade-off between precision and recall.

A baseline corresponding to the positive-class prevalence is included for comparison.

This visualization is especially useful because the dataset contains only 6.41% positive observations.

### 8.4 Model Comparison Chart

The model comparison chart presents accuracy, precision, recall, F1-score, ROC-AUC, and Average Precision for all three classifiers.

### 8.5 Saved Project Outputs

**Figures**

- `outputs/figures/svm_confusion_matrix.png`
- `outputs/figures/random_forest_confusion_matrix.png`
- `outputs/figures/xgboost_confusion_matrix.png`
- `outputs/figures/roc_curve_comparison.png`
- `outputs/figures/precision_recall_curve_comparison.png`
- `outputs/figures/model_comparison.png`

**Metrics**

- `outputs/metrics/test_metrics.json`
- `outputs/metrics/cross_validation_results.json`
- `outputs/metrics/cross_validation_fold_scores.csv`

**Trained Models**

- `outputs/models/svm.joblib`
- `outputs/models/random_forest.joblib`
- `outputs/models/xgboost.joblib`

## 9. Limitations and Future Improvements

### 9.1 Limitations

1. **Small dataset:** The dataset contains only 858 observations.
2. **Few positive cases:** Only 55 observations belong to the positive class.
3. **Small test minority class:** The test set contains only 11 positive cases, making recall estimates sensitive to individual predictions.
4. **Low precision:** All three models produce relatively low positive-class precision.
5. **Limited ranking performance:** The test ROC-AUC and Average Precision scores indicate that further model improvement is needed.
6. **Dataset representativeness:** Results from this dataset cannot automatically be generalized to other patient populations.
7. **Clinical validation:** The models have not undergone independent clinical validation.

### 9.2 Future Improvements

Possible future improvements include:

- Evaluating additional classification algorithms.
- Investigating different sampling strategies.
- Performing feature engineering and feature selection.
- Evaluating classification thresholds using validation data.
- Improving precision while maintaining an acceptable recall level.
- Testing on larger and independent datasets.
- Conducting external validation before considering any clinical use.

## 10. Conclusion

This project successfully implements an end-to-end machine learning pipeline for high-imbalance clinical disease prediction.

The workflow includes dataset loading, missing-value analysis, leakage prevention, preprocessing, SMOTE, hyperparameter tuning, cross-validation, test-set evaluation, and visualization generation.

Three classifiers—SVM, Random Forest, and XGBoost—were trained and compared.

SVM achieved the highest test recall of 45.45%, while Random Forest achieved the highest test ROC-AUC of 0.6443. XGBoost achieved the highest mean cross-validation recall and ROC-AUC.

The experiment demonstrates that handling class imbalance requires more than simply maximizing accuracy. Recall, precision, F1-score, ROC-AUC, and Average Precision provide a more informative assessment of model behavior.

Although the pipeline completed successfully, the results are not sufficient to establish a clinically reliable diagnostic model. Further experimentation, larger datasets, and independent validation are necessary.

---

**Project:** Progree Remote Machine Learning Internship — Task 4  
**Project Focus:** High-Imbalance Clinical Disease Predictor  
**Implementation:** Python and Scikit-learn ecosystem  
**Status:** Model training, evaluation, and output generation completed.