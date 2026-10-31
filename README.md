# Progree — Machine Learning Remote Internship

A collection of hands-on Machine Learning projects completed as part of my one-month Progree Remote Machine Learning Internship (20 October – 20 November 2026).

This repository documents my learning journey, practical implementations, model evaluation, and project deliverables across four internship tasks.

## Internship Overview

| Detail | Information |
|---|---|
| Organization | Progree |
| Role | Machine Learning Intern |
| Internship Type | Remote |
| Duration | 20 October 2026 – 20 November 2026 |
| Program Length | One Month |

## Repository Structure

```text
Progree_Internship/
│
├── task-1-linkedin-announcement/
│   ├── linkedin_announcement.png
│   └── README.md
│
├── progree-task-2-linear-regression/
│
├── progree-task-3-cnn/
│
├── progree-task-4-clinical-disease-predictor/
│
├── .gitignore
└── README.md
```

## Internship Tasks

### Task 1 — LinkedIn Internship Announcement

Published a LinkedIn announcement sharing my selection for the Progree Machine Learning Remote Internship.

**Focus Areas:**
- Internship introduction and learning goals.
- Professional communication.
- Sharing career development milestones.

**Deliverables:** LinkedIn announcement screenshot and documentation.

### Task 2 — Linear Regression

Developed an end-to-end regression pipeline using the Ames Housing dataset to predict house sale prices.

**Key Implementation:**
- Data preprocessing and missing-value imputation.
- Numerical feature scaling and categorical encoding.
- Ridge Regression.
- Hyperparameter tuning using GridSearchCV.
- Model evaluation and visualization.

**Results:**
- Best Alpha: 10
- Cross-Validation R²: 0.8166
- Test R²: 0.8780
- Test RMSE: $30,592.53

**Technologies:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib.

### Task 3 — CNN Image Classification

Built a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset.

**Key Implementation:**
- Image preprocessing and normalization.
- Data augmentation.
- CNN architecture using convolution and pooling layers.
- Model training with Adam optimizer.
- Early stopping and performance evaluation.

**Results:**
- Training Epochs: 20
- Best Validation Accuracy: 74.80%
- Test Accuracy: 73.17%
- Model Parameters: 402,986

**Technologies:** Python, TensorFlow, Keras, NumPy, Matplotlib.

### Task 4 — High-Imbalance Clinical Disease Predictor

Developed a Machine Learning classification pipeline for cervical cancer biopsy prediction using the UCI Cervical Cancer Risk Factors dataset.

**Key Implementation:**
- Data cleaning and preprocessing.
- Handling missing values.
- Addressing class imbalance using SMOTE within an imbalanced-learn pipeline.
- Training SVM, Random Forest, and XGBoost models.
- Hyperparameter tuning using GridSearchCV with F2 scoring.
- Evaluation using precision, recall, F1-score, ROC-AUC, and Average Precision.

**Test Results:**

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| SVM | 0.8256 | 0.1724 | 0.4545 | 0.2500 |
| Random Forest | 0.8314 | 0.0909 | 0.1818 | 0.1212 |
| XGBoost | 0.7616 | 0.0833 | 0.2727 | 0.1277 |

**Technologies:** Python, Pandas, NumPy, Scikit-learn, imbalanced-learn, XGBoost, Matplotlib.

## Skills Demonstrated

- Python Programming
- Data Cleaning and Preprocessing
- Exploratory Data Analysis
- Regression and Classification
- Deep Learning and CNNs
- Handling Imbalanced Datasets
- Hyperparameter Tuning
- Model Evaluation
- Data Visualization
- Git and GitHub

## Repository Purpose

This repository serves as a documented portfolio of my practical work during the Progree Machine Learning Remote Internship. It includes project source code, documentation, trained models, evaluation metrics, and visualizations.

---

**Author:** Misbah Sajjad  
**Field:** Artificial Intelligence & Data Science  
**GitHub:** [misbahs10](https://github.com/misbahs10)
