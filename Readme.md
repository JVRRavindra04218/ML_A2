# BITS Pilani ML Assignment 2

## Problem Statement

The objective of this project is to implement and compare multiple classification models to predict a target variable (TARGET_COL) from a given dataset. The dataset contains student performance data and other attributes, with the goal of classifying individuals into binary categories (e.g., successful placement vs. not).

## Dataset Description

- **Source**: `DataSet.csv` (Provided)
- **Size**: 500 instances, 14 columns.
- **Features**: gender, age, ssc_percentage, hsc_percentage, hsc_stream, degree_percentage, technical_skills_score, soft_skills_score, aptitude_score, communication_score, work_experience_months, salary_lpa.
- **Target**: `TARGET_COL` (0 or 1).

## Models Used

| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| :------------------ | :------- | :----- | :-------- | :----- | :----- | :----- |
| Logistic Regression | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree       | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| KNN                 | 0.9600   | 1.0000 | 0.9545    | 1.0000 | 0.9767 | 0.8461 |
| Naive Bayes         | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| Random Forest       | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |

## Observations

- **Logistic Regression, Decision Tree, Naive Bayes, and Random Forest** all achieved perfect scores on the hold-out test set, suggesting the dataset might be linearly separable or have very clear boundaries.
- **KNN** performed slightly lower on accuracy (96%) but still maintained perfect AUC, indicating high discriminative power with minor misclassifications.
- The models are saved in the `model/` directory and served via a **Streamlit** dashboard.