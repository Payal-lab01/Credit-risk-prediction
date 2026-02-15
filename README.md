# Credit Risk Prediction

This project focuses on building, evaluating, and comparing machine learning models
for credit risk prediction. The work explores how credit risk models behave under
different economic and regulatory environments and is designed to be extensible
to explainability and decision systems.

## Current Scope
- Credit risk modeling using structured financial data
- Model comparison using Logistic Regression and Random Forest
- Evaluation using ROC-AUC, KS Statistic, Recall, and Precision
- Cross-economy analysis using:
  - Indian credit bureau–style data (CIBIL)
  - US peer-to-peer lending data (LendingClub)

## Datasets
- Indian credit dataset (bureau-style, anonymized)
- LendingClub public loan dataset

## Models Used
- Logistic Regression (with threshold calibration)
- Random Forest Classifier

## Evaluation Metrics
- ROC-AUC
- KS Statistic
- Recall (Default class)
- Precision (Default class)

## Key Insights (So Far)
- Indian credit risk shows strong linear separability, making Logistic Regression effective.
- US credit risk exhibits non-linear and policy-sensitive behavior, requiring calibration.
- Model performance depends more on economic context than on algorithm choice alone.

## Reproducibility
Trained models are saved using `joblib` along with preprocessing artifacts such as
feature schemas, scalers, and decision thresholds.

## Future Work
- SHAP and LIME for model explainability
- NLP-based borrower text analysis
- End-to-end loan decision system

## Author
Payal Yadav
