# Credit Risk Prediction

## An Explainable Machine Learning Framework for Credit Risk Assessment Across Indian and US Lending Environments

[Live Demo](https://credit-risk-prediction-g3os.onrender.com/) | [GitHub Repository](https://github.com/Payal-lab01/Credit-risk-prediction)

---

## Abstract

This project presents a machine learning-based framework for credit risk prediction and explainability across two distinct lending environments: an Indian credit bureau-style dataset and the US LendingClub peer-to-peer lending dataset.

The study evaluates Logistic Regression and Random Forest models using credit-risk-oriented evaluation metrics, including ROC-AUC, Kolmogorov-Smirnov (KS) Statistic, Recall, and Precision. In addition to predictive performance, the framework incorporates Explainable Artificial Intelligence (XAI) techniques using SHAP and LIME to provide local and feature-level explanations for individual predictions.

The project further examines how model behavior and predictive performance vary across datasets representing different credit and lending environments. A web-based application was developed using Flask, HTML, CSS, and JavaScript to provide an interactive interface for prediction, model comparison, and explainability.

---

## 1. Project Overview

Credit risk assessment is a fundamental component of lending and financial decision-making. Machine learning techniques can identify complex patterns in borrower and loan characteristics; however, predictive performance alone is insufficient in applications where transparency and interpretability are important.

This project addresses both aspects by developing an end-to-end credit risk prediction system that combines:

- Machine learning-based credit risk prediction
- Cross-dataset model evaluation
- Credit-risk-specific performance metrics
- Decision threshold analysis
- SHAP-based model explanations
- LIME-based local explanations
- Interactive prediction interfaces
- REST API-based model serving
- Web-based visualization and deployment

The study uses two datasets to examine whether model behavior varies across different lending environments.

---

## 2. Objectives

The primary objectives of the project are:

1. Develop machine learning models for credit risk prediction.
2. Evaluate model performance using ROC-AUC, KS Statistic, Recall, and Precision.
3. Compare credit-risk modeling behavior across Indian and US lending datasets.
4. Investigate the effect of class imbalance and decision thresholds on model predictions.
5. Apply SHAP and LIME to explain individual model predictions.
6. Develop an interactive web application for credit risk assessment.
7. Provide an end-to-end framework connecting machine learning models, preprocessing, explainability, and deployment.

---

## 3. Datasets

### 3.1 Indian Credit Bureau-Style Dataset

The Indian dataset contains anonymized borrower-level credit information representing a credit bureau-style environment.

After preprocessing and feature construction, the modeling dataset contains:

- 51,336 observations
- 83 final features
- Binary credit-risk target

The features represent borrower characteristics, credit history, loan accounts, payment behavior, credit enquiries, income, and related credit attributes.

### 3.2 LendingClub Dataset

The LendingClub dataset contains historical US peer-to-peer lending and borrower information.

The original dataset contains approximately:

- 2.26 million records
- 51 features

Due to computational constraints, a 25% sample containing approximately 565,000 records was used for model development.

The final modeling representation contains:

- 11 continuous features
- 40 categorical dummy variables
- 51 model features

---

## 4. Machine Learning Models

### Logistic Regression

Logistic Regression is used as an interpretable baseline model for binary credit risk classification.

For the Indian dataset, Logistic Regression is the deployed prediction model. A class-balanced variant was evaluated during model development to examine the effect of class weighting.

For LendingClub, Logistic Regression was evaluated using a calibrated probability threshold to account for the imbalanced default classification problem.

### Random Forest

Random Forest is evaluated on the LendingClub dataset as a tree-based ensemble model capable of representing non-linear relationships and interactions between features.

The Random Forest model uses class weighting and a standard probability threshold of 0.50.

---

## 5. Model Evaluation

The project evaluates model performance using metrics relevant to credit-risk classification.

### Evaluation Metrics

**ROC-AUC**

Measures the model's ability to distinguish between the two classes across different classification thresholds.

**KS Statistic**

Measures the maximum separation between the cumulative distributions of predicted scores for the two classes.

**Recall**

Measures the proportion of actual default cases correctly identified by the model.

**Precision**

Measures the proportion of predicted default cases that are actually defaults.

---

## 6. Experimental Results

### Indian Credit Bureau Dataset

| Model | ROC-AUC | KS Statistic | Recall | Precision |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8363 | 0.5407 | 0.7252 | 0.6715 |

### LendingClub Dataset

| Model | ROC-AUC | KS Statistic | Recall | Precision | Threshold |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7121 | 0.3086 | 0.6696 | 0.2097 | 0.13 |
| Random Forest | 0.7048 | 0.2949 | 0.7251 | 0.1934 | 0.50 |

The results are provided to support comparative analysis of the evaluated models and datasets. They should not be interpreted as evidence that one model is universally superior to another across all credit-risk applications.

---

## 7. Explainable Artificial Intelligence

Explainability is an important component of the proposed system because credit-risk predictions may influence financial decisions.

The project uses two post-hoc explainability techniques: SHAP and LIME.

### 7.1 SHAP

SHAP (SHapley Additive exPlanations) is used to estimate the contribution of individual features to a model prediction.

The implementation uses:

- LinearExplainer for Logistic Regression
- TreeExplainer for Random Forest

The application provides both positive and negative feature contributions for individual predictions.

In the implemented interpretation:

- Positive SHAP contribution indicates movement toward higher predicted default risk.
- Negative SHAP contribution indicates movement toward lower predicted default risk.

### 7.2 LIME

LIME (Local Interpretable Model-Agnostic Explanations) is used to generate local explanations for individual predictions.

LIME approximates the behavior of the prediction model around a particular input and identifies features that contribute to the local prediction.

### 7.3 Explainability Interpretation

SHAP and LIME explanations describe how the trained model uses features when producing a prediction. Feature contributions should not be interpreted as causal effects.

---

## 8. Web Application

The project includes an interactive web application that provides a unified interface for prediction and explainability.

### Application Components

**Overview**

Provides an overview of the project, datasets, models, and evaluation metrics.

**CIBIL Prediction**

Allows users to enter borrower-level information and obtain a credit-risk prediction using the Indian credit model.

**LendingClub Prediction**

Allows users to enter loan and borrower information and evaluate the LendingClub models.

**Model Comparison**

Provides comparative evaluation of the implemented models using ROC-AUC, KS Statistic, Recall, and Precision.

**Explainability**

Provides SHAP and LIME explanations for individual model predictions.

---

## 9. System Architecture

```text
                        Web Application
                              |
                              v
                     Flask REST API
                              |
              +---------------+---------------+
              |                               |
              v                               v
       CIBIL Pipeline                  LendingClub Pipeline
              |                               |
              v                               v
      Logistic Regression          +----------+----------+
                                   |                     |
                                   v                     v
                            Logistic Regression    Random Forest
                                   |                     |
                                   +----------+----------+
                                              |
                                              v
                                   Explainability Engine
                                        SHAP + LIME
````

---

## 10. Technology Stack

### Programming and Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

### Explainable AI

* SHAP
* LIME

### Backend

* Flask
* Gunicorn

### Frontend

* HTML
* CSS
* JavaScript
* Chart.js

### Deployment

* Render

### Development Environment

* Google Colab
* GitHub
* Antigravity IDE

---

## 11. Repository Structure

```text
Credit-risk-prediction/
│
├── Models/
│   ├── cibil_logistic_model.pkl
│   ├── cibil_scaler.pkl
│   ├── cibil_feature_columns.pkl
│   ├── lc_logistic_model.pkl
│   ├── lc_random_forest_model.pkl
│   ├── lc_scaler.pkl
│   ├── lc_feature_columns.pkl
│   ├── lc_rf_feature_columns.pkl
│   └── lc_threshold.pkl
│
├── Notebooks/
│   └── Model development notebooks
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── api.js
│       └── app.js
│
├── templates/
│   └── index.html
│
├── app.py
├── preprocessing.py
├── explainability.py
├── requirements.txt
└── README.md
```

---

## 12. Installation

Clone the repository:

```bash
git clone https://github.com/Payal-lab01/Credit-risk-prediction.git
cd Credit-risk-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

The local application will be available at:

```text
http://127.0.0.1:5001
```

---

## 13. API Endpoints

The Flask backend exposes the following endpoints:

| Endpoint                     | Method | Purpose                                |
| ---------------------------- | ------ | -------------------------------------- |
| `/`                          | GET    | Serves the web application             |
| `/api/health`                | GET    | System and model health check          |
| `/api/metrics`               | GET    | Model evaluation metrics               |
| `/api/presets/<dataset>`     | GET    | Sample borrower profiles               |
| `/api/predict/cibil`         | POST   | CIBIL credit-risk prediction           |
| `/api/predict/lendingclub`   | POST   | LendingClub credit-risk prediction     |
| `/api/explainability/status` | GET    | Explainability engine status           |
| `/api/explain/cibil`         | POST   | CIBIL SHAP and LIME explanations       |
| `/api/explain/lendingclub`   | POST   | LendingClub SHAP and LIME explanations |

---

## 14. Deployment

The application is deployed as a Flask web service using Render.

### Live Application

[Credit Risk Prediction Dashboard](https://credit-risk-prediction-g3os.onrender.com/)

The deployed application provides access to:

* Credit risk prediction
* Model comparison
* Risk assessment
* SHAP explanations
* LIME explanations

---

## 15. Reproducibility

The trained models and preprocessing artifacts are stored using Joblib.

The repository includes:

* Trained model files
* Feature schemas
* Scalers
* Decision threshold configuration
* Preprocessing pipeline
* Explainability implementation
* Web application source code

This allows the prediction system to be reproduced without retraining the models.

---

## 16. Limitations

The current implementation has several limitations:

1. The LendingClub dataset represents historical US peer-to-peer lending activity and may not reflect current lending conditions.
2. A 25% sample of the original LendingClub dataset was used for computational feasibility.
3. The CIBIL prediction interface does not expose every one of the 83 model features directly; features not supplied by the user are initialized using the baseline feature distribution used by the deployment pipeline.
4. CIBIL categorical encodings in the current deployment are treated as provisional because the original fitted LabelEncoder objects were not persisted.
5. SHAP and LIME provide model explanations rather than causal explanations.
6. The system is a research prototype and should not be used as a standalone automated lending decision system.

---

## 17. Future Work

Potential extensions include:

* Evaluation of additional machine learning and gradient-boosting models
* Counterfactual explanations
* Fairness and bias analysis
* Explanation stability evaluation
* Improved cross-dataset feature mapping
* Model calibration analysis
* Automated model monitoring
* NLP-based borrower information analysis
* Integration with a complete lending decision workflow

---

## 18. Research Contribution

The project focuses on the intersection of credit-risk prediction, cross-dataset comparison, and explainable artificial intelligence.

Rather than evaluating predictive performance in isolation, the framework combines model evaluation with feature-level explanations and examines model behavior across two distinct credit-risk datasets.

This provides a basis for further investigation into how dataset characteristics and lending environments may influence model performance, decision thresholds, and explainability.

---

## 19. Author

**Payal Yadav**

B.Tech Computer Science and Engineering
Manipal University Jaipur

---

## 20. License

This project is intended for academic and research purposes.

