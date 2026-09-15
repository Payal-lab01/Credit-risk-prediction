"""
Model Explainability Engine (SHAP & LIME)
Provides post-hoc local and feature-level explanations for Indian Credit Bureau (CIBIL)
and US Peer-to-Peer (LendingClub) credit risk prediction models.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
from lime import lime_tabular

import preprocessing as pp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")

# -------------------------------------------------------------------------
# Load Saved Models and Feature Definitions
# -------------------------------------------------------------------------
cibil_model = joblib.load(os.path.join(MODELS_DIR, "cibil_logistic_model.pkl"))
cibil_scaler = joblib.load(os.path.join(MODELS_DIR, "cibil_scaler.pkl"))
cibil_cols = joblib.load(os.path.join(MODELS_DIR, "cibil_feature_columns.pkl"))

lc_logistic = joblib.load(os.path.join(MODELS_DIR, "lc_logistic_model.pkl"))
lc_rf = joblib.load(os.path.join(MODELS_DIR, "lc_random_forest_model.pkl"))
lc_scaler = joblib.load(os.path.join(MODELS_DIR, "lc_scaler.pkl"))
lc_cols = joblib.load(os.path.join(MODELS_DIR, "lc_feature_columns.pkl"))
lc_threshold = float(joblib.load(os.path.join(MODELS_DIR, "lc_threshold.pkl")))

# CIBIL User Input Feature Names (distinct from baseline bureau distribution)
CIBIL_USER_INPUT_KEYS = {
    "Credit_Score", "AGE", "GENDER", "MARITALSTATUS", "EDUCATION",
    "NETMONTHLYINCOME", "Time_With_Curr_Empr", "Total_TL", "Tot_Active_TL",
    "Tot_Closed_TL", "Tot_Missed_Pmnt", "num_times_delinquent", "tot_enq",
    "last_prod_enq2", "first_prod_enq2", "pct_active_tl", "pct_closed_tl",
    "pct_of_active_TLs_ever"
}

# -------------------------------------------------------------------------
# Human-Readable Feature Descriptions
# -------------------------------------------------------------------------
FEATURE_DESCRIPTIONS = {
    "Credit_Score": "CIBIL Credit Score",
    "AGE": "Applicant Age",
    "GENDER": "Borrower Gender",
    "MARITALSTATUS": "Marital Status",
    "EDUCATION": "Education Level",
    "NETMONTHLYINCOME": "Net Monthly Income",
    "Time_With_Curr_Empr": "Employment Tenure (Months)",
    "Total_TL": "Total Trade Lines",
    "Tot_Active_TL": "Active Accounts",
    "Tot_Closed_TL": "Closed Accounts",
    "Tot_Missed_Pmnt": "Lifetime Missed Payments",
    "num_times_delinquent": "Delinquency Frequency",
    "tot_enq": "Credit Inquiries",
    "pct_active_tl": "Ratio of Active Accounts",
    "pct_closed_tl": "Ratio of Closed Accounts",
    "last_prod_enq2": "Last Loan Type Enquired",
    "first_prod_enq2": "First Loan Type Enquired",
    "loan_amnt": "Requested Loan Amount",
    "int_rate": "Loan Interest Rate",
    "installment": "Monthly Installment",
    "annual_inc": "Annual Income",
    "log_annual_inc": "Log Transformed Annual Income",
    "dti": "Debt-to-Income Ratio (DTI)",
    "revol_bal": "Revolving Credit Balance",
    "revol_util": "Revolving Credit Utilization",
    "open_acc": "Open Credit Lines",
    "total_acc": "Total Credit Lines",
    "delinq_2yrs": "Delinquencies in Past 2 Years",
    "inq_last_6mths": "Credit Inquiries in Last 6 Months",
    "grade": "LendingClub Assigned Grade",
    "sub_grade": "LendingClub Sub-Grade Tier",
}

DISCLAIMER = "Feature contributions explain the model's prediction and do not represent causal effects."
CIBIL_BASELINE_DISCLOSURE = (
    "Note: The CIBIL model evaluates an 83-dimensional bureau feature vector. "
    "Primary applicant inputs override specific columns, while secondary deep trade-line features "
    "are populated using the model's training distribution baseline (cibil_scaler.mean_)."
)

# -------------------------------------------------------------------------
# Pre-instantiate and Cache SHAP & LIME Explainers (High Performance)
# -------------------------------------------------------------------------
print("Initializing SHAP & LIME explainers (caching in memory)...")

# 1. CIBIL SHAP (LinearExplainer on zero-mean scaled space)
cibil_masker = np.zeros((1, len(cibil_cols)))
cibil_shap_explainer = shap.LinearExplainer(cibil_model, cibil_masker)

# 2. CIBIL LIME (Standard normal in scaled space since mean=0, std=1)
np.random.seed(42)
cibil_lime_bg = np.random.normal(0, 1, size=(200, len(cibil_cols)))
cibil_lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=cibil_lime_bg,
    feature_names=cibil_cols,
    class_names=["Approved", "Default"],
    mode="classification",
    random_state=42,
    discretize_continuous=False,
)

# 3. LendingClub Logistic Regression SHAP (LinearExplainer)
lc_logreg_masker = np.zeros((1, len(lc_cols)))
lc_logreg_shap_explainer = shap.LinearExplainer(lc_logistic, lc_logreg_masker)

# 4. LendingClub Random Forest SHAP (TreeExplainer)
lc_rf_shap_explainer = shap.TreeExplainer(lc_rf)

# 5. LendingClub LIME Explainers
# Background: zero-centered for scaled continuous and binary dummies
lc_bg_scaled = np.zeros((200, len(lc_cols)))
lc_logreg_lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=lc_bg_scaled,
    feature_names=lc_cols,
    class_names=["Approved", "Default"],
    mode="classification",
    random_state=42,
    discretize_continuous=False,
)

# Background for RF: raw continuous distribution means + 0 dummies
lc_bg_raw = np.zeros((200, len(lc_cols)))
for i, col in enumerate(lc_scaler.feature_names_in_):
    col_idx = lc_cols.index(col)
    lc_bg_raw[:, col_idx] = lc_scaler.mean_[i]

lc_rf_lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=lc_bg_raw,
    feature_names=lc_cols,
    class_names=["Approved", "Default"],
    mode="classification",
    random_state=42,
    discretize_continuous=False,
)

print("Explainability engine initialized successfully.")


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def generate_human_interpretation(feature_name: str, impact: float, is_user_input: bool = True) -> str:
    """
    Translates mathematical feature attribution into cautious, non-causal human language.
    Positive impact (> 0) = pushed model toward higher default probability (Risk).
    Negative impact (< 0) = pushed model toward approval (Creditworthiness).
    """
    human_name = FEATURE_DESCRIPTIONS.get(feature_name, feature_name.replace("_", " "))
    source_tag = "" if is_user_input else " (bureau baseline)"

    if impact < 0:
        return f"{human_name}{source_tag} influenced the model toward approval."
    else:
        return f"{human_name}{source_tag} contributed to higher estimated default risk."


def clean_feature_label(raw_name: str) -> str:
    """Extracts base feature name from LIME condition strings if present."""
    for op in [" <= ", " < ", " >= ", " > ", " = "]:
        if op in raw_name:
            parts = raw_name.split(op)
            for p in parts:
                p_clean = p.strip()
                if p_clean in cibil_cols or p_clean in lc_cols:
                    return p_clean
    return raw_name.strip()


# -------------------------------------------------------------------------
# CIBIL Explainability
# -------------------------------------------------------------------------
def explain_cibil(df_features: pd.DataFrame, scaled_features: np.ndarray, prediction_meta: dict) -> dict:
    """
    Computes both SHAP and LIME explanations for a CIBIL prediction.
    Explains the exact 83-dimensional feature vector passed to cibil_logistic_model.pkl.
    """
    try:
        # 1. SHAP Linear Attribution
        shap_res = cibil_shap_explainer(scaled_features)
        shap_vals = shap_res.values[0]  # shape (83,)

        shap_factors = []
        for i, col in enumerate(cibil_cols):
            val = float(shap_vals[i])
            is_user = col in CIBIL_USER_INPUT_KEYS
            shap_factors.append({
                "feature": col,
                "display_name": FEATURE_DESCRIPTIONS.get(col, col.replace("_", " ")),
                "raw_value": round(float(df_features[col].iloc[0]), 4),
                "shap_value": round(val, 4),
                "direction": "Risk (+)" if val > 0 else "Approval (-)",
                "source": "user_input" if is_user else "baseline_bureau_mean",
                "human_text": generate_human_interpretation(col, val, is_user),
            })

        # Sort by absolute SHAP attribution magnitude
        shap_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        top_shap_approval = [f for f in shap_factors if f["shap_value"] < 0][:5]
        top_shap_risk = [f for f in shap_factors if f["shap_value"] > 0][:5]

        # 2. LIME Local Explanation
        lime_exp = cibil_lime_explainer.explain_instance(
            data_row=scaled_features[0],
            predict_fn=cibil_model.predict_proba,
            num_features=10,
        )

        lime_factors = []
        for feat_expr, weight in lime_exp.as_list():
            clean_name = clean_feature_label(feat_expr)
            is_user = clean_name in CIBIL_USER_INPUT_KEYS
            val = float(weight)
            lime_factors.append({
                "feature": clean_name,
                "display_name": FEATURE_DESCRIPTIONS.get(clean_name, clean_name.replace("_", " ")),
                "lime_weight": round(val, 4),
                "direction": "Risk (+)" if val > 0 else "Approval (-)",
                "source": "user_input" if is_user else "baseline_bureau_mean",
                "human_text": generate_human_interpretation(clean_name, val, is_user),
            })

        return {
            "status": "success",
            "model_name": "Logistic Regression (Balanced)",
            "default_probability": prediction_meta.get("default_probability"),
            "risk_label": prediction_meta.get("risk_label"),
            "disclaimer": DISCLAIMER,
            "baseline_disclosure": CIBIL_BASELINE_DISCLOSURE,
            "shap": {
                "explainer_type": "SHAP LinearExplainer",
                "top_approval_factors": top_shap_approval,
                "top_risk_factors": top_shap_risk,
                "all_top_factors": shap_factors[:10],
            },
            "lime": {
                "explainer_type": "LIME Tabular Explainer",
                "factors": lime_factors,
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": f"Explainability generation failed: {str(e)}",
            "disclaimer": DISCLAIMER,
        }


# -------------------------------------------------------------------------
# LendingClub Explainability
# -------------------------------------------------------------------------
def explain_lendingclub(X_raw: pd.DataFrame, X_scaled_for_logistic: pd.DataFrame, prediction_meta: dict) -> dict:
    """
    Computes both SHAP and LIME explanations for BOTH LendingClub models:
    1. Calibrated Logistic Regression (51 features: scaled continuous + unscaled dummies)
    2. Random Forest Ensemble (51 features: raw continuous + dummies)
    """
    try:
        # A. LOGISTIC REGRESSION EXPLANATIONS
        # 1. SHAP Linear
        shap_log_res = lc_logreg_shap_explainer(X_scaled_for_logistic.values)
        shap_log_vals = shap_log_res.values[0]  # shape (51,)

        log_shap_factors = []
        for i, col in enumerate(lc_cols):
            val = float(shap_log_vals[i])
            log_shap_factors.append({
                "feature": col,
                "display_name": FEATURE_DESCRIPTIONS.get(col, col.replace("_", " ")),
                "raw_value": round(float(X_raw[col].iloc[0]), 4),
                "shap_value": round(val, 4),
                "direction": "Risk (+)" if val > 0 else "Approval (-)",
                "human_text": generate_human_interpretation(col, val, True),
            })
        log_shap_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        # 2. LIME Logistic
        lime_log_exp = lc_logreg_lime_explainer.explain_instance(
            data_row=X_scaled_for_logistic.values[0],
            predict_fn=lc_logistic.predict_proba,
            num_features=8,
        )
        lime_log_factors = []
        for feat_expr, weight in lime_log_exp.as_list():
            clean_name = clean_feature_label(feat_expr)
            val = float(weight)
            lime_log_factors.append({
                "feature": clean_name,
                "display_name": FEATURE_DESCRIPTIONS.get(clean_name, clean_name.replace("_", " ")),
                "lime_weight": round(val, 4),
                "direction": "Risk (+)" if val > 0 else "Approval (-)",
                "human_text": generate_human_interpretation(clean_name, val, True),
            })

        # B. RANDOM FOREST EXPLANATIONS
        # 1. SHAP TreeExplainer
        # TreeExplainer returns shape (1, 51, 2) where class 1 is Default
        rf_shap_res = lc_rf_shap_explainer(X_raw.values)
        if len(rf_shap_res.values.shape) == 3:
            rf_shap_vals = rf_shap_res.values[0, :, 1]
        else:
            rf_shap_vals = rf_shap_res.values[0]

        rf_shap_factors = []
        for i, col in enumerate(lc_cols):
            val = float(rf_shap_vals[i])
            rf_shap_factors.append({
                "feature": col,
                "display_name": FEATURE_DESCRIPTIONS.get(col, col.replace("_", " ")),
                "raw_value": round(float(X_raw[col].iloc[0]), 4),
                "shap_value": round(val, 4),
                "direction": "Risk (+)" if val > 0 else "Approval (-)",
                "human_text": generate_human_interpretation(col, val, True),
            })
        rf_shap_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        # 2. LIME Random Forest
        lime_rf_exp = lc_rf_lime_explainer.explain_instance(
            data_row=X_raw.values[0],
            predict_fn=lc_rf.predict_proba,
            num_features=8,
        )
        lime_rf_factors = []
        for feat_expr, weight in lime_rf_exp.as_list():
            clean_name = clean_feature_label(feat_expr)
            val = float(weight)
            lime_rf_factors.append({
                "feature": clean_name,
                "display_name": FEATURE_DESCRIPTIONS.get(clean_name, clean_name.replace("_", " ")),
                "lime_weight": round(val, 4),
                "direction": "Risk (+)" if val > 0 else "Approval (-)",
                "human_text": generate_human_interpretation(clean_name, val, True),
            })

        return {
            "status": "success",
            "dataset": "lendingclub",
            "disclaimer": DISCLAIMER,
            "logistic_regression": {
                "model_name": "Logistic Regression (Calibrated 0.13 Threshold)",
                "default_probability": prediction_meta.get("evaluated_models", {}).get("logistic_regression", {}).get("default_probability"),
                "risk_label": prediction_meta.get("evaluated_models", {}).get("logistic_regression", {}).get("risk_label"),
                "shap": {
                    "explainer_type": "SHAP LinearExplainer",
                    "top_approval_factors": [f for f in log_shap_factors if f["shap_value"] < 0][:5],
                    "top_risk_factors": [f for f in log_shap_factors if f["shap_value"] > 0][:5],
                    "all_top_factors": log_shap_factors[:8],
                },
                "lime": {
                    "explainer_type": "LIME Tabular Explainer",
                    "factors": lime_log_factors,
                },
            },
            "random_forest": {
                "model_name": "Random Forest Classifier (Ensemble)",
                "default_probability": prediction_meta.get("evaluated_models", {}).get("random_forest", {}).get("default_probability"),
                "risk_label": prediction_meta.get("evaluated_models", {}).get("random_forest", {}).get("risk_label"),
                "shap": {
                    "explainer_type": "SHAP TreeExplainer",
                    "top_approval_factors": [f for f in rf_shap_factors if f["shap_value"] < 0][:5],
                    "top_risk_factors": [f for f in rf_shap_factors if f["shap_value"] > 0][:5],
                    "all_top_factors": rf_shap_factors[:8],
                },
                "lime": {
                    "explainer_type": "LIME Tabular Explainer",
                    "factors": lime_rf_factors,
                },
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": f"Explainability generation failed: {str(e)}",
            "disclaimer": DISCLAIMER,
        }


# -------------------------------------------------------------------------
# Status Endpoint Helper
# -------------------------------------------------------------------------
def get_explainability_status() -> dict:
    """Returns engine readiness for SHAP and LIME across both datasets."""
    return {
        "status": "active",
        "libraries": {
            "shap": shap.__version__,
            "lime": lime_tabular.__name__,
        },
        "cibil": {
            "shap_explainer": "LinearExplainer (cached)",
            "lime_explainer": "LimeTabularExplainer (cached)",
            "feature_count": len(cibil_cols),
        },
        "lending_club": {
            "logistic_regression": {
                "shap_explainer": "LinearExplainer (cached)",
                "lime_explainer": "LimeTabularExplainer (cached)",
            },
            "random_forest": {
                "shap_explainer": "TreeExplainer (cached)",
                "lime_explainer": "LimeTabularExplainer (cached)",
            },
            "feature_count": len(lc_cols),
        },
    }
