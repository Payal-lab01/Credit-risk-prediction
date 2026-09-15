"""
Credit Risk Prediction Web Application Backend (Flask REST API)
Serves prediction engines, presets, metrics, and health diagnostics
for Indian Credit Bureau (CIBIL) and US Peer-to-Peer (LendingClub) models.
"""

import os
from datetime import datetime, timezone
import joblib
from flask import Flask, request, jsonify, render_template

import preprocessing as pp
import explainability as exp_engine

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")

# -------------------------------------------------------------------------
# Load Pretrained ML Artifacts
# -------------------------------------------------------------------------
print("Loading CIBIL models and scalers...")
cibil_model = joblib.load(os.path.join(MODELS_DIR, "cibil_logistic_model.pkl"))
cibil_scaler = joblib.load(os.path.join(MODELS_DIR, "cibil_scaler.pkl"))
cibil_cols = joblib.load(os.path.join(MODELS_DIR, "cibil_feature_columns.pkl"))

print("Loading LendingClub models, scalers, and calibrated threshold...")
lc_logistic = joblib.load(os.path.join(MODELS_DIR, "lc_logistic_model.pkl"))
lc_rf = joblib.load(os.path.join(MODELS_DIR, "lc_random_forest_model.pkl"))
lc_scaler = joblib.load(os.path.join(MODELS_DIR, "lc_scaler.pkl"))
lc_cols = joblib.load(os.path.join(MODELS_DIR, "lc_feature_columns.pkl"))
lc_threshold = float(joblib.load(os.path.join(MODELS_DIR, "lc_threshold.pkl")))

print("All ML artifacts loaded successfully.")

# -------------------------------------------------------------------------
# Authentic Repository Evaluation Metrics
# -------------------------------------------------------------------------
REPOSITORY_METRICS = {
    "cibil": {
        "dataset_name": "Indian Credit Bureau (CIBIL-style)",
        "economic_context": "Bureau trade-line data exhibiting strong linear separability.",
        "models": {
            "logistic_regression": {
                "name": "Logistic Regression (Standard)",
                "roc_auc": 0.8363,
                "ks_statistic": 0.5407,
                "recall": 0.7252,
                "precision": 0.6715,
                "threshold": 0.50,
                "deployed": False,
                "notes": "Linear baseline model evaluated on scaled bureau features.",
            },
            "logistic_regression_balanced": {
                "name": "Logistic Regression (Balanced Class Weight)",
                "roc_auc": 0.8363,
                "ks_statistic": 0.5403,
                "recall": 0.7249,
                "precision": 0.6714,
                "threshold": 0.50,
                "deployed": True,
                "notes": "Deployed production model in Models/cibil_logistic_model.pkl.",
            },
        },
    },
    "lending_club": {
        "dataset_name": "US Peer-to-Peer Lending (LendingClub)",
        "economic_context": "Market-driven P2P lending data with non-linear default risk.",
        "models": {
            "logistic_regression": {
                "name": "Logistic Regression (Calibrated Threshold)",
                "roc_auc": 0.7121,
                "ks_statistic": 0.3086,
                "recall": 0.6696,
                "precision": 0.2097,
                "threshold": lc_threshold,  # 0.13
                "deployed": True,
                "notes": "Decision threshold calibrated to 0.13 to optimize default recall under severe class imbalance.",
            },
            "random_forest": {
                "name": "Random Forest Classifier (Balanced)",
                "roc_auc": 0.7048,
                "ks_statistic": 0.2949,
                "recall": 0.7251,
                "precision": 0.1934,
                "threshold": 0.50,
                "deployed": True,
                "notes": "Captures complex interactions across loan terms and borrower credit history without scaling.",
            },
        },
    },
    "cross_economy_insights": {
        "linear_separability": "Indian bureau data separates cleanly with linear hyperplanes (KS = 0.54), whereas US P2P data exhibits higher entropy and non-linear interactions.",
        "policy_calibration": "US subprime lending requires policy-level decision threshold tuning (0.13 vs 0.50) to catch risky defaults.",
    },
}

# -------------------------------------------------------------------------
# Test Presets / Borrower Personas
# -------------------------------------------------------------------------
PRESETS = {
    "cibil": [
        {
            "id": "cibil_prime",
            "name": "Prime Borrower (Low Risk)",
            "description": "Established salaried borrower with high score (780), zero past delinquencies, and healthy income.",
            "data": {
                "Credit_Score": 780,
                "AGE": 38,
                "GENDER": "M",
                "MARITALSTATUS": "Married",
                "EDUCATION": "POST-GRADUATE",
                "NETMONTHLYINCOME": 85000,
                "Time_With_Curr_Empr": 120,
                "Total_TL": 6,
                "Tot_Active_TL": 2,
                "Tot_Closed_TL": 4,
                "Tot_Missed_Pmnt": 0,
                "num_times_delinquent": 0,
                "tot_enq": 2,
                "last_prod_enq2": "HL",
                "first_prod_enq2": "HL",
            },
        },
        {
            "id": "cibil_near_prime",
            "name": "Near-Prime Borrower (Moderate Risk)",
            "description": "Mid-tier borrower with average score (675), 1 missed payment, and moderate credit enquiries.",
            "data": {
                "Credit_Score": 675,
                "AGE": 29,
                "GENDER": "F",
                "MARITALSTATUS": "Single",
                "EDUCATION": "GRADUATE",
                "NETMONTHLYINCOME": 35000,
                "Time_With_Curr_Empr": 48,
                "Total_TL": 4,
                "Tot_Active_TL": 2,
                "Tot_Closed_TL": 2,
                "Tot_Missed_Pmnt": 1,
                "num_times_delinquent": 1,
                "tot_enq": 4,
                "last_prod_enq2": "PL",
                "first_prod_enq2": "ConsumerLoan",
            },
        },
        {
            "id": "cibil_subprime",
            "name": "Subprime Borrower (High Risk)",
            "description": "High-risk applicant with low score (580), multiple historical delinquencies, and heavy recent credit inquiries.",
            "data": {
                "Credit_Score": 580,
                "AGE": 25,
                "GENDER": "M",
                "MARITALSTATUS": "Single",
                "EDUCATION": "SSC",
                "NETMONTHLYINCOME": 18000,
                "Time_With_Curr_Empr": 14,
                "Total_TL": 8,
                "Tot_Active_TL": 5,
                "Tot_Closed_TL": 3,
                "Tot_Missed_Pmnt": 4,
                "num_times_delinquent": 5,
                "tot_enq": 10,
                "last_prod_enq2": "PL",
                "first_prod_enq2": "CC",
            },
        },
    ],
    "lendingclub": [
        {
            "id": "lc_prime",
            "name": "Prime Grade A Borrower (Low Risk)",
            "description": "High income ($110,000), low debt-to-income (9.5%), zero delinquencies, prime interest rate.",
            "data": {
                "loan_amnt": 12000,
                "int_rate": 7.25,
                "installment": 371.84,
                "annual_inc": 110000,
                "dti": 9.5,
                "revol_bal": 4200,
                "revol_util": 18.5,
                "open_acc": 9,
                "total_acc": 22,
                "delinq_2yrs": 0,
                "inq_last_6mths": 0,
                "grade": "A",
                "sub_grade": "A3",
            },
        },
        {
            "id": "lc_moderate",
            "name": "Mid-Tier Grade C Borrower (Moderate Risk)",
            "description": "Average income ($55,000), moderate DTI (19.8%), standard credit utilization, 1 inquiry.",
            "data": {
                "loan_amnt": 16000,
                "int_rate": 14.5,
                "installment": 376.5,
                "annual_inc": 55000,
                "dti": 19.8,
                "revol_bal": 14500,
                "revol_util": 58.2,
                "open_acc": 11,
                "total_acc": 24,
                "delinq_2yrs": 0,
                "inq_last_6mths": 1,
                "grade": "C",
                "sub_grade": "C2",
            },
        },
        {
            "id": "lc_subprime",
            "name": "Subprime Grade E Borrower (High Risk)",
            "description": "High debt-to-income (31.5%), high interest (21.5%), prior 2-year delinquency, high credit inquiry activity.",
            "data": {
                "loan_amnt": 28000,
                "int_rate": 21.5,
                "installment": 765.2,
                "annual_inc": 42000,
                "dti": 31.5,
                "revol_bal": 28900,
                "revol_util": 89.4,
                "open_acc": 14,
                "total_acc": 30,
                "delinq_2yrs": 1,
                "inq_last_6mths": 3,
                "grade": "E",
                "sub_grade": "E4",
            },
        },
    ],
}


# -------------------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------------------
@app.errorhandler(pp.ValidationError)
def handle_validation_error(err):
    return jsonify({
        "status": "error",
        "error_type": "ValidationError",
        "message": err.message,
        "field": err.field,
    }), 400


@app.errorhandler(Exception)
def handle_general_exception(err):
    return jsonify({
        "status": "error",
        "error_type": type(err).__name__,
        "message": str(err),
    }), 500


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Serves the main Credit Risk Prediction dashboard."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """System health check and artifact verification."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_loaded": {
            "cibil_logistic": cibil_model is not None,
            "cibil_scaler": cibil_scaler is not None,
            "cibil_features_count": len(cibil_cols),
            "lc_logistic": lc_logistic is not None,
            "lc_random_forest": lc_rf is not None,
            "lc_scaler": lc_scaler is not None,
            "lc_features_count": len(lc_cols),
            "lc_calibrated_threshold": lc_threshold,
        },
        "provisional_encodings": {
            "cibil_label_encoder_provisional": pp.CIBIL_CATEGORICAL_MAPPINGS_PROVISIONAL,
            "note": "CIBIL LabelEncoders follow standard alphabetical order on unique training classes and are treated as provisional.",
        },
    }), 200


@app.route("/api/metrics", methods=["GET"])
def metrics():
    """Returns exact authentic evaluation metrics for CIBIL and LendingClub."""
    return jsonify({
        "status": "success",
        "data": REPOSITORY_METRICS,
    }), 200


@app.route("/api/presets/<dataset>", methods=["GET"])
def presets(dataset):
    """Returns sample borrower presets for instant evaluation."""
    dataset_key = dataset.lower().replace("-", "").replace("_", "")
    if dataset_key in ["cibil", "india", "indian"]:
        key = "cibil"
    elif dataset_key in ["lendingclub", "lc", "us"]:
        key = "lendingclub"
    else:
        return jsonify({
            "status": "error",
            "message": f"Unknown dataset '{dataset}'. Valid options: 'cibil', 'lendingclub'.",
        }), 404

    return jsonify({
        "status": "success",
        "dataset": key,
        "presets": PRESETS[key],
    }), 200


@app.route("/api/predict/cibil", methods=["POST"])
def predict_cibil():
    """
    Evaluates credit risk for an Indian bureau applicant.
    Takes user fields, merges onto baseline distribution (83 features),
    scales via cibil_scaler.pkl, and executes cibil_logistic_model.pkl.
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({
            "status": "error",
            "error_type": "BadRequest",
            "message": "Missing JSON payload in request body.",
        }), 400

    # Build 83-dimensional feature DataFrame
    df_features = pp.build_cibil_features(payload)

    # Scale complete 83-feature vector
    scaled_features = cibil_scaler.transform(df_features)

    # Predict default probability (Class 1 = Default / Risk)
    default_prob = float(cibil_model.predict_proba(scaled_features)[0, 1])
    threshold = 0.50
    is_default = int(default_prob >= threshold)

    # Determine risk tier and label
    if default_prob < 0.35:
        risk_tier = "Low Risk"
        risk_label = "Approved / Prime Profile"
        action = "Approve"
    elif default_prob < 0.50:
        risk_tier = "Moderate Risk"
        risk_label = "Borderline / Manual Review"
        action = "Manual Review"
    else:
        risk_tier = "High Risk"
        risk_label = "Decline / High Risk of Default"
        action = "Decline"

    # Risk score index (scaled 0-100 where higher is safer)
    risk_score_index = int(round((1.0 - default_prob) * 100))

    return jsonify({
        "status": "success",
        "dataset": "cibil",
        "model_used": "Logistic Regression (Balanced)",
        "prediction": is_default,
        "action": action,
        "risk_label": risk_label,
        "risk_tier": risk_tier,
        "default_probability": round(default_prob, 4),
        "approval_probability": round(1.0 - default_prob, 4),
        "decision_threshold": threshold,
        "risk_score_index": risk_score_index,
        "summary_features": {
            "Credit_Score": df_features["Credit_Score"].iloc[0],
            "AGE": df_features["AGE"].iloc[0],
            "NETMONTHLYINCOME": df_features["NETMONTHLYINCOME"].iloc[0],
            "Tot_Missed_Pmnt": df_features["Tot_Missed_Pmnt"].iloc[0],
            "num_times_delinquent": df_features["num_times_delinquent"].iloc[0],
            "Total_TL": df_features["Total_TL"].iloc[0],
        },
        "explainability_ready": {
            "status": "hook_available",
            "method": "SHAP_LinearExplainer",
            "ready": True,
            "feature_dim": 83,
        },
    }), 200


@app.route("/api/predict/lendingclub", methods=["POST"])
def predict_lendingclub():
    """
    Evaluates credit risk for a US P2P loan applicant.
    Evaluates both:
    1. Calibrated Logistic Regression (threshold = 0.13, scaled continuous features)
    2. Random Forest Classifier (threshold = 0.50, unscaled continuous features)
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({
            "status": "error",
            "error_type": "BadRequest",
            "message": "Missing JSON payload in request body.",
        }), 400

    # Build 51-dimensional features (raw and scaled)
    X_raw, X_scaled_for_logistic = pp.build_lendingclub_features(payload)

    # 1. Evaluate Calibrated Logistic Regression
    prob_logistic = float(lc_logistic.predict_proba(X_scaled_for_logistic)[0, 1])
    is_default_logistic = int(prob_logistic >= lc_threshold)
    label_logistic = "Decline / High Risk" if is_default_logistic == 1 else "Approved / Low Risk"
    tier_logistic = "High Risk" if prob_logistic >= lc_threshold else ("Moderate Risk" if prob_logistic >= 0.08 else "Low Risk")

    # 2. Evaluate Random Forest Classifier
    prob_rf = float(lc_rf.predict_proba(X_raw)[0, 1])
    is_default_rf = int(prob_rf >= 0.50)
    label_rf = "Decline / High Risk" if is_default_rf == 1 else "Approved / Low Risk"
    tier_rf = "High Risk" if prob_rf >= 0.50 else ("Moderate Risk" if prob_rf >= 0.35 else "Low Risk")

    # Primary consensus recommendation
    if is_default_logistic == 1 and is_default_rf == 1:
        consensus = "High Risk (Consensus Decline)"
        primary_action = "Decline"
    elif is_default_logistic == 0 and is_default_rf == 0:
        consensus = "Low Risk (Consensus Approved)"
        primary_action = "Approve"
    else:
        consensus = "Divergent / Policy Sensitive (Calibrated Threshold Triggered)"
        primary_action = "Manual Review / Policy Override"

    return jsonify({
        "status": "success",
        "dataset": "lendingclub",
        "primary_recommendation": {
            "consensus": consensus,
            "action": primary_action,
            "calibrated_decision": label_logistic,
        },
        "evaluated_models": {
            "logistic_regression": {
                "model_name": "Logistic Regression (Calibrated Threshold)",
                "default_probability": round(prob_logistic, 4),
                "approval_probability": round(1.0 - prob_logistic, 4),
                "decision_threshold": lc_threshold,
                "prediction": is_default_logistic,
                "risk_label": label_logistic,
                "risk_tier": tier_logistic,
                "notes": f"Evaluated against calibrated decision threshold of {lc_threshold}.",
            },
            "random_forest": {
                "model_name": "Random Forest Classifier",
                "default_probability": round(prob_rf, 4),
                "approval_probability": round(1.0 - prob_rf, 4),
                "decision_threshold": 0.50,
                "prediction": is_default_rf,
                "risk_label": label_rf,
                "risk_tier": tier_rf,
                "notes": "Ensemble probability evaluated on unscaled features at standard 0.50 threshold.",
            },
        },
        "loan_summary": {
            "loan_amnt": float(payload["loan_amnt"]),
            "int_rate": float(payload["int_rate"]),
            "annual_inc": float(payload["annual_inc"]),
            "dti": float(payload["dti"]),
            "grade": payload.get("grade", "B").upper(),
            "sub_grade": payload.get("sub_grade", "B1").upper(),
        },
        "explainability_ready": {
            "status": "hook_available",
            "method": "SHAP_TreeExplainer_and_LinearExplainer",
            "ready": True,
            "feature_dim": 51,
        },
    }), 200


# -------------------------------------------------------------------------
# Explainability Endpoints (SHAP & LIME)
# -------------------------------------------------------------------------
@app.route("/api/explainability/status", methods=["GET"])
def explainability_status():
    """Returns status and configuration of SHAP and LIME engines."""
    return jsonify(exp_engine.get_explainability_status()), 200


@app.route("/api/explain/cibil", methods=["POST"])
def explain_cibil_endpoint():
    """
    Computes SHAP and LIME explanations for an Indian bureau prediction.
    Uses the exact same preprocessing pipeline (preprocessing.build_cibil_features)
    and scaled feature vector.
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({
            "status": "error",
            "message": "Missing JSON payload in request body.",
        }), 400

    # 1. Preprocess using the EXACT same pipeline as prediction
    df_features = pp.build_cibil_features(payload)
    scaled_features = cibil_scaler.transform(df_features)

    # 2. Get prediction probability
    default_prob = float(cibil_model.predict_proba(scaled_features)[0, 1])
    is_default = int(default_prob >= 0.50)
    risk_label = "Decline / High Risk of Default" if is_default == 1 else "Approved / Prime Profile"

    prediction_meta = {
        "default_probability": round(default_prob, 4),
        "risk_label": risk_label,
        "prediction": is_default,
    }

    # 3. Compute explanation
    explanation = exp_engine.explain_cibil(df_features, scaled_features, prediction_meta)
    status_code = 200 if explanation.get("status") == "success" else 500
    return jsonify(explanation), status_code


@app.route("/api/explain/lendingclub", methods=["POST"])
def explain_lendingclub_endpoint():
    """
    Computes SHAP and LIME explanations for both LendingClub models.
    Uses the exact same preprocessing pipeline (preprocessing.build_lendingclub_features).
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({
            "status": "error",
            "message": "Missing JSON payload in request body.",
        }), 400

    # 1. Preprocess using the EXACT same pipeline as prediction
    X_raw, X_scaled_for_logistic = pp.build_lendingclub_features(payload)

    # 2. Model probabilities
    prob_log = float(lc_logistic.predict_proba(X_scaled_for_logistic)[0, 1])
    is_default_log = int(prob_log >= lc_threshold)
    label_log = "Decline / High Risk" if is_default_log == 1 else "Approved / Low Risk"

    prob_rf = float(lc_rf.predict_proba(X_raw)[0, 1])
    is_default_rf = int(prob_rf >= 0.50)
    label_rf = "Decline / High Risk" if is_default_rf == 1 else "Approved / Low Risk"

    prediction_meta = {
        "evaluated_models": {
            "logistic_regression": {
                "default_probability": round(prob_log, 4),
                "risk_label": label_log,
                "prediction": is_default_log,
            },
            "random_forest": {
                "default_probability": round(prob_rf, 4),
                "risk_label": label_rf,
                "prediction": is_default_rf,
            },
        }
    }

    # 3. Compute explanation
    explanation = exp_engine.explain_lendingclub(X_raw, X_scaled_for_logistic, prediction_meta)
    status_code = 200 if explanation.get("status") == "success" else 500
    return jsonify(explanation), status_code


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting Credit Risk Prediction API on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
