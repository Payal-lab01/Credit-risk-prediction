"""
Credit Risk Prediction Preprocessing Engine
Handles schema validation, feature transformations, baseline distribution merges,
and dual model preprocessing for Indian (CIBIL) and US (LendingClub) credit models.
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")

# -------------------------------------------------------------------------
# Load Saved Artifacts
# -------------------------------------------------------------------------
cibil_feature_columns = joblib.load(os.path.join(MODELS_DIR, "cibil_feature_columns.pkl"))
cibil_scaler = joblib.load(os.path.join(MODELS_DIR, "cibil_scaler.pkl"))

lc_feature_columns = joblib.load(os.path.join(MODELS_DIR, "lc_feature_columns.pkl"))
lc_scaler = joblib.load(os.path.join(MODELS_DIR, "lc_scaler.pkl"))
lc_threshold = float(joblib.load(os.path.join(MODELS_DIR, "lc_threshold.pkl")))

# -------------------------------------------------------------------------
# CIBIL Provisional Categorical Mappings
# IMPORTANT NOTICE:
# The CIBIL LabelEncoder instances were not exported as separate pickle files.
# The mappings below follow scikit-learn's standard LabelEncoder alphabetical
# ordering on the known unique categories from training. These mappings are
# treated as PROVISIONAL pending direct encoder artifacts or raw training data verification.
# -------------------------------------------------------------------------
CIBIL_CATEGORICAL_MAPPINGS_PROVISIONAL = True

CIBIL_PROVISIONAL_CATEGORICAL_MAPPINGS = {
    "MARITALSTATUS": {
        "Married": 0,
        "Single": 1,
    },
    "EDUCATION": {
        "12TH": 0,
        "GRADUATE": 1,
        "OTHERS": 2,
        "POST-GRADUATE": 3,
        "PROFESSIONAL": 4,
        "SSC": 5,
        "UNDER GRADUATE": 6,
    },
    "GENDER": {
        "F": 0,
        "M": 1,
    },
    "last_prod_enq2": {
        "AL": 0,
        "CC": 1,
        "ConsumerLoan": 2,
        "GL": 3,
        "HL": 4,
        "PL": 5,
        "others": 6,
    },
    "first_prod_enq2": {
        "AL": 0,
        "CC": 1,
        "ConsumerLoan": 2,
        "GL": 3,
        "HL": 4,
        "PL": 5,
        "others": 6,
    },
}

# Pre-computed mean vector from cibil_scaler for all 83 features
CIBIL_BASELINE_MEANS = dict(zip(cibil_feature_columns, cibil_scaler.mean_))

# -------------------------------------------------------------------------
# LendingClub Constants
# -------------------------------------------------------------------------
LC_NUMERICAL_COLS = [
    "loan_amnt",
    "int_rate",
    "installment",
    "log_annual_inc",
    "dti",
    "revol_bal",
    "revol_util",
    "open_acc",
    "total_acc",
    "delinq_2yrs",
    "inq_last_6mths",
]

VALID_GRADES = ["A", "B", "C", "D", "E", "F", "G"]
VALID_SUB_GRADES = [f"{g}{i}" for g in VALID_GRADES for i in range(1, 6)]


class ValidationError(Exception):
    """Raised when user input violates validation boundaries."""
    def __init__(self, message, field=None):
        super().__init__(message)
        self.message = message
        self.field = field


# -------------------------------------------------------------------------
# CIBIL Preprocessing Logic
# -------------------------------------------------------------------------
def build_cibil_features(user_input: dict) -> pd.DataFrame:
    """
    Transforms user input into the strict 83-dimensional feature vector
    expected by cibil_scaler.pkl and cibil_logistic_model.pkl.

    1. Validates primary applicant values.
    2. Encodes categorical variables (provisional label encoding).
    3. Starts with the baseline mean vector derived from cibil_scaler.mean_.
    4. Overlays user-provided primary inputs and any advanced bureau overrides.
    5. Computes derived ratio features (e.g. pct_active_tl, pct_closed_tl).
    6. Returns a 1x83 pandas DataFrame ordered exactly as in cibil_feature_columns.pkl.
    """
    if not isinstance(user_input, dict):
        raise ValidationError("Request payload must be a JSON object.")

    # Validate essential numerical fields if provided
    credit_score = user_input.get("Credit_Score", CIBIL_BASELINE_MEANS["Credit_Score"])
    try:
        credit_score = float(credit_score)
        if not (300 <= credit_score <= 900):
            raise ValidationError("Credit_Score must be between 300 and 900.", "Credit_Score")
    except (ValueError, TypeError):
        raise ValidationError("Credit_Score must be a valid number.", "Credit_Score")

    age = user_input.get("AGE", CIBIL_BASELINE_MEANS["AGE"])
    try:
        age = float(age)
        if not (18 <= age <= 100):
            raise ValidationError("AGE must be between 18 and 100.", "AGE")
    except (ValueError, TypeError):
        raise ValidationError("AGE must be a valid number.", "AGE")

    income = user_input.get("NETMONTHLYINCOME", CIBIL_BASELINE_MEANS["NETMONTHLYINCOME"])
    try:
        income = float(income)
        if income < 0:
            raise ValidationError("NETMONTHLYINCOME cannot be negative.", "NETMONTHLYINCOME")
    except (ValueError, TypeError):
        raise ValidationError("NETMONTHLYINCOME must be a valid number.", "NETMONTHLYINCOME")

    time_emp = user_input.get("Time_With_Curr_Empr", CIBIL_BASELINE_MEANS["Time_With_Curr_Empr"])
    try:
        time_emp = float(time_emp)
        if time_emp < 0:
            raise ValidationError("Time_With_Curr_Empr cannot be negative.", "Time_With_Curr_Empr")
    except (ValueError, TypeError):
        raise ValidationError("Time_With_Curr_Empr must be a valid number.", "Time_With_Curr_Empr")

    total_tl = user_input.get("Total_TL", CIBIL_BASELINE_MEANS["Total_TL"])
    try:
        total_tl = float(total_tl)
        if total_tl < 0:
            raise ValidationError("Total_TL cannot be negative.", "Total_TL")
    except (ValueError, TypeError):
        raise ValidationError("Total_TL must be a valid number.", "Total_TL")

    tot_active_tl = user_input.get("Tot_Active_TL", CIBIL_BASELINE_MEANS["Tot_Active_TL"])
    try:
        tot_active_tl = float(tot_active_tl)
        if tot_active_tl < 0:
            raise ValidationError("Tot_Active_TL cannot be negative.", "Tot_Active_TL")
    except (ValueError, TypeError):
        raise ValidationError("Tot_Active_TL must be a valid number.", "Tot_Active_TL")

    tot_closed_tl = user_input.get("Tot_Closed_TL", CIBIL_BASELINE_MEANS["Tot_Closed_TL"])
    try:
        tot_closed_tl = float(tot_closed_tl)
        if tot_closed_tl < 0:
            raise ValidationError("Tot_Closed_TL cannot be negative.", "Tot_Closed_TL")
    except (ValueError, TypeError):
        raise ValidationError("Tot_Closed_TL must be a valid number.", "Tot_Closed_TL")

    tot_missed_pmnt = user_input.get("Tot_Missed_Pmnt", CIBIL_BASELINE_MEANS["Tot_Missed_Pmnt"])
    try:
        tot_missed_pmnt = float(tot_missed_pmnt)
        if tot_missed_pmnt < 0:
            raise ValidationError("Tot_Missed_Pmnt cannot be negative.", "Tot_Missed_Pmnt")
    except (ValueError, TypeError):
        raise ValidationError("Tot_Missed_Pmnt must be a valid number.", "Tot_Missed_Pmnt")

    num_delinq = user_input.get("num_times_delinquent", CIBIL_BASELINE_MEANS["num_times_delinquent"])
    try:
        num_delinq = float(num_delinq)
        if num_delinq < 0:
            raise ValidationError("num_times_delinquent cannot be negative.", "num_times_delinquent")
    except (ValueError, TypeError):
        raise ValidationError("num_times_delinquent must be a valid number.", "num_times_delinquent")

    tot_enq = user_input.get("tot_enq", CIBIL_BASELINE_MEANS["tot_enq"])
    try:
        tot_enq = float(tot_enq)
        if tot_enq < 0:
            raise ValidationError("tot_enq cannot be negative.", "tot_enq")
    except (ValueError, TypeError):
        raise ValidationError("tot_enq must be a valid number.", "tot_enq")

    # Start with complete baseline dictionary
    feature_vector = CIBIL_BASELINE_MEANS.copy()

    # Apply validated primary numeric fields
    feature_vector["Credit_Score"] = credit_score
    feature_vector["AGE"] = age
    feature_vector["NETMONTHLYINCOME"] = income
    feature_vector["Time_With_Curr_Empr"] = time_emp
    feature_vector["Total_TL"] = total_tl
    feature_vector["Tot_Active_TL"] = tot_active_tl
    feature_vector["Tot_Closed_TL"] = tot_closed_tl
    feature_vector["Tot_Missed_Pmnt"] = tot_missed_pmnt
    feature_vector["num_times_delinquent"] = num_delinq
    feature_vector["tot_enq"] = tot_enq

    # Derived ratios
    if total_tl > 0:
        feature_vector["pct_active_tl"] = min(1.0, tot_active_tl / total_tl)
        feature_vector["pct_closed_tl"] = min(1.0, tot_closed_tl / total_tl)
        feature_vector["pct_of_active_TLs_ever"] = feature_vector["pct_active_tl"]

    # Encode categorical fields
    for cat_col, mapping in CIBIL_PROVISIONAL_CATEGORICAL_MAPPINGS.items():
        if cat_col in user_input:
            val = user_input[cat_col]
            if isinstance(val, (int, float)):
                feature_vector[cat_col] = float(val)
            elif isinstance(val, str):
                if val not in mapping:
                    valid_choices = list(mapping.keys())
                    raise ValidationError(
                        f"Invalid value '{val}' for {cat_col}. Valid choices: {valid_choices}",
                        cat_col,
                    )
                feature_vector[cat_col] = float(mapping[val])
            else:
                raise ValidationError(f"Invalid format for {cat_col}.", cat_col)

    # Optional: Apply advanced bureau overrides if supplied
    advanced = user_input.get("advanced_bureau_overrides", {})
    if isinstance(advanced, dict):
        for k, v in advanced.items():
            if k in feature_vector:
                try:
                    feature_vector[k] = float(v)
                except (ValueError, TypeError):
                    raise ValidationError(f"Override for {k} must be a number.", k)

    # Construct ordered DataFrame matching training schema exactly
    df_cibil = pd.DataFrame([[feature_vector[col] for col in cibil_feature_columns]], columns=cibil_feature_columns)
    return df_cibil


# -------------------------------------------------------------------------
# LendingClub Preprocessing Logic
# -------------------------------------------------------------------------
def build_lendingclub_features(user_input: dict):
    """
    Transforms user input into the strict 51-dimensional feature space for LendingClub:
    1. 11 numerical features (with log_annual_inc = np.log1p(annual_inc)).
    2. 40 dummy variables (grade_B...grade_G, sub_grade_A2...sub_grade_G5).

    Returns:
    - X_raw: 1x51 DataFrame with raw numerical features + dummy indicators (for Random Forest).
    - X_scaled_for_logistic: 1x51 DataFrame where the 11 continuous features are scaled
      using lc_scaler.pkl, while the 40 dummy indicators remain unscaled (for Logistic Regression).
    """
    if not isinstance(user_input, dict):
        raise ValidationError("Request payload must be a JSON object.")

    # Required numerical fields
    required_num = [
        "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
        "revol_bal", "revol_util", "open_acc", "total_acc", "delinq_2yrs", "inq_last_6mths"
    ]
    extracted_num = {}

    for field in required_num:
        if field not in user_input:
            raise ValidationError(f"Missing required field: '{field}'", field)
        try:
            val = float(user_input[field])
        except (ValueError, TypeError):
            raise ValidationError(f"Field '{field}' must be a valid number.", field)

        if field in ["loan_amnt", "annual_inc"] and val <= 0:
            raise ValidationError(f"'{field}' must be greater than 0.", field)
        if field in ["int_rate", "installment", "dti", "revol_bal", "revol_util", "open_acc", "total_acc", "delinq_2yrs", "inq_last_6mths"] and val < 0:
            raise ValidationError(f"'{field}' cannot be negative.", field)

        extracted_num[field] = val

    # Validate grade and sub_grade
    grade = user_input.get("grade", "B")
    if not isinstance(grade, str) or grade.upper() not in VALID_GRADES:
        raise ValidationError(f"Invalid grade '{grade}'. Valid options: {VALID_GRADES}", "grade")
    grade = grade.upper()

    sub_grade = user_input.get("sub_grade", f"{grade}1")
    if not isinstance(sub_grade, str) or sub_grade.upper() not in VALID_SUB_GRADES:
        raise ValidationError(f"Invalid sub_grade '{sub_grade}'. Valid options: {VALID_SUB_GRADES}", "sub_grade")
    sub_grade = sub_grade.upper()

    # Ensure grade matches sub_grade prefix (e.g. Grade B requires B1-B5)
    if not sub_grade.startswith(grade):
        raise ValidationError(f"sub_grade '{sub_grade}' does not match grade '{grade}'.", "sub_grade")

    # Compute log_annual_inc (matches Cell 22 in CRP_LendingClub.ipynb)
    log_annual_inc = float(np.log1p(extracted_num["annual_inc"]))

    # Prepare DataFrame with all 51 features initialized to 0.0
    X_raw = pd.DataFrame(0.0, index=[0], columns=lc_feature_columns)

    # Assign 11 continuous features
    X_raw["loan_amnt"] = extracted_num["loan_amnt"]
    X_raw["int_rate"] = extracted_num["int_rate"]
    X_raw["installment"] = extracted_num["installment"]
    X_raw["log_annual_inc"] = log_annual_inc
    X_raw["dti"] = extracted_num["dti"]
    X_raw["revol_bal"] = extracted_num["revol_bal"]
    X_raw["revol_util"] = extracted_num["revol_util"]
    X_raw["open_acc"] = extracted_num["open_acc"]
    X_raw["total_acc"] = extracted_num["total_acc"]
    X_raw["delinq_2yrs"] = extracted_num["delinq_2yrs"]
    X_raw["inq_last_6mths"] = extracted_num["inq_last_6mths"]

    # Assign dummy variables (Grade A and Sub-Grade A1 are baseline/dropped in drop_first=True)
    if grade != "A":
        grade_col = f"grade_{grade}"
        if grade_col in X_raw.columns:
            X_raw[grade_col] = 1.0

    if sub_grade != "A1":
        sub_grade_col = f"sub_grade_{sub_grade}"
        if sub_grade_col in X_raw.columns:
            X_raw[sub_grade_col] = 1.0

    # Prepare scaled version for Logistic Regression
    # Note: Only the 11 continuous features are scaled via lc_scaler.pkl (n_features_in_ = 11)
    # The dummy variables remain unscaled binary values.
    num_df = X_raw[LC_NUMERICAL_COLS].copy()
    scaled_num = lc_scaler.transform(num_df)

    X_scaled_for_logistic = X_raw.copy()
    X_scaled_for_logistic[LC_NUMERICAL_COLS] = scaled_num

    return X_raw, X_scaled_for_logistic
