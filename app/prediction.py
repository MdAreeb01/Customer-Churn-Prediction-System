import joblib 
import pandas as pd 

# Load Model
model = joblib.load("../models/xgboost_model.pkl")

# Load Scaler
scaler = joblib.load("../models/scaler.pkl")

# Load Feature Names
feature_columns = joblib.load("../models/feature_columns.pkl")

# Prediction Function
def predict_customer(
    gender,
    senior,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
):

    total_charges = tenure * monthly_charges
    
    # Arrange columns exactly like training data 
    input_df = pd.DataFrame(
        0, 
        index = [0],
        columns = feature_columns
    )

    # Binary Columns
    input_df["gender"] = 1 if gender == "Male" else 0
    input_df["SeniorCitizen"] = 1 if senior == "Yes" else 0
    input_df["Partner"] = 1 if partner == "Yes" else 0
    input_df["Dependents"] = 1 if dependents == "Yes" else 0
    input_df["PhoneService"] = 1 if phone_service == "Yes" else 0
    input_df["PaperlessBilling"] = 1 if paperless_billing == "Yes" else 0

    # Categorical Columns
    if multiple_lines == "Yes":
        input_df["MultipleLines_Yes"] = 1

    elif multiple_lines == "No phone service":
        input_df["MultipleLines_No phone service"] = 1

    if internet_service == "Fiber optic":
        input_df["InternetService_Fiber optic"] = 1
    
    elif internet_service == "No":
        input_df["InternetService_No"] = 1

    if online_security == "Yes":
        input_df["OnlineSecurity_Yes"] = 1
    
    elif online_security == "No internet service":
        input_df["OnlineSecurity_No internet service"] = 1

    if online_backup == "Yes":
        input_df["OnlineBackup_Yes"] = 1
    
    elif online_backup == "No internet service":
        input_df["OnlineBackup_No internet service"] = 1

    if device_protection == "Yes":
        input_df["DeviceProtection_Yes"] = 1
    
    elif device_protection == "No internet service":
        input_df["DeviceProtection_No internet service"] = 1

    if tech_support == "Yes":
        input_df["TechSupport_Yes"] = 1
    
    elif tech_support == "No internet service":
        input_df["TechSupport_No internet service"] = 1

    if streaming_tv == "Yes":
        input_df["StreamingTV_Yes"] = 1
    
    elif streaming_tv == "No internet service":
        input_df["StreamingTV_No internet service"] = 1

    if streaming_movies == "Yes":
        input_df["StreamingMovies_Yes"] = 1
    
    elif streaming_movies == "No internet service":
        input_df["StreamingMovies_No internet service"] = 1

    if contract == "One year":
        input_df["Contract_One year"] = 1
    
    elif contract == "Two year":
        input_df["Contract_Two year"] = 1

    if payment_method == "Credit card (automatic)":
        input_df["PaymentMethod_Credit card (automatic)"] = 1
    
    elif payment_method == "Electronic check":
        input_df["PaymentMethod_Electronic check"] = 1
    
    elif payment_method == "Mailed check":
        input_df["PaymentMethod_Mailed check"] = 1

    # Numerical Columns
    input_df["tenure"] = tenure
    input_df["MonthlyCharges"] = monthly_charges
    input_df["TotalCharges"] = total_charges

    numerical_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    input_df[numerical_cols] = scaler.transform(
        input_df[numerical_cols]
    )

    probability = float(model.predict_proba(input_df)[0][1])

    prediction = int(model.predict(input_df)[0])

    if probability < 0.30:
        risk = "Low"

    elif probability < 0.70:
        risk = "Medium"

    else:
        risk = "High"

    return prediction, probability, risk