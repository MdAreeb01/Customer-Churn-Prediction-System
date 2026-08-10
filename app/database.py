import mysql.connector
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path

# Find .env in project root
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

# Connect to MySQL
connection = mysql.connector.connect(
    host = os.getenv("MYSQL_HOST"),
    port = int(os.getenv("MYSQL_PORT")),
    user = os.getenv("MYSQL_USER"),
    password = os.getenv("MYSQL_PASSWORD"),
    database = os.getenv("MYSQL_DATABASE")
)

cursor = connection.cursor()

# Create save_prediction
def save_prediction(
    gender,
    senior,
    partner,
    dependents,
    tenure,
    phone_service,
    internet_service,
    contract,
    monthly_charges,
    prediction,
    probability,
    risk
):

    total_charges = tenure * monthly_charges

    query = """
    INSERT INTO churn_predictions(

        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        InternetService,
        Contract,
        MonthlyCharges,
        TotalCharges,
        ChurnPrediction,
        ChurnProbability,
        RiskLevel
        
    )

    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)

    """
    try: 
        cursor.execute(
            query,
    
            (
                int(1 if gender == "Male" else 0),
                int(1 if senior == "Yes" else 0),
                int(1 if partner == "Yes" else 0),
                int(1 if dependents == "Yes" else 0),
                int(tenure),
                1 if phone_service == "Yes" else 0,
                str(internet_service),
                str(contract),
                float(monthly_charges),
                float(total_charges),
                "Yes" if int(prediction) == 1 else "No",
                float(round(probability * 100, 2)),
                str(risk)
            )
        )
        connection.commit()
        return True

    except mysql.connector.Error as err:
        import streamlit as st
        st.error(f"Database Error: {err}")
        return False

def get_prediction_dataframe():

    query = """
    SELECT *
    FROM churn_predictions
    ORDER BY prediction_date DESC
    """

    return pd.read_sql(query, connection)