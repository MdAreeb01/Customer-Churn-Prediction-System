import streamlit as st
import plotly.express as px
from datetime import datetime
import mysql.connector
import joblib
from prediction import predict_customer
from database import (save_prediction, get_prediction_dataframe)
from model_performance import show_model_performance
from model_comparison import model_comparison_content
from hero_section import hero_section_details
from about import about_page

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

if "form_version" not in st.session_state:
    st.session_state.form_version = 0

if st.session_state.get("show_saved_toast", False):
    st.toast("Prediction saved successfully!")

    st.html(
        """
        <script>
            window.parent.document
                .querySelector('section.main')
                .scrollTo({
                    top: 0,
                    behavior: 'smooth'
                });
        </script>
        """,
        unsafe_allow_javascript=True
    )
    
    st.session_state.show_saved_toast = False

def check_model():

    try:
        joblib.load("../models/xgboost_model.pkl")
        return "🟢 Loaded"

    except:
        return "🔴 Not Loaded"

def check_database():

    try:
        connection = mysql.connector.connect(
            host = "127.0.0.1",
            port = 3306,
            user = "root",
            password = "3306",
            database = "customer_churn_db"
        )

        connection.close()

        return "🟢 Connected"

    except:
        return "🔴 Disconnected"
        
# st.sidebar.image("logo.png", width=120)
st.sidebar.title("📊 Customer Churn")
st.sidebar.caption("### Machine Learning Dashboard")
st.sidebar.divider()

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Predict Customer Churn",
        "Prediction History",
        "Model Performance",
        "Model Comparison",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.success("Model: XGBoost")

st.sidebar.caption("Version 1.0")

if page == "Home":

    current_time = datetime.now()

    hero_section_details()

    st.divider()

    st.subheader("System Status")

    col1, col2 = st.columns(2)

    with col1:

        st.write(f"**Model:** {check_model()}")
    
        st.write(f"**Database:** {check_database()}")
    
        st.write("**Prediction:** 🟢 Ready")

    with col2:

        st.write(
            f"**Date:** {current_time.strftime('%d %B %Y')}"
        )
    
        st.write(
            f"**Time:** {current_time.strftime('%I:%M:%S %p')}"
        )
        
elif page == "Predict Customer Churn":

    st.title("🔮 Predict Customer Churn")
    st.caption("### Predict whether a telecom customer is likely to churn using the trained XGBoost model.")
    st.divider()
    
    st.subheader("Customer Information")

    form_version = st.session_state.form_version

    # Initialize save flag
    if "saved" not in st.session_state:
        st.session_state.saved = False

    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.pills("Gender:", ["Female", "Male"], selection_mode = "single", default = None, key = f"gender_{form_version}")
    
        senior = st.pills("Senior Citizen:", ["Yes", "No"], selection_mode = "single", default = None, key = f"senior_{form_version}")
    
        partner = st.pills("Partner", ["Yes", "No"], selection_mode = "single", default = None, key = f"partner_{form_version}")
    
        dependents = st.pills("Dependents", ["Yes", "No"], selection_mode = "single", default = None, key = f"dependents_{form_version}")

        paperless_billing = st.pills("Paperless Billing", ["Yes", "No"], selection_mode = "single", default = None, key = f"paperless_billing_{form_version}")
        
        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ],
            key = f"contract_{form_version}"
        )
    
        payment_method = st.selectbox(
            "Payment Method", 
            [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
            ],
            key = f"payment_method_{form_version}"
        )
        
        monthly_charges = st.number_input(
            "Monthly Charges",
            18.0, 
            120.0,
            70.0,
            key = f"monthly_charges_{form_version}"
        )

        tenure = st.slider(
            "Tenure (Months)",
            0,
            72, 
            12,
            key = f"tenure_{form_version}"
        )

    with col2:

        phone_service = st.pills("Phone Service", ["Yes", "No"], 
                                 selection_mode = "single", default = None, 
                                 key = f"phone_service_{form_version}")

        if phone_service == "No":

            multiple_lines = "No phone service"

            col1, col2 = st.columns([2, 3])

            with col1:
                st.pills(
                    "Multiple Lines", ["No phone service"],
                    default = "No phone service", disabled = True, key = f"multiple_lines_disabled_{form_version}")

            with col2:
                st.caption("Multiple Lines is automatically set to No phone service because Phone Service is No.")

        elif phone_service == "Yes":

            multiple_lines = st.pills(
            "Multiple Lines", ["Yes", "No"],
            selection_mode = "single",
            default = None,
            key = f"multiple_lines_{form_version}"
            )

        else:
            multiple_lines = None

            st.pills(
                "Multiple Lines",
                ["No", "Yes"],
                disabled = True,
                key = f"multiple_lines_empty_{form_version}"
            )
        
        internet_service = st.pills(
            "Internet Service", ["DSL", "Fiber optic", "No"],
            selection_mode = "single", default = None, key = f"internet_service_{form_version}")

        if internet_service == "No":
            
            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"

            col1, col2 = st.columns([2, 3])

            with col1:
                st.pills(
                    "Online Security", ["No internet service"],
                    default = "No internet service", disabled = True, key = f"online_security_disabled_{form_version}")

                st.pills(
                    "Online Backup", ["No internet service"],
                    default = "No internet service", disabled = True, key = f"online_backup_disabled_{form_version}")

                st.pills(
                    "Device Protection", ["No internet service"],
                    default = "No internet service", disabled = True, key = f"device_protection_disabled_{form_version}")

                st.pills(
                    "Tech Support", ["No internet service"],
                    default = "No internet service", disabled = True, key = f"tech_support_disabled_{form_version}")

                st.pills(
                    "Streaming TV", ["No internet service"],
                    default = "No internet service", disabled = True, key = f"streaming_tv_disabled_{form_version}")

                st.pills(
                    "Streaming Movies", ["No internet service"],
                    default = "No internet service", disabled = True, key = f"streaming_movies_disabled_{form_version}")

            with col2:
                st.caption("Related services are automatically set to"
                           "'No internet service' because Internet Service is No.")

        elif internet_service == "DSL" or internet_service == "Fiber optic":

            online_security = st.pills(
            "Online Security", ["Yes", "No"],
            selection_mode = "single",
            default = None,
            key = f"online_security_{form_version}"
            )

            online_backup = st.pills(
            "Online Backup", ["Yes", "No"],
            selection_mode = "single",
            default = None,
            key = f"online_backup_{form_version}"
            )

            device_protection = st.pills(
            "Device Protection", ["Yes", "No"],
            selection_mode = "single",
            default = None,
            key = f"device_protection_{form_version}"
            )

            tech_support = st.pills(
            "Tech Support", ["Yes", "No"],
            selection_mode = "single",
            default = None,
            key = f"tech_support_{form_version}"
            )

            streaming_tv = st.pills(
            "Streaming TV", ["Yes", "No"],
            selection_mode = "single",
            default = None,
            key = f"streaming_tv_{form_version}"
            )

            streaming_movies = st.pills(
            "Streaming Movies", ["Yes", "No"],
            selection_mode = "single",
            default = None,
            key = f"streaming_movies_{form_version}"
            )

        else:
            online_security = None

            st.pills(
                "Online Security",
                ["No", "Yes"],
                disabled = True,
                key = f"online_security_empty_{form_version}"
            )

            online_backup = None

            st.pills(
                "Online Backup",
                ["No", "Yes"],
                disabled = True,
                key = f"online_backup_empty_{form_version}"
            )

            device_protection = None

            st.pills(
                "Device Protection",
                ["No", "Yes"],
                disabled = True,
                key = f"device_protection_empty_{form_version}"
            )

            tech_support = None

            st.pills(
                "Tech Support",
                ["No", "Yes"],
                disabled = True,
                key = f"tech_support_empty_{form_version}"
            )

            streaming_tv = None

            st.pills(
                "Streaming TV",
                ["No", "Yes"],
                disabled = True,
                key = f"streaming_tv_empty_{form_version}"
            )

            streaming_movies = None

            st.pills(
                "Streaming Movies",
                ["No", "Yes"],
                disabled = True,
                key = f"streaming_movies_empty_{form_version}"
            )
            
    predict_clicked = st.button("Predict", type = "primary", use_container_width = True)

    if predict_clicked:

        required_fields = {
                "Gender": gender,
                "Senior Citizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "Paperless Billing": paperless_billing,
                "Phone Service": phone_service,
                "Multiple Lines": multiple_lines,
                "Internet Service": internet_service,
                "Online Security": online_security,
                "Online Backup": online_backup,
                "Device Protection": device_protection,
                "Tech Support": tech_support,
                "Streaming TV": streaming_tv,
                "Streaming Movies": streaming_movies
            }

        missing = [k for k, v in required_fields.items() if v is None]

        if missing:
            st.error(f"Please select: {', '.join(missing)}")

        else:
            with st.spinner("Predicting customer churn..."):
                prediction, probability, risk = predict_customer(
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
                )
    
                st.session_state.prediction = prediction
                st.session_state.probability = probability
                st.session_state.risk = risk
    
                # Reset save flag for this new prediction
                st.session_state.saved = False
        
                # Store the input values too
                st.session_state.customer_data = {
                    "gender": gender,
                    "senior": senior,
                    "partner": partner,
                    "dependents": dependents,
                    "tenure": tenure,
                    "phone_service": phone_service,
                    "internet_service": internet_service,
                    "contract": contract,
                    "monthly_charges": monthly_charges
                }

    st.divider()

    if "prediction" in st.session_state:

        prediction = st.session_state.prediction
        probability = st.session_state.probability
        risk = st.session_state.risk

        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)

        # Probability card
        with col1:
            st.metric(
                "Churn Probability",
                f"{probability: .2%}"
            )

        # Risk level
        with col2:
            st.metric(
                "Risk Level",
                risk
            )

        with col3:
            st.metric(
                "Estimated Lifetime Charges",
                f"${tenure * monthly_charges:.2f}"
            )
            
        # Progress bar
        st.progress(float(probability))

        # Prediction Explanation Panel
        st.divider()

        st.subheader("Prediction Explaination")

        reasons = []

        if tenure < 12:
            reasons.append(
                "• Customer has a short tenure, which is associated with higher churn."
            )
        
        if contract == "Month-to-month":
            reasons.append(
                "• Month-to-month contracts have the highest churn rate."
            )
        
        if internet_service == "Fiber optic":
            reasons.append(
                "• Fiber optic customers tend to churn more frequently."
            )
        
        if monthly_charges > 70:
            reasons.append(
                "• Monthly charges are relatively high."
            )
        
        if tech_support == "No":
            reasons.append(
                "• Customer does not have Tech Support."
            )
        
        if online_security == "No":
            reasons.append(
                "• Customer does not use Online Security."
            )

        if reasons:

            for reason in reasons:
                st.write(reason)
        
        else:

            st.success(
                "No major churn indicators were detected."
            )

        st.divider()

        # Suggested Retention Strategy
        st.subheader("Suggested Retention Strategy")

        if prediction == 1:

            st.info("""
        
            • Offer a One-Year or Two-Year contract.
        
            • Provide a loyalty discount.
        
            • Recommend Tech Support.
        
            • Recommend Online Security.
        
            • Contact the customer before renewal.
        
            """)
        
        else:
        
            st.success("""
        
            Customer appears to have low churn risk.
        
            Continue maintaining customer satisfaction.
        
            """)
        
        # Display result
        if prediction == 1:
            st.error(f"⚠️ Customer is likely to churn")
                
        else:
            st.success(f"✅ Customer is likely to stay")

        save_clicked = st.button(
            "💾 Save Prediction",
            type = "primary",
            use_container_width = True,
            disabled = st.session_state.saved
        )

        if save_clicked:
    
            data = st.session_state.customer_data
    
            saved = save_prediction(
                data["gender"],
                data["senior"],
                data["partner"],
                data["dependents"],
                data["tenure"],
                data["phone_service"],
                data["internet_service"],
                data["contract"],
                data["monthly_charges"],
                prediction,
                probability,
                risk
            )
    
            if saved:

                # Show toast after page refresh
                st.session_state.show_saved_toast = True

                # Remove previous prediction
                st.session_state.pop("prediction", None)
                st.session_state.pop("probability", None)
                st.session_state.pop("risk", None)
                st.session_state.pop("customer_data", None)

                # Create a completely new set of input widgets
                st.session_state.form_version += 1
    
                # Refresh Page
                st.rerun()
                
            else:
                st.error("Unable to save prediction. "
                         "Please check the database connection.")

elif page == "Prediction History":
    
    st.title("📜Prediction History")

    st.divider()

    df = get_prediction_dataframe()

    filtered_df = df.copy()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Predictions", 
            len(filtered_df)
        )

    with col2:
        st.metric(
            "High Risk",
            len(filtered_df[filtered_df["RiskLevel"] == "High"])
        )

    with col3:
        st.metric(
            "Average Probability",
            f"{filtered_df['ChurnProbability'].mean():.2f}%"
        )

    with col4:
        if len(filtered_df) > 0:
            
            churn_rate = (
               len(filtered_df[filtered_df["ChurnPrediction"] == "Yes"])
               / len(filtered_df)
            ) * 100
        else: 
            churn_rate = 0
            
        st.metric(
            "Predicted Churn Rate",
            f"{churn_rate:.2f}%"
        )

    st.divider()

    st.subheader("Filter Predictions")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:

        risk_filter = st.selectbox(
            "Risk_Level",
            ["All"] + sorted(df["RiskLevel"].unique().tolist())
        )

    with filter_col2:

        contract_filter = st.selectbox(
            "Contract",
            ["All"] + sorted(df["Contract"].unique().tolist())
        )

    with filter_col3:

        internet_filter = st.selectbox(
            "Internet Service",
            ["All"] + sorted(df["InternetService"].unique().tolist())
        )

    if risk_filter != "All":
        filtered_df = filtered_df[
            filtered_df["RiskLevel"] == risk_filter
        ]

    if contract_filter != "All":
        filtered_df = filtered_df[
            filtered_df["Contract"] == contract_filter
        ]

    if internet_filter != "All":
        filtered_df = filtered_df[
            filtered_df["InternetService"] == internet_filter
        ]

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Risk Level Distribution")
    
        risk_counts = (
            filtered_df["RiskLevel"].value_counts().reset_index()
        )
    
        risk_counts.columns = ["Risk Level", "Count"]
    
        fig = px.bar(
            risk_counts,
            x = "Risk Level",
            y = "Count",
            text = "Count",
            color = "Risk Level",
            color_discrete_map = {
                "High": "#5E4637",   
                "Medium": "#9D755D", 
                "Low": "#E8D8CE"  
            },
            title = "Customer Risk Distribution"
        )

        fig.update_traces(
            insidetextfont=dict(color="#2C1603", weight = "bold"),
        )
            
        fig.update_layout(
            xaxis_title = "Risk Level",
            yaxis_title = "Number of Customers",
            template = "plotly_white",
            showlegend = False
        )
    
        st.plotly_chart(
            fig,
            use_container_width = True
        )

    with chart_col2:

        st.subheader("Prediction Outcome")
    
        prediction_counts = (
            filtered_df["ChurnPrediction"].value_counts().reset_index()
        )
    
        prediction_counts.columns = [
            "Prediction", 
            "Count"
        ]
    
        prediction_counts["Prediction"] = prediction_counts["Prediction"].replace({
            "Yes": "Churn",
            "No": "Stay"
        })
    
        fig = px.pie(
            prediction_counts,
            names = "Prediction",
            values = "Count",
            hole = 0.5,
            title = "Predicted Customer Outcomes",
            color_discrete_sequence = ["#9D755D", "#E8D8CE"] 
        )
    
        fig.update_traces(
            textposition = "inside",
            textinfo = "percent + label",
            insidetextfont=dict(color='#2C1603', weight = "bold"),
        )
    
        fig.update_layout(
            template = "plotly_white"
        )
    
        st.plotly_chart(
            fig,
            use_container_width = True
        )

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:

        st.subheader("Contract Distribution")

        contract_counts = (
            filtered_df["Contract"].value_counts().reset_index()
        )

        contract_counts.columns = [
            "Contract",
            "Count"
        ]

        fig = px.bar(
            contract_counts,
            x = "Contract",
            y = "Count",
            text = "Count",
            color = "Contract",
            color_discrete_map = {
                 "Month-to-month": "#5E4637",   
                 "One year": "#9D755D", 
                 "Two year": "#E8D8CE"  
            },
            title = "Contracts in Prediction History"
        )

        fig.update_traces(
            insidetextfont=dict(color="#2C1603", weight = "bold"),
        )

        fig.update_layout(
            template = "plotly_white",
            xaxis_title = "Contract",
            yaxis_title = "Number of Customers",
            showlegend = False
        )

        st.plotly_chart(
            fig, 
            use_container_width = True
        )

    with chart_col4:

        st.subheader("Internet Service Distribution")

        internet_counts = (
            filtered_df["InternetService"].value_counts().reset_index()
        )

        internet_counts.columns = [
            "Internet Service",
            "Count"
        ]

        fig = px.pie(
            internet_counts,
            names = "Internet Service",
            values = "Count",
            hole = 0.3,
            title = "Internet Service Types",
            color_discrete_sequence = ["#5E4637", "#9D755D", "#E8D8CE"]
        )

        fig.update_traces(
            insidetextfont=dict(color="#2C1603", weight = "bold"),
        )

        fig.update_layout(
            template = "plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width = True
        )
        
    st.divider()

    st.subheader("Prediction History")

    csv = filtered_df.to_csv(index = False).encode("utf-8")

    st.download_button(
        label = "Download CSV",
        data = csv,
        file_name = "prediction_history.csv",
        mime = "text/csv",
        disabled = filtered_df.empty
    )
    
    display_df = filtered_df[
        [
            "prediction_id",
            "gender",
            "SeniorCitizen",
            "Partner",
            "tenure",
            "InternetService",
            "Contract",
            "MonthlyCharges",
            "ChurnPrediction",
            "ChurnProbability",
            "RiskLevel",
            "prediction_date"
        ]
    ]

    display_df = display_df.rename(
        columns = {
            "prediction_id": "Prediction ID",
            "gender": "Gender",
            "SeniorCitizen": "Senior Citizen",
            "Partner": "Partner",
            "tenure": "Tenure",
            "InternetService": "Internet Service",
            "Contract": "Contract",
            "MonthlyCharges": "Monthly Charges",
            "ChurnPrediction": "Prediction",
            "ChurnProbability": "Probability (%)",
            "RiskLevel": "Risk",
            "prediction_date": "Prediction Date"
        }
    )

    st.caption(
        f"Showing {len(display_df)} prediction(s)."
    )
    
    if display_df.empty:

        st.info(
            "📭 No predictions available yet.\n\nGenerate and save a prediction to view the history."
        )
    
    else:
    
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height = 450
        )
            
elif page == "Model Performance":
    show_model_performance()

elif page == "Model Comparison":
    model_comparison_content()

elif page == "About":
    about_page()