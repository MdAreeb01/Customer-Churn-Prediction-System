import streamlit as st

def hero_section_details():

    st.title("🏠 Customer Churn Prediction System")
    
    st.markdown("""
    ### Predict Telecom Customer Churn using Machine Learning
    
    This application predicts whether a telecom customer is likely to churn using an optimized   **XGBoost** model.
    
    The project integrates **Machine Learning, SQL, Streamlit, Plotly, Groq AI, and Power BI** into a complete analytics solution.
    """)

    st.divider()
    
    # KPI Cards
    col1,col2,col3,col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Model",
            "XGBoost"
        )
    
    with col2:
        st.metric(
            "Accuracy",
            "79.34%"
        )
    
    with col3:
        st.metric(
            "Dataset",
            "Telco"
        )
    
    with col4:
        st.metric(
            "Records",
            "7043"
        )
    
    # Feature Cards
    st.divider()
    
    st.subheader("Key Features")
    
    st.markdown("""
    
    -> Customer Churn Prediction
    
    -> Prediction History Dashboard
    
    -> Model Performance Dashboard
    
    -> Interactive Plotly Charts
    
    -> MySQL Database Integration
    
    -> CSV Export
    
    -> Power BI Ready
    
    -> AI Ready
    
    """)
    
    # Workflow
    st.divider()
    
    st.subheader("Project Workflow")
    
    st.code("""
    Dataset
    ↓
    Data Cleaning
    ↓
    EDA
    ↓
    Feature Engineering
    ↓
    Model Training
    ↓
    XGBoost
    ↓
    Churn Prediction
    ↓
    MySQL
    ↓
    Power BI
    ↓
    Groq AI
    ↓
    Actionable Insights
    """)
    
    # Quick Navigation
    st.divider()
    
    st.subheader("Quick Start")
    
    st.info("""
    
    1️⃣ Go to Predict Customer Churn
    
    2️⃣ Enter Customer Details
    
    3️⃣ Generate Prediction
    
    4️⃣ Generate AI-Powered Explanation
    
    5️⃣ Save Prediction

    6️⃣ Analyze Prediction History
    
    """)