import streamlit as st 

def about_page():
    
    st.title("ℹ️ About This Project")

    # Project Overview
    st.markdown("""
    ## Customer Churn Prediction System
    
    This project is an end-to-end Machine Learning application developed to predict whether a telecom customer is likely to churn.
    
    The application integrates Machine Learning, SQL Database Management, Data Visualization, AI-Powered Churn Analysis, and Interactive Dashboards to provide actionable business insights.
    """)

    st.divider()

    # Technology Stack
    st.subheader("Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.markdown("""
        
        ##### Programming
        
        - Python
        - Pandas
        - Numpy
        
        """)

    with tech2:
        st.markdown("""
        
        ##### Machine Learning
        
        - Scikit-learn
        - XGBoost
        - Joblib
        
        """)

    with tech3:
        st.markdown("""
        
        ##### Development
        
        - Streamlit
        - MySQL
        - Plotly
        
        """)

    st.divider()
        
    # Machine Learning Workflow
    st.subheader("Machine Learning Workflow")

    st.markdown("""
    
    1. Data Cleaning
    2. Exploratory Data Analysis
    3. Feature Engineering
    4. Model Training
    5. Hyperparameter Tuning
    6. Model Evaluation
    7. Deployment using Streamlit
    8. Prediction Storage using MySQL
    9. Interactive Dashboard
    10. AI-Powered Explanation Generation
    
    """)

    st.divider()

    # Dataset Information
    st.subheader("Dataset")

    st.markdown("""
    
    Dataset Name: **Telco Customer Churn**

    Records: **7043**

    Features: **21 Original Features**

    Target Variable: **Churn**
    
    """)

    st.divider()

    # Final Model
    st.subheader("Final Model")

    st.success("""
    
    Selected Model: **XGBoost**
    
    Reasons for Selection:
    
    - Highest Cross Validation Performance 
    - Strong Generalization 
    - Balanced Precision & Recall
    - Good ROC-AUC
    - Robust Performance after Hyperparameter Tuning
    
    """)

    st.divider()

    # Developer

    st.subheader("Developer")

    st.info("""
    
    Developed by: **Mohd Areeb**
    
    Machine Learning | Data Science | Python | SQL | Power BI
    
    """)

    st.divider()
    
    # Bonus Touch
    st.caption("Version 2.0 | Customer Churn Prediction System")