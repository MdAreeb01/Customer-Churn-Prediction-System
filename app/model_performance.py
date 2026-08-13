import joblib
from pathlib import Path    
import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent.parent

# Load Evaluation Metrics
metrics = joblib.load(BASE_DIR / "artifacts" / "model_metrics.pkl")

# Load Confusion Metrics
cm = joblib.load(BASE_DIR / "artifacts" / "confusion_matrix.pkl")

# Load ROC Data
roc_data = joblib.load(BASE_DIR / "artifacts" / "roc_curve.pkl")

# Load Feature Importance
feature_importance = joblib.load(BASE_DIR / "artifacts" / "feature_importance.pkl")

def show_model_performance():

    st.title("📈 Model Performance")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Accuracy",
            f"{metrics['Accuracy']:.2%}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{metrics['Precision']:.2%}"
        )

    with col3:
        st.metric(
            "Recall",
            f"{metrics['Recall']:.2%}"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{metrics['F1 Score']:.2%}"
        )

    with col5:
        st.metric(
            "ROC-AUC",
            f"{metrics['ROC-AUC']:.2%}"
        )

    st.divider()
    
    st.subheader("Confusion Matrix")
    
    # Convert Confusion Matrix to Data Frame
    cm_df = pd.DataFrame(
        cm, 
        index = ["Actual Stay", "Actual Churn"],
        columns = ["Predicted Stay", "Predicted Churn"]
    )
    
    # Create Heatmap
    fig = px.imshow(
        cm_df,
        text_auto = True,
        color_continuous_scale = "Blues",
        title = "Confusion Matrix"
    )
    
    fig.update_layout(
        xaxis_title = "Predicted Class",
        yaxis_title = "Actual Class"
    )
    
    st.plotly_chart(
        fig,
        use_container_width = True
    )
    
    st.divider()
    
    # ROC Curve
    st.subheader("ROC Curve")
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x = roc_data["fpr"],
            y = roc_data["tpr"],
            mode = "lines",
            name = f"AUC = {metrics['ROC-AUC']:.3f}",
            line=dict(color="#86786C")
        )
    )
    
    # Add Random Classifier
    fig.add_trace(
        go.Scatter(
            x = [0, 1],
            y = [0, 1],
            mode = "lines",
            name = "Random",
            line = dict(color = "red", dash = "dash")

        )
    )
    
    # Layout
    fig.update_layout(
        title = "ROC Curve",
        xaxis_title = "False Positive Rate",
        yaxis_title = "True Positive Rate",
        template = "plotly_white"
    )
    
    # Display
    st.plotly_chart(
        fig, 
        use_container_width = True
    )

    st.divider()
    
    # Feature Importance
    st.subheader("Top 10 Important Features")

    top_features = (
        feature_importance.sort_values(by = "Importance", ascending = False).head(10))

    # Plot
    fig = px.bar(
        top_features,
        x = "Importance",
        y = "Feature",
        text = "Importance",
        orientation = "h",
        title = "Top 10 Features",
        color = "Feature",
        color_discrete_sequence = 
        [
            "#2C1603",
            "#422E1D",
            "#594737",
            "#6F6052",
            "#86786C",
            "#9C9186",
            "#B3AAA1",
            "#C9C2BB",
            "#E0DBD5",
            "#F7F4F0",
        ]
    )

    fig.update_traces(
        outsidetextfont=dict(color="white",  weight = "bold"),
        texttemplate="%{text:.2}%",
        textposition="outside"
        )

    fig.update_layout(
        template = "plotly_white",
        showlegend = False
    )

    st.plotly_chart(
        fig, 
        use_container_width = True
    )

    st.divider()
    
    # Model Summary
    st.success("""
    ### Selected Model: XGBoost
    
    XGBoost was selected as the final model because it achieved the best balance between cross- validation performance and test performance among all evaluated algorithms.
    
    Key Strengths:
    
    - High predictive performance
    - Good generalization
    - Strong ROC-AUC
    - Balanced Precision and Recall
    - Handles nonlinear relationships effectively
    - Robust against overfitting after hyperparameter tuning

    """)