import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load the CSV
comparison = pd.read_csv(BASE_DIR / "artifacts" / "model_comparison.csv")

display_comparison = comparison[
[
    "Model",
    "Training_Accuracy",
    "Test_Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "CV_Accuracy"
]
]

display_comparison = display_comparison.rename(
    columns = {
        "Model": "Model",
        "Training_Accuracy": "Training Accuracy",
        "Test_Accuracy": "Test Accuracy",
        "Precision": "Precision Score",
        "Recall": "Recall Score",
        "F1": "F1 Score",
        "ROC_AUC": "ROC-AUC Score",
        "CV_Accuracy": "Cross-Validation Accuracy"
    }
)

def model_comparison_content():
    
    # Show Table
    st.title("⚖️ Model Comparison")
    
    st.dataframe(
        display_comparison,
        use_container_width = True,
        hide_index = True
    )

    st.divider()
    
    # Accuracy Comparison Chart
    fig = px.bar(
        display_comparison,
        x = "Model",
        y = "Test Accuracy",
        text = "Test Accuracy",
        title = "Model Accuracy Comparison",
        color = "Model",
        color_discrete_sequence = 
        [
            "#422E1D",
            "#594737",
            "#6F6052",
            "#86786C",
            "#9C9186",
            "#B3AAA1"
        ]
    )

    fig.update_traces(
        insidetextfont=dict(color="#2C1603", weight = "bold")
    )

    fig.update_layout(
        showlegend = False
    )
    
    st.plotly_chart(
        fig,
        use_container_width = True
    )

    st.divider()
    
    # F1 Score Comparison
    fig = px.bar(
        display_comparison,
        x = "Model",
        y = "F1 Score",
        text = "F1 Score",
        title = "F1 Score Comnparison",
        color = "Model",
        color_discrete_sequence = 
        [
            "#422E1D",
            "#594737",
            "#6F6052",
            "#86786C",
            "#9C9186",
            "#B3AAA1"
        ]
    )

    fig.update_traces(
        insidetextfont=dict(color="#2C1603", weight = "bold")
    )

    fig.update_layout(
        showlegend = False
    )
    
    st.plotly_chart(
        fig,
        use_container_width = True
    )

    st.divider()
    
    # ROC-AUC Comparison
    fig = px.bar(
        display_comparison,
        x = "Model",
        y = "ROC-AUC Score",
        text = "ROC-AUC Score",
        title = "ROC-AUC Comparison",
        color = "Model",
            color_discrete_sequence = 
            [
                "#422E1D",
                "#594737",
                "#6F6052",
                "#86786C",
                "#9C9186",
                "#B3AAA1"
            ]
        )
    
    fig.update_traces(
        insidetextfont=dict(color="#2C1603", weight = "bold")
    )

    fig.update_layout(
        showlegend = False
    )
    
    st.plotly_chart(
        fig,
        use_container_width = True
    )

    st.divider()
    
    # Final Selection Summary
    st.success("""
    ### Final Model Selection
    
    After evaluating six machine learning algorithms, XGBoost was selected as the final deployment model.
    
    Reasons:
    
    - Strong cross-validation performance.
    - Excellent balance between precision and recall.
    - Competitive ROC-AUC score.
    - Good generalization after hyperparameter tuning.
    - Robust performance on unseen data.
    """)