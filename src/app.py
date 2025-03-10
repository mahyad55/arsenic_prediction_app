## Import Libraries
import numpy as np
import streamlit as st
import pickle
import shap
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils import ModelEvaluation
import plotly.express as px
from streamlit_plotly_events import plotly_events
from utils import StreamLit
from data_loader import (
    model, optuna_model, explainer,
    X_train, X_test, y_train, y_test, X_huc,
    y_test_results, y_train_results, y_indent_results, y_huc_results,
    y_train_pred_proba,y_test_pred_proba
)

## Initiate app
sl = StreamLit()
st.set_page_config(
    page_title="Arsenic Prediction",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

## Import data
evaluation = ModelEvaluation()
if "thr_method" not in st.session_state:
    st.session_state.thr_method = "auc"
best_thr = evaluation.best_threshold(y_test, y_test_pred_proba, method=st.session_state.thr_method)
y_train_pred = np.where(y_train_pred_proba > best_thr, 1, 0)
y_test_pred = np.where(y_test_pred_proba > best_thr, 1, 0)

# Initialize session state for page selection
if "page" not in st.session_state:
    st.session_state.page = "Page 1"


# Sidebar Navigation
st.sidebar.title("""🧪 Predicting As Levels in U.S.Subwatersheds Using Machine Learning and Explainable AI""")
st.sidebar.markdown("---")  # Adds a horizontal separator

# Buttons for switching pages
if st.sidebar.button("🗺️ HeatMap", use_container_width=True):
    st.session_state.page = "Page 1"

if st.sidebar.button("✅ Model Evaluation", use_container_width=True):
    st.session_state.page = "Page 2"

if st.sidebar.button("📊️ Feature Analysing", use_container_width=True):
    st.session_state.page = "Page 3"

st.sidebar.markdown("---")  # Adds another separator for styling

justified_text = """
<div style="text-align: justify;">
This study applies machine learning to predict arsenic (As) levels in groundwater across U.S. subwatersheds.
SHAP analysis identified key factors influencing As contamination, including land cover, agriculture,
and atmospheric pollutants. The findings provide a data-driven approach for environmental management,
helping prioritize high-risk areas for monitoring and remediation.
</div>
"""
st.sidebar.markdown(justified_text, unsafe_allow_html=True)

# Display the selected page
if st.session_state.page == "Page 1":
    st.title("🗺 HeatMap")

elif st.session_state.page == "Page 2":
    st.title("✅ Classifier Performance Dashboard")
    st.write("Model Evaluation")

    st.divider()

elif st.session_state.page == "Page 3":
    st.title("📊️ Feature Analysing")
    st.write("Correlation and Variance Inflation of Features")
    st.divider()