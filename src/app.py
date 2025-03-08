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
    model, optuna_model, explainer, shap_values,  shap_values_model,
    explainer_full, shap_values_full,  shap_values_model_full,
    X_train, X_test, y_train, y_test, X_huc, X_huc2,
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

