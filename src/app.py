## Import Libraries
import numpy as np
import streamlit as st
import pickle
import shap
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils import ModelEvaluation, StreamLit
import plotly.express as px
from streamlit_plotly_events import plotly_events
from utils import StreamLit
from streamlit_extras.metric_cards import style_metric_cards
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

# Custom CSS to fix font color in dark mode
st.markdown("""
    <style>
        /* Fix metric card text color */
        div[data-testid="stMetric"] label {
            color: #bbb !important;  /* Light gray label */
        }

        div[data-testid="stMetric"] div {
            color: #1DB954 !important;  /* Green metric value */
            font-weight: bold;
        }

        /* Optional: Change background of metric card */
        div[data-testid="metric-container"] {
            background-color: #262730 !important;  /* Dark background */
            border-radius: 10px;
            padding: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Increase font size for the table */
        .stDataFrame div[data-testid="stVerticalBlock"] {
            font-size: 36px !important;
        }
        
        /* Increase font size for column headers */
        .stDataFrame th {
            font-size: 36px !important;
            font-weight: bold !important;
    </style>
""", unsafe_allow_html=True)

## Import data
evaluation = ModelEvaluation()
st_utils = StreamLit()
if "thr_method" not in st.session_state:
    st.session_state.thr_method = "auc"
st.session_state.best_thr = evaluation.best_threshold(y_test, y_test_pred_proba, method=st.session_state.thr_method)
y_train_pred = np.where(y_train_pred_proba > st.session_state.best_thr, 1, 0)
y_test_pred = np.where(y_test_pred_proba > st.session_state.best_thr, 1, 0)

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
    _, col1_t, col2_t, _ = st.columns([1,4,4,1])


    with col1_t:
        st.subheader('Target Distribution in Train Set')
        fig_target_gb = evaluation.target_distribution(
            y_train, show=False, target_name='target', Indeterminate=False,record_id='HUC_12'
        )
        st.pyplot(fig_target_gb)

    with col2_t:
        st.subheader('Target Distribution in Test Set')
        fig_target_gb = evaluation.target_distribution(
            y_test, show=False, target_name='target', Indeterminate=False,record_id='HUC_12'
        )
        st.pyplot(fig_target_gb)
    st.divider()

    # **Section 1: Static Metrics**
    st.subheader("Overall Model Performance (Independent of Threshold)")
    auc_col1, auc_col2, ks_col3, ks_col4 = st.columns(4)

    col1, col2, col3 = st.columns(3)



    roc_auc_scores_train, roc_auc_scores_test = evaluation.metric_auc(
        y_train, y_test, y_train_pred_proba, y_test_pred_proba
    )
    ks_stat_train, ks_stat_test = evaluation.metric_ks(
        y_train, y_test, y_train_pred_proba, y_test_pred_proba
    )


    auc_col1.metric("AUC-ROC (Train Set)", f"{roc_auc_scores_train:.2%}")
    auc_col2.metric("AUC-ROC (Test Set)", f"{roc_auc_scores_test:.2%}")

    ks_col3.metric("KS-Statistic (Train Set)", f"{ks_stat_train:.2%}")
    ks_col4.metric("KS-Statistic (Test Set)", f"{ks_stat_test:.2%}")
    # Apply the modern style
    style_metric_cards()

    @st.cache_data
    def generate_evaluation_plots(y_test, y_test_pred, y_train, y_train_pred):
        fig_clf_report = evaluation.classification_report_overall1x2(
            y_train, y_train_pred, y_test, y_test_pred,
            target_names=['good', 'bad']
        )
        fig_clf_cm = evaluation.confusion_plot1x2(
            y_train=y_train, y_train_pred=y_train_pred,
            y_test=y_test, y_test_pred=y_test_pred
        )
        return fig_clf_report, fig_clf_cm


    # Generate updated figures
    fig_clf_roc = evaluation.roc_auc_plot1x2(
        y_train=y_train, y_train_pred_proba=y_train_pred_proba,
        y_test=y_test, y_test_pred_proba=y_test_pred_proba
    )

    fig_ks_train = evaluation.ks_plot(y_train, y_train_pred_proba, show=False)
    fig_ks_test = evaluation.ks_plot(y_test, y_test_pred_proba, show=False)

    _, col2_roc, _ = st.columns([1, 4, 1])
    with col2_roc:
        st.subheader('Receiver operating characteristic')
        st.pyplot(fig_clf_roc)
    st.divider()

    col2_ks_train, col2_ks_test = st.columns([1,1])
    with col2_ks_train:
        st.subheader('Kolmogorov-Smirnov (KS) Plot for Train Set')
        st.pyplot(fig_ks_train)

    with col2_ks_test:
        st.subheader('Kolmogorov-Smirnov (KS) Plot for Test Set')
        st.pyplot(fig_ks_test)
    st.divider()

    # **Section 2: Threshold Selection**
    st.subheader("Metrics Based on Selected Threshold")
    st.subheader("Threshold Selection")
    threshold_method = st.selectbox("Select Thresholding Method", ["AUC-ROC", "F1-optimal", "KS-statistic", "Manual"])

    # Handle Threshold Selection
    if threshold_method == "AUC-ROC":
        st.session_state.best_thr = evaluation.best_threshold(y_test, y_test_pred_proba, method='auc')
        y_train_pred = np.where(y_train_pred_proba > st.session_state.best_thr, 1, 0)
        y_test_pred = np.where(y_test_pred_proba > st.session_state.best_thr, 1, 0)

    elif threshold_method == "F1-optimal":
        st.session_state.best_thr = evaluation.best_threshold(y_test, y_test_pred_proba, method='f1')
        y_train_pred = np.where(y_train_pred_proba > st.session_state.best_thr, 1, 0)
        y_test_pred = np.where(y_test_pred_proba > st.session_state.best_thr, 1, 0)

    elif threshold_method == "KS-statistic":
        st.session_state.best_thr = evaluation.best_threshold(y_test, y_test_pred_proba, method='ks')
        y_train_pred = np.where(y_train_pred_proba > st.session_state.best_thr, 1, 0)
        y_test_pred = np.where(y_test_pred_proba > st.session_state.best_thr, 1, 0)
    else:  # Manual Selection
        st.session_state.best_thr = st.slider("Set Manual Threshold", 0.0, 1.0, 0.5, 0.01)
        y_train_pred = np.where(y_train_pred_proba > st.session_state.best_thr, 1, 0)
        y_test_pred = np.where(y_test_pred_proba > st.session_state.best_thr, 1, 0)

    fig_clf_report, fig_clf_cm = generate_evaluation_plots(y_test, y_test_pred, y_train, y_train_pred)

    # **Section 3: Metrics Dependent on Threshold**
    st.write(f'Best Threshold based on "{threshold_method}": "{st.session_state.best_thr:.2f}"')

    clf_report_train = classification_report(y_train, y_train_pred, target_names=['good', 'bad'], output_dict=True)
    df_clf_report_train = pd.DataFrame(clf_report_train).transpose()

    clf_report_test = classification_report(y_test, y_test_pred, target_names=['good', 'bad'], output_dict=True)
    df_clf_report_test = pd.DataFrame(clf_report_test).transpose()
    st.divider()

    _,acc_train, _, acc_test, _ = st.columns(5)
    acc_train.metric("Accuracy of Train Set", f"{df_clf_report_train.loc['accuracy', 'precision']:.2%}")
    acc_test.metric("Accuracy of Test Set", f"{df_clf_report_test.loc['accuracy', 'precision']:.2%}")
    st.divider()

    df_clf_report_train = df_clf_report_train.drop(['accuracy'], axis=0)
    df_clf_report_test = df_clf_report_test.drop(['accuracy'], axis=0)

    styled_df_clf_report_train = df_clf_report_train.style \
        .applymap(lambda val: st_utils.heatmap_style(df_clf_report_train, val), subset=['precision', 'recall', 'f1-score']) \
        .format({'precision': '{:.1%}', 'recall': '{:.1%}', 'f1-score': '{:.1%}', 'support': '{:.0f}'}) \
        .set_properties(**{'text-align': 'center'}) \
        .set_table_styles([{
            'selector': 'th',
            'props': [('font-weight', 'bold'), ('text-align', 'center')]
    }])

    styled_df_clf_report_test = df_clf_report_test.style \
        .applymap(lambda val: st_utils.heatmap_style(df_clf_report_test, val), subset=['precision', 'recall', 'f1-score']) \
        .format({'precision': '{:.1%}', 'recall': '{:.1%}', 'f1-score': '{:.1%}', 'support': '{:.0f}'}) \
        .set_properties(**{'text-align': 'center'}) \
        .set_table_styles([{
            'selector': 'th',
            'props': [('font-weight', 'bold'), ('text-align', 'center')]
    }])

    st.write(f'Classification Results for Train Set')
    st.dataframe(styled_df_clf_report_train, use_container_width=True)
    st.write(f'Classification Results for Test Set')
    st.dataframe(styled_df_clf_report_test, use_container_width=True)
    st.divider()


    # Add your seaborn plots here
    st.subheader('Confusion Matrix for Train and Test set')
    st.pyplot(fig_clf_cm)
    st.divider()

elif st.session_state.page == "Page 3":
    st.title("📊️ Feature Analysing")
    st.write("Correlation and Variance Inflation of Features")
    st.divider()