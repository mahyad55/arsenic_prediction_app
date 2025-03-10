import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc, f1_score, precision_score, recall_score

# Simulated Data (Replace with your own model outputs)
y_true = np.random.randint(0, 2, 500)  # True labels
y_scores = np.random.rand(500)  # Predicted scores

# Compute Static Metrics
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# UI Layout
st.title("Classifier Performance Dashboard")

# **Section 1: Static Metrics**
st.subheader("Overall Model Performance (Independent of Threshold)")
col1, col2 = st.columns(2)
col1.metric("AUC-ROC", f"{roc_auc:.3f}")
col2.metric("Max F1 Score", f"{max(2 * (precision * recall) / (precision + recall)):.3f}")

# **Section 2: Threshold Selection**
st.subheader("Threshold Selection")
threshold_method = st.selectbox("Select Thresholding Method", ["F1-optimal", "KS-statistic", "Manual"])

# Compute Optimal Thresholds
best_f1_thr = thresholds[np.argmax(2 * (precision * recall) / (precision + recall))]
ks_thr = thresholds[np.argmax(tpr - fpr)]

# Handle Threshold Selection
if threshold_method == "F1-optimal":
    threshold = best_f1_thr
elif threshold_method == "KS-statistic":
    threshold = ks_thr
else:  # Manual Selection
    threshold = st.slider("Set Manual Threshold", 0.0, 1.0, 0.5, 0.01)

# **Section 3: Metrics Dependent on Threshold**
st.subheader("Metrics Based on Selected Threshold")
y_pred = (y_scores >= threshold).astype(int)

col3, col4, col5 = st.columns(3)
col3.metric("Precision", f"{precision_score(y_true, y_pred):.3f}")
col4.metric("Recall", f"{recall_score(y_true, y_pred):.3f}")
col5.metric("F1-score", f"{f1_score(y_true, y_pred):.3f}")

# **Section 4: Plots**
st.subheader("Performance Plots")

# Static Plots (ROC & Precision-Recall)
st.subheader("ROC Curve (Static)")
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig)

st.subheader("Precision-Recall Curve (Static)")
fig, ax = plt.subplots()
ax.plot(recall, precision, label="PR Curve")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend()
st.pyplot(fig)

# Dynamic Plot: Confusion Matrix
st.subheader("Confusion Matrix (Changes with Threshold)")
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots()
cax = ax.matshow(cm, cmap="coolwarm")
fig.colorbar(cax)
st.pyplot(fig)
