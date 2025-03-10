import os

import pickle
import pandas as pd
import shap

my_path = r".\data"

## Upload model
model = pickle.load(open(rf'{my_path}\model.pkl', 'rb'))
optuna_model = pickle.load(open(rf'{my_path}\optuna_model.pkl', 'rb'))

## Upload shap values
explainer = shap.TreeExplainer(model)

## Upload data
X_train = pd.read_csv(rf'{my_path}\X_train_final.csv', index_col='HUC_12')
X_test = pd.read_csv(rf'{my_path}\X_test_final.csv', index_col='HUC_12')
y_train = pd.read_csv(rf'{my_path}\y_train.csv', index_col='HUC_12')
y_test = pd.read_csv(rf'{my_path}\y_test.csv', index_col='HUC_12')
X_huc = pd.read_csv(rf'{my_path}\X_huc_final.csv', index_col='HUC_12')

y_test_results = pd.read_csv(rf'{my_path}\y_test_results.csv', index_col='HUC_12')
y_train_results = pd.read_csv(rf'{my_path}\y_train_results.csv', index_col='HUC_12')
y_indent_results = pd.read_csv(rf'{my_path}\y_indent_results.csv', index_col='HUC_12')
y_huc_results = pd.read_csv(rf'{my_path}\y_huc_results.csv', index_col='HUC_12')
y_huc_results = y_huc_results.drop('pred', axis=1, errors='ignore')

y_train_pred_proba = model.predict_proba(X_train)[:,1]
y_test_pred_proba = model.predict_proba(X_test)[:,1]

