import pickle
import pandas as pd

from .model_evaluation import ModelEvaluation
## Import Data
my_path = r'.\data'
## Upload model
model = pickle.load(open(r'.\data\model.pkl', 'rb'))
optuna = pickle.load(open(r'.\data\optuna.pkl', 'rb'))

## Upload shap values
explainer = pickle.load(open(rf'{my_path}\explainer.pkl', 'rb'))
shap_values = pickle.load(open(rf'{my_path}\shap_values.pkl', 'rb'))
shap_values_model = pickle.load(open(rf'{my_path}\shap_values_lgbm.pkl', 'rb'))

## Upload shap values
explainer_full = pickle.load(open(rf'{my_path}\explainer_full.pkl', 'rb'))
shap_values_full = pickle.load(open(rf'{my_path}\shap_values_full.pkl', 'rb'))
shap_values_model_full = pickle.load(open(rf'{my_path}\shap_values_lgbm_full.pkl', 'rb'))

## Upload data
X_train = pd.read_csv(rf'{my_path}\X_train_final.csv', index_col='HUC_12')
X_test = pd.read_csv(rf'{my_path}\X_test_final.csv', index_col='HUC_12')
y_train = pd.read_csv(rf'{my_path}\y_train.csv', index_col='HUC_12')
y_test = pd.read_csv(rf'{my_path}\y_test.csv', index_col='HUC_12')
X_huc = pd.read_csv(rf'{my_path}\X_huc_final.csv', index_col='HUC_12')
X_huc2 = X_huc.reset_index(drop=False)

y_test_results = pd.read_csv(rf'{my_path}\y_test_results.csv', index_col='HUC_12')
y_train_results = pd.read_csv(rf'{my_path}\y_train_results.csv', index_col='HUC_12')
y_indent_results = pd.read_csv(rf'{my_path}\y_indent_results.csv', index_col='HUC_12')
y_huc_results = pd.read_csv(rf'{my_path}\y_huc_results.csv', index_col='HUC_12')
y_huc_results = y_huc_results.drop('pred', axis=1, errors='ignore')
evaluation: ModelEvaluation = ModelEvaluation()

y_train_pred_proba = model.predict_proba(X_train)[:,1]
y_test_pred_proba = model.predict_proba(X_test)[:,1]