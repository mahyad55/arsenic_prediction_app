# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 12:06:26 2025

@author: MahYad
"""
###################### Builin Packages ######################
import warnings
warnings.filterwarnings("ignore")
import random
import sys
import os
import re
import json
import math
import copy
import inspect

import traceback
import logging
import itertools
from dateutil.relativedelta import relativedelta
import hashlib

####################### common packages #####################
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm  
from joblib import Parallel, delayed

################## Visualization packages ###################
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as po
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots

from sklearn.calibration import CalibrationDisplay
from matplotlib.gridspec import GridSpec
# calibrate model on validation data
from sklearn.calibration import CalibratedClassifierCV
from matplotlib import pyplot
from sklearn.calibration import calibration_curve
import matplotlib.cm as cm
import matplotlib.colors as mcolors
####################### Text Packages #####################
# from persiantools.jdatetime import JalaliDate
# from arabic_reshaper import ArabicReshaper
# from bidi.algorithm import get_display
# from persiantools import digits, characters

######################## Statistical ########################
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

######################### Metrics ###########################
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score, precision_score,
    fbeta_score, f1_score, cohen_kappa_score, matthews_corrcoef, log_loss,
    brier_score_loss, recall_score
)

# from scikitplot.helpers import binary_ks_curve

######################## Preprocessing ########################
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import (
    RobustScaler, OneHotEncoder, StandardScaler, OrdinalEncoder, Normalizer, MinMaxScaler
)
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import make_pipeline, Pipeline

######################### Split Data ###########################
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_validate

######################### Missing Values #######################
# import missingno as msno


############################# Models ############################
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.experimental import enable_hist_gradient_boosting
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.model_selection import StratifiedKFold
# from catboost import CatBoostClassifier
# from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

######################### Hyperparameters #########################
# import optuna
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

########################## User Packages ##########################
# from utils import FeatureSelection

######################### Jupyter Notebook ########################
from sklearn import set_config






# %%
class ModelEvaluation:
        
    def _ks_threshold(self, y_true, pred_probs):
        # Create a dataframe to hold predictions and actual labels
        # data = pd.DataFrame({'y_true': y_true.values, 'pred_probs': pred_probs})
        data = pd.DataFrame(columns=['y_true','pred_probs'])
        data['y_true'] = y_true
        data['pred_probs'] = pred_probs
        # Sort the data by predicted probabilities in descending order
        data = data.sort_values(by='pred_probs', ascending=False)
        
        # Calculate cumulative percentage of positives and negatives
        data['cum_pos_pct'] = data['y_true'].cumsum() / data['y_true'].sum()  # CDF of Positives
        data['cum_neg_pct'] = (1 - data['y_true']).cumsum() / (1 - data['y_true']).sum()  # CDF of Negatives
        
        # Calculate the KS statistic
        data['ks_stat'] = np.abs(data['cum_pos_pct'] - data['cum_neg_pct'])
        
        # Find the maximum KS statistic and corresponding threshold
        # ks_max = data['ks_stat'].max()
        threshold_ks = data.loc[data['ks_stat'].idxmax(), 'pred_probs']
        
        return threshold_ks
    
    def _f1_threshold(self, y_true, pred_probs):
        precision, recall, thresholds = precision_recall_curve(y_true, pred_probs)
    
        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        threshold_f1 = thresholds[ix]
        return threshold_f1
    
    def _auc_threshold(self, y_true, pred_probs):
        # # calculate pr curve
        fpr, tpr, thresholds = roc_curve(y_true, pred_probs)
    
    
        # # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1 - fpr))
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)
        threshold_auc = thresholds[ix]
        return threshold_auc
    
    def best_threshold(self, y_true, pred_probs, method='f1'):
        if method=='ks':
            return self._ks_threshold(y_true, pred_probs)
        elif method=='f1':
            return self._f1_threshold(y_true, pred_probs)
        elif method=='auc':
            return self._auc_threshold(y_true, pred_probs)
    
    # Create annotations for the heatmap
    def _create_annotations(self, cm, cm_perc):
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = np.sum(cm, axis=1)[i]
                    annot[i, j] = f'{p:.1f}%\n{c}/{s}'
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = f'{p:.1f}%\n{c}'
        return annot

    def confusion_plot1x2(self, y_train, y_train_pred, y_test, y_test_pred):

        # Create confusion matrices
        cm_train = confusion_matrix(y_train, y_train_pred)
        cm_test = confusion_matrix(y_test, y_test_pred)

        # Normalize confusion matrices (percentages)
        cm_train_norm = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis] * 100
        cm_test_norm = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis] * 100

        # Create annotation matrices
        annot_train = self._create_annotations(cm_train, cm_train_norm)
        annot_test = self._create_annotations(cm_test, cm_test_norm)

        # Labels
        labels = ['Negative', 'Positive']

        # Create diagonal and off-diagonal masks
        def create_diagonal_mask(cm):
            mask = np.zeros_like(cm, dtype=bool)
            np.fill_diagonal(mask, True)
            return mask

        # Create masks for diagonal and off-diagonal cells
        mask_train_diag = create_diagonal_mask(cm_train)
        mask_train_off_diag = ~mask_train_diag

        mask_test_diag = create_diagonal_mask(cm_test)
        mask_test_off_diag = ~mask_test_diag

        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Custom colormaps: Green for diagonal (true predictions), Red for off-diagonal (errors)
        cmap_green = sns.color_palette(["#00A170"], as_cmap=True)  # Green for correct
        cmap_red = sns.color_palette(["#9B2335"], as_cmap=True)  # Red for errors

        # Plot diagonal (green) for train
        sns.heatmap(cm_train_norm, mask=mask_train_off_diag, annot=annot_train, fmt='', annot_kws={'size': 14},
                    cmap=cmap_green, cbar=False, vmin=0, vmax=100,
                    yticklabels=labels, xticklabels=labels, ax=ax1)

        # Plot off-diagonal (red) for train
        sns.heatmap(cm_train_norm, mask=mask_train_diag, annot=annot_train, fmt='', annot_kws={'size': 14},
                    cmap=cmap_red, cbar=False, yticklabels=labels, xticklabels=labels, ax=ax1)

        # Plot diagonal (green) for test
        sns.heatmap(cm_test_norm, mask=mask_test_off_diag, annot=annot_test, fmt='', annot_kws={'size': 14},
                    cmap=cmap_green, cbar=False, vmin=0, vmax=100,
                    yticklabels=labels, xticklabels=labels, ax=ax2)

        # Plot off-diagonal (red) for test
        sns.heatmap(cm_test_norm, mask=mask_test_diag, annot=annot_test, fmt='', annot_kws={'size': 14},
                    cmap=cmap_red, cbar=False, yticklabels=labels, xticklabels=labels, ax=ax2)

        # Add additional text annotations
        additional_texts = ['(True Negative)', '(True Positive)', '(False Positive)', '(False Negative)']

        for text_elt, additional_text in zip(ax1.texts, additional_texts):
            ax1.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
                     ha='center', va='top', size=14)

        for text_elt, additional_text in zip(ax2.texts, additional_texts):
            ax2.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
                     ha='center', va='top', size=14)

        # Set titles and labels
        ax1.set_title('Train Confusion Matrix', size=12)
        ax1.tick_params(labelsize=12, length=0)
        ax1.set_xlabel('Predicted Values', size=12)
        ax1.set_ylabel('Actual Values', size=12)

        ax2.set_title('Test Confusion Matrix', size=12)
        ax2.tick_params(labelsize=12, length=0)
        ax2.set_xlabel('Predicted Values', size=12)
        ax2.set_ylabel('Actual Values', size=12)

        plt.tight_layout()
        plt.show()
        return fig
    
    def roc_auc_plot1x2(self, y_train, y_train_pred_proba, y_test, y_test_pred_proba):
        # Calculate ROC-AUC metrics
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
        roc_auc_train = auc(fpr_train, tpr_train)

        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)
        roc_auc_test = auc(fpr_test, tpr_test)

        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Set titles
        fig.suptitle('ROC-AUC Plots', fontsize=16)

        # Plot ROC-AUC for train in the first row and first column
        axes[0].plot(
            fpr_train,
            tpr_train,
            label=f'Train ROC-AUC: {roc_auc_train:.2f}',
            color='blue'
        )
        axes[0].fill_between(fpr_train, 0, tpr_train, color='blue', alpha=0.3)
        axes[0].set_title(f'Train ROC-AUC: {roc_auc_train:.2%}')
        axes[0].set(xlabel='False Positive Rate', ylabel='True Positive Rate')
        # Add diagonal dashed line
        axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)

        # Plot ROC-AUC for test in the first row and second column
        axes[1].plot(
            fpr_test,
            tpr_test,
            label=f'Test ROC-AUC: {roc_auc_test:.2f}',
            color='orange'
        )
        axes[1].fill_between(fpr_test, 0, tpr_test, color='orange', alpha=0.3)
        axes[1].set_title(f'Test ROC-AUC: {roc_auc_test:.2%}')
        axes[1].set(xlabel='False Positive Rate', ylabel='True Positive Rate')
        # Add diagonal dashed line
        axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Show the plot
        plt.show()  # Assume you have y_train, y_train_pred, y_test, and
        return fig
    
    def classification_report_overall1x2(self, y_train, y_train_pred, y_test, y_test_pred,
        target_names=['good', 'bad']):
        # Prepare train and test data for heatmap
        heatmap_data_train, annot_text_train = self._prepare_heatmap_overall(
            y_train, y_train_pred, target_names=target_names, percentage=True
        )
        heatmap_data_test, annot_text_test = self._prepare_heatmap_overall(
            y_test, y_test_pred, target_names=target_names, percentage=True
        )

        # Create 1x2 subplot
        fig, axes = plt.subplots(1, 2, figsize=(10, 2))
        fig.suptitle('Classification Results', fontsize=12)

        # Plot the heatmap for the train set
        index_name = ['macro avg', 'weighted avg']
        sns.heatmap(
            heatmap_data_train, annot=annot_text_train.values, fmt='',
            annot_kws={'size': 12}, cmap="rocket", cbar=False,
            vmin=0, vmax=1, ax=axes[0], yticklabels=index_name,
            xticklabels=heatmap_data_train.columns
        )
        axes[0].set_title("Train Classification Report Heatmap")

        # Plot the heatmap for the test set
        sns.heatmap(
            heatmap_data_test, annot=annot_text_test.values, fmt='',
            annot_kws={'size': 12}, cmap="rocket", cbar=False,
            vmin=0, vmax=1, ax=axes[1], yticklabels=index_name,
            xticklabels=heatmap_data_test.columns
        )
        axes[1].set_title("Test Classification Report Heatmap")

        # Adjust layout
        # Increase space between the two subplots
        plt.subplots_adjust(hspace=0.9, left=0.1, right=0.15, wspace=0.05) 
        plt.tight_layout()
        plt.show()
        return fig

    def confusion_plot(
            self, y, y_pred, dataset='Test',save=False, title=None,
            export_name=None, save_path=r'.\results'
        ):
    
        # Create confusion matrices
        cm = confusion_matrix(y, y_pred)
    
        # Normalize confusion matrices (percentages)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
        # Create annotation matrices
        annot = self._create_annotations(cm, cm_norm)
    
        # Labels
        labels = ['Negative', 'Positive']
    
        # Create diagonal and off-diagonal masks
        def create_diagonal_mask(cm):
            mask = np.zeros_like(cm, dtype=bool)
            np.fill_diagonal(mask, True)
            return mask
    
        # Create masks for diagonal and off-diagonal cells
        mask_diag = create_diagonal_mask(cm)
        mask_off_diag = ~mask_diag
    
    
        # Create the figure
        fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))
    
        # Custom colormaps: Green for diagonal (true predictions), Red for off-diagonal (errors)
        cmap_green = sns.color_palette(["#00A170"], as_cmap=True)  # Green for correct
        cmap_red = sns.color_palette(["#9B2335"], as_cmap=True)    # Red for errors
    
        # Plot diagonal (green) for train
        sns.heatmap(cm_norm, mask=mask_off_diag, annot=annot, fmt='', annot_kws={'size': 14},
                    cmap=cmap_green, cbar=False, vmin=0, vmax=100,
                    yticklabels=labels, xticklabels=labels, ax=ax1)
    
        # Plot off-diagonal (red) for train
        sns.heatmap(
            cm_norm, mask=mask_diag, annot=annot, fmt='',
            annot_kws={'size': 14}, cmap=cmap_red, cbar=False,
            yticklabels=labels, xticklabels=labels, ax=ax1
        )
    
        # Add additional text annotations
        additional_texts = ['(True Negative)', '(True Positive)', '(False Positive)', '(False Negative)']
    
        for text_elt, additional_text in zip(ax1.texts, additional_texts):
            ax1.text(
                *text_elt.get_position(),
                '\n' + additional_text,
                color=text_elt.get_color(),
                ha='center', va='top', size=14
            )
    
        # Set titles and labels
        if title is None:
            title = f'Confusion Matrix for {dataset} set'
        
        ax1.set_title(title, size=12)
        ax1.tick_params(labelsize=12, length=0)
        ax1.set_xlabel('Predicted Values', size=12)
        ax1.set_ylabel('Actual Values', size=12)
    
        plt.tight_layout()
        
        # Save the plot (optional)
        if save:
            if export_name==None:
                export_name = title
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                try:
                    plt.savefig(rf'.\{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
                except:
                    plt.savefig(rf'{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
                    
        plt.show()
        
    
    def roc_precision_recall_plot(
            self, y, y_pred_proba, dataset='Test',
            save=False, save_path=r'.\results', export_name=None
        ):
    
        # Calculate ROC-AUC metrics
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc_scores = auc(fpr, tpr)
    
        # Calculate precision-recall metrics
        precision_scores, recall_scores, _ = precision_recall_curve(y, y_pred_proba)
    
        # Calculate Average Precision (AP) scores
        # ap = average_precision_score(y, y_pred_proba)
        ap = auc(recall_scores, precision_scores)
    
        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
        # Set titles
        fig.suptitle(f'ROC-Precision-Recall Plots for {dataset} set', fontsize=16)
    
        # Plot ROC-AUC for train in the first row and first column
        axes[0].plot(
            fpr,
            tpr,
            label=f'ROC-AUC: {roc_auc_scores:.2f}',
            color='blue'
        )
        axes[0].fill_between(fpr, 0, tpr, color='blue', alpha=0.3)
        axes[0].set_title(f'ROC-AUC: {roc_auc_scores:.2%}')
        axes[0].set(xlabel='False Positive Rate', ylabel='True Positive Rate')
        # Add diagonal dashed line
        axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
    
        # Plot Precision-Recall for train in the second row and first column
        axes[1].plot(
            recall_scores,
            precision_scores,
            label=f'Train Precision-Recall: AP={ap:.2f}',
            color='green'
        )
        axes[1].fill_between(recall_scores, 0, precision_scores, color='green', alpha=0.3)
        
        axes[1].set_title(f'Precision-Recall: AP={ap:.2%}')
        axes[1].set(xlabel='Recall', ylabel='Precision')
        # Add diagonal dashed line
        axes[1].plot([0, 1], [1, 0], linestyle='--', color='gray', linewidth=2)
    
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        

        # Save the plot (optional)
        if save:
            if export_name==None:
                export_name = f'ROC-Precision-Recall Plots for {dataset} set'
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                try:
                    plt.savefig(rf'.\{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
                except:
                    plt.savefig(rf'{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
                    
        plt.show()# Assume you have y_train, y_train_pred, y_test, and y_test_pred
    
    # Function to prepare data for heatmap
    def _prepare_heatmap_data(self, y_true, y_pred, target_names):
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        
        # Convert to a DataFrame and reset index
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.reset_index().rename(columns={'index': 'class'})
        
        # Create heatmap for numerical values (drop accuracy, macro avg, weighted avg)
        heatmap_data = report_df.drop(columns=["class"]).iloc[:-3]
        
        # Format the annotations
        annot_text = heatmap_data.copy()
        annot_text['precision'] = heatmap_data['precision'].apply(lambda x: f'{x:.2f}')
        annot_text['recall'] = heatmap_data['recall'].apply(lambda x: f'{x:.2f}')
        annot_text['f1-score'] = heatmap_data['f1-score'].apply(lambda x: f'{x:.2f}')
        annot_text['support'] = heatmap_data['support'].apply(lambda x: f'{x:,.0f}')
        
        return heatmap_data, annot_text
    
    # Function to prepare data for heatmap
    def _prepare_heatmap_overall(self, y_true, y_pred, target_names, percentage=False):
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        
        # Convert to a DataFrame and reset index
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.reset_index().rename(columns={'index': 'class'})
        
        # Create heatmap for numerical values (drop accuracy, macro avg, weighted avg)
        heatmap_data = report_df.drop(columns=["class"]).iloc[-2:]
        
        # Format the annotations
        annot_text = heatmap_data.copy()
        if percentage:
            annot_text['precision'] = heatmap_data['precision'].apply(lambda x: f'{x:.1%}')
            annot_text['recall'] = heatmap_data['recall'].apply(lambda x: f'{x:.1%}')
            annot_text['f1-score'] = heatmap_data['f1-score'].apply(lambda x: f'{x:.1%}')
        else:
            annot_text['precision'] = heatmap_data['precision'].apply(lambda x: f'{x:.2f}')
            annot_text['recall'] = heatmap_data['recall'].apply(lambda x: f'{x:.2f}')
            annot_text['f1-score'] = heatmap_data['f1-score'].apply(lambda x: f'{x:.2f}')
            
        annot_text['support'] = heatmap_data['support'].apply(lambda x: f'{x:,.0f}')
    
        return heatmap_data, annot_text
    
    
    def classification_report_plot(
            self, y, y_pred, target_names=['good', 'bad'], dataset='Test',
            export_name=None,save=False, save_path=r'.\results'):
        # Prepare train and test data for heatmap
        heatmap_data, annot_text = self._prepare_heatmap_data(y, y_pred, target_names=target_names)
        heatmap_data_overall, annot_text_overall = self._prepare_heatmap_overall(
            y, y_pred, target_names=target_names
        )
    
        # Create 1x2 subplot
        fig, axes = plt.subplots(1, 2, figsize=(10, 2))
        fig.suptitle(f'Classification Results for {dataset} set', fontsize=12)
    
        # Plot the heatmap for the train set
        sns.heatmap(heatmap_data, annot=annot_text.values, fmt='', annot_kws={'size': 12},
                    cmap="rocket", cbar=False, vmin=0, vmax=1, ax=axes[0],
                    yticklabels=target_names, xticklabels=heatmap_data.columns)
        axes[0].set_title("Classification Report Heatmap")
    
    
        index_name = ['macro avg', 'weighted avg']
        # Plot the heatmap for the train set
        sns.heatmap(
            heatmap_data_overall, annot=annot_text_overall.values,
            fmt='', annot_kws={'size': 12}, cmap="rocket", cbar=False,
            vmin=0, vmax=1, ax=axes[1], yticklabels=index_name,
            xticklabels=heatmap_data_overall.columns
        )
        axes[1].set_title("Overall Classification Report Heatmap")
    
        # Adjust layout
        # Increase space between the two subplots
        plt.subplots_adjust(hspace=0.9, left=0.1, right=0.15, wspace=0.05)  
        plt.tight_layout()
        
        # Save the plot (optional)
        if save:
            if export_name==None:
                export_name = 'Overall Classification Report Heatmap'
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                try:
                    plt.savefig(rf'.\{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
                except:
                    plt.savefig(rf'{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
                    
        plt.show()
    
    def ks_plot(self, y, y_pred_proba, show=True, save=False, save_path=r'.\results', export_name=None):
        # Create a DataFrame with true labels and predicted probabilities
        data_ks = pd.DataFrame()
        data_ks['true_label'] = y
        data_ks['predicted_proba'] = y_pred_proba
        
        # Sort the probabilities
        data_ks = data_ks.sort_values(by='predicted_proba')
        data_ks = data_ks.reset_index(drop=True)
        
        # Separate the probabilities for each class (0 and 1)
        cdf_0 = np.cumsum(data_ks['true_label'] == 0) / sum(data_ks['true_label'] == 0)
        cdf_1 = np.cumsum(data_ks['true_label'] == 1) / sum(data_ks['true_label'] == 1)
        
        # Compute the KS statistic (the maximum difference between the two CDFs)
        ks_stat = np.max(np.abs(cdf_0 - cdf_1))
        ks_location = np.argmax(np.abs(cdf_0 - cdf_1))
        
        # Create the plot
        fig = plt.figure(figsize=(8, 4), dpi=200)
        
        # Plot the CDFs
        plt.plot(data_ks['predicted_proba'], cdf_0, label='Negative Class (0)', color='blue', lw=2)
        plt.plot(data_ks['predicted_proba'], cdf_1, label='Positive Class (1)', color='orange', lw=2)
        
        # Plot the maximum KS statistic
        plt.vlines(data_ks['predicted_proba'].values[ks_location], ymin=cdf_0[ks_location], ymax=cdf_1[ks_location], 
                   color='red', linestyle='--', label=f'KS Statistic = {ks_stat:.3f}')
        
        # Add labels and title
        plt.title('Kolmogorov-Smirnov (KS) Plot', fontsize=16)
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Cumulative Distribution', fontsize=12)
        
        # Add a grid and legend
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        # Show the plot
        plt.tight_layout()
        
        # Save the plot (optional)
        if save:
            if export_name==None:
                export_name = 'Kolmogorov-Smirnov (KS) Plot'
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                try:
                    plt.savefig(rf'.\{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
                except:
                    plt.savefig(rf'{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig
    
    def calibrationCurve(self, 
            y_train, y_train_pred_proba, y_test, y_test_pred_proba,
            num_bins=10, strategy='uniform',annot_text=True,
            save=False, save_path=r'\results', export_name=None
        ):
        
        fig = plt.figure(figsize=(12, 9), dpi=200)
        gs = GridSpec(2, 2)
        colors = plt.get_cmap("Dark2")
        ax_calibration_curve = fig.add_subplot(gs[0, :2])
        calibration_displays = {}
    
        display_train = CalibrationDisplay.from_predictions(
            y_train, y_train_pred_proba, strategy=strategy,n_bins=num_bins,
            color=colors(0), ax=ax_calibration_curve,name='Train'
        )
        display_test = CalibrationDisplay.from_predictions(
            y_test, y_test_pred_proba, strategy=strategy,n_bins=num_bins,
            color=colors(1), ax=ax_calibration_curve,name='Test'
        )
    
        calibration_displays['Train'] = display_train
        calibration_displays['Test'] = display_test
    
        ax_calibration_curve.grid()
        ax_calibration_curve.set_title("Probability Calibration Curves")
    
        if strategy=='uniform':
    
            # Add histogram
            ax1 = fig.add_subplot(gs[1, 0])
            # sns.displot(x=calibration_displays['Train'].y_prob, ax=ax1)
            arr1 = ax1.hist(
                calibration_displays['Train'].y_prob,
                range=(0, 1),
                bins=num_bins,
                label='Train',
                color=colors(0),
                edgecolor='black',
                linewidth=1.2
            )
            
            if annot_text:
                for i in range(num_bins):
                    if arr1[0][i] > 0:
                        plt.text(
                            arr1[1][i],
                            arr1[0][i] + 1,
                            str(int(arr1[0][i])),
                            fontsize=10
                        )
    
            ax1.set(title='Train', xlabel="Mean predicted probability ", ylabel="Count")
    
            ax2 = fig.add_subplot(gs[1, 1])
            # sns.displot(x=calibration_displays['Test'].y_prob, ax=ax2)
            arr2 = ax2.hist(
                calibration_displays['Test'].y_prob,
                range=(0, 1),
                bins=num_bins,
                label='Test',
                color=colors(1),
                edgecolor='black',
                linewidth=1.2
            )
    
            if annot_text:
                for i in range(num_bins):
                    if arr2[0][i] > 0:
                        plt.text(
                            arr2[1][i],
                            arr2[0][i] + 1,
                            str(int(arr2[0][i])),
                            fontsize=10
                        )
    
            ax2.set(title='Test', xlabel="Mean predicted probability", ylabel="Count")
    
            plt.tight_layout()
            # Save the plot (optional)
            if save:
                if export_name==None:
                    export_name = "Probability Calibration Curves"
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    try:
                        plt.savefig(rf'.\{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
                    except:
                        plt.savefig(rf'{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
        elif strategy=='quantile':
            interval_train = pd.qcut(display_train.y_prob, q=num_bins)
            qcut_result_train = pd.DataFrame(interval_train.value_counts()).reset_index()
            qcut_result_train.columns = ['Interval', 'Count']
            qcut_result_train['Interval'] = qcut_result_train['Interval'].astype(str)
    
            interval_test = pd.qcut(display_test.y_prob, q=num_bins)
            qcut_result_test = pd.DataFrame(interval_test.value_counts()).reset_index()
            qcut_result_test.columns = ['Interval', 'Count']
            qcut_result_test['Interval'] = qcut_result_test['Interval'].astype(str)
    
            # # Add histogram
            ax1 = fig.add_subplot(gs[1, 0])
            arr1 = ax1.bar(
                x=qcut_result_train['Interval'],
                height=qcut_result_train['Count'],
                label='Train',
                color=colors(0),
                edgecolor='black',
                linewidth=1.2
            )
            if annot_text:
                for i in range(num_bins):
                    if qcut_result_train['Count'].iloc[i] > 0:
                        plt.text(
                            qcut_result_train['Interval'].iloc[i],
                            qcut_result_train['Count'].iloc[i] + 1,
                            str(int(qcut_result_train['Count'].iloc[i])),
                            fontsize=10
                        )
            ax1.set(title='Train', xlabel="Mean predicted probability ", ylabel="Count")
            plt.xticks(rotation=45)
    
            ax2 = fig.add_subplot(gs[1, 1])
            arr2 = ax2.bar(
                x=qcut_result_test['Interval'],
                height=qcut_result_test['Count'],
                label='Test',
                color=colors(0),
                edgecolor='black',
                linewidth=1.2
            )
            
            if annot_text:
                for i in range(num_bins):
                    if qcut_result_test['Count'].iloc[i] > 0:
                        plt.text(
                            qcut_result_test['Interval'].iloc[i],
                            qcut_result_test['Count'].iloc[i] + 1,
                            str(int(qcut_result_test['Count'].iloc[i])),
                            fontsize=10
                        )
            ax2.set(title='Test', xlabel="Mean predicted probability", ylabel="Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the plot (optional)
            if save:
                if export_name==None:
                    export_name = "Probability Calibration Curves"
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    try:
                        plt.savefig(rf'.\{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
                    except:
                        plt.savefig(rf'{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
    def target_distribution(self, 
            df_target, target_name='target', show=True,
            record_id='CustomerId', Indeterminate=False,
            save=False, save_path=r'.\results', export_name=None
        ):
        
        if not isinstance(df_target, pd.DataFrame):
            df_target = pd.DataFrame(df_target, index=df_target.index)
            df_target = df_target.reset_index()
            df_target = pd.DataFrame(df_target, columns=[target_name, record_id])
        else:
            if df_target.shape[1]==1:
                df_target = pd.DataFrame(df_target, index=df_target.index)
                df_target = df_target.reset_index()
                df_target = pd.DataFrame(df_target, columns=[target_name, record_id])
                
        
        if not Indeterminate:
            Indeterminate_list = ['Indeterminate', 'indeterminate', 0.5, '0.5']
            df_target = df_target.loc[~df_target[target_name].isin(Indeterminate_list)]

        num_target_labels = df_target.groupby(target_name)[record_id].nunique()
        num_target_labels = pd.DataFrame(num_target_labels).reset_index()
        num_target_labels.columns=['target', 'count']
        count_all = num_target_labels['count'].sum()
        num_target_labels['percentage'] = num_target_labels['count']/count_all
        
        
        
        if Indeterminate:
            target_map = {0:'Good', 1:'Bad', 0.5:'Indeterminate'}
            target_map2 = {'good':'Good', 'bad':'Bad', 'indeterminate':'Indeterminate'}
            custom_dict = {'Good': 0,'Indeterminate':0.5, 'Bad': 1} 
            
            # Set the color dictionary and hatch patterns
            color_dict = {'Good': 'green','Indeterminate': 'yellow','Bad': 'red'}
            hatch_patterns = {'Good': '//','Indeterminate': 'xx','Bad': '\\'}
            order = ['Good', 'Indeterminate', 'Bad']
        
        else:
            
            target_map = {0:'Good', 1:'Bad'}
            target_map2 = {'good':'Good', 'bad':'Bad'}
            custom_dict = {'Good': 0, 'Bad': 1} 
            
            # Set the color dictionary and hatch patterns
            color_dict = {'Good': 'green','Bad': 'red'}
            hatch_patterns = {'Good': '//','Bad': '\\'}
            order = ['Good', 'Bad']
            
        if 'good' in num_target_labels['target'].values:
            num_target_labels['target'] = num_target_labels['target'].map(target_map2)
            
        elif (0 in num_target_labels['target'].values) | ('0' in num_target_labels['target'].values):
            num_target_labels['target'] = num_target_labels['target'].map(target_map)
            
        else:
            pass
            
        # Set the style for a more professional look
        sns.set(style="whitegrid")
        num_target_labels = num_target_labels.sort_values(
            by=['target'],
            key=lambda x: x.map(custom_dict),
            ascending=True
        )
        
        # Create the bar plot
        fig = plt.figure(figsize=(6, 6))    
        bars = sns.barplot(x='target', y='percentage', data=num_target_labels, 
                           palette=color_dict, edgecolor='black', order=order)
        
        # add title and labels with larger font sizes
        plt.title('Percentage of each Target Label', fontsize=16, family='Arial')
        plt.xlabel('Target Label', fontsize=14, family='Arial')
        plt.ylabel('Percentage', fontsize=14, family='Arial')
        
        # set different fill patterns for each bar
        for bar, category in zip(bars.patches, num_target_labels['target']):
            bar.set_hatch(hatch_patterns[category])
        
        # show values on top of bars as percentages with specified font settings
        for bar, percentage in zip(bars.patches, num_target_labels['percentage']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height*1.01, f'{percentage:.1%}', 
                     ha='center', va='bottom', fontsize=12, family='Arial')
        
        # Add gridlines for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Customize ticks
        plt.xticks(fontsize=12, family='Arial')
        
        # set custom y-ticks
        max_value = num_target_labels['percentage'].max()
        y_ticks = np.arange(0, max_value * 1.2, 0.1)  # Create ticks from 0 to max_value with an interval of 1.5
        plt.yticks(y_ticks, fontsize=12, family='Arial')
        
        # Format y-ticks as percentages
        def percent_formatter(x, pos):
            return f'{x:.0%}'
        
        plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
        
        # Show the plot
        plt.tight_layout()
        
        # Save the plot (optional)
        if save:
            if export_name==None:
                export_name = 'Percentage of each Target Label'
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                try:
                    plt.savefig(rf'.\{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
                except:
                    plt.savefig(rf'{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig
    
    def human_readable_number(self, num,precision=2):
        """
        Convert a number into a human-readable format with appropriate symbols.
        Handles very small numbers, negative numbers, and large numbers.
    
        Parameters;
            num (int or float0: The number to format.
            precision (int); The number of decimal places to keep 9default is 2).
    
        Returns:
            str; the formatted human-readable number.
        """
        # handle negative numbers
        is_negative = num < 0
        num = abs(num)
    
        # Define thresholds and suffixes
        thresholds = [
            (1e18, 'Qt'),  # Quintillion
            (1e15, 'Qr'),  # Quadrillion
            (1e12, 'T'),  # Trillion
            (1e9, 'B'),   # Billion
            (1e6, 'M'),   # Million
            (1e3, 'k'),  # Thousand
            (1, ''),     # Units
            (1e-3, 'm'), # Milli
            (1e-6, ''), # Micro
            (1e-9, 'n'), # Nano
            (1e-12, 'p') # Pico
        ]
    
    
        # find the appropriate suffix
        for threshold, suffix in thresholds:
            if num >= threshold:
                num = num / threshold
                break
        else:
            # For very small numbers, use scientific notation
            return f"{'-' if is_negative else ''}{num:.{precision}e}"
    
        # Format the number
        formatted_num = f"{num:.{precision}f}{suffix}"
        return f"{'-' if is_negative else ''}{formatted_num}"

    def humanize_interval(self, interval, precision=2):
        """
        Convert numbers in an interval string 
        (e.g., from pd.cut or pd.qcut0 into human-readable format.
        Handles brackets, parentheses, 'inf', '-inf', and negative numbers.
        Humanizes both sides of the interval, even if one side contains 'inf' or '-inf'.
    
        Parameters;
            interval (str): the interval string 
            (e.g., "[-inf, -2651651]', "[186464684, inf]", "[-1000, 0)").
            precision (int0: the number of decimal places to keep 9default is 2).
    
        Returns:
            str: the human-readable interval string.
        """
        # extract brackets/parentheses
        brackets = re.findall(r"[\[\]()]", interval)
    
        # split the interval into left and right parts
        parts = re.split(r",\s*", interval.strip('[]()'))
        if len(parts) != 2:
            return interval  # Return original if the format is invalid
    
        left_org, right_org = parts
    
        # Function to humanize a single side of the interval
        def humanize_side(side):
            if side in ["inf", "-inf"]:
                return side  # leave inf or -inf unchanged
            elif float(side)==0:
                return 0
            try:
                num = float(side)
                return self.human_readable_number(num,precision)        
            except ValueError:
                return side  # Return original if conversion fails
    
        # Humanize both sides
        left = humanize_side(left_org)
        right = humanize_side(right_org)
        # reconstruct the interval string
        return f"{brackets[0]}{left}, {right}{brackets[1]}"
        
    
    def calculate_vif(self, data: pd.DataFrame, column_index: int, column_name: str) -> pd.DataFrame:
        """
        Compute the Variance Inflation Factor (VIF) for a given column in the dataset.
    
        Args:
            data (pd.DataFrame): The dataset containing feature values.
            column_index (int): The index of the column to compute VIF for.
            column_name (str): The name of the column.
    
        Returns:
            pd.DataFrame: A DataFrame containing the feature name and its VIF value.
        """
        vif_value = variance_inflation_factor(data, column_index)
        output = pd.DataFrame(
            [[column_name, vif_value]],
            columns=["feature_name", "variance_inflation"]
        )
        
        return output
        
    def remove_high_vif_features(self, X_train: pd.DataFrame, vif_threshold: float = 10.0) -> list:
        """
        Iteratively remove features with high Variance Inflation Factor (VIF).
    
        Args:
            X_train (pd.DataFrame): The training dataset with independent variables.
            vif_threshold (float, optional): The threshold above which a feature is removed. Default is 10.0.
    
        Returns:
            list: The list of selected features after removing high VIF features.
        """
        
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        selected_features = X_train_scaled.columns.tolist()
        step = 0
    
        while True:
            step += 1
    
            # Compute VIF for all selected features in parallel
            vif_results = Parallel(n_jobs=-1)(
                delayed(self.calculate_vif)(X_train_scaled[selected_features].values, idx, col)
                for idx, col in tqdm(enumerate(selected_features), desc=f"Step {step}")
            )
    
            # Combine results into a DataFrame and sort by VIF in descending order
            vif_df = pd.concat(vif_results).sort_values(
                by="variance_inflation", ascending=False
            ).reset_index(drop=True)
    
            # Check if the highest VIF exceeds the threshold
            max_vif = vif_df["variance_inflation"].max()
            if max_vif >= vif_threshold:
                feature_to_remove = vif_df.iloc[0, 0]
                print(f"Step {step}: VIF {max_vif:.2f} (Removing '{feature_to_remove}')")
                selected_features.remove(feature_to_remove)
            else:
                break
    
        return selected_features
    
    def variance_inflation_plot(self, df_var_inf, save=False, save_path=r'.\results', export_name=None):
        #### Plot Variance inflaction results on selected features
        # Set up the matplotlib figure
        df_var_inf = df_var_inf.sort_values(by='variance_inflation', ascending=False)
        
        fig = plt.figure(figsize=(12, 8))  # Adjust size to accommodate 200+ bars
        sns.set(style="whitegrid")  # Set style
        
        # Create a color palette that transitions from high to low importance
        colors = sns.color_palette("viridis", len(df_var_inf))  # Use a perceptually uniform colormap
        
        # Create the barplot
        ax = sns.barplot(x='variance_inflation', y='feature', data=df_var_inf, palette=colors)
        
        # Customize the plot
        ax.set_title('Feature variance_inflation', fontsize=18, fontweight='bold', pad=20)  # Title
        ax.set_xlabel('Variance Inflation', fontsize=14, fontweight='bold', labelpad=15)  # X-axis label
        ax.set_ylabel('Features', fontsize=14, fontweight='bold', labelpad=15)  # Y-axis label
        
        # Customize tick labels
        ax.tick_params(axis='x', labelsize=12)  # X-axis tick font size
        ax.tick_params(axis='y', labelsize=12)  # Y-axis tick font size
        
        # Rotate y-axis labels for better readability
        plt.yticks(rotation=0)
        
        # add annotations (importance values) on each bar
        for p in ax.patches:
            width = p.get_width()
            value = f'{self.human_readable_number(width,precision=1)}'
            ax.text(width * 1.01, p.get_y() + p.get_height() / 2, value, 
                    ha='left', va='center', fontsize=10, color='black')
        
        # Set borders for bars
        for bar in ax.patches:
            bar.set_edgecolor('black')  # Set border color for bars
        
        # adjust layout to prevent overlapping
        plt.tight_layout()
        
        # Save the plot (optional)
        # Save the plot (optional)
        if save:
            if export_name==None:
                export_name = 'Feature variance_inflation'
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                try:
                    plt.savefig(rf'.\{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
                except:
                    plt.savefig(rf'{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
                    
        
        # Show the plot
        plt.show()
        return fig
    
    def feature_correlation(self, X, method='pearson', save=False, save_path=r'.\results', export_name=None):
        
        sns.set_theme(style='white')
        fig = plt.figure(figsize=(14,14))
        corr_matrix = X.corr(method=method)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        # np.fill_diagonal(mask, False)
        # mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230,20, as_cmap=True)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            center=0,
            annot=True,
            fmt='.0%',
            annot_kws={'size':10, 'color':'black'},
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
        )
        plt.title(f'{method.capitalize()} Correlation Matrix', fontsize=20, pad=20, fontweight='bold')
        plt.xticks(fontsize=14, rotation=45, ha='right')
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        # Save the plot (optional)
        if save:
            if export_name==None:
                export_name = f'{method.capitalize()} Correlation Matrix',
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                try:
                    plt.savefig(rf'.\{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
                except:
                    plt.savefig(rf'{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
        # Show the plot
        plt.show()
        return fig
    
    def optuna_opt(self, 
            X_train, X_test, y_train, y_test, clf, metric, n_trials=1000,objective='binary',
            thr_method='auc_roc', random_state=42, class_weight=None, n_jobs=-1
        ):
    
        # Define the objective function for Optuna
        def objective(trial):
            # Suggest hyperparameters
            if clf in ['xgb', 'xgboost', 'xgbclassifier', 'xgboostclassifier', 'xgboost_classifier']:
                xgb_params = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'eta': trial.suggest_loguniform('eta', 1e-3, 0.3),
                    'alpha': trial.suggest_float('alpha', 0, 1),
                    'lambda': trial.suggest_float('lambda', 0, 1),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'booster': 'gbtree',
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)  # Adjust for imbalance
                }
                model = XGBClassifier(
                    **xgb_params,
                    class_weight=class_weight,
                    n_jobs=n_jobs,
                    random_state=random_state
                )
    
            elif clf in ['lgb', 'lgbm', 'lgboost', 'lgbmclassifier', 'lgbm_classifier']: 
                lgb_param = {
                    'boosting_type': 'gbdt',
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'max_depth': trial.suggest_int('max_depth', -1, 15),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                }
                
                if objective=='cross_entropy':
                    lgbm_objective = "cross_entropy_lambda"
                    lgbm_metric = 'cross_entropy'
                else:
                    lgbm_objective = "binary"
                    lgbm_metric = 'binary_logloss'
                    
                model = LGBMClassifier(
                    **lgb_param, 
                    is_unbalance=True,  # If your dataset is imbalanced
                    metric=lgbm_metric,
                    objective=lgbm_objective,
                    class_weight=class_weight,
                    n_jobs=n_jobs, 
                    random_state=random_state,
                    verbose=-1, 
                )
    
    
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
    
            
            best_thr = self.best_threshold(y_test, y_pred_proba, method=thr_method)
            y_pred = np.where(y_pred_proba > best_thr, 1, 0)
            if metric=='auc':
                score = roc_auc_score(y_test, y_pred_proba)
            elif metric=='auprc':
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)  
                score = auc(recall, precision)
            elif metric=='precision':
                score = precision_score(y_test, y_pred)  
            elif metric=='recall':
                score = recall_score(y_test, y_pred)  # For maximizing recall
            elif metric=='f1':
                score = f1_score(y_test, y_pred)  # Uncomment this line to maximize F1 score
            elif metric=='accuracy':
                score = accuracy_score(y_test, y_pred)  # Uncomment this line to maximize F1 score
    
            return score
    
        # Create the Optuna study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
    
        best_params = study.best_params
        best_value = study.best_value
        print(f"Best Parameters: {best_params}")
        print(f"Best Score: {best_value}")
        return study, best_params, best_value
        
    def optuna_plot(self, optuna_trials):
        # Note: 'value' is typically the AUC score in binary classification problems with Optuna
        
        if not isinstance(optuna_trials, pd.DataFrame):
            optuna_trials = optuna_trials.trials_dataframe()

        drop_cols = ['number', 'value', 'datetime_start', 
                     'datetime_complete', 'duration', 'state']
        selected_hyperparameters = optuna_trials.drop(drop_cols, axis=1).columns.tolist()

                
        # Add the objective value (AUC score) to the list
        data = optuna_trials[selected_hyperparameters + ['value']]
        
        # Rename 'value' column to something more intuitive (like AUC score)
        hyperparameters_name = [re.sub('params_','',hp) for hp in selected_hyperparameters]
        axis_labels = dict(zip(selected_hyperparameters,hyperparameters_name))
        data = data.rename(columns={'value': 'auc_score'})
        axis_labels['auc_score']= 'AUC Score'
        
        # Create the parallel coordinates plot with enhanced style
        fig = px.parallel_coordinates(
            data,
            dimensions=selected_hyperparameters,   # Use hyperparameters for the axes
            color='auc_score',            # Color lines by AUC score
            labels=axis_labels,           # Custom labels
            # color_continuous_scale=px.colors.sequential.Plasma,  # A vivid color scale
            color_continuous_scale=px.colors.diverging.RdYlGn,  # A vivid color scale
            # color_continuous_scale=px.colors.sequential.speed,  # A vivid color scale
        
            # range_color=[data['auc_score'].min(), data['auc_score'].max()],  # Dynamic color range
            range_color=[0.86, data['auc_score'].max()],  # Dynamic color range
            title="Optimized Hyperparameters Effect on AUC Score",
        )
        
        # Update plot aesthetics
        fig.update_layout(
            height=800,
            width=1400,
            font=dict(size=14, family="Arial"),   # Set the font style and size
            title_font_size=20,                   # Title font size
            title_x=0.5,                          # Center the title
            plot_bgcolor='rgba(0,0,0,0)',         # Transparent background
            # paper_bgcolor='rgba(0,0,0,0)',  # Light gray paper background
            paper_bgcolor='rgba(240,240,240,1)',  # Light gray paper background
        )
        
        # Show the plot
        fig.show()
                
    def optuna_hyperparameter_effect(self, opt_df, hyperparameter, save=False, save_path=r'.\results', export_name=None):
            
        df_hyper = opt_df[[hyperparameter, 'value']].copy()
        df_hyper.columns = ['hyperparameter', 'AUC']
        
        # Normalize the color mapping based on AUC values
        norm = mcolors.Normalize(vmin=df_hyper["AUC"].min(), vmax=df_hyper["AUC"].max())
        cmap = cm.viridis  # Use the Viridis color map
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy array for colorbar
    
        # Normalize size values
        min_size, max_size = 20, 200
        # size = np.interp(df_hyper["AUC"], (df_hyper["AUC"].min(), df_hyper["AUC"].max()), (min_size, max_size))
    
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))  # Use fig, ax to manage axes
    
        # Create the scatter plot
        scatter = sns.scatterplot(
            data=df_hyper,
            x="hyperparameter",
            y="AUC",
            hue=df_hyper["AUC"],  # Color based on AUC
            palette="viridis",
            size="AUC",  # Dynamic marker size
            sizes=(min_size, max_size),
            edgecolor='black',
            legend=False,  # Disable default Seaborn legend
            ax=ax  # Explicitly set axis
        )
    
        # Add LOWESS trendline
        sns.regplot(
            data=df_hyper,
            x="hyperparameter",
            y="AUC",
            scatter=False, 
            lowess=True,
            color='black',
            line_kws={"linewidth": 2},
            ax=ax  # Explicitly set axis
        )
    
        # Customize the plot
        ax.set_title(f'Impact of "{hyperparameter}" on AUC (Optuna Analysis)', fontsize=16, loc='center')
        ax.set_xlabel('Hyperparameter Value', fontsize=14)
        ax.set_ylabel('AUC Score', fontsize=14)
    
        # Add colorbar (Explicitly link it to `fig`)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("AUC Score", fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    
        # Customize the grid and axis lines
        ax.grid(True, linestyle='--', alpha=0.6)
    
        # Customize tick formats and axis limits
        ax.tick_params(axis='both', which='both', labelsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
        # Adjust layout for clarity
        export_name = f'Impact of {hyperparameter} on AUC (Optuna Analysis)'
        plt.tight_layout()
        
        
        # Save the plot (optional)
        if save:
            if export_name==None:
                export_name = f'Impact of {hyperparameter} on AUC (Optuna Analysis)',
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                try:
                    plt.savefig(rf'.\{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
                except:
                    plt.savefig(rf'{save_path}\{export_name}.png', dpi=300, bbox_inches='tight')
        
        # Show the plot
        plt.show()
