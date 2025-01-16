# %% Script for the development and evaluation of a Gradient Boosted Trees (GBT) classification model, 
# following the processing of the PTBXL database in ptbxl_ecg_cvd_exploration.py. 

# NOTE: This script considers ONLY the "pure" ECG recordings of PTBXL (those corresponding to a single CVD superclass).

# Dataset Description:
# The processed dataset consists of aproximately 21,000 ECG recordings characterized by 27 features and corresponding to 19,000 patients.
# Each ECG recording is labelled with one or more cardiovascular disease superclasses:
    # - NORM (normal) - Normal ECG recordings without any detectable cardiovascular abnormalities.
    # - MI (myocardial infarction) - Various forms of heart attack, characterized by ischemic changes in the ECG, 
    #                                such as ST elevation or Q waves.
    # - STTC (ST/T change) - Changes in the ST segment and T wave of the ECG.
    #                        Includes ischemia-related changes, non-specific ST-T abnormalities, 
    #                        and other conditions that affect repolarization.
    # - CD (conduction disturbance) - Conditions affecting the electrical conduction pathways of the heart:
    #                                 bundle branch block, atrioventricular (AV) blocks, intraventricular conduction delays, tachyarhythmia, etc.
    # - HYP (hypertrophy)] - Enlargement or thickening of the heart muscle (particularly the ventricle(s)).
# Not all of the CVD superclass labels in the dataset were validated by humans: approx. 6,000 of the ECG recordings were not human-validated.
# The prevalence of coexisting CVD superclasses in individual ECG recordings is significant and reflects the reality of CVDs (i.e., comorbidities).

# Pure observations only:
# To simplify training and evaluation, in this initial script only the "pure" ECG recordings demonstrating a single CVD superclass are considered.
# Further scripts will consider all ECG recordings and multi-label training and evaluation via more complex schemes.

# Classification Model Justification:
# Following the suggestions in ptbxl_ecg_cvd_exploration.py., Gradient Boosted Trees will be used as the classification model.
# Their non-linearity and ability to capture complex patterns in data make them a good candidate in this processed dataset where there was little CVD superclass cluster separability.

# Evaluation Metric Justification:
# Given the significant class imbalance of the dataset, F1 will be the driving classification metric in this evaluation and  performances of 0.8+ are targetted.

# Script outline:
# This script consists of the following sections:
    # 1. Loading processed featureset and labels.
    # 2. Splitting into training/validation and test sets.
    # 3. Grid search cross-validation.
    # 4. Optimal model training.
    # 5. Evaluation.
    # 6. Conclusions.
    

# %% 0. Importing Libraries + Defining global parameters and flags.

# % 0.1 Importing libraries
import os
import ast
import wfdb

import scipy 
import numpy as np
import pandas as pd
import collections as colls

import seaborn as sns
import matplotlib.pyplot as plt

# % 0.2 Global parameters and flags
rand_state = 93 # Fixing random state for reproducibility.
flag_save = True # Flag to save results and figures.


# %% 1. Loading Datasets

# % 1.1 Loading X, y and processing parameters

# Define project path.
prj_path = 'C:/Users/rcord/OneDrive/Documentos/GitHub/python-ml-portfolio/ptbxl/'

# Load X, y, and params
X = pd.read_csv(prj_path + 'X.csv', index_col='ecg_id') # Obtained via ptbxl_ecg_cvd_exploration.py.
y = pd.read_csv(prj_path + 'y.csv', index_col='ecg_id')
y['scp_codes'] = y['scp_codes'].apply(ast.literal_eval) # Converting labels to list of strings, rather than string bookended with '[]'.

proc_params = pd.read_csv(prj_path + 'params.csv')

# % 1.2 Considering only pure ECG recordings (no coexisting CVDs).
pure_ecg_ids = y.index[y['scp_codes'].apply(len) == 1].tolist()

X = X.loc[pure_ecg_ids, :]
y = y.loc[pure_ecg_ids, :]

del pure_ecg_ids


# %% 2. Splitting into training/validation and test sets.
 
bool_tr = y['strat_fold'] < 9 # Includes validation set because cross-validation will be implemented.
bool_test = y['strat_fold'] < 9

X_tr = X.loc[bool_tr, :]
X_test = X.loc[bool_test, :]

y_tr   = y.loc[bool_tr, :]
y_test = y.loc[bool_test, :]    
