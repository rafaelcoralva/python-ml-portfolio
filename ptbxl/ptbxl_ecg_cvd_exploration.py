# Script for the exploration and analysis of the PTBXL and PTBXL+ ECG database.
# The database consists of some approx. 22,000 12-lead ECG recordings corresponding to 19,000 patients and labelled with one or more cardiovascular disease superclasses:
    # - NORM (normal) - Normal ECG recordings without any detectable cardiovascular abnormalities.
    # - MI (myocardial infarction) - Various forms of heart attack, characterized by ischemic changes in the ECG, 
    #                                such as ST elevation or Q waves.
    # - STTC (ST/T change) - Changes in the ST segment and T wave of the ECG.
    #                        Includes ischemia-related changes, non-specific ST-T abnormalities, 
    #                        and other conditions that affect repolarization.
    # - CD (conduction disturbance) - Conditions affecting the electrical conduction pathways of the heart:
    #                                 bundle branch block, atrioventricular (AV) blocks, intraventricular conduction delays, tachyarhythmia, etc.
    # - HYP (hypertrophy)] - Enlargement or thickening of the heart muscle (particularly the ventricle(s)).
# The prevalence of coexisting CVD superclasses in individual ECG recordings is significant and reflects the reality of CVDs (i.e., comorbidities).
# This script loads the PTBXL database (patient demographic, recording, and label information), 
# as well as the PTBXL+ database (extracted ECG features from the PTBXL database using public ECG analysis libraries).
# Features of these databases are selected, engineering and analyzed for their further use in a Machine-Learning-based classification framework.
# The following analysis are performed here:
    # 1. Preliminary feature selection.
    # 2. Combination of PTBXL+ features common to GE Healthcare 12SL and University of Glasgow featuresets.
    # 3. Feature exploration and wrangling.
    # 4. Target CVD superclass exploration and wrangling.
    # 5. Feature engineering
    # 6. Visualization of CVD superclass clustering.
    # 7. Saving outputs for further ML classification framework.

# Rafael Cordero, December 2024.


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

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE

# % 0.2 Global parameters and flags
rand_state = 93 # Fixing random state for reproducibility
flag_combine_shared_feats = False # Flag to combined shared features between PTBXL+ 12L and Uni G featuresets.
flag_save = True # Flag to save output X and y


# %% 1. Loading Datasets

# % 1.1 Loading PTBXL and PTBXL+ Datasets

# Define public dataset path.
data_path = 'C:/Users/rcord/OneDrive/Documentos/Data/Public/'

# Load both PTBXL and PTBXL+ (12SL + Uni G)
df_ptbxl           = pd.read_csv(data_path + 'PTBXL/ptbxl_database.csv', index_col='ecg_id') # Download from https://physionet.org/content/ptb-xl/1.0.3/
df_ptbxl_plus_12sl = pd.read_csv(data_path + 'PTBXL+/features/12sl_features.csv', index_col='ecg_id') # Download from https://physionet.org/content/ptb-xl-plus/1.0.1/. 
df_ptbxl_plus_unig = pd.read_csv(data_path + 'PTBXL+/features/unig_features.csv', index_col='ecg_id') 
# PTBXL+ ECGDeli featureset not considered due to suspiciously high feature values for _Dur_ and _Int_ features.

# % 1.2 Extracting target variable (code lifted from example_physionet.py)
y_dic = df_ptbxl['scp_codes'].apply(lambda x: ast.literal_eval(x)) 

df_ptbxl_agg = pd.read_csv(data_path +'/PTBXL/scp_statements.csv', index_col=0)
df_ptbxl_agg = df_ptbxl_agg[df_ptbxl_agg.diagnostic == 1]

def aggregate_diagnostic(y_dict):
    tmp = []
    for key in y_dict.keys():
        if key in df_ptbxl_agg.index:
            tmp.append(df_ptbxl_agg.loc[key].diagnostic_class)
    return list(set(tmp))

# Extract CVD superclass labels
cvd_labels = y_dic.apply(aggregate_diagnostic)

# Extract cross-validation stratification fold for future train/val/test splitting
strat_fold = df_ptbxl['strat_fold'] # Database authors recommend 10-fold train-test splits obtained via author-defined stratified sampling. 
                                    # Respects patient assignment (i.e., all ECG recordings of a particular patient were assigned to the same fold). 
                                    # ECG recordings in fold 9 and 10 underwent at least one human evaluation and are therefore of a particularly high label quality. 
                                    # Database authors therefore propose to use folds 1-8 as training set, fold 9 as validation set and fold 10 as test set.

# Packaging target class output (with stratification fold + raw 500Hz ECG file location)
y = pd.concat([cvd_labels, strat_fold, df_ptbxl['filename_hr']], axis=1)

# Defining troubleshooting function to extract and plot the raw ECG of an input ECG ID
def extract_plot_ecg(ecg_id, y, root_path=data_path, flag_plot=True):
    
    # Extraction
    ecg_filename = y['filename_hr'][ecg_id]
    ecg_path = root_path + ecg_filename
    ecg_signal = wfdb.rdsamp(ecg_path)
    
    # Plotting
    if flag_plot:
        
        # Extracting from ecg_signal structure
        fs = ecg_signal[1]['fs']
        leads = ecg_signal[1]['sig_name']
        units = ecg_signal[1]['units']
        sigs = ecg_signal[0]
        
        time = np.arange(sigs.shape[0]) / fs # Time vector.
        
        # Plotting each lead
        plt.figure(figsize=(12, 6))
        for ii, lead in enumerate(leads):
            plt.subplot(6, 2, ii + 1)
            plt.plot(time, sigs[:, ii])
            if (ii == 10 or ii == 11): # Only plot x labels and ticks on bottom subplots
                plt.xlabel('Time (s)')
            else:
                plt.xticks([])
            plt.ylabel(f'{units[ii]}')
            plt.legend([lead], loc='center left')

        plt.suptitle(f"ECG {str(ecg_id)}, CVD = {y.loc[ecg_id, 'scp_codes']}")
        plt.tight_layout()
        plt.show()

    return ecg_signal

del y_dic, df_ptbxl_agg, cvd_labels, strat_fold


# %% 2 Preliminary Feature selection
# PTBXL and PTBXL+ feature sets are huge (27 and 782+748 columns respectively).
# There is a lot of irrelevant, duplicated, or highly granular information.
# Some preliminary feature selection is performed.

# % 2.1. PTBXL preliminary feature selection
# Lots of non-physiological features irrelevant to CVD classifiation: 
    # - patient_id,
    # - nurse, 
    # - site, 
    # - device, 
    # - recording_date, 
    # - scp_codes : target CVD superclass (already extracted),
    # - report, 
    # - validated_by
    # - second_opinion,
    # - initial_autogenerated_report,
    # - validated_by_human,
    # - static_noise,
    # - burst_noise,
    # - electrodes_problems,
    # - strat_fold : useful for train/val/test set splitting (already extracted).
    # - filename_lr, 
    # - filename_hr
# -> Discard these non-physiological/irrelevant features.
    
df_ptbxl.drop(labels=['patient_id', 'nurse','site','device', 'recording_date', 'scp_codes',
                      'report', 'validated_by', 'second_opinion', 'initial_autogenerated_report',
                      'validated_by_human', 'static_noise', 'burst_noise',
                      'electrodes_problems', 'strat_fold', 'filename_lr', 'filename_hr'], 
              axis=1,
              inplace=True)

# % 2.2. PTBXL+ 12sl preliminary feature selection
# - Very high-dimensional: 782 features! 
#   There are a lot of "repeating" features across the 12 leads.
#   E.g.: P_Dur_ (P wave duration) is computed for each of the 12 leads + '_Global' (median beat-derived).
#   -> Discard repeated/lead-specific features so we are left only with '_Global' feautres and any other non-lead-specific feaures.
# - There exist '_On', '_Off', '_Dur' features for each ECG waveform (i.e., P, Q, QRS, and T). E.g., P_On_Global, P_Off_Global and P_Duration_Global.
#   '_On' denotes the onset of the wave, '_Off' the offset, and '_Dur' the duration (i.e., the difference of these). E.g., P_Duration_Global = P_Off_Global - P_On_Global
#   -> Discard '_On', and '_Off' features and keep '_Dur' features.
# - 'P_Found_Global' is essentially just a boolean to denote if P waves were detected. If it is 0, P wave-related features are set to NaN.
#    -> Discard feature.
   
# Defining lead suffixes to identify lead-specific features in df_ptbxl_plus and discard
lead_suffixes = ['_I', '_II', '_III', '_aVR', '_aVL', '_aVF', '_V1', '_V2', '_V3', '_V4', '_V5', '_V6']    

# Function to check if a feature ends with any of the lead suffixes
def is_lead_specific(feature_name, lead_suffixes):
    return any(feature_name.endswith(suffix) for suffix in lead_suffixes)

# Discarding lead-specific features
lead_specific_feats = [feat for feat in df_ptbxl_plus_12sl.columns if is_lead_specific(feat, lead_suffixes)]
df_ptbxl_plus_12sl.drop(lead_specific_feats, axis=1, inplace=True)
 
# Discarding Onset and Offset features
bool_on_off_feats = [('_On_' in feat) or ('_Off_' in feat) for feat in df_ptbxl_plus_12sl.columns]
on_off_feats = df_ptbxl_plus_12sl.columns[bool_on_off_feats]

df_ptbxl_plus_12sl.drop(columns=on_off_feats, inplace=True)
    
# Discarding P_Found_Global
df_ptbxl_plus_12sl.drop(columns=['P_Found_Global'], inplace=True)

del lead_specific_feats, bool_on_off_feats, on_off_feats

# % 2.3. PTBXL+ unig preliminary feature selection
# Similar issues + solutions as for PTBXL+ 12SL:
# - Too high-dimensional: 748 features! 
#   There are a lot of "repeating" features across the leads.
#   -> Discard "repeat" lead-specific features so we are left only with '_Global' feautres and any other non-lead-specific feaures (e.g., 'HR_Sinus_Global').
# - There exist '_On', '_Off', '_Dur' features for each ECG waveform (i.e., P, Q, QRS, and T). E.g., P_On_Global, P_Off_Global and P_Duration_Global.
#   -> Discard '_On', and '_Off' features and keep '_Dur' features.
# - 'P_Found_Global' is essentially just a boolean to denote if P waves were detected. If it is 0, P wave-related features are set to NaN.
#    -> Discard feature, NaNs in P wave features can be imputed later.
# Additional issues specific to PTBXL+ Uni G:
# - QRS_PV48/58/68/78SpatVel_Global : Highly specific QRS spatial velocity features. Too granular and redundant with other QRS velocity metrics.
#   -> Discard features.
# - VM_Vel_18/28/38/48/58/68/78QRS, VM_Vel_-30/-10/-20/+10/+20/30msQRS, QRS velocity metrics at various points in the QRS complex (very specific and redundant).
#   -> Discard features.

# Discarding lead-specific features
lead_specific_feats = [feat for feat in df_ptbxl_plus_unig.columns if is_lead_specific(feat, lead_suffixes)]
df_ptbxl_plus_unig.drop(lead_specific_feats, axis=1, inplace=True)
 
# Discarding Onset and Offset features
bool_on_off_feats = [('_On_' in feat) or ('_Off_' in feat) for feat in df_ptbxl_plus_unig.columns]
on_off_feats = df_ptbxl_plus_unig.columns[bool_on_off_feats]

df_ptbxl_plus_unig.drop(columns=on_off_feats, inplace=True)
    
# Discarding P_Found_Global
df_ptbxl_plus_unig.drop(columns=['P_Found_Global'], inplace=True)

# Discarding highly specific QRS spatial velocity features
bool_QRSspatvel_feats = ['SpatVel_' in feat for feat in df_ptbxl_plus_unig.columns]
QRS_spat_vel_feats = df_ptbxl_plus_unig.columns[bool_QRSspatvel_feats]

df_ptbxl_plus_unig.drop(columns=QRS_spat_vel_feats, inplace=True)

# Discarding highly specific QRS velocity features
bool_QRSvel_feats = [('VM_Vel_' in feat) and ('QRS' in feat) and ('Max' not in feat) for feat in df_ptbxl_plus_unig.columns] # Do not include VM_Vel_MaxQRS for discarding, it's "general".
QRS_vel_feats = df_ptbxl_plus_unig.columns[bool_QRSvel_feats]

df_ptbxl_plus_unig.drop(columns=QRS_vel_feats, inplace=True)

del lead_suffixes, lead_specific_feats, bool_on_off_feats, on_off_feats, bool_QRSspatvel_feats, QRS_spat_vel_feats, bool_QRSvel_feats, QRS_vel_feats


# %% 3. Combining PTBXL and the PTBLX+ dataframes

# % 3.1 Combining PTBXL+ dataframes into unique PTBXL+ dataframe
# There are several features shared in both 12SL and Uni G featuresets.
# If flag_combine_shared_feats is True, these are mean-averaged IF they are similar enough (within 0.5 average IQRs).
# If they are not similar enough, or if flag_combine_shared_feats is False, the 12SL feature values are considered and the Uni G feature values are ignored.

# Initializing the new combined PTBXL+ DataFrame
df_ptbxl_plus = pd.DataFrame(index=df_ptbxl_plus_12sl.index) 

# Identifying shared and unique features
shared_feats      = df_ptbxl_plus_12sl.columns.intersection(df_ptbxl_plus_unig.columns)
unique_12sl_feats = df_ptbxl_plus_12sl.columns.difference(shared_feats)
unique_unig_feats = df_ptbxl_plus_unig.columns.difference(shared_feats)

# Add unique features from both datasets
df_ptbxl_plus = pd.concat([df_ptbxl_plus, df_ptbxl_plus_12sl[unique_12sl_feats]], axis=1)
df_ptbxl_plus = pd.concat([df_ptbxl_plus, df_ptbxl_plus_unig[unique_unig_feats]], axis=1)

# Adding shared features
if flag_combine_shared_feats: # Mean-averaging shared feature values if similar enough, or by using only 12SL values if not similar enough
    for feat in shared_feats:
    
        # Compute differences between 12SL and Uni G feature values
        differences = df_ptbxl_plus_12sl[feat] - df_ptbxl_plus_unig[feat]
    
        # Compute IQR for the 12SL feature and Uni G feature values
        q1_12sl, q3_12sl = df_ptbxl_plus_12sl[feat].quantile(0.25), df_ptbxl_plus_12sl[feat].quantile(0.75)
        q1_unig, q3_unig = df_ptbxl_plus_unig[feat].quantile(0.25), df_ptbxl_plus_unig[feat].quantile(0.75)
        iqr_12sl = q3_12sl - q1_12sl
        iqr_unig = q3_unig - q1_unig
        
        # Compute mean IQR
        iqr_avg = (iqr_12sl  + iqr_unig) / 2
        
        # Normalize the differences by the mean IQR
        normalized_differences = differences / iqr_avg
    
        # Plotting overlapped 12SL and Uni G histograms to show feature value similarity
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Assesment of {feat} similarity in 12SL and Uni G', fontsize=16)
    
        # Left subplot: overlapping histograms of the distributions
        axes[0].hist(df_ptbxl_plus_12sl[feat], bins=100, alpha=0.5, label='12SL', color='blue')
        axes[0].hist(df_ptbxl_plus_unig[feat], bins=100, alpha=0.5, label='Uni G', color='orange')
        axes[0].set_title(f'Distribution of {feat} (12SL vs Uni G)')
        axes[0].set_xlabel(feat)
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
    
        # Right subplot: histogram of the normalized differences
        axes[1].hist(normalized_differences, bins=np.arange(-10, 10.5, 0.5), color='green')
        axes[1].set_title(f'Normalized Difference of {feat} (12SL - Uni G)')
        axes[1].set_xlabel('Normalized Difference (by Avg IQR)')
        axes[1].set_xlim([-10, 10]) 
        axes[1].set_xticks(list(range(-10, 11)))
        axes[1].set_ylabel('Frequency')
    
        axes[1].axvline(x=-0.5, color='red', linestyle='--', linewidth=1) # Condition to average or not (if difference within red lines)
        axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=1)
        
        # Adjust layout and show plots
        plt.tight_layout()
        plt.show()
        
        # Averaging if similar enough and adding to PTBXL+ Featureset
        df_ptbxl_plus[feat] = np.where(abs(normalized_differences) < 0.5,  # Condition: abs(qvg IQR-normalized difference) < 0.5 * IQR
                                       (df_ptbxl_plus_12sl[feat] + df_ptbxl_plus_unig[feat]) / 2,  # True cases - take average.
                                       df_ptbxl_plus_12sl[feat]) # False cases - use 12SL values.
    
    del feat, differences, q1_12sl, q1_unig, q3_12sl, q3_unig, iqr_12sl, iqr_unig, iqr_avg, normalized_differences, fig, axes
    
    # Shared feature averaging observations:
    # - Almost all shared features demonstrated the great majority of their values being similar between 12SL and Uni G featuresets (i.e., within 0.5 IQRs of each other)
    # - The following features showed a systematic difference between 12SL and UniG values:
        # - QT_IntCorr_Global: Systematic difference of ~50msec (~0.5 IQRs)
        #                      -> Flag feature for further inspection.
        # - P/T_AxisFront_GLobal: Some observations show differences of ~8.5 IQRs (corresponding to ~360 degrees).
        #                         This is because 12SL reports angles in [-90, 270] and Uni G in [-180, 180] range, the feature values aren't actually different, but they appear as if they are 360 degrees away.
        #                         No action required, defaulting to the 12SL value when absolute difference exceeds 0.5 IQRs corrects for this.
        # - P_Dur_Global: Uni G values shows sharp large peaks in a normal-ish-looking distribution and a strong negative skew not present in 12SL.
        #                 Lower-end Uni G values do not appear physiologically realistic when considering typical or pathological P wave duration values.
        #                 No action required, defaulting to the 12SL value corrects for this.

else: # Considering only the 12SL feature values and ignoring the Uni G feature values
    df_ptbxl_plus[shared_feats] = df_ptbxl_plus_12sl[shared_feats]
    
del df_ptbxl_plus_12sl, df_ptbxl_plus_unig, shared_feats, unique_12sl_feats, unique_unig_feats, 

# % 3.2 Combining PTBXL+ dataframes with PTBXL dataframes
X = pd.concat([df_ptbxl, df_ptbxl_plus], axis=1)

del df_ptbxl, df_ptbxl_plus


# %% 4. Feature Exploration and Wrangling

# % 4.1 Exploration of basic feature stats
print(X.info())

# Observations:
    # Several features are very sparsely populated (lots of NaNs):
        # height : Could have been interesting to combine with weight to get Body Mass Index but too sparse (6974 of 21799 pbservations) -> Discard feature,
        # weight : (9241 of 21799 observations) -> Discard feature,
        # heart_axis : (13331 observations). Already exists (fully-populated and in P/R/T resolution) in the PTBXL+ features -> Discard feature,
        
    # Several other features appear sparsely populated but really they're cateogricals where the non-case is NaN:
        # infarction_stadium1 : Is including this feature kind of cheating? Won't it correlate highly to the MI superclass? 
        #                       -> Discard feature,
        # infarction_stadium2 : Same concern as above 
        #                       -> Discard feature,
        # baseline_drift : Extremely high cardinality (318 unique values) due to feature values existing as combinations of all (strings) ECG leads demonstrating baseline drift. 
        #                  Further, a priori, baseline drift is not reported as a useful predictor of CVD 
        #                  -> Discard feature,
        # extra_beats : Extremely high cardinality (129 unique values), also high chance this will correlate highly with CD superclass 
        #               (kind of cheating - this should  be extracted holistically from the signal, and e.g. RR_StDev_Global_iqr may already capture this 
        #               -> Discard feature,
        # pacemaker : Very confusing feature values [nan, 'ja, pacemaker', 'PACE????, nan', 'ja, nan', '?']]. 
        #             Also, will probably correlate strongly with conduction_disturbances (cheating again) 
        #             -> Discard feature.

# Discarding sparsely-populated features or "cheating" features
X.drop(labels=['height', 'weight', 'heart_axis', 
               'infarction_stadium1', 'infarction_stadium2',
               'extra_beats', 'baseline_drift', 'pacemaker'],
       axis=1, 
       inplace=True)

# Reorganizing feature order in X
desired_column_order = ['age', 'sex', # Demographic features.
                        'QRS_Count_Global', 'RR_Mean_Global', # Heart Rate features.
                        'HR_Atrial_Global', 'HR_Sinus_Global', 'HR_Ventr_Global', 'HR__Global', 
                        'HR_Var_Global', 'RR_StdDev_Global', # Heart Rate Variability features.
                        'P_Dur_Global', 'QRS_Dur_Global', 'T_Dur_Global',  # Wave duration features
                        'PR_Int_Global', 'ST_Dur_Global', # Interval features
                        'QT_Int_Global', 'QT_IntBazett_Global', 'QT_IntFridericia_Global', 'QT_IntFramingham_Global', 'QT_IntHodge_Global', 'QT_IntCorr_Global', # QT inteval features.
                        'QT_Disp_Global',
                        'P_AxisFront_Global', 'R_AxisFrontal_Global', 'QRS_AxisFront_Global', 'QRS_AxisIndet_Global', 'ST_AxisFront_Global', 'T_AxisFront_Global', # Net vector cardiography features.  
                        'VM_AngFront_MaxQRS', 'VM_AngFront_MaxT', 'VM_AngRSag_MaxQRS', 'VM_AngRSag_MaxT', 'VM_AngTrans_MaxQRS', 'VM_AngTrans_MaxT', 'VM_Ang_QRS-T', # Max point vector cardiography features.
                        'VM_LenFront_MaxQRS', 'VM_LenFront_MaxT', 'VM_LenRSag_MaxQRS', 'VM_LenRSag_MaxT', 'VM_LenTrans_MaxQRS', 'VM_LenTrans_MaxT',
                        'VM_Mag_MaxQRS', 'VM_Mag_MaxT', 
                        'VM_Pos_MaxQRS', 'VM_Pos_MaxT',
                        'VM_Vel_MaxQRS', 'VM_Vel_MaxT',
                        'LVH_Strain_Global'] # LV features.
X = X[desired_column_order]

del desired_column_order

# % 4.2 Visualizing feature distributions (with detected outliers)
df_feature_SW_normalities = pd.DataFrame(index=X.columns) # Initialize DataFrame with Shapiro-Wilk test results to assess normality of each feature.
df_bool_outliers = pd.DataFrame(columns=X.columns) # Initialize DataFrame for outlier detection - DataFrame with booleans if an observation's feature value is an outlier. 

for col in X.columns:
    plt.figure(figsize=(12, 6))

    # Left subplot: Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=X.loc[:,col])
    plt.title(f'Box-Whisker plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Feature units')
    
    # Right subplot: Histograms
    plt.subplot(1, 2, 2)
    sns.histplot(y=X.loc[:,col], kde=False, bins=50)
    plt.title(f'Distribution of {col}')
    plt.xlabel('Frequency')
    plt.ylabel('')
    
    plt.tight_layout()

    # % 4.3 Testing and displaying normality of feature distributions with Shapiro-Wilk test.
    
    # Applying SW test statistic    
    col_sw_stat, col_p = scipy.stats.shapiro(X.loc[:, col].dropna().sample(n=5000, random_state=rand_state)) # A sample of 5000 randomly-selected non-NaN observations is considered due to limitations in the scipy shapiro function.
    col_normality = "Normal" if col_p > 0.05 else "Non-Normal" # Assess normality on 95th confidence interval.
     
    # Add distriution normality label to histogram
    plt.legend([f"Shapiro-Wilk Test: {col_normality}"])
    plt.show()     
       
    # Store results
    df_feature_SW_normalities.loc[col, "SW_Statistic"] = col_sw_stat
    df_feature_SW_normalities.loc[col, "P_Value"] = col_p
    df_feature_SW_normalities.loc[col, "Normality_Assumption"] = col_normality
    
# Feature distribution observations:
    # - Almost all features appear somewhat Gaussian but fail a Shapiro-Wilk normality distribution test.
    # - RR_Mean_Global was the most normally-distributed feature (but p is still << 0.5).
    # - Most features demonstrated noticeable positive skewness (may be the reason for failing the Shapiro-Wilk test).
    # - VM_AngRSag_MaxQRS has an 'awkward' distribution: its peak is split on the extremes since it occure at 180 = -180 degrees.
    #   This makes it look like there are two peaks at the these extremes but really these peaks are one and the same due to circular wrapping.
    #   -> Wrap the feature values 180 degrees (i.e. flipping the reference axis) so that the peak occurs at 0 degrees.  

# Feature quality and outlier analysis observations:
    # - Significant amount of outliers for each feature ('o's in the boxplots - exceeding 1.5 x IQR from the 25th and 75th quartiles).
    #   PTBXL+ dataset could potentially contain some noise or some computed features could be unreliable. 
    #   -> Analyze each feature for realistic value ranges and identify suspicious outliers citing relevant clinical literature:
        # - age : Patient age in years.
        #         285 unrealistic outliers with ages of 300 years. 
        #         From the database documentation, any age higher than 89 was set to 300 for legal reasons.
        #         -> Correct by setting ages of 300 to 90. 
        # - ST_Dur_Global : Duration of ST segment in msec. Reflects period of electrical neutrality between ventricular depolarization and repolarization.
        #                   In healthy conditions [80,120] msec (Braunwald, E., 2020: "Heart Disease: A Textbook of Cardiovascular Medicine."). 
        #                   In pathological conditions can be shorter or longer, but it can never be negative.
        #                   2 unrealistic outliers with negative ST_Dur_Global .
        #                   -> Discard observations.
        # - Vectorcardiography Features (VM_Len/Mag/Vel...MaxQRS/T):
        #   - Represent spatial and velocity properties of QRS and T waves in mV or mV/sec.
        #   - Healthy ranges depend on feature type and wave (Goldberger et al., 2015: "VCG Studies"):
        #     - Length/Magnitude: [0.5, 6] mV for QRS, [0, 3] mV for T wave.
        #     - Velocity: [0.5, 3] mV/sec for QRS.
        #   - Pathological values may extend these ranges but never below zero or above ~10 mV or ~10 mV/sec.
        # - 1 unrealistic outlier > 10mV or 10mV/sec common to several of these features. Likely due to signal artefact.
        #   -> Discard observation.
        # - LVH_Strain_Global : Left Ventricular Hypertrophy (LVH) strain. Reflects strain-related changes in ventricular repolarization.
        #                       3 unique values: [-3276 (746 observations), 0 (20111 obs), 1 (938 obs)].
        #                       Unexplained in PTBXL+ documentation but typically this feature is categorical with 3 classes: 
        #                        - positive LVH strain (cardiopathological) [likely class 1], 
        #                        - no LVH strain [likely class 0],
        #                        - invalid ECG recording for LVH strain assesment [likely class -3276].
        #                       -> set -3276 values to NaN.              
            
del  col, col_sw_stat, col_p, col_normality  
           
# % 4.4 Correcting features and discarding/NaNing unrealistic outliers

# Correcting age outlier observations (n=285)
bool_outlier_age = X['age'] == 300
X.loc[bool_outlier_age, 'age'] = 90

# Wrapping VM_AngRSag_MaxQRS 
X['VM_AngRSag_MaxQRS'] = np.where(X['VM_AngRSag_MaxQRS'] < 0, # Condition.
                                  X['VM_AngRSag_MaxQRS'] + 360, # True case.
                                  X['VM_AngRSag_MaxQRS']) - 180 # False case.
        
# Discarding ST_Dur_Global and vector cardiography features
df_bool_outliers = pd.DataFrame(columns=X.columns)

df_bool_outliers['ST_Dur_Global']      = X['ST_Dur_Global'] < 0
df_bool_outliers['VM_LenFront_MaxQRS'] = (X['VM_LenFront_MaxQRS'] < 0) | (X['VM_LenFront_MaxQRS'] > 10)
df_bool_outliers['VM_LenFront_MaxT']   = (X['VM_LenFront_MaxT'] < 0) | (X['VM_LenFront_MaxT'] > 10)
df_bool_outliers['VM_LenRSag_MaxT']    = (X['VM_LenRSag_MaxT'] < 0) | (X['VM_LenRSag_MaxT'] > 10)
df_bool_outliers['VM_LenTrans_MaxQRS'] = (X['VM_LenTrans_MaxQRS'] < 0) | (X['VM_LenTrans_MaxQRS'] > 10)
df_bool_outliers['VM_LenTrans_MaxT']   = (X['VM_LenTrans_MaxT'] < 0) | (X['VM_LenTrans_MaxT'] > 10)
df_bool_outliers['VM_Mag_MaxQRS']      = (X['VM_Mag_MaxQRS'] < 0) | (X['VM_Mag_MaxQRS'] > 10)
df_bool_outliers['VM_Mag_MaxT']        = (X['VM_Mag_MaxT'] < 0) | (X['VM_Mag_MaxT'] > 10)
df_bool_outliers['VM_Vel_MaxQRS']      = (X['VM_Vel_MaxQRS'] < 0) | (X['VM_Vel_MaxQRS'] > 3)

bool_outliers = df_bool_outliers.any(axis=1) # Logical OR operation across columns.
ecg_id_outliers = X.index[bool_outliers]

X.drop(ecg_id_outliers, axis=0, inplace=True)
y.drop(ecg_id_outliers, axis=0, inplace=True)

print(f'{sum(bool_outliers)} observations with unrealistic outliers ({100*sum(bool_outliers)/X.shape[0]:.2f}% of total observations) deleted.') 

# Setting LVH_Strain_Global -3276 values to NaN
bool_invalid_LVH_strain = X['LVH_Strain_Global'] == -3276
X.loc[bool_invalid_LVH_strain, 'LVH_Strain_Global'] = float('nan')

del bool_outlier_age, df_bool_outliers, bool_outliers, ecg_id_outliers, bool_invalid_LVH_strain
    
# % 4.5 Missing values statistics
print(f'There are a total of {np.isnan(X.values).sum()} NaNs in X.')
print(f'These are distributed over {(X.isna().sum(axis=1) > 0).sum()} observations')
print(f'Each feature has the following number of NaNs: \n{X.isna().sum()}')

# Observations:
    # -> For now, no observation discarding or feature value imputation due to NaNs 
    #    (some tree-based methods e.g. LightGBM are robust to NaNs)
    

# %% 5. Target CVD superclass Exploration and Wrangling

# Observations following inspection of target 'y':
    # - The target y exists as a Pandas Series of lists of ALL the CVD superclasses present in that index's ECG recording. 
    # - Multiple CVD superclasses may coexist in a single ECG recording due to presence of multiple diseases (e.g, comorbidities). 
    # - Only 16244 of 21799 ECG recordings(75.9%) are "pure" (i.e., demonstrate a single CVD superclass) most of which are 'NORM'.
    # - The coexistence of several CVD superclasses is a true reflection of the complexity of CVD in the real world.
    #   This significantly complicates training and evaluation since there are so many instances of coexisting CVDs superclasses.
    # - 411 ECG recordings had an empty label in the target 'y' -> Discard.
    
all_cvds = ['NORM', 'MI', 'STTC', 'CD', 'HYP'] # CVD superclasses (in decrasing order of frequency of occurence).

# % 5.1 Keeping only labelled ECG recordings
bool_labelled_y = [True if len(superclass) > 0 else False for superclass in y['scp_codes']]

X = X.loc[bool_labelled_y, :]
y = y.loc[bool_labelled_y, :]

del bool_labelled_y

# % 5.2 Computing total counts of each target CVD superclass, regardless of coexisting CVDs
all_cvds_counts = pd.DataFrame(data = [[0, 0, 0, 0, 0]], # Initialization of total counts DataFrame
                               index = ['Total'], 
                               columns= all_cvds)

# Going through each target CVD superclass label and counting occurence of each CVD superclass
for ii in y.index:
    for cvd in all_cvds:
        if cvd in y['scp_codes'][ii]:
            all_cvds_counts[cvd] += 1
    
# Plotting bar chart of total counts of each target CVD superclass regardless of coexisting CVDs
all_cvds_colors = ['green', 'purple', 'blue', 'orange', 'red']

plt.figure(figsize=(10, 6))
fig_bars = plt.bar(all_cvds, all_cvds_counts.loc['Total',:].tolist(), color=all_cvds_colors, width=0.5)
  
plt.title('Total counts of each CVD superclass (regardless of coexisting CVDs)')
plt.xlabel('CVD Superclasses')
plt.ylabel('Total counts')
plt.legend(fig_bars, all_cvds, title='CVDs', loc='upper right')
 
# Adding % of total counts of each CVD regardless of coexisting CVDs in X (note: these %s will NOT add to 100 due to CVD coexistence!) 
for ii, bar in enumerate(fig_bars):
    bar_count = all_cvds_counts.iloc[0,ii]
    bar_prct_count = 100*(bar_count/X.shape[0])
    plt.text(bar.get_x() + bar.get_width() / 2, bar_count + 0.5, f'{bar_prct_count:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# Observations:
    # - There is a notable class imbalance in terms of total occurence of CVDs:
        # - NORM is the dominant class (present in 44.5% of ECG recordings).
        # - MI, STTC, and CD are quite similarly represented (present in 23-26% of ECG recordings).
        # - HYP is the minorty class (present in aprox 12% of ECG recordings).
    # - Recall, there is a significant proportion of these CVDs that coexist in the ECG recordings:
        # - 16244 observations (75.9%) are "pure" (one CVD superclass).
        # - 4068 observations (19.02%) have 2 coexisting CVDs.
        # - 919 observations (4.30%) have 3 coexisting CVDs.
        # - 157 observations (0.73%) have 4 coexisitng CVDs.

del ii, bar, fig_bars, bar_count, bar_prct_count

# % 5.3 Computing total counts of each target CVD superclass in pure ECG recordings only.
all_pure_cvds_counts = pd.DataFrame(data = [[0, 0, 0, 0, 0]], # Initialization of total counts DataFrame
                                    index = ['Total'], 
                                    columns= all_cvds)

# Identifying pure ECG recordings (single CVD superclass)
idxs_pure = [idx for idx, labels in y['scp_codes'].items() if len(labels) == 1]

# Going through each target CVD superclass label and counting occurence of each CVD superclass
for ii in y.loc[idxs_pure, :].index:
    for cvd in all_cvds:
        if cvd in y.loc[idxs_pure, 'scp_codes'][ii]:
            all_pure_cvds_counts[cvd] += 1
    
# Plotting pie chart of total counts of each target CVD superclass in pure EXG recordings only
plt.figure(figsize=(8, 8))

# Custom function to display both percentage and total count
def autopct_func(pct, all_vals):
    absolute = int(round(pct / 100. * sum(all_vals)))
    return f"{pct:.1f}%\n({absolute})"

# Plotting the pie chart
plt.pie(all_pure_cvds_counts.loc['Total', :], labels=all_pure_cvds_counts.columns, autopct=lambda pct: autopct_func(pct, all_pure_cvds_counts.loc['Total', :]))
plt.title('Total counts of each CVD superclass in pure ECG recordings only')
plt.show()

# Observations:
    # - There is an even more significant class imbalance in terms of occurence of pure CVDs:
        # - NORM is even more dominant (represents in 55.8% of pure ECG recordings).
        # - MI, STTC, and CD are less but still quite similarly represented (present in 10.5-15.6% of pure ECG recordings).
        # - HYP is the even more of a minorty (present in only 3.3% of pure ECG recordings).

del ii 

# % 5.4 Computing total count of all observed coexisting CVD combinations.
all_cvd_combs = y['scp_codes'].apply(lambda ii: frozenset(ii))  # frozenset() instead of set to make it mutable for subsquent Counter() call.
all_cvd_combs_counts = colls.Counter(all_cvd_combs)
all_cvd_combs_counts = {','.join(key) # Cleaning the keys of the combinations_counts dictionairy.
                        if isinstance(key, frozenset) else key: value 
                        for key, value in all_cvd_combs_counts.items()} 

# Plotting pie chart with all coexisting CVD combinations for each CVD superclass
for cvd in all_cvds:
    cvd_combs_counts = {key: value for key, value in all_cvd_combs_counts.items() if cvd in key}
    
    # Reordering keys in cvd_combs_counts dictionairy so current CVD is always the first CVD in the key
    cvd_combs_counts_ord = {}
    for cvd_comb, cvd_comb_count in cvd_combs_counts.items():
        coexisting_cvds = cvd_comb.split(',')        
        reordered_key = ','.join([cvd] + 
                                 sorted([c for c in coexisting_cvds if c != cvd], key=lambda x: all_cvds.index(x))) # Respecting the order of all_cvds.
        cvd_combs_counts_ord[reordered_key] = cvd_comb_count
      
   # Sort cvd_combinations_counts_ord in descending order of occurences of each CVD combination
    cvd_combs_counts_reord = sorted(cvd_combs_counts_ord.items(), key=lambda x: x[1], reverse=True)
    cvd_combs_reord, cvd_combs_counts_reord = zip(*cvd_combs_counts_reord) # Unpacking for plotting
    
    # Plotting
    plt.figure(figsize=(8, 8))
    plt.pie(cvd_combs_counts_reord, labels=cvd_combs_reord, autopct='%1.1f%%')
    plt.title(f'{cvd} CVD Combinations Proportions', fontsize=14)
    plt.show()

# Observations: 
    # - NORM: - 95% of the time occurs "purely" (no coexisting CVDs).
    #         - <5% it occurs along with CD.
    # - MI: - 46% of the time it occurs purely.
    #       - 24% of the time it occurs with CD.
    #       - 11% of the time it occurs with STTC.
    #       - 7% of the time it occurs with STTC AND HYP.
    # - STTC: - 46% of the time it occurs purely.
    #         - 15% of the time it occurs with HYP.
    #         - 11% of the time it occurs with MI.
    #         - 9% of the time it occurs with CD.
    #         - 7% of the time it occurs with MI and HYP.
    # - CD: - 35% of the time it occurs purely.
    #       - 27% of the time it occurs with MI.
    #       - 10% of the time it occurs with STTC.
    #       - 8% of the time it occurs with NORM.
    #       - 6% of the time it occurs with HYP.
    # - HYP: - 30% of the time it occurs with SSTC (its most frequent occurence is not pure!). 
    #        - 20% of the time it occurs purely.
    #        - 14% of the time it occurs with MI and STTC.
    #        - 11% of the time it occurs with CD.
    #        - 8% of the time it occurs with STTC.
    #        - 7% of the time it occurs with MI.
    #        - 6% of the time it occurs with MI/STTC/CD.
    # - The extent of CVD coexistence is generally very high. 
    #   -> For classification of mixed ECG recordings (as well as pure), 
    #      consider weighting training set observations based on number of coexisting CVDs in each observation 
    #      (in addition to general class imabalance weighting).
    #      E.g., a pure observation will have a weighting of 1, an observation with 3 coexisting CVDs will have a weighting of 1/3.
    # - As we progress towards minority classes, the extent of CVD coexistence increases. 
    #   This will likely make it even harder to classify mixed ECG recordings of the minority classes.
    
del all_cvd_combs, cvd, cvd_combs_counts, cvd_combs_counts_ord, cvd_comb, cvd_comb_count, coexisting_cvds, reordered_key, cvd_combs_reord, cvd_combs_counts_reord


# %% 6. Feature Engineering

# % 6.1 Handling categorical features
# Following the preliminary feature selection of %% 2., and %% 4 there is only 3 categorical features:
    # - sex.
    # - QRS_AxisIndet_Global.
    # - LVH_Strain_Global. 
# All 3 are already in numerical (boolean) form. Hence, no further categorical feature encoding is required.

# % 6.2 New feature creation (AV parity)

# Mismatch in atrial and ventricular rates can be indicative of AV node pathologies or atrial/ventricular arrhythmias.
X['HR_AV_Mismatch_Global'] = abs(X['HR_Atrial_Global'] - X['HR_Ventr_Global']) 

# % 6.3 Feature Selection
             
# 7.3.1 Selecting between all HR-related features
# The following features contain heart rate information and are likely strongly intercorrelated:
    # - QRS_Count_Global
    # - RR_Mean_Global
    # - HR_Atrial_Global
    # - HR_Sinus_Global
    # - HR_Ventr_Global
    # - HR__Global

# HR feature correlations
plt.figure(figsize=(10, 6))
sns.heatmap(X.loc[:, ['QRS_Count_Global', 
                      'RR_Mean_Global', 
                      'HR_Atrial_Global', 
                      'HR_Sinus_Global', 
                      'HR_Ventr_Global', 
                      'HR__Global']].corr())
plt.title('HR Feature Correlations')
plt.xlabel('HR Feature')
plt.ylabel('HR Feature')

# Observations:
    # - QRS_Count_Global has a very strong correlation with HR_Sinus_Global (r=0.98)
    #   Expected since both essentially convey the same information.
    #   -> Discard QRS_Count_Global since this is a less "universal" feature than HR_Sinus_Global.
    # - RR_Mean_Global has a very strong (negative) correlation with HR_Ventr_Global (-0.94) and HR__Global (-0.94).
    #   Expected since, similarly, both essentially convey the same information (heart rate [especially QRS-derived] being inversely proportional to RR interval). 
    #  -> Discard RR_Mean_Global since this is a less "universal" feature than HR_Ventr_Global.
    # - HR__Global has a very strong (negative) correlation with RR_Mean_Global (-0.94) and (positive) correlation with HR_Ventr_Global (1.00).
    #   Expected for same reasons as discussed for RR_Mean_Global.
    #   -> Discard HR__Global since its info is already captured by HR_Ventr_Global.
    
X.drop(['QRS_Count_Global', 'RR_Mean_Global', 'HR__Global'], 
        axis=1, 
        inplace=True)    

# 6.3.2 Selecting between all QT Int features
# The following features all represent different corrections of the QT interval and are likely strongly intercorrelated:
    # - QT_Int_Global
    # - QT_IntBazett_Global
    # - QT_IntFridericia_Global
    # - QT_IntFramingham_Global
    # - QT_IntHodge_Global
    # - QT_IntCorr_Global
# Only one QT Interval is necessary, the rest must be discarded.

# QTInt feature correlations
plt.figure(figsize=(10, 6))
sns.heatmap(X.loc[:, ['QT_Int_Global', 
                      'QT_IntBazett_Global', 
                      'QT_IntFridericia_Global', 
                      'QT_IntFramingham_Global', 
                      'QT_IntHodge_Global', 
                      'QT_IntCorr_Global']].corr())
plt.title('QT Int Feature Correlations')
plt.xlabel('QT Int Feature')
plt.ylabel('QT Int Feature')

# Observations:
    # - QT_Int_Global is not corrected for varying RR intervals.
    #   -> Discard.
    # - QT_IntBazett_Global ( = QT_Int/(RR**0.5)). Common in clinical settings but less accurate at extreme heart rates.
    #   -> Discard.
    # - QT_IntFridericia_Global ( = QT_Int/(RR**0.333)). More accurate than Bazett at extremes but less clinically used.
    #   -> Discard.
    # - QT_IntFramingham_Global ( = QT_Int + 0.154*(1-RR)). Robust corrections across wide range of heart rates and increasingly favored in research due to its accuracy.
    #   -> Keep!
    # - QT_IntHodge_Global ( = QT_Int + 1.75*(HR-60)). Useful for heart rates close to normal but less accurate at extremes. Also doesn't correlate well to other QT int features (suspicious).
    #   -> Discard.
    # - QT_IntCorr_Global - Precise correction method (or averaging method of other corrections) not specified. Also demonstrated noticeble difference between 12SL and Uni G values in %% 3.
    # -> Discard.
    
X.drop(['QT_Int_Global', 'QT_IntBazett_Global', 'QT_IntFridericia_Global', 'QT_IntHodge_Global', 'QT_IntCorr_Global'], 
       axis=1, 
       inplace=True)         

# 6.3.3 Selecting between all VM features

# The following 19 features convey different and highly granular vectorcardiography information:
    # - VM_AngFront_MaxQRS,
    # - VM_AngFront_MaxT,
    # - VM_AngRSag_MaxQRS
    # - VM_AngRSag_MaxT,
    # - VM_AngTrans_MaxQRS,
    # - VM_AngTrans_MaxT,
    # - VM_Ang_QRS-T,
    # - VM_LenFront_MaxQRS,
    # - VM_LenFront_MaxT, 
    # - VM_LenRSag_MaxQRS,
    # - VM_LenRSag_MaxT, 
    # - VM_LenTrans_MaxQRS,
    # - VM_LenTrans_MaxT,
    # - VM_Mag_MaxQRS,
    # - VM_Mag_MaxT, 
    # - VM_Pos_MaxQRS, 
    # - VM_Pos_MaxT,
    # - VM_Vel_MaxQRS,
    # - VM_Vel_MaxT
# Ideally, we could reduce the number of these granular vectorcardiography features and keep the most general ones.

# VM feature correlations
plt.figure(figsize=(10, 6))
sns.heatmap(X.loc[:, ['VM_AngFront_MaxQRS','VM_AngFront_MaxT', 
                      'VM_AngRSag_MaxQRS', 'VM_AngRSag_MaxT',
                      'VM_AngTrans_MaxQRS', 'VM_AngTrans_MaxT', 'VM_Ang_QRS-T',
                      'VM_LenFront_MaxQRS', 'VM_LenFront_MaxT', 
                      'VM_LenRSag_MaxQRS', 'VM_LenRSag_MaxT', 
                      'VM_LenTrans_MaxQRS', 'VM_LenTrans_MaxT',
                      'VM_Mag_MaxQRS', 'VM_Mag_MaxT', 
                      'VM_Pos_MaxQRS', 'VM_Pos_MaxT',
                      'VM_Vel_MaxQRS', 'VM_Vel_MaxT']].corr())
plt.title('VM Feature Correlations')
plt.xlabel('VM Feature')
plt.ylabel('VM Feature')

# Observations:
    # - The VM_Len... features are highly granular and redundant. 
    #   The projected length of the QRS and T complex/wave onto each cartesian plane can be derived from the angles and the vector magnitude features.
    #   -> Discard features.
    
X.drop(['VM_LenFront_MaxQRS', 'VM_LenFront_MaxT', 
        'VM_LenRSag_MaxQRS', 'VM_LenRSag_MaxT', 
        'VM_LenTrans_MaxQRS', 'VM_LenTrans_MaxT'], 
       axis=1, 
       inplace=True)    
    
# % 6.4 Distributions of selected features grouped by class for pure ECG recordings (prevent confounding distributions)
y_pure_labels = y.loc[idxs_pure, 'scp_codes']
y_pure_labels = y_pure_labels.apply(lambda x: x[0])  # Unpacking from list.

# Create a dictionary to map classes to colors
cvd_color_map = dict(zip(all_cvds, all_cvds_colors))

for col in X.columns:
    plt.figure(figsize=(12, 6))
    
    # Extracting current feature values of pure ECG recordings
    col_X_pure = X.loc[idxs_pure, col]

    # Left subplot: Boxplots
    plt.subplot(1, 2, 1)
    sns.boxplot(x=y_pure_labels, y=col_X_pure, palette=cvd_color_map)
    plt.title(f'Grouped box-whisker plots of {col} (pure obs only)')
    plt.xlabel('True Class')
    plt.ylabel(col)

    # Right subplot: Overlapped histograms
    plt.subplot(1, 2, 2)
    for cvd, color in zip(all_cvds, all_cvds_colors):
        cvd_col_X_pure = col_X_pure[y_pure_labels == cvd]
        sns.histplot(y=cvd_col_X_pure, kde=False, label=f'{cvd}', bins=50, alpha=0.5, color=color)
    plt.title(f'Grouped histogram of {col} (pure obs only)')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.legend(all_cvds)

    plt.tight_layout()
    plt.show()

# % 6.5 Handling numerical features (standardization or normlization)
# The idea is to first try tree-based models, which are invariant to feature scale.
# However, for visualization of CVD superclass clustering (e.g., using PCA or ICA), feature ranges must be comparable.
# All features demonstrated a non-normal distribution, hence standardization is not appropriate.
# -> Apply normalization.

# Initialization - Create a MinMaxScaler instance 
scaler = MinMaxScaler()

# Fit the scaler to the training/val data and transform it 
X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

del scaler


# %% 7. Visualizing train/val set class (i.e. CVD superclass clusters)
# This analysis is performed on train/val set to prevent data leakage from test set.

# Thought:
# It would be interesting to differentiate the pure ECG recordings (i.e., those with a single CVD superclass) from those with coexisting CVDs.
# One might expect the pure observations to exist closer to their CVD superclass centroid, and the mixed CVD observations to exist on the boundary regions and overlap with those of the other coexisting CVDs.

# % 7.1 Principal component analysis visualization

# De-mean?

# Isolating pure ECG recordings (single CVD superclass)
X_scaled_pure = X_scaled.loc[idxs_pure].dropna()  # Drop observations with NaNs
y_pure = y.loc[X_scaled_pure.index]

# Removing 'sex' feature - since its a boolean and quite 50/50 split in the data, it will dominate the variance and hence the PC1 loadings
X_scaled_pure.drop(columns=["sex"], inplace=True)

# Assign colormap disctionairy for vizualization
color_map = {cvd: color for cvd, color in zip(all_cvds, all_cvds_colors)}

# 3D PCA
pca = PCA(n_components=3, random_state=rand_state)
X_scaled_pure_pca = pca.fit_transform(X_scaled_pure)

# Extracting and ranking (absolute) loadings
loadings = pd.DataFrame(abs(pca.components_.T), columns=[f"PC{ii+1}" for ii in range(pca.n_components_)], index=X_scaled_pure.columns)
loadings_top_5 = {f"PC{ii+1}": loadings[f"PC{ii+1}"].abs().sort_values(ascending=False).head(5) for ii in range(pca.n_components_)}

# PC top 5 loadings:
for pc, sorted_features in loadings_top_5.items():
    print(f"\nTop 5 features for {pc} (sorted by absolute loading):")
    print(sorted_features)
    
# Plotting 3D PCA
fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(111, projection='3d')
for ii, (ii_cvd, ii_color) in enumerate(color_map.items()):
    indexes_ii_cvd_pure = [jj for jj, labels in enumerate(y_pure) if labels[0] == ii_cvd]
    ax.scatter(X_scaled_pure_pca[indexes_ii_cvd_pure, 0], 
               X_scaled_pure_pca[indexes_ii_cvd_pure, 1], 
               X_scaled_pure_pca[indexes_ii_cvd_pure, 2],
               c=ii_color, label=ii_cvd, alpha=0.4, marker='.')

ax.set_title("3D PCA Visualization of Pure ECG Recordings", fontsize=14)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")
ax.legend(title="CVD Superclass", loc='upper right', fontsize=10)
plt.show()

# Observations:
    # Not super clearly separable classes or distinct clusters:
        # - Large NORM cloud spanning most of the PC observation space.
        # - STTC and MI clouds can be somewhat discerned but with significant NORM contamination.
        # - CD and HYP seem to be distributed throughout the NORM cloud.
    # Variance does not seem to be a great separator of CVD superclasses.  
    
del idxs_pure, pca, X_scaled_pure_pca, pc, sorted_features, loadings, loadings_top_5, fig, ax, ii, ii_cvd, ii_color, indexes_ii_cvd_pure    

# % 7.2 Independant component analysis visualization
ica = FastICA(n_components=3, random_state=rand_state, max_iter=1000)
X_scaled_pure_ica = ica.fit_transform(X_scaled_pure)
    
# Plotting 3D ICA
fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(111, projection='3d')
for ii, (ii_cvd, ii_color) in enumerate(color_map.items()):
    indexes_ii_cvd_pure = [jj for jj, labels in enumerate(y_pure) if labels[0] == ii_cvd]
    ax.scatter(X_scaled_pure_ica[indexes_ii_cvd_pure, 0], 
               X_scaled_pure_ica[indexes_ii_cvd_pure, 1], 
               X_scaled_pure_ica[indexes_ii_cvd_pure, 2],
               c=ii_color, label=ii_cvd, alpha=0.4, marker='.')

ax.set_title("3D ICA Visualization of Pure ECG Recordings", fontsize=14)
ax.set_xlabel("IC 1")
ax.set_ylabel("IC 2")
ax.set_zlabel("IC 3")
ax.legend(title="CVD Superclass", loc='upper right', fontsize=10)
plt.show()

del ica, X_scaled_pure_ica, fig, ax, ii, ii_cvd, ii_color, indexes_ii_cvd_pure
    
# Observations:
    # Slightly more separable classes than PCA but still no distinct clusters:
        # Cloud of NORM observations has region that is distinct and somewhat compact, but a few NORM observations exist beyond it.
        # MI and STTC clouds exist mixed together in two main regions above and below the NORM cloud.
        # CD cloud permeates the entire mass and exists somewhat alone to the left of the NORM cloud.
    # Statistical independence seems a better separator of CVD superclasses than variance (as in PCA).
    # Perhahps non-linear methods (e.g., tSNE) will better separate the classes.
    
# % 7.3 tSNE visualization
tsne = TSNE(n_components=3, random_state=rand_state, perplexity=30, n_iter=1000, n_jobs=-1)
X_scaled_pure_tsne = tsne.fit_transform(X_scaled_pure)

# Plotting 3D t-SNE
fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(111, projection='3d')
for ii, (ii_cvd, ii_color) in enumerate(color_map.items()):
    indexes_ii_cvd_pure = [jj for jj, labels in enumerate(y_pure) if labels[0] == ii_cvd]
    ax.scatter(X_scaled_pure_tsne[indexes_ii_cvd_pure, 0], 
               X_scaled_pure_tsne[indexes_ii_cvd_pure, 1], 
               X_scaled_pure_tsne[indexes_ii_cvd_pure, 2],
               c=ii_color, label=ii_cvd, alpha=0.4, marker='.')

ax.set_title("3D t-SNE Visualization of Pure ECG Recordings", fontsize=14)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
ax.legend(title="CVD Superclass", loc='upper right', fontsize=10)
plt.show()

del tsne, X_scaled_pure_tsne, fig, ax, ii, (ii_cvd, ii_color), indexes_ii_cvd_pure

# The fact that CVD superclasses were not clearly separable into distinct clusters, even when using pure-label observations suggests that the mixed-label observations would be even harder to distinguish.


# %% 9. Saving relevant outputs
if flag_save:
    X.to_csv(os.path.join(os.getcwd(), 'X.csv'))
    y.to_csv(os.path.join(os.getcwd(), 'y.csv'))


# %% 10. Troubleshooting

# Displaying specific ECGs
ecg_id_troublshoot = 1012
extract_plot_ecg(ecg_id_troublshoot, y, data_path+'PTBXL/', True)