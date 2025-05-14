# -*- coding: utf-8 -*-
"""
Created February 2025

@author: Krzysztof Raczynski
"""

import pandas as pd
import time
import numpy as np
from PyFra import *
import nolds
import warnings
warnings.filterwarnings("ignore")

# EDIT BLOCK
# ---- INPUT DATA ----
path = "path_to_data"
output_path = "output_path"
# ---- COLUMNS TO PROCESS FOR BATCHING ----
colstart = range(1430,1510,10)


# COMPUTE BLOCK 
data = pd.read_csv(path)
data.set_index('datetime', inplace=True, drop=True)

def computation(series):
    # Mean values and nomalize
    mean_D = np.array(series.groupby(pd.PeriodIndex(series.index, freq="D")).mean())
    mean_D = (mean_D - np.mean(mean_D)) / np.std(mean_D)
    mean_W = np.array(series.groupby(pd.PeriodIndex(series.index, freq="W")).mean())
    mean_W = (mean_W - np.mean(mean_W)) / np.std(mean_W)
    mean_M = np.array(series.groupby(pd.PeriodIndex(series.index, freq="M")).mean())
    mean_M = (mean_M - np.mean(mean_M)) / np.std(mean_M)
    mean_Q = np.array(series.groupby(pd.PeriodIndex(series.index, freq="Q")).mean())
    mean_Q = (mean_Q - np.mean(mean_Q)) / np.std(mean_Q)
    mean_A = np.array(series.groupby(pd.PeriodIndex(series.index, freq="A")).mean())
    mean_A = (mean_A - np.mean(mean_A)) / np.std(mean_A)
    # Max values and nomalize
    up_W = np.array(series.groupby(pd.PeriodIndex(series.index, freq="W")).max())
    up_W = (up_W - np.mean(up_W)) / np.std(up_W)
    up_M = np.array(series.groupby(pd.PeriodIndex(series.index, freq="M")).max())
    up_M = (up_M - np.mean(up_M)) / np.std(up_M)
    up_Q = np.array(series.groupby(pd.PeriodIndex(series.index, freq="Q")).max())
    up_Q = (up_Q - np.mean(up_Q)) / np.std(up_Q)
    up_A = np.array(series.groupby(pd.PeriodIndex(series.index, freq="A")).max())
    up_A = (up_A - np.mean(up_A)) / np.std(up_A)
    # Min values and nomalize
    low_W = np.array(series.groupby(pd.PeriodIndex(series.index, freq="W")).min())
    low_W = (low_W - np.mean(low_W)) / np.std(low_W)
    low_M = np.array(series.groupby(pd.PeriodIndex(series.index, freq="M")).min())
    low_M = (low_M - np.mean(low_M)) / np.std(low_M)
    low_Q = np.array(series.groupby(pd.PeriodIndex(series.index, freq="Q")).min())
    low_Q = (low_Q - np.mean(low_Q)) / np.std(low_Q)
    low_A = np.array(series.groupby(pd.PeriodIndex(series.index, freq="A")).min())  
    low_A = (low_A - np.mean(low_A)) / np.std(low_A)
    # -- RS analysis --
    rs = [RS_analysis(low_W),RS_analysis(low_M),RS_analysis(low_Q),
                      RS_analysis(low_A),RS_analysis(mean_D),RS_analysis(mean_W),
                      RS_analysis(mean_M),RS_analysis(mean_Q),RS_analysis(mean_A),
                      RS_analysis(up_W),RS_analysis(up_M),RS_analysis(up_Q),
                      RS_analysis(up_A)]
    
    # -- DFA analysis
    dfa = [DFA(low_W),DFA(low_M),DFA(low_Q),DFA(low_A),DFA(mean_D),
                       DFA(mean_W),DFA(mean_M),DFA(mean_Q),DFA(mean_A),DFA(up_W),
                       DFA(up_M),DFA(up_Q),DFA(up_A)]
    
    # -- MF DFA
    mfdfa = [MF_DFA(low_W),MF_DFA(low_M),MF_DFA(low_Q),
                      MF_DFA(low_A),MF_DFA(mean_D),MF_DFA(mean_W),
                      MF_DFA(mean_M),MF_DFA(mean_Q),MF_DFA(mean_A),
                      MF_DFA(up_W),MF_DFA(up_M),MF_DFA(up_Q),
                      MF_DFA(up_A)]
  
    max_scales = []
    max_modmax = []
    max_slopes = []
    max_coeffs = []
    mean_scales = []
    mean_modmax = []
    mean_slopes = []
    mean_coeffs = []
    norm_scales = []
    norm_modmax = []
    norm_slopes = []
    norm_coeffs = []   
    for i in [low_W, low_M, low_Q, low_A, mean_D, mean_W, mean_M, mean_Q, mean_A,
              up_W, up_M, up_Q, up_A]:
        max10_scales, max10_modmax = _WTMM(i, 'max', 'cmor0.5-1.0')
        max10_slope, max10_coeffs = WTMM(i, 'max', 'cmor0.5-1.0')
        mean10_scales, mean10_modmax = _WTMM(i, 'mean', 'cmor0.5-1.0')
        mean10_slope, mean10_coeffs = WTMM(i, 'mean', 'cmor0.5-1.0')
        norm10_scales, norm10_modmax = _WTMM(i, 'norm', 'cmor0.5-1.0')
        norm10_slope, norm10_coeffs = WTMM(i, 'norm', 'cmor0.5-1.0')
        max15_scales, max15_modmax = _WTMM(i, 'max', 'cmor1.0-1.5')
        max15_slope, max15_coeffs = WTMM(i, 'max', 'cmor1.0-1.5')
        mean15_scales, mean15_modmax = _WTMM(i, 'mean', 'cmor1.0-1.5')
        mean15_slope, mean15_coeffs = WTMM(i, 'mean', 'cmor1.0-1.5')
        norm15_scales, norm15_modmax = _WTMM(i, 'norm', 'cmor1.0-1.5')
        norm15_slope, norm15_coeffs = WTMM(i, 'norm', 'cmor1.0-1.5')
        max20_scales, max20_modmax = _WTMM(i, 'max', 'cmor1.5-2.0')
        max20_slope, max20_coeffs = WTMM(i, 'max', 'cmor1.5-2.0')
        mean20_scales, mean20_modmax = _WTMM(i, 'mean', 'cmor1.5-2.0')
        mean20_slope, mean20_coeffs = WTMM(i, 'mean', 'cmor1.5-2.0')
        norm20_scales, norm20_modmax = _WTMM(i, 'norm', 'cmor1.5-2.0')
        norm20_slope, norm20_coeffs = WTMM(i, 'norm', 'cmor1.5-2.0')
        
        max_scales.append(max10_scales)
        max_scales.append(max15_scales)
        max_scales.append(max20_scales)
        max_modmax.append(max10_modmax)
        max_modmax.append(max15_modmax)
        max_modmax.append(max20_modmax)
        max_slopes.append(max10_slope)
        max_slopes.append(max15_slope)
        max_slopes.append(max20_slope)
        max_coeffs.append(max10_coeffs)
        max_coeffs.append(max15_coeffs)
        max_coeffs.append(max20_coeffs)
        mean_scales.append(mean10_scales)
        mean_scales.append(mean15_scales)
        mean_scales.append(mean20_scales)
        mean_modmax.append(mean10_modmax)
        mean_modmax.append(mean15_modmax)
        mean_modmax.append(mean20_modmax)
        mean_slopes.append(mean10_slope)
        mean_slopes.append(mean15_slope)
        mean_slopes.append(mean20_slope)
        mean_coeffs.append(mean10_coeffs)
        mean_coeffs.append(mean15_coeffs)
        mean_coeffs.append(mean20_coeffs)
        norm_scales.append(norm10_scales)
        norm_scales.append(norm15_scales)
        norm_scales.append(norm20_scales)
        norm_modmax.append(norm10_modmax)
        norm_modmax.append(norm15_modmax)
        norm_modmax.append(norm20_modmax)
        norm_slopes.append(norm10_slope)
        norm_slopes.append(norm15_slope)
        norm_slopes.append(norm20_slope)
        norm_coeffs.append(norm10_coeffs)
        norm_coeffs.append(norm15_coeffs)
        norm_coeffs.append(norm20_coeffs)
    
    gHs = []
    for i in [low_W, low_M, low_Q, low_A, mean_D, mean_W, mean_M, mean_Q, mean_A,
              up_W, up_M, up_Q, up_A]:
        qv = list(range(1,int(np.ceil(len(i)/4))))
        gH = nolds.mfhurst_b(i, qvals=qv)
        gHs.append(gH)
    
    sampEns = []
    for i in [low_W, low_M, low_Q, low_A, mean_D, mean_W, mean_M, mean_Q, mean_A,
              up_W, up_M, up_Q, up_A]:
        sampEn = nolds.sampen(i)
        sampEns.append(sampEn)
    
    return rs, dfa, mfdfa, max_slopes,mean_slopes,norm_slopes,max_scales,mean_scales,norm_scales,max_modmax,mean_modmax,norm_modmax,max_coeffs,mean_coeffs,norm_coeffs,gHs,sampEns
    

for start in colstart:
    print("Initializing " + str(start))
    end = start + 100
    counter = 0
    rs = pd.DataFrame(columns=['l_W', 'l_M', 'l_Q', 'l_A', 'm_D', 'm_W', 'm_M', 'm_Q', 'm_A', 'u_W', 'u_M', 'u_Q', 'u_A'])
    dfa = pd.DataFrame(columns=['l_W', 'l_M', 'l_Q', 'l_A', 'm_D', 'm_W', 'm_M', 'm_Q', 'm_A', 'u_W', 'u_M', 'u_Q', 'u_A'])
    mfdfa = pd.DataFrame(columns=['l_W', 'l_M', 'l_Q', 'l_A', 'm_D', 'm_W', 'm_M', 'm_Q', 'm_A', 'u_W', 'u_M', 'u_Q', 'u_A'])
    genH = pd.DataFrame(columns=['l_W', 'l_M', 'l_Q', 'l_A', 'm_D', 'm_W', 'm_M', 'm_Q', 'm_A', 'u_W', 'u_M', 'u_Q', 'u_A'])
    sampEnt = pd.DataFrame(columns=['l_W', 'l_M', 'l_Q', 'l_A', 'm_D', 'm_W', 'm_M', 'm_Q', 'm_A', 'u_W', 'u_M', 'u_Q', 'u_A'])
    wtmm_slopes_max = pd.DataFrame(columns = [ 'l_W_10', 'l_W_15', 'l_W_20', 'l_M_10', 'l_M_15', 'l_M_20',  'l_Q_10', 'l_Q_15', 'l_Q_20', 'l_A_10', 'l_A_15', 'l_A_20',  'm_D_10', 'm_D_15', 'm_D_20', 'm_W_10', 'm_W_15', 'm_W_20',  'm_M_10', 'm_M_15', 'm_M_20', 'm_Q_10', 'm_Q_15', 'm_Q_20', 'm_A_10', 'm_A_15', 'm_A_20', 'u_W_10', 'u_W_15', 'u_W_20',  'u_M_10', 'u_M_15', 'u_M_20', 'u_Q_10', 'u_Q_15', 'u_Q_20',  'u_A_10', 'u_A_15', 'u_A_20'])
    wtmm_slopes_mean = pd.DataFrame(columns = ['l_W_10', 'l_W_15', 'l_W_20', 'l_M_10', 'l_M_15', 'l_M_20',  'l_Q_10', 'l_Q_15', 'l_Q_20', 'l_A_10', 'l_A_15', 'l_A_20',  'm_D_10', 'm_D_15', 'm_D_20', 'm_W_10', 'm_W_15', 'm_W_20',  'm_M_10', 'm_M_15', 'm_M_20', 'm_Q_10', 'm_Q_15', 'm_Q_20', 'm_A_10', 'm_A_15', 'm_A_20', 'u_W_10', 'u_W_15', 'u_W_20',  'u_M_10', 'u_M_15', 'u_M_20', 'u_Q_10', 'u_Q_15', 'u_Q_20',  'u_A_10', 'u_A_15', 'u_A_20'])
    wtmm_slopes_norm = pd.DataFrame(columns = [ 'l_W_10', 'l_W_15', 'l_W_20', 'l_M_10', 'l_M_15', 'l_M_20',  'l_Q_10', 'l_Q_15', 'l_Q_20', 'l_A_10', 'l_A_15', 'l_A_20',  'm_D_10', 'm_D_15', 'm_D_20', 'm_W_10', 'm_W_15', 'm_W_20',  'm_M_10', 'm_M_15', 'm_M_20', 'm_Q_10', 'm_Q_15', 'm_Q_20', 'm_A_10', 'm_A_15', 'm_A_20', 'u_W_10', 'u_W_15', 'u_W_20',  'u_M_10', 'u_M_15', 'u_M_20', 'u_Q_10', 'u_Q_15', 'u_Q_20',  'u_A_10', 'u_A_15', 'u_A_20'])
    wtmm_scales_max = pd.DataFrame(columns = [ 'l_W_10', 'l_W_15', 'l_W_20', 'l_M_10', 'l_M_15', 'l_M_20',  'l_Q_10', 'l_Q_15', 'l_Q_20', 'l_A_10', 'l_A_15', 'l_A_20',  'm_D_10', 'm_D_15', 'm_D_20', 'm_W_10', 'm_W_15', 'm_W_20',  'm_M_10', 'm_M_15', 'm_M_20', 'm_Q_10', 'm_Q_15', 'm_Q_20', 'm_A_10', 'm_A_15', 'm_A_20', 'u_W_10', 'u_W_15', 'u_W_20',  'u_M_10', 'u_M_15', 'u_M_20', 'u_Q_10', 'u_Q_15', 'u_Q_20',  'u_A_10', 'u_A_15', 'u_A_20'])
    wtmm_scales_mean = pd.DataFrame(columns = [ 'l_W_10', 'l_W_15', 'l_W_20', 'l_M_10', 'l_M_15', 'l_M_20',  'l_Q_10', 'l_Q_15', 'l_Q_20', 'l_A_10', 'l_A_15', 'l_A_20',  'm_D_10', 'm_D_15', 'm_D_20', 'm_W_10', 'm_W_15', 'm_W_20',  'm_M_10', 'm_M_15', 'm_M_20', 'm_Q_10', 'm_Q_15', 'm_Q_20', 'm_A_10', 'm_A_15', 'm_A_20', 'u_W_10', 'u_W_15', 'u_W_20',  'u_M_10', 'u_M_15', 'u_M_20', 'u_Q_10', 'u_Q_15', 'u_Q_20',  'u_A_10', 'u_A_15', 'u_A_20'])
    wtmm_scales_norm = pd.DataFrame(columns = [ 'l_W_10', 'l_W_15', 'l_W_20', 'l_M_10', 'l_M_15', 'l_M_20',  'l_Q_10', 'l_Q_15', 'l_Q_20', 'l_A_10', 'l_A_15', 'l_A_20',  'm_D_10', 'm_D_15', 'm_D_20', 'm_W_10', 'm_W_15', 'm_W_20',  'm_M_10', 'm_M_15', 'm_M_20', 'm_Q_10', 'm_Q_15', 'm_Q_20', 'm_A_10', 'm_A_15', 'm_A_20', 'u_W_10', 'u_W_15', 'u_W_20',  'u_M_10', 'u_M_15', 'u_M_20', 'u_Q_10', 'u_Q_15', 'u_Q_20',  'u_A_10', 'u_A_15', 'u_A_20'])
    wtmm_modmax_max = pd.DataFrame(columns = [ 'l_W_10', 'l_W_15', 'l_W_20', 'l_M_10', 'l_M_15', 'l_M_20',  'l_Q_10', 'l_Q_15', 'l_Q_20', 'l_A_10', 'l_A_15', 'l_A_20',  'm_D_10', 'm_D_15', 'm_D_20', 'm_W_10', 'm_W_15', 'm_W_20',  'm_M_10', 'm_M_15', 'm_M_20', 'm_Q_10', 'm_Q_15', 'm_Q_20', 'm_A_10', 'm_A_15', 'm_A_20', 'u_W_10', 'u_W_15', 'u_W_20',  'u_M_10', 'u_M_15', 'u_M_20', 'u_Q_10', 'u_Q_15', 'u_Q_20',  'u_A_10', 'u_A_15', 'u_A_20'])
    wtmm_modmax_mean = pd.DataFrame(columns = [ 'l_W_10', 'l_W_15', 'l_W_20', 'l_M_10', 'l_M_15', 'l_M_20',  'l_Q_10', 'l_Q_15', 'l_Q_20', 'l_A_10', 'l_A_15', 'l_A_20',  'm_D_10', 'm_D_15', 'm_D_20', 'm_W_10', 'm_W_15', 'm_W_20',  'm_M_10', 'm_M_15', 'm_M_20', 'm_Q_10', 'm_Q_15', 'm_Q_20', 'm_A_10', 'm_A_15', 'm_A_20', 'u_W_10', 'u_W_15', 'u_W_20',  'u_M_10', 'u_M_15', 'u_M_20', 'u_Q_10', 'u_Q_15', 'u_Q_20',  'u_A_10', 'u_A_15', 'u_A_20'])
    wtmm_modmax_norm = pd.DataFrame(columns = [ 'l_W_10', 'l_W_15', 'l_W_20', 'l_M_10', 'l_M_15', 'l_M_20',  'l_Q_10', 'l_Q_15', 'l_Q_20', 'l_A_10', 'l_A_15', 'l_A_20',  'm_D_10', 'm_D_15', 'm_D_20', 'm_W_10', 'm_W_15', 'm_W_20',  'm_M_10', 'm_M_15', 'm_M_20', 'm_Q_10', 'm_Q_15', 'm_Q_20', 'm_A_10', 'm_A_15', 'm_A_20', 'u_W_10', 'u_W_15', 'u_W_20',  'u_M_10', 'u_M_15', 'u_M_20', 'u_Q_10', 'u_Q_15', 'u_Q_20',  'u_A_10', 'u_A_15', 'u_A_20'])
    wtmm_coeffs_max = pd.DataFrame(columns = [ 'l_W_10', 'l_W_15', 'l_W_20', 'l_M_10', 'l_M_15', 'l_M_20',  'l_Q_10', 'l_Q_15', 'l_Q_20', 'l_A_10', 'l_A_15', 'l_A_20',  'm_D_10', 'm_D_15', 'm_D_20', 'm_W_10', 'm_W_15', 'm_W_20',  'm_M_10', 'm_M_15', 'm_M_20', 'm_Q_10', 'm_Q_15', 'm_Q_20', 'm_A_10', 'm_A_15', 'm_A_20', 'u_W_10', 'u_W_15', 'u_W_20',  'u_M_10', 'u_M_15', 'u_M_20', 'u_Q_10', 'u_Q_15', 'u_Q_20',  'u_A_10', 'u_A_15', 'u_A_20'])
    wtmm_coeffs_mean = pd.DataFrame(columns = [ 'l_W_10', 'l_W_15', 'l_W_20', 'l_M_10', 'l_M_15', 'l_M_20',  'l_Q_10', 'l_Q_15', 'l_Q_20', 'l_A_10', 'l_A_15', 'l_A_20',  'm_D_10', 'm_D_15', 'm_D_20', 'm_W_10', 'm_W_15', 'm_W_20',  'm_M_10', 'm_M_15', 'm_M_20', 'm_Q_10', 'm_Q_15', 'm_Q_20', 'm_A_10', 'm_A_15', 'm_A_20', 'u_W_10', 'u_W_15', 'u_W_20',  'u_M_10', 'u_M_15', 'u_M_20', 'u_Q_10', 'u_Q_15', 'u_Q_20',  'u_A_10', 'u_A_15', 'u_A_20'])
    wtmm_coeffs_norm = pd.DataFrame(columns = [ 'l_W_10', 'l_W_15', 'l_W_20', 'l_M_10', 'l_M_15', 'l_M_20',  'l_Q_10', 'l_Q_15', 'l_Q_20', 'l_A_10', 'l_A_15', 'l_A_20',  'm_D_10', 'm_D_15', 'm_D_20', 'm_W_10', 'm_W_15', 'm_W_20',  'm_M_10', 'm_M_15', 'm_M_20', 'm_Q_10', 'm_Q_15', 'm_Q_20', 'm_A_10', 'm_A_15', 'm_A_20', 'u_W_10', 'u_W_15', 'u_W_20',  'u_M_10', 'u_M_15', 'u_M_20', 'u_Q_10', 'u_Q_15', 'u_Q_20',  'u_A_10', 'u_A_15', 'u_A_20'])
    if start <= 3100:
        data_slice = data.iloc[:,start:end]
    else:
        data_slice = data.iloc[:,start:]
        
        
    for column in data_slice.columns:
        temp_data = data_slice[column]
        temp_data.replace(0, np.nan, inplace=True)
        missing_ratio = temp_data.isna().sum() / len(temp_data)
        
        if missing_ratio > 0:
            if missing_ratio < 0.05:
                print('Filling ' + column + "'s " + str(round(missing_ratio*100,2)) + '% missing values')
                temp_data.interpolate(inplace=True)
                temp_data = temp_data.dropna()
            else:
                print('Skipping ' + column + ' due to ' + str(round(missing_ratio*100,2)) + '% missing values')
                continue
        else:
            print("--- Computing: " + column)
    
        start_time = time.time()
        
        _rs, _dfa, _mfdfa, _max_slopes, _mean_slopes, _norm_slopes, _max_scales, _mean_scales, _norm_scales, _max_modmax, _mean_modmax, _norm_modmax, _max_coeffs, _mean_coeffs, _norm_coeffs, _gHs, _sampEns = computation(temp_data)
        
        rs.loc[column] = _rs
        dfa.loc[column] = _dfa
        mfdfa.loc[column] = _mfdfa
        wtmm_slopes_max.loc[column] = _max_slopes
        wtmm_slopes_mean.loc[column] = _mean_slopes
        wtmm_slopes_norm.loc[column] = _norm_slopes
        wtmm_scales_max.loc[column] = _max_scales
        wtmm_scales_mean.loc[column] = _mean_scales
        wtmm_scales_norm.loc[column] = _norm_scales
        wtmm_modmax_max.loc[column] = _max_modmax
        wtmm_modmax_mean.loc[column] = _mean_modmax
        wtmm_modmax_norm.loc[column] = _norm_modmax
        wtmm_coeffs_max.loc[column] = _max_coeffs
        wtmm_coeffs_mean.loc[column] = _mean_coeffs
        wtmm_coeffs_norm.loc[column] = _norm_coeffs
        genH.loc[column] = _gHs
        sampEnt.loc[column] = _sampEns
        
        counter += 1
        print("Finished " + column + " in: " + str((time.time() - start_time)/60) + " min")
        print("Currently finished: " + str(counter) + " ---")
           
    rs.to_csv(output_path + 'rs_' + str(end) + '.csv')
    dfa.to_csv(output_path + 'dfa_' + str(end) + '.csv')
    mfdfa.to_csv(output_path + 'mfdfa_' + str(end) + '.csv')
    genH.to_csv(output_path + 'genH_' + str(end) + '.csv')
    sampEnt.to_csv(output_path + 'sampEnt_' + str(end) + '.csv')
    wtmm_slopes_max.to_csv(output_path + 'wtmm_slopes_max_' + str(end) + '.csv')
    wtmm_slopes_mean.to_csv(output_path + 'wtmm_slopes_mean_' + str(end) + '.csv')
    wtmm_slopes_norm.to_csv(output_path + 'wtmm_slopes_norm_' + str(end) + '.csv')
    wtmm_scales_max.to_csv(output_path + 'wtmm_scales_max_' + str(end) + '.csv')
    wtmm_scales_mean.to_csv(output_path + 'wtmm_scales_mean_' + str(end) + '.csv')
    wtmm_scales_norm.to_csv(output_path + 'wtmm_scales_norm_' + str(end) + '.csv')
    wtmm_modmax_max.to_csv(output_path + 'wtmm_modmax_max_' + str(end) + '.csv')
    wtmm_modmax_mean.to_csv(output_path + 'wtmm_modmax_mean_' + str(end) + '.csv')
    wtmm_modmax_norm.to_csv(output_path + 'wtmm_modmax_norm_' + str(end) + '.csv')
    wtmm_coeffs_max.to_csv(output_path + 'wtmm_coeffs_max_' + str(end) + '.csv')
    wtmm_coeffs_mean.to_csv(output_path + 'wtmm_coeffs_mean_' + str(end) + '.csv')
    wtmm_coeffs_norm.to_csv(output_path + 'wtmm_coeffs_norm_' + str(end) + '.csv')