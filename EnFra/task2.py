# -*- coding: utf-8 -*-
"""
Created February 2025

@author: Krzysztof Raczynski
"""

import pandas as pd
import numpy as np
from scipy.ndimage import label
#from joblib import Parallel, delayed
from pyts.image import RecurrencePlot
from PyFra import *

# EDIT BLOCK
# ---- INPUT DATA ----
data_path = "[path to input data here]"
out_path = "[path to output data here]"
# ---- COLUMNS TO PROCESS FOR BATCHING ----
colstart = range(1430,1510,10)


# COMPUTE BLOCK 
data = pd.read_csv(data_path)
data.set_index('datetime', inplace=True, drop=True)
data.index = pd.to_datetime(data.index)

# Precompute period indexes once
index_D = pd.PeriodIndex(data.index, freq="D")
index_W = pd.PeriodIndex(data.index, freq="W")
index_M = pd.PeriodIndex(data.index, freq="M")
index_Q = pd.PeriodIndex(data.index, freq="Q")
index_A = pd.PeriodIndex(data.index, freq="Y")


def second_task(data):
    series = data.reshape(1,-1)
    rp = RecurrencePlot(threshold='point', percentage=20)
    X_rp = rp.fit_transform(series)[0]
    #xrps.append(X_rp)
    total_points = X_rp.shape[0] ** 2
    rec_points = np.sum(X_rp)
    RR = rec_points / total_points
    diag_lines = count_diagonal_lines(X_rp, min_length=2)
    DET = np.sum(diag_lines) / rec_points if rec_points > 0 else 0
    L_max = max(diag_lines) if diag_lines else 0
    ENTR = -np.sum((np.array(diag_lines)/np.sum(diag_lines)) * np.log(np.array(diag_lines)/np.sum(diag_lines))) if diag_lines else 0
    DIV = 1 / L_max if L_max > 0 else 0
    vert_lines = count_vertical_lines(X_rp, min_length=2)
    LAM = np.sum(vert_lines) / rec_points if rec_points > 0 else 0
    TT = np.mean(vert_lines) if vert_lines else 0
    
    return X_rp, RR, DET, ENTR, L_max, DIV, LAM, TT

def process_column(column, series):
    s = series.copy()
    s.replace(0, np.nan, inplace=True)
    missing_ratio = s.isna().mean()
    if missing_ratio > 0:
        if missing_ratio < 0.05:
            print(f"Filling {column}: {missing_ratio*100:.2f}% missing")
            s.interpolate(inplace=True)
            s.dropna(inplace=True)
        else:
            print(f"Skipping {column}: {missing_ratio*100:.2f}% missing")
            return None  # Skip column if too many missing values

    try:
        # Compute aggregated series once
        mean_D = normalize(s.groupby(index_D).mean().to_numpy())
        mean_W = normalize(s.groupby(index_W).mean().to_numpy())
        mean_M = normalize(s.groupby(index_M).mean().to_numpy())
        mean_Q = normalize(s.groupby(index_Q).mean().to_numpy())
        mean_A = normalize(s.groupby(index_A).mean().to_numpy())
        
        up_W = normalize(s.groupby(index_W).max().to_numpy())
        up_M = normalize(s.groupby(index_M).max().to_numpy())
        up_Q = normalize(s.groupby(index_Q).max().to_numpy())
        up_A = normalize(s.groupby(index_A).max().to_numpy())
        
        low_W = normalize(s.groupby(index_W).min().to_numpy())
        low_M = normalize(s.groupby(index_M).min().to_numpy())
        low_Q = normalize(s.groupby(index_Q).min().to_numpy())
        low_A = normalize(s.groupby(index_A).min().to_numpy())
    except:
        try:
            # Mean values and nomalize
            mean_D = normalize(series.groupby(pd.PeriodIndex(series.index, freq="D")).mean().to_numpy())
            mean_W = normalize(series.groupby(pd.PeriodIndex(series.index, freq="W")).mean().to_numpy())
            mean_M = normalize(series.groupby(series.index, freq="M").mean().to_numpy())
            mean_Q = normalize(series.groupby(pd.PeriodIndex(series.index, freq="Q")).mean().to_numpy())
            mean_A = normalize(series.groupby(pd.PeriodIndex(series.index, freq="A")).mean().to_numpy())
            # Max values and nomalize
            up_W = normalize(series.groupby(pd.PeriodIndex(series.index, freq="W")).max().to_numpy())
            up_M = normalize(series.groupby(series.index, freq="M").max().to_numpy())
            up_Q = normalize(series.groupby(pd.PeriodIndex(series.index, freq="Q")).max().to_numpy())
            up_A = normalize(series.groupby(pd.PeriodIndex(series.index, freq="A")).max().to_numpy())
            # Min values and nomalize
            low_W = normalize(series.groupby(pd.PeriodIndex(series.index, freq="W")).min().to_numpy())
            low_M = normalize(series.groupby(series.index, freq="M").min().to_numpy())
            low_Q = normalize(series.groupby(pd.PeriodIndex(series.index, freq="Q")).min().to_numpy())
            low_A = normalize(series.groupby(pd.PeriodIndex(series.index, freq="A")).min().to_numpy())
        except:
            return None
    
    # Order of aggregated series (same as in your original code):
    series_list = [low_W, low_M, low_Q, low_A,
                   mean_D, mean_W, mean_M, mean_Q, mean_A,
                   up_W, up_M, up_Q, up_A]
    #start_time = time.time()
    try:
        # Parallelize RS, DFA, MF_DFA computations (each returns one value per series)
        rests = [second_task(sitem) for sitem in series_list]
    except:
        return None
        
    #print("Finished in: " + str((time.time() - start_time)/60) + " min")
    
    s_low_W, s_low_M, s_low_Q, s_low_A = rests[0], rests[1],rests[2],rests[3]
    s_mean_D, s_mean_W, s_mean_M, s_mean_Q, s_mean_A = rests[4],rests[5],rests[6],rests[7],rests[8]
    s_up_W, s_up_M, s_up_Q, s_up_A = rests[9],rests[10],rests[11],rests[12]
    
    return {
        'column': column,
        's_low_W': s_low_W, 
        's_low_M': s_low_M, 
        's_low_Q': s_low_Q, 
        's_low_A': s_low_A ,
        's_mean_D': s_mean_D, 
        's_mean_W': s_mean_W, 
        's_mean_M': s_mean_M, 
        's_mean_Q': s_mean_Q, 
        's_mean_A': s_mean_A ,
        's_up_W': s_up_W, 
        's_up_M': s_up_M, 
        's_up_Q': s_up_Q, 
        's_up_A': s_up_A 
    }


for start in colstart:
    print("Processing batch starting at column:", start)
    end = start + 10
    # Slice the data (if start exceeds available columns, adjust slicing)
    if start == 3280:
        data_slice = data.iloc[:, start:] if start < data.shape[1] else pd.DataFrame()
    else:
        data_slice = data.iloc[:, start:end] if start < data.shape[1] else pd.DataFrame()
    if data_slice.empty:
        continue
    # Use parallel processing for all columns in the slice
    results = [process_column(col, data_slice[col]) for col in data_slice.columns]
    # Filter out any columns that were skipped (returned None)
    results = [res for res in results if res is not None]

    
    # Initialize DataFrames for each group with the same column names as before
    RQA_low_W = pd.DataFrame(columns=['X_rp', 'RR', 'DET', 'ENTR', 'L_max', 'DIV', 'LAM', 'TT'])
    RQA_low_M = pd.DataFrame(columns=RQA_low_W.columns)
    RQA_low_Q = pd.DataFrame(columns=RQA_low_W.columns)
    RQA_low_A = pd.DataFrame(columns=RQA_low_W.columns)
    RQA_mean_D = pd.DataFrame(columns=RQA_low_W.columns) 
    RQA_mean_W = pd.DataFrame(columns=RQA_low_W.columns) 
    RQA_mean_M = pd.DataFrame(columns=RQA_low_W.columns) 
    RQA_mean_Q = pd.DataFrame(columns=RQA_low_W.columns) 
    RQA_mean_A = pd.DataFrame(columns=RQA_low_W.columns)
    RQA_up_W = pd.DataFrame(columns=RQA_low_W.columns)
    RQA_up_M = pd.DataFrame(columns=RQA_low_W.columns) 
    RQA_up_Q = pd.DataFrame(columns=RQA_low_W.columns)
    RQA_up_A = pd.DataFrame(columns=RQA_low_W.columns)

    # Assemble the results from each processed column into the DataFrames.
    for res in results:
        colname = res['column']
        RQA_low_W.loc[colname] = res['s_low_W']
        RQA_low_M.loc[colname] = res['s_low_M']
        RQA_low_Q.loc[colname] = res['s_low_Q']
        RQA_low_A.loc[colname] = res['s_low_A']
        RQA_mean_D.loc[colname] = res['s_mean_D']
        RQA_mean_W.loc[colname] = res['s_mean_W']
        RQA_mean_M.loc[colname] = res['s_mean_M']
        RQA_mean_Q.loc[colname] = res['s_mean_Q']
        RQA_mean_A.loc[colname] = res['s_mean_A']
        RQA_up_W.loc[colname] = res['s_up_W']
        RQA_up_M.loc[colname] = res['s_up_M']
        RQA_up_Q.loc[colname] = res['s_up_Q']
        RQA_up_A.loc[colname] = res['s_up_A']

    print(f'Finished processing columns {start} to {end}')
    
    # Optionally, write your DataFrames to CSV files   
    RQA_low_W.to_csv(out_path + 'RQA_low_W_' + str(end) + '.csv')
    RQA_low_M.to_csv(out_path + 'RQA_low_M_' + str(end) + '.csv')
    RQA_low_Q.to_csv(out_path + 'RQA_low_Q_' + str(end) + '.csv')
    RQA_low_A.to_csv(out_path + 'RQA_low_A_' + str(end) + '.csv')
    RQA_mean_D.to_csv(out_path + 'RQA_mean_D_' + str(end) + '.csv')
    RQA_mean_W.to_csv(out_path + 'RQA_mean_W_' + str(end) + '.csv')
    RQA_mean_M.to_csv(out_path + 'RQA_mean_M_' + str(end) + '.csv')
    RQA_mean_Q.to_csv(out_path + 'RQA_mean_Q_' + str(end) + '.csv')
    RQA_mean_A.to_csv(out_path + 'RQA_mean_A_' + str(end) + '.csv')
    RQA_up_W.to_csv(out_path + 'RQA_up_W_' + str(end) + '.csv')
    RQA_up_M.to_csv(out_path + 'RQA_up_M_' + str(end) + '.csv')
    RQA_up_Q.to_csv(out_path + 'RQA_up_Q_' + str(end) + '.csv')
    RQA_up_A.to_csv(out_path + 'RQA_up_A_' + str(end) + '.csv')