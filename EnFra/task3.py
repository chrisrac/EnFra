# -*- coding: utf-8 -*-
"""
Created February 2025

@author: Krzysztof Raczynski
"""

import pandas as pd
import time
import numpy as np
import nolds
from nolitsa import dimension
from joblib import Parallel, delayed
from PyFra import *
import warnings
warnings.filterwarnings("ignore")

# EDIT BLOCK
# ---- INPUT DATA ----
data_path = "[path to input data here]"
out_path = "[path to output data here]"
# ---- COLUMNS TO PROCESS FOR BATCHING ----
colstart = range(0,1000,100)

# COMPUTE BLOCK 
data = pd.read_csv(data_path)
data.set_index('datetime', inplace=True, drop=True)
data.index = pd.to_datetime(data.index)  # ensure index is datetime
# --- Precompute PeriodIndex Objects ---
index_D = pd.PeriodIndex(data.index, freq="D")
index_W = pd.PeriodIndex(data.index, freq="W")
index_M = pd.PeriodIndex(data.index, freq="M")
index_Q = pd.PeriodIndex(data.index, freq="Q")
index_A = pd.PeriodIndex(data.index, freq="A")
# --- Define Helper Functions ---

def compute_series(series):
    """
    Compute the false nearest neighbors (FNN) based embedding dimension and corresponding 
    matrix dimension, then calculate the Lyapunov exponent.
    """
    dims = np.arange(1, 51)
    #p = psutil.Process()
    #p.cpu_affinity([0,1])
    try:
        # Compute the percentage of false nearest neighbors for a range of dimensions.
        fnn_percentages = dimension.fnn(series, dim=dims)
        # Find the first dimension (offset by 1)
        dim_n = np.where(fnn_percentages == 0)[1][0] + 1
        matrix_candidates = valid_matrix_dims(dim_n)
        dim_m = max(matrix_candidates)
        # Calculate the Lyapunov exponent with chosen parameters.
        exponent = nolds.lyap_e(series, emb_dim=dim_n, matrix_dim=dim_m)
        return exponent, dim_m, dim_n
    except Exception:
        return np.nan, np.nan, np.nan
    
       
# --- Main Processing Loop ---

for start in colstart:
    print("Initializing columns starting from index:", start)
    end = start + 100
    # DataFrames to store results
    columns_out = ['l_W', 'l_M', 'l_Q', 'l_A', 'm_D', 'm_W', 'm_M', 'm_Q', 'm_A', 
                   'u_W', 'u_M', 'u_Q', 'u_A']
    lyapunov_exp = pd.DataFrame(columns=columns_out)
    lyapunov_dimM = pd.DataFrame(columns=columns_out)
    lyapunov_dimN = pd.DataFrame(columns=columns_out)
    if start==3200:
        data_slice = data.iloc[:, start:]
    else:
        data_slice = data.iloc[:, start:end]
    for column in data_slice.columns:
        # --- Data Preparation ---
        temp_data = data_slice[column].copy()  # Work on a copy of the series
        temp_data.replace(0, np.nan, inplace=True)
        missing_ratio = temp_data.isna().mean()  # fraction of missing values
        if missing_ratio > 0:
            if missing_ratio < 0.05:
                print('Filling missing values in', column)
                temp_data.interpolate(inplace=True)
                temp_data.dropna(inplace=True)
            else:
                print('Skipping', column, 'due to excessive missing values')
                continue
        start_time = time.time()
        # --- Compute Aggregated Series Using Precomputed Indexes ---
        try:
            mean_D = normalize(temp_data.groupby(index_D).mean().to_numpy())
            mean_W = normalize(temp_data.groupby(index_W).mean().to_numpy())
            mean_M = normalize(temp_data.groupby(index_M).mean().to_numpy())
            mean_Q = normalize(temp_data.groupby(index_Q).mean().to_numpy())
            mean_A = normalize(temp_data.groupby(index_A).mean().to_numpy())
 
            up_W = normalize(temp_data.groupby(index_W).max().to_numpy())
            up_M = normalize(temp_data.groupby(index_M).max().to_numpy())
            up_Q = normalize(temp_data.groupby(index_Q).max().to_numpy())
            up_A = normalize(temp_data.groupby(index_A).max().to_numpy())
 
            low_W = normalize(temp_data.groupby(index_W).min().to_numpy())
            low_M = normalize(temp_data.groupby(index_M).min().to_numpy())
            low_Q = normalize(temp_data.groupby(index_Q).min().to_numpy())
            low_A = normalize(temp_data.groupby(index_A).min().to_numpy())
        except:
            try:
                mean_D = normalize(temp_data.groupby(pd.PeriodIndex(temp_data.index, freq="D")).mean().to_numpy())
                mean_W = normalize(temp_data.groupby(pd.PeriodIndex(temp_data.index, freq="W")).mean().to_numpy())
                mean_M = normalize(temp_data.groupby(pd.PeriodIndex(temp_data.index, freq="M")).mean().to_numpy())
                mean_Q = normalize(temp_data.groupby(pd.PeriodIndex(temp_data.index, freq="Q")).mean().to_numpy())
                mean_A = normalize(temp_data.groupby(pd.PeriodIndex(temp_data.index, freq="A")).mean().to_numpy())
 
                up_W = normalize(temp_data.groupby(pd.PeriodIndex(temp_data.index, freq="W")).max().to_numpy())
                up_M = normalize(temp_data.groupby(pd.PeriodIndex(temp_data.index, freq="M")).max().to_numpy())
                up_Q = normalize(temp_data.groupby(pd.PeriodIndex(temp_data.index, freq="Q")).max().to_numpy())
                up_A = normalize(temp_data.groupby(pd.PeriodIndex(temp_data.index, freq="A")).max().to_numpy())
 
                low_W = normalize(temp_data.groupby(pd.PeriodIndex(temp_data.index, freq="W")).min().to_numpy())
                low_M = normalize(temp_data.groupby(pd.PeriodIndex(temp_data.index, freq="M")).min().to_numpy())
                low_Q = normalize(temp_data.groupby(pd.PeriodIndex(temp_data.index, freq="Q")).min().to_numpy())
                low_A = normalize(temp_data.groupby(pd.PeriodIndex(temp_data.index, freq="A")).min().to_numpy())
            except:
                continue
        # List of all aggregated series in the desired order
        series_list = [low_W, low_M, low_Q, low_A, 
                       mean_D, mean_W, mean_M, mean_Q, mean_A, 
                       up_W, up_M, up_Q, up_A]
        # --- Parallelize the Computations for the Aggregated Series ---
        results = Parallel(n_jobs=2)(
            delayed(compute_series)(s) for s in series_list
        )
        # Unzip the results into separate lists
        lex, ldm, ldn = zip(*results)
        # Record the results in the corresponding DataFrames
        lyapunov_exp.loc[column] = lex
        lyapunov_dimM.loc[column] = ldm
        lyapunov_dimN.loc[column] = ldn 
        elapsed = (time.time() - start_time) / 60
        print(f"Finished {column} in: {elapsed:.2f} min")
    # Save results for this batch of columns
    lyapunov_exp.to_csv(out_path + 'lyapunov_exp_' + str(end) + '.csv')
    lyapunov_dimM.to_csv(out_path + 'lyapunov_dimM_' + str(end) + '.csv')
    lyapunov_dimN.to_csv(out_path + 'lyapunov_dimN_' + str(end) + '.csv')